"""Integration tests: auth, sessions, runs, cancellation, SSE streaming and the Celery task."""
import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from openai import APIConnectionError

from app.agent.runner import run_channel
from app.core.security import create_access_token
from app.models.agent import RunStatus, RunStep
from app.routers.stream import _event_stream
from app.tasks import agent_tasks
from app.tasks.agent_tasks import execute_agent_run
from app.tests.conftest import (
    TEST_PASSWORD,
    auth_headers,
    drain_events,
    get_steps,
    reload_run,
    text_response,
    wait_for_subscriber,
)


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


# --- Auth -------------------------------------------------------------------------

async def test_register_login_and_me(client):
    register = await client.post("/auth/register", json={"email": "new@test.com", "password": TEST_PASSWORD})
    assert register.status_code == 201
    token = register.json()["access_token"]
    assert register.json()["expires_in"] == 30 * 60

    me = await client.get("/auth/me", headers=auth_headers(token))
    assert me.status_code == 200
    assert me.json()["email"] == "new@test.com"

    duplicate = await client.post("/auth/register", json={"email": "NEW@test.com", "password": TEST_PASSWORD})
    assert duplicate.status_code == 409

    login = await client.post("/auth/token", data={"username": "NEW@test.com", "password": TEST_PASSWORD})
    assert login.status_code == 200

    wrong = await client.post("/auth/token", data={"username": "new@test.com", "password": "WrongPass123"})
    assert wrong.status_code == 401


async def test_invalid_and_expired_tokens_are_rejected(client, user_factory):
    user, _ = await user_factory()
    expired = create_access_token(user.id, expires_minutes=-1)

    assert (await client.get("/auth/me")).status_code == 401
    assert (await client.get("/auth/me", headers=auth_headers("not-a-jwt"))).status_code == 401
    assert (await client.get("/auth/me", headers=auth_headers(expired))).status_code == 401


# --- Sessions -----------------------------------------------------------------------

async def test_create_session_validates_tools(client, user_factory):
    _, token = await user_factory()

    created = await client.post(
        "/sessions",
        json={"name": "Research Assistant", "tools_enabled": ["web_search", "calculator", "web_search"]},
        headers=auth_headers(token),
    )
    assert created.status_code == 201
    assert created.json()["tools_enabled"] == ["web_search", "calculator"]

    rejected = await client.post(
        "/sessions",
        json={"name": "Bad", "tools_enabled": ["web_search", "fly_to_moon"]},
        headers=auth_headers(token),
    )
    assert rejected.status_code == 422
    assert "fly_to_moon" in rejected.text


async def test_sessions_are_isolated_per_user(client, user_factory, session_factory):
    owner, owner_token = await user_factory()
    _, intruder_token = await user_factory()
    session = await session_factory(owner)

    assert len((await client.get("/sessions", headers=auth_headers(owner_token))).json()) == 1
    assert (await client.get("/sessions", headers=auth_headers(intruder_token))).json() == []
    assert (await client.get(f"/sessions/{session.id}", headers=auth_headers(intruder_token))).status_code == 404
    assert (await client.delete(f"/sessions/{session.id}", headers=auth_headers(intruder_token))).status_code == 404


# --- Runs ---------------------------------------------------------------------------

async def test_submit_run_enqueues_task_and_returns_immediately(
    client, db, celery_mock, user_factory, session_factory
):
    user, token = await user_factory()
    session = await session_factory(user)

    response = await client.post(
        f"/sessions/{session.id}/run", json={"message": "Hello"}, headers=auth_headers(token)
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.json()["status"] == "queued"
    assert celery_mock["enqueued"] == [{"args": [run_id], "task_id": run_id}]
    assert (await reload_run(db, run_id)).status == RunStatus.QUEUED


async def test_submit_run_validation_and_ownership(client, user_factory, session_factory):
    owner, owner_token = await user_factory()
    _, intruder_token = await user_factory()
    session = await session_factory(owner)

    blank = await client.post(f"/sessions/{session.id}/run", json={"message": "   "}, headers=auth_headers(owner_token))
    assert blank.status_code == 422

    foreign = await client.post(
        f"/sessions/{session.id}/run", json={"message": "hi"}, headers=auth_headers(intruder_token)
    )
    assert foreign.status_code == 404


async def test_submit_run_returns_503_when_broker_is_down(
    client, db, monkeypatch, user_factory, session_factory
):
    user, token = await user_factory()
    session = await session_factory(user)

    def broken_apply_async(*args, **kwargs):
        raise ConnectionError("broker unreachable")

    monkeypatch.setattr(execute_agent_run, "apply_async", broken_apply_async)

    response = await client.post(f"/sessions/{session.id}/run", json={"message": "hi"}, headers=auth_headers(token))

    assert response.status_code == 503
    runs = (await client.get(f"/sessions/{session.id}", headers=auth_headers(token))).json()["recent_runs"]
    assert runs[0]["status"] == "failed"
    steps = await get_steps(db, runs[0]["id"])
    assert steps[-1].step_type == "error"


async def test_full_run_flow_status_and_steps(client, fake_llm, user_factory, session_factory, make_runner):
    user, token = await user_factory()
    session = await session_factory(user)
    fake_llm.queue(text_response("All done."))

    run_id = (
        await client.post(f"/sessions/{session.id}/run", json={"message": "Do it"}, headers=auth_headers(token))
    ).json()["run_id"]
    await make_runner().run(run_id)  # what the Celery worker would do

    status = (await client.get(f"/runs/{run_id}/status", headers=auth_headers(token))).json()
    assert status["status"] == "completed"
    assert status["final_answer"] == "All done."
    assert status["tokens_used"] == 20
    assert status["step_count"] == 3

    steps = (await client.get(f"/runs/{run_id}/steps", headers=auth_headers(token))).json()
    assert [s["step_type"] for s in steps] == ["context", "llm_call", "final_answer"]

    detail = (await client.get(f"/sessions/{session.id}", headers=auth_headers(token))).json()
    assert detail["recent_runs"][0]["id"] == run_id


async def test_cancel_queued_run(
    client, db, fake_redis, celery_mock, user_factory, session_factory, run_factory, make_runner
):
    user, token = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session)
    pubsub = fake_redis.pubsub()
    await pubsub.subscribe(run_channel(run.id))

    response = await client.delete(f"/runs/{run.id}", headers=auth_headers(token))

    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    assert celery_mock["revoked"] == [{"task_id": run.id, "terminate": True, "signal": "SIGUSR1"}]
    assert (await get_steps(db, run.id))[-1].step_type == "cancelled"
    assert await drain_events(pubsub) == [{"step_type": "done", "status": "cancelled"}]
    await pubsub.aclose()

    # A late worker delivery must not resurrect the run
    await make_runner().run(run.id)
    assert (await reload_run(db, run.id)).status == RunStatus.CANCELLED


async def test_cancel_finished_run_returns_conflict(client, user_factory, session_factory, run_factory):
    user, token = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session, status=RunStatus.COMPLETED, final_answer="done")

    response = await client.delete(f"/runs/{run.id}", headers=auth_headers(token))

    assert response.status_code == 409


async def test_run_endpoints_are_isolated_per_user(client, user_factory, session_factory, run_factory):
    owner, _ = await user_factory()
    _, intruder_token = await user_factory()
    run = await run_factory(await session_factory(owner))
    headers = auth_headers(intruder_token)

    for method, path in [
        ("GET", f"/runs/{run.id}/status"),
        ("GET", f"/runs/{run.id}/steps"),
        ("GET", f"/runs/{run.id}/stream"),
        ("DELETE", f"/runs/{run.id}"),
    ]:
        response = await client.request(method, path, headers=headers)
        assert response.status_code == 404, path


# --- SSE streaming --------------------------------------------------------------------

async def test_stream_replays_finished_run_and_closes(
    client, db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, token = await user_factory()
    run = await run_factory(await session_factory(user))
    fake_llm.queue(text_response("Finished."))
    await make_runner().run(run.id)

    response = await client.get(f"/runs/{run.id}/stream", headers=auth_headers(token))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e["step_type"] for e in events] == ["context", "llm_call", "final_answer", "done"]
    assert events[-1] == {"step_type": "done", "status": "completed"}


async def test_stream_accepts_query_token_and_resumes_after_last_event_id(
    client, db, user_factory, session_factory, run_factory, make_runner
):
    user, token = await user_factory()
    run = await run_factory(await session_factory(user))
    await make_runner().run(run.id)
    first_step = (await get_steps(db, run.id))[0]

    response = await client.get(
        f"/runs/{run.id}/stream",
        params={"access_token": token},
        headers={"Last-Event-ID": first_step.id},
    )

    assert response.status_code == 200
    assert [e["step_type"] for e in parse_sse(response.text)] == ["llm_call", "final_answer", "done"]


async def test_stream_requires_auth(client, user_factory, session_factory, run_factory):
    user, _ = await user_factory()
    run = await run_factory(await session_factory(user))

    assert (await client.get(f"/runs/{run.id}/stream")).status_code == 401


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


async def test_stream_relays_live_events_without_duplicates(
    db, fake_redis, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    run = await run_factory(await session_factory(user), status=RunStatus.RUNNING)
    replayed = RunStep(run_id=run.id, step_type="context", payload={"memories": []})
    db.add(replayed)
    await db.commit()
    channel = run_channel(run.id)

    frames: list[str] = []

    async def consume() -> None:
        async for frame in _event_stream(ConnectedRequest(), fake_redis, run.id, None):
            if not frame.startswith(":"):
                frames.append(frame)

    consumer = asyncio.create_task(consume())
    await wait_for_subscriber(fake_redis, channel)
    await fake_redis.publish(channel, json.dumps({"id": replayed.id, "step_type": "context", "payload": {}}))
    await fake_redis.publish(channel, json.dumps({"id": "live-1", "step_type": "tool_call", "payload": {"name": "calculator"}}))
    await fake_redis.publish(channel, json.dumps({"step_type": "done", "status": "completed"}))
    await asyncio.wait_for(consumer, timeout=5)

    events = parse_sse("".join(frames))
    assert [e["step_type"] for e in events] == ["context", "tool_call", "done"]
    assert events[1]["payload"] == {"name": "calculator"}
    assert await fake_redis.pubsub_numsub(channel) == [(channel, 0)]


# --- Celery task --------------------------------------------------------------------

@pytest.fixture
def worker_resources(monkeypatch, make_runner, fake_redis):
    """Point the task's resource factory at the test DB, fake Redis and fake LLM."""

    @asynccontextmanager
    async def fake_resources():
        yield make_runner(), fake_redis

    monkeypatch.setattr(agent_tasks, "_runner_resources", fake_resources)


async def test_worker_executes_run_and_releases_lock(
    db, fake_redis, worker_resources, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    run = await run_factory(await session_factory(user))

    assert await agent_tasks._run(run.id) == "done"

    assert (await reload_run(db, run.id)).status == RunStatus.COMPLETED
    assert await fake_redis.exists(agent_tasks._run_lock_key(run.id)) == 0


async def test_worker_skips_run_locked_by_another_worker(
    db, fake_redis, fake_llm, worker_resources, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    run = await run_factory(await session_factory(user))
    await fake_redis.set(agent_tasks._run_lock_key(run.id), "another-worker")

    assert await agent_tasks._run(run.id) == "skipped"

    assert (await reload_run(db, run.id)).status == RunStatus.QUEUED
    assert fake_llm.chat_calls == []


async def test_redelivered_finished_run_is_a_no_op(
    db, fake_llm, worker_resources, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    run = await run_factory(await session_factory(user), status=RunStatus.COMPLETED, final_answer="done")

    await agent_tasks._run(run.id)

    assert fake_llm.chat_calls == []
    assert await get_steps(db, run.id) == []


async def test_worker_failure_retry_and_cancel_helpers(
    db, worker_resources, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    session = await session_factory(user)
    failing = await run_factory(session, status=RunStatus.RUNNING)
    cancelling = await run_factory(session, status=RunStatus.RUNNING)

    await agent_tasks._record_retry(failing.id, 1, TimeoutError("slow"), 5.0)
    await agent_tasks._mark_failed(failing.id, ValueError("bad"))
    await agent_tasks._mark_cancelled(cancelling.id)

    assert [s.step_type for s in await get_steps(db, failing.id)] == ["retry", "error"]
    assert (await reload_run(db, failing.id)).status == RunStatus.FAILED
    assert (await reload_run(db, cancelling.id)).status == RunStatus.CANCELLED


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://llm.test/v1/chat/completions"))


def test_task_marks_run_failed_on_unexpected_error(monkeypatch):
    calls: dict = {}

    async def fake_run(run_id):
        raise ValueError("boom")

    async def fake_mark_failed(run_id, error):
        calls["failed"] = (run_id, str(error))

    monkeypatch.setattr(agent_tasks, "_run", fake_run)
    monkeypatch.setattr(agent_tasks, "_mark_failed", fake_mark_failed)

    assert execute_agent_run("run-1") == "failed"
    assert calls["failed"] == ("run-1", "boom")


def test_task_marks_run_cancelled_on_revoke_signal(monkeypatch):
    calls: dict = {}

    async def fake_run(run_id):
        raise SoftTimeLimitExceeded()

    async def fake_mark_cancelled(run_id):
        calls["cancelled"] = run_id

    monkeypatch.setattr(agent_tasks, "_run", fake_run)
    monkeypatch.setattr(agent_tasks, "_mark_cancelled", fake_mark_cancelled)

    assert execute_agent_run("run-2") == "cancelled"
    assert calls["cancelled"] == "run-2"


def test_task_retries_transient_errors(monkeypatch):
    calls: dict = {}

    async def fake_run(run_id):
        raise _connection_error()

    async def fake_record_retry(run_id, attempt, error, countdown):
        calls["retry"] = (run_id, attempt, countdown)

    monkeypatch.setattr(agent_tasks, "_run", fake_run)
    monkeypatch.setattr(agent_tasks, "_record_retry", fake_record_retry)

    # Called directly (not inside a worker), Celery re-raises the original error on retry
    with pytest.raises(APIConnectionError):
        execute_agent_run("run-3")

    run_id, attempt, countdown = calls["retry"]
    assert (run_id, attempt) == ("run-3", 1)
    assert 5 <= countdown <= 7


def test_task_gives_up_after_max_retries(monkeypatch):
    calls: dict = {}

    async def fake_run(run_id):
        raise _connection_error()

    async def fake_mark_failed(run_id, error):
        calls["failed"] = run_id

    monkeypatch.setattr(agent_tasks, "_run", fake_run)
    monkeypatch.setattr(agent_tasks, "_mark_failed", fake_mark_failed)
    monkeypatch.setattr(execute_agent_run, "max_retries", 0)

    assert execute_agent_run("run-4") == "failed"
    assert calls["failed"] == "run-4"