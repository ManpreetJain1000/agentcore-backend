"""Shared fixtures: test database, fake Redis, fake LLM, API client and model factories."""
import os

# Settings are read at import time, so point them at the test database first
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://agentcore:agentcore@localhost:5432/agentcore_test",
)
os.environ["LLM_API_KEY"] = "test-key"

import asyncio
import copy
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import fakeredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

import app.models  # noqa: F401  (registers every table on Base.metadata)
from app.agent.planner import LLMResponse, ToolCall
from app.agent.runner import AgentRunner
from app.core.config import settings
from app.core.security import create_access_token, hash_password
from app.db.base import AsyncSessionLocal, Base, engine
from app.main import create_app
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep
from app.models.memory import EMBEDDING_DIM
from app.models.user import User
from app.tasks.agent_tasks import execute_agent_run
from app.tasks.celery_app import celery_app

_db_name = settings.DATABASE_URL.rsplit("/", 1)[-1]
if "test" not in _db_name:
    raise RuntimeError(f"Tests must run against a test database, got '{_db_name}'")

TEST_PASSWORD = "StrongPass123"
_PASSWORD_HASH = hash_password(TEST_PASSWORD)  # bcrypt is slow; hash once


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run every async test on one session-wide event loop (shared DB pool)."""
    session_loop = pytest.mark.asyncio(loop_scope="session")
    for item in items:
        if pytest_asyncio.is_async_test(item):
            item.add_marker(session_loop, append=False)


# --- Fake LLM -----------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "is", "am", "are", "do", "does", "to",
    "in", "of", "that", "what", "which", "you", "it", "and",
}


def fake_embedding(text_value: str) -> list[float]:
    """Deterministic bag-of-words embedding: texts sharing words are 'similar'."""
    vector = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"[a-z0-9]+", text_value.lower()):
        if word in _STOPWORDS:
            continue
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [x / norm for x in vector]


def text_response(content: str, tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        assistant_message={"role": "assistant", "content": content},
        finish_reason="stop",
        prompt_tokens=tokens,
        completion_tokens=tokens,
    )


def tool_response(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    tool_calls = [
        ToolCall(id=f"call_{uuid4().hex[:8]}", name=name, arguments=json.dumps(args))
        for name, args in calls
    ]
    return LLMResponse(
        content=None,
        tool_calls=tool_calls,
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ],
        },
        finish_reason="tool_calls",
        prompt_tokens=5,
        completion_tokens=5,
    )


class FakeLLM:
    """Stands in for LLMClient: returns queued responses and records every call."""

    model = "fake-model"

    def __init__(self) -> None:
        self.responses: list[LLMResponse | Exception] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[str] = []
        self.fail_embed = False

    def queue(self, *responses: LLMResponse | Exception) -> "FakeLLM":
        self.responses.extend(responses)
        return self

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.chat_calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        if not self.responses:
            return text_response("Fake final answer")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def embed(self, text_value: str) -> list[float]:
        self.embed_calls.append(text_value)
        if self.fail_embed:
            raise RuntimeError("embedding service unavailable")
        return fake_embedding(text_value)

    async def aclose(self) -> None:
        return None


# --- Small helpers ------------------------------------------------------------

def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def reload_run(db: AsyncSession, run_id: str) -> AgentRun:
    return await db.get(
        AgentRun,
        run_id,
        populate_existing=True,
        options=[noload(AgentRun.steps), noload(AgentRun.session)],
    )


async def get_steps(db: AsyncSession, run_id: str) -> list[RunStep]:
    stmt = (
        select(RunStep)
        .where(RunStep.run_id == run_id)
        .order_by(RunStep.occurred_at, RunStep.id)
        .execution_options(populate_existing=True)
    )
    return list((await db.scalars(stmt)).all())


async def drain_events(pubsub, idle_rounds: int = 3) -> list[dict[str, Any]]:
    """Collect every pending pub/sub message as parsed JSON."""
    events: list[dict[str, Any]] = []
    idle = 0
    while idle < idle_rounds:
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
        if message is None:
            idle += 1
            continue
        idle = 0
        events.append(json.loads(message["data"]))
    return events


async def wait_for_subscriber(redis, channel: str, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        counts = await redis.pubsub_numsub(channel)
        if counts and counts[0][1] > 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Nobody subscribed to {channel}")


# --- Infrastructure fixtures --------------------------------------------------

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database() -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_tables() -> AsyncIterator[None]:
    yield
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def celery_mock(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Record enqueue/revoke calls instead of talking to a real broker."""
    calls: dict[str, list[dict[str, Any]]] = {"enqueued": [], "revoked": []}

    def fake_apply_async(args=None, task_id=None, **kwargs):
        calls["enqueued"].append({"args": args, "task_id": task_id})

    def fake_revoke(task_id, **kwargs):
        calls["revoked"].append({"task_id": task_id, **kwargs})

    monkeypatch.setattr(execute_agent_run, "apply_async", fake_apply_async)
    monkeypatch.setattr(celery_app.control, "revoke", fake_revoke)
    return calls


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield redis
    await redis.aclose()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(fake_redis, fake_llm) -> AsyncIterator[AsyncClient]:
    application = create_app()
    application.state.redis = fake_redis
    application.state.llm = fake_llm
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def make_runner(fake_redis, fake_llm):
    """Build AgentRunners on their own DB session, like a real worker."""
    sessions: list[AsyncSession] = []

    def factory(llm: Any = None, **kwargs: Any) -> AgentRunner:
        session = AsyncSessionLocal()
        sessions.append(session)
        return AgentRunner(session, fake_redis, llm or fake_llm, **kwargs)

    yield factory
    for session in sessions:
        await session.close()


# --- Factory fixtures ---------------------------------------------------------

@pytest_asyncio.fixture
async def user_factory(db: AsyncSession):
    async def create(email: str | None = None) -> tuple[User, str]:
        user = User(
            email=email or f"user-{uuid4().hex[:8]}@test.com",
            hashed_password=_PASSWORD_HASH,
        )
        db.add(user)
        await db.commit()
        return user, create_access_token(user.id)

    return create


@pytest_asyncio.fixture
async def session_factory(db: AsyncSession):
    async def create(
        user: User,
        tools: list[str] | None = None,
        system_prompt: str = "You are a test assistant.",
        name: str = "Test session",
    ) -> AgentSession:
        session = AgentSession(
            user_id=user.id,
            name=name,
            system_prompt=system_prompt,
            tools_enabled=list(tools or []),
        )
        db.add(session)
        await db.commit()
        return session

    return create


@pytest_asyncio.fixture
async def run_factory(db: AsyncSession):
    async def create(
        session: AgentSession,
        message: str = "Hello agent",
        status: RunStatus = RunStatus.QUEUED,
        final_answer: str | None = None,
    ) -> AgentRun:
        run = AgentRun(
            session_id=session.id,
            user_message=message,
            status=status,
            final_answer=final_answer,
        )
        db.add(run)
        await db.commit()
        return run

    return create