import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import select

from app.agent.runner import run_channel
from app.core.dependencies import DbSession, RedisClient, StreamUser, load_owned_run
from app.db.base import AsyncSessionLocal
from app.models.agent import AgentRun, RunStatus, RunStep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

HEARTBEAT_SECONDS = 15.0
TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


def _sse(event: dict[str, Any], event_id: str | None = None) -> str:
    """Format one Server-Sent Events frame."""
    frame = f"id: {event_id}\n" if event_id else ""
    return f"{frame}data: {json.dumps(event, default=str)}\n\n"


def _step_frame(step: RunStep) -> str:
    return _sse({"step_type": step.step_type, "payload": step.payload}, step.id)


def _done_frame(status: str) -> str:
    return _sse({"step_type": "done", "status": status})


async def _load_snapshot(run_id: str) -> tuple[RunStatus | None, list[RunStep]]:
    """Current status + persisted steps, using a short-lived DB session."""
    async with AsyncSessionLocal() as db:
        status = await db.scalar(select(AgentRun.status).where(AgentRun.id == run_id))
        steps = (
            await db.scalars(
                select(RunStep)
                .where(RunStep.run_id == run_id)
                .order_by(RunStep.occurred_at, RunStep.id)
            )
        ).all()
    return status, list(steps)


async def _event_stream(
    request: Request,
    redis: Redis,
    run_id: str,
    last_event_id: str | None,
) -> AsyncIterator[str]:
    channel = run_channel(run_id)
    pubsub = redis.pubsub()
    sent: set[str] = set()

    # Subscribe BEFORE reading the DB so no event can fall into the gap
    await pubsub.subscribe(channel)
    try:
        # 1. Replay what already happened
        status, steps = await _load_snapshot(run_id)
        skipping = last_event_id is not None and any(s.id == last_event_id for s in steps)
        for step in steps:
            sent.add(step.id)
            if skipping:
                if step.id == last_event_id:
                    skipping = False
                continue
            yield _step_frame(step)

        if status is None or status in TERMINAL_STATUSES:
            yield _done_frame(status.value if status else "failed")
            return

        # 2. Relay live events
        while True:
            if await request.is_disconnected():
                return

            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=HEARTBEAT_SECONDS
            )

            if message is None:
                # Heartbeat keeps proxies from closing the connection, and the
                # DB check closes the stream even if a 'done' event was lost
                yield ": ping\n\n"
                status, steps = await _load_snapshot(run_id)
                for step in steps:
                    if step.id not in sent:
                        sent.add(step.id)
                        yield _step_frame(step)
                if status is None or status in TERMINAL_STATUSES:
                    yield _done_frame(status.value if status else "failed")
                    return
                continue

            event = json.loads(message["data"])
            step_id = event.get("id")
            if step_id:
                if step_id in sent:
                    continue
                sent.add(step_id)

            if event.get("step_type") == "done":
                yield _done_frame(event.get("status", "completed"))
                return

            yield _sse(
                {"step_type": event.get("step_type"), "payload": event.get("payload", {})},
                step_id,
            )

    except asyncio.CancelledError:
        logger.debug("Client disconnected from stream of run %s", run_id)
        raise
    finally:
        # Shield cleanup so it still runs when the request is being cancelled
        with anyio.CancelScope(shield=True):
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("Error while closing pubsub for run %s", run_id, exc_info=True)


@router.get(
    "/runs/{run_id}/stream",
    response_class=StreamingResponse,
    summary="Stream run steps in real time (Server-Sent Events)",
    responses={
        200: {
            "description": "SSE stream. Each frame: `data: {\"step_type\": ..., \"payload\": {...}}`. "
            "The stream ends with `data: {\"step_type\": \"done\", \"status\": ...}`.",
            "content": {"text/event-stream": {}},
        },
        401: {"description": "Missing or invalid token"},
        404: {"description": "Run not found"},
    },
)
async def stream_run(
    run_id: str,
    request: Request,
    current_user: StreamUser,
    db: DbSession,
    redis: RedisClient,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    run = await load_owned_run(db, run_id, current_user.id)
    return StreamingResponse(
        _event_stream(request, redis, run.id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable buffering behind nginx
        },
    )