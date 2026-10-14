import asyncio
import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from celery.exceptions import SoftTimeLimitExceeded
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import LockError
from sqlalchemy.exc import InterfaceError, OperationalError

from app.agent.runner import AgentRunner
from app.core.config import settings
from app.db.base import create_worker_sessionmaker
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RUN_LOCK_TTL_SECONDS = 30 * 60

# Errors worth retrying: the next attempt has a real chance of succeeding
TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    RedisConnectionError,
    OperationalError,
    InterfaceError,
)


def _run_lock_key(run_id: str) -> str:
    return f"agentcore:run:{run_id}:lock"


@asynccontextmanager
async def _runner_resources() -> AsyncIterator[tuple[AgentRunner, Redis]]:
    """Fresh DB engine + Redis client bound to the current event loop."""
    engine, session_factory = create_worker_sessionmaker()
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        async with session_factory() as db:
            yield AgentRunner(db, redis), redis
    finally:
        await redis.aclose()
        await engine.dispose()


async def _run(run_id: str) -> str:
    async with _runner_resources() as (runner, redis):
        lock = redis.lock(_run_lock_key(run_id), timeout=RUN_LOCK_TTL_SECONDS)
        if not await lock.acquire(blocking=False):
            logger.info("Run %s is already being executed by another worker, skipping", run_id)
            return "skipped"
        try:
            await runner.run(run_id)
        finally:
            try:
                await lock.release()
            except LockError:
                pass
    return "done"


async def _mark_failed(run_id: str, error: BaseException) -> None:
    async with _runner_resources() as (runner, _):
        await runner.mark_failed(run_id, error)


async def _mark_cancelled(run_id: str) -> None:
    async with _runner_resources() as (runner, _):
        await runner.mark_cancelled(run_id)


async def _record_retry(run_id: str, attempt: int, error: BaseException, countdown: float) -> None:
    async with _runner_resources() as (runner, _):
        await runner.record_retry(run_id, attempt, error, countdown)


@celery_app.task(bind=True, name="agentcore.execute_agent_run", max_retries=MAX_RETRIES)
def execute_agent_run(self, run_id: str) -> str:
    """Execute one agent run in the background."""
    try:
        return asyncio.run(_run(run_id))

    except SoftTimeLimitExceeded:
        # Raised by Celery when the task is revoked with signal=SIGUSR1
        logger.info("Run %s was cancelled while running", run_id)
        asyncio.run(_mark_cancelled(run_id))
        return "cancelled"

    except TRANSIENT_ERRORS as exc:
        attempt = self.request.retries + 1
        if self.request.retries < self.max_retries:
            countdown = min(5 * 2**self.request.retries, 60) + random.uniform(0, 2)
            logger.warning(
                "Run %s hit a transient error (%s); retry %s/%s in %.1fs",
                run_id, type(exc).__name__, attempt, self.max_retries, countdown,
            )
            asyncio.run(_record_retry(run_id, attempt, exc, countdown))
            raise self.retry(exc=exc, countdown=countdown)
        logger.error("Run %s failed after %s retries", run_id, self.max_retries)
        asyncio.run(_mark_failed(run_id, exc))
        return "failed"

    except Exception as exc:  # noqa: BLE001
        logger.exception("Run %s failed", run_id)
        asyncio.run(_mark_failed(run_id, exc))
        return "failed"