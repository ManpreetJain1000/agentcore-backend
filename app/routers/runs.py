import json
import logging

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select, update
from starlette.concurrency import run_in_threadpool

from app.agent.runner import run_channel
from app.core.dependencies import DbSession, OwnedRun, OwnedSession, RedisClient
from app.models.agent import AgentRun, RunStatus, RunStep
from app.schemas.agent import RunRequest, RunResponse, RunStatusOut, StepOut
from app.tasks.agent_tasks import execute_agent_run
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])

ACTIVE_STATUSES = (RunStatus.QUEUED, RunStatus.RUNNING)


@router.post(
    "/sessions/{session_id}/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a message; the agent runs in the background",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Session not found"},
        503: {"description": "Task queue unavailable"},
    },
)
async def create_run(payload: RunRequest, session: OwnedSession, db: DbSession) -> RunResponse:
    run = AgentRun(session_id=session.id, user_message=payload.message)
    db.add(run)
    # Commit BEFORE enqueueing so the worker can always find the row
    await db.commit()
    await db.refresh(run, attribute_names=["created_at"])

    try:
        # task_id = run_id, so the run can be revoked by its own id
        await run_in_threadpool(execute_agent_run.apply_async, args=[run.id], task_id=run.id)
    except Exception as exc:  # noqa: BLE001 - broker unreachable
        logger.exception("Failed to enqueue run %s", run.id)
        run.status = RunStatus.FAILED
        db.add(
            RunStep(
                run_id=run.id,
                step_type="error",
                payload={"error": "Task queue unavailable", "type": type(exc).__name__},
            )
        )
        await db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Task queue unavailable, try again later")

    return RunResponse(run_id=run.id, status=run.status, created_at=run.created_at)


@router.get(
    "/runs/{run_id}/status",
    response_model=RunStatusOut,
    summary="Poll the status of a run",
    responses={401: {"description": "Missing or invalid token"}, 404: {"description": "Run not found"}},
)
async def get_run_status(run: OwnedRun, db: DbSession) -> RunStatusOut:
    step_count = await db.scalar(
        select(func.count()).select_from(RunStep).where(RunStep.run_id == run.id)
    )
    return RunStatusOut(
        run_id=run.id,
        status=run.status,
        tokens_used=run.tokens_used,
        step_count=step_count or 0,
        final_answer=run.final_answer,
        created_at=run.created_at,
    )


@router.get(
    "/runs/{run_id}/steps",
    response_model=list[StepOut],
    summary="Full step-by-step trace of a run",
    responses={401: {"description": "Missing or invalid token"}, 404: {"description": "Run not found"}},
)
async def get_run_steps(run: OwnedRun, db: DbSession) -> list[RunStep]:
    stmt = (
        select(RunStep)
        .where(RunStep.run_id == run.id)
        .order_by(RunStep.occurred_at, RunStep.id)
    )
    return list((await db.scalars(stmt)).all())


@router.delete(
    "/runs/{run_id}",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel a queued or running run",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Run not found"},
        409: {"description": "Run already finished"},
    },
)
async def cancel_run(run: OwnedRun, db: DbSession, redis: RedisClient) -> RunResponse:
    run_id = run.id
    created_at = run.created_at
    was_queued = run.status == RunStatus.QUEUED

    # Conditional update: never overwrite a run that finished in the meantime
    result = await db.execute(
        update(AgentRun)
        .where(AgentRun.id == run_id, AgentRun.status.in_(ACTIVE_STATUSES))
        .values(status=RunStatus.CANCELLED)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        await db.rollback()
        current_status = await db.scalar(select(AgentRun.status).where(AgentRun.id == run_id))
        raise HTTPException(status.HTTP_409_CONFLICT, f"Run is already {current_status.value}")
    await db.commit()

    # Stop the worker mid-loop: SIGUSR1 raises SoftTimeLimitExceeded inside the task
    await run_in_threadpool(
        celery_app.control.revoke, run_id, terminate=True, signal="SIGUSR1"
    )

    if was_queued:
        # No worker will pick this up, so close the trace and the stream here
        db.add(RunStep(run_id=run_id, step_type="cancelled", payload={"reason": "Cancelled before start"}))
        await db.commit()
        await redis.publish(
            run_channel(run_id), json.dumps({"step_type": "done", "status": "cancelled"})
        )

    return RunResponse(run_id=run_id, status=RunStatus.CANCELLED, created_at=created_at)