import logging
from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import select
from sqlalchemy.orm import noload

from app.agent.memory_manager import ShortTermMemory
from app.core.dependencies import CurrentUser, DbSession, OwnedSession, RedisClient
from app.models.agent import AgentRun, AgentSession
from app.schemas.agent import (
    AgentSessionCreate,
    AgentSessionDetail,
    AgentSessionOut,
    RunSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

RECENT_RUNS_LIMIT = 10


@router.get(
    "",
    response_model=list[AgentSessionOut],
    summary="List the caller's agent sessions",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_sessions(
    current_user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AgentSession]:
    stmt = (
        select(AgentSession)
        .where(AgentSession.user_id == current_user.id)
        .options(noload(AgentSession.runs))
        .order_by(AgentSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.scalars(stmt)
    return list(result.all())


@router.post(
    "",
    response_model=AgentSessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent session",
    responses={
        401: {"description": "Missing or invalid token"},
        422: {"description": "Validation error (e.g. unknown tool in tools_enabled)"},
    },
)
async def create_session(
    payload: AgentSessionCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> AgentSession:
    session = AgentSession(
        user_id=current_user.id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        tools_enabled=payload.tools_enabled,
    )
    db.add(session)
    await db.commit()
    # created_at is set by the DB; load just that column (async can't lazy-load)
    await db.refresh(session, attribute_names=["created_at"])
    return session


@router.get(
    "/{session_id}",
    response_model=AgentSessionDetail,
    summary="Get a session with its most recent runs",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Session not found"},
    },
)
async def get_session(session: OwnedSession, db: DbSession) -> AgentSessionDetail:
    stmt = (
        select(AgentRun)
        .where(AgentRun.session_id == session.id)
        .options(noload(AgentRun.steps))
        .order_by(AgentRun.created_at.desc())
        .limit(RECENT_RUNS_LIMIT)
    )
    runs = (await db.scalars(stmt)).all()

    return AgentSessionDetail(
        **AgentSessionOut.model_validate(session).model_dump(),
        recent_runs=[RunSummary.model_validate(run) for run in runs],
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session, all of its runs and its short-term memory",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Session not found"},
    },
)
async def delete_session(session: OwnedSession, db: DbSession, redis: RedisClient) -> None:
    session_id = session.id
    # Runs and steps are removed by ON DELETE CASCADE in Postgres
    await db.delete(session)
    await db.commit()

    try:
        await ShortTermMemory(redis).clear(session_id)
    except Exception:  # noqa: BLE001 - the session is gone; a stale Redis key will expire on its own
        logger.warning("Could not clear short-term memory for session %s", session_id, exc_info=True)