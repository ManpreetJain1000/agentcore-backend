from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from app.core.security import TokenError, decode_access_token
from app.db.base import AsyncSessionLocal
from app.models.agent import AgentRun, AgentSession
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """One DB session per request, always closed afterwards."""
    async with AsyncSessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_redis(request: Request) -> Redis:
    """Shared Redis client created in the app lifespan."""
    return request.app.state.redis


RedisClient = Annotated[Redis, Depends(get_redis)]


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DbSession,
) -> User:
    """Decode the bearer token and inject the matching active User."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_access_token(token)
    except TokenError:
        raise credentials_exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_owned_session(
    session_id: str,
    current_user: CurrentUser,
    db: DbSession,
) -> AgentSession:
    """Load a session that belongs to the caller, or 404."""
    stmt = (
        select(AgentSession)
        .where(AgentSession.id == session_id, AgentSession.user_id == current_user.id)
        .options(noload(AgentSession.runs))
    )
    session = await db.scalar(stmt)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session


OwnedSession = Annotated[AgentSession, Depends(get_owned_session)]


async def get_owned_run(
    run_id: str,
    current_user: CurrentUser,
    db: DbSession,
) -> AgentRun:
    """Load a run whose session belongs to the caller, or 404."""
    stmt = (
        select(AgentRun)
        .join(AgentSession, AgentRun.session_id == AgentSession.id)
        .where(AgentRun.id == run_id, AgentSession.user_id == current_user.id)
        .options(noload(AgentRun.steps), noload(AgentRun.session))
    )
    run = await db.scalar(stmt)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run


OwnedRun = Annotated[AgentRun, Depends(get_owned_run)]