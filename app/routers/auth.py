from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.dependencies import CurrentUser, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.user import TokenResponse, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_for(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user and get an access token",
    responses={409: {"description": "Email already registered"}},
)
async def register(payload: UserCreate, db: DbSession) -> TokenResponse:
    email = payload.email.lower()

    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    # bcrypt is CPU-heavy; run it off the event loop
    hashed = await run_in_threadpool(hash_password, payload.password)
    user = User(email=email, hashed_password=hashed)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:  # race: same email registered concurrently
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    return _token_for(user)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Log in with email + password (OAuth2 form)",
    responses={401: {"description": "Incorrect email or password"}},
)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
) -> TokenResponse:
    # OAuth2 form uses the field name "username"; we treat it as the email
    user = await db.scalar(select(User).where(User.email == form.username.lower()))

    valid = user is not None and await run_in_threadpool(
        verify_password, form.password, user.hashed_password
    )
    if not valid or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_for(user)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Return the currently authenticated user",
    responses={401: {"description": "Missing or invalid token"}},
)
async def me(current_user: CurrentUser) -> User:
    return current_user