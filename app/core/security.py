from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenError(Exception):
    """Raised when a JWT is missing, malformed, expired or of the wrong type."""


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT access token. `subject` is the user id."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": subject,
        "iat": now,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Decode and validate an access token. Returns the user id (sub)."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError as exc:  # covers bad signature, expiry, malformed token
        raise TokenError("Invalid or expired token") from exc

    if payload.get("type") != "access":
        raise TokenError("Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise TokenError("Token missing subject")
    return user_id