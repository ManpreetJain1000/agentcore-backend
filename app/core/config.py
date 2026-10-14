from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "AgentCore"
    DEBUG: bool = False

    # Infrastructure
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # LLM provider (any OpenAI-compatible endpoint)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Agent loop
    AGENT_MAX_ITERATIONS: int = 10
    TOOL_TIMEOUT_SECONDS: float = 5.0

    # Memory
    SHORT_TERM_CONTEXT_MESSAGES: int = 10
    SHORT_TERM_MAX_TURNS: int = 20
    LONG_TERM_TOP_K: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()