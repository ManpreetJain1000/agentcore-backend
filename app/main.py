from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.base import engine
from app.routers import auth, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown: close DB connection pool cleanly
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="AgentCore — AI agent sessions with tool calling, memory and streaming.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(sessions.router)

    @app.get("/health", tags=["health"], summary="Liveness check")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()