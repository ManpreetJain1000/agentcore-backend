"""Short-term (Redis) and long-term (pgvector) memory."""
import json
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import LLMClient
from app.core.config import settings
from app.models.memory import LongTermMemory


class ShortTermMemory:
    """Recent chat turns per session, stored as a capped Redis list (newest first)."""

    def __init__(self, redis: Redis, max_turns: int | None = None) -> None:
        self.redis = redis
        self.max_turns = max_turns or settings.SHORT_TERM_MAX_TURNS

    @staticmethod
    def key(session_id: str) -> str:
        return f"agentcore:stm:session:{session_id}"

    async def get_recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        """Return up to `limit` messages in chronological order."""
        turns_needed = (limit + 1) // 2
        raw_turns = await self.redis.lrange(self.key(session_id), 0, turns_needed - 1)

        messages: list[dict[str, Any]] = []
        for raw in reversed(raw_turns):  # oldest -> newest
            turn = json.loads(raw)
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages[-limit:] if limit else []

    async def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self.key(session_id)
        turn = json.dumps({"user": user_message, "assistant": assistant_message})
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.lpush(key, turn)
            pipe.ltrim(key, 0, self.max_turns - 1)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self.redis.delete(self.key(session_id))


class LongTermMemoryStore:
    """Per-user facts with semantic retrieval via pgvector cosine distance."""

    def __init__(self, db: AsyncSession, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm

    async def add(self, user_id: str, content: str, source_run_id: str | None = None) -> LongTermMemory:
        embedding = await self.llm.embed(content)
        memory = LongTermMemory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            source_run_id=source_run_id,
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory, attribute_names=["created_at"])
        return memory

    async def search_by_embedding(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[LongTermMemory]:
        stmt = (
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)  # user isolation
            .order_by(LongTermMemory.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        return list((await self.db.scalars(stmt)).all())

    async def search(self, user_id: str, query: str, limit: int) -> list[LongTermMemory]:
        embedding = await self.llm.embed(query)
        return await self.search_by_embedding(user_id, embedding, limit)