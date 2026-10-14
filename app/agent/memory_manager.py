"""Short-term (Redis) and long-term (pgvector) memory."""
import json
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.agent.planner import LLMClient
from app.core.config import settings
from app.models.memory import LongTermMemory


class ShortTermMemory:
    """Recent chat turns per session, stored as a capped Redis list (newest first).

    Eviction: LTRIM keeps the last N turns, and the key expires after a period
    of inactivity so abandoned sessions don't hold memory forever.
    """

    def __init__(
        self,
        redis: Redis,
        max_turns: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis = redis
        self.max_turns = max_turns or settings.SHORT_TERM_MAX_TURNS
        self.ttl_seconds = ttl_seconds or settings.SHORT_TERM_TTL_SECONDS

    @staticmethod
    def key(session_id: str) -> str:
        return f"agentcore:stm:session:{session_id}"

    async def get_recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        """Return up to `limit` messages in chronological order."""
        if limit <= 0:
            return []
        turns_needed = (limit + 1) // 2
        raw_turns = await self.redis.lrange(self.key(session_id), 0, turns_needed - 1)

        messages: list[dict[str, Any]] = []
        for raw in reversed(raw_turns):  # oldest -> newest
            turn = json.loads(raw)
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages[-limit:]

    async def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self.key(session_id)
        turn = json.dumps({"user": user_message, "assistant": assistant_message})
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.lpush(key, turn)
            pipe.ltrim(key, 0, self.max_turns - 1)
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self.redis.delete(self.key(session_id))


class LongTermMemoryStore:
    """Per-user facts with semantic retrieval via pgvector cosine distance.

    Every query filters on user_id, so one user's memories can never be
    returned for another user.
    """

    def __init__(self, db: AsyncSession, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm

    async def add(
        self, user_id: str, content: str, source_run_id: str | None = None
    ) -> tuple[LongTermMemory, bool]:
        """Store a fact. Returns (memory, created); near-duplicates are not stored again."""
        embedding = await self.llm.embed(content)

        nearest = await self.search_by_embedding_with_scores(user_id, embedding, limit=1)
        if nearest and nearest[0][1] <= settings.LONG_TERM_DEDUP_DISTANCE:
            return nearest[0][0], False

        memory = LongTermMemory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            source_run_id=source_run_id,
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory, attribute_names=["created_at"])
        return memory, True

    async def list_for_user(self, user_id: str, limit: int, offset: int) -> list[LongTermMemory]:
        stmt = (
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)
            .options(defer(LongTermMemory.embedding))  # don't ship 1536 floats per row
            .order_by(LongTermMemory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.db.scalars(stmt)).all())

    async def search_by_embedding_with_scores(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[tuple[LongTermMemory, float]]:
        """Nearest memories with their cosine distance (0 = identical, 2 = opposite)."""
        distance = LongTermMemory.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(LongTermMemory, distance)
            .where(LongTermMemory.user_id == user_id)
            .options(defer(LongTermMemory.embedding))
            .order_by(distance)
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).all()
        return [(row[0], float(row[1])) for row in rows]

    async def search_by_embedding(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[LongTermMemory]:
        results = await self.search_by_embedding_with_scores(user_id, embedding, limit)
        return [memory for memory, _ in results]

    async def search_with_scores(
        self, user_id: str, query: str, limit: int
    ) -> list[tuple[LongTermMemory, float]]:
        embedding = await self.llm.embed(query)
        return await self.search_by_embedding_with_scores(user_id, embedding, limit)

    async def delete(self, user_id: str, memory_id: str) -> bool:
        result = await self.db.execute(
            delete(LongTermMemory).where(
                LongTermMemory.id == memory_id,
                LongTermMemory.user_id == user_id,
            )
        )
        await self.db.commit()
        return result.rowcount > 0