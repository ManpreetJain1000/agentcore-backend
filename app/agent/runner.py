"""ReAct-style agent loop: build context -> LLM -> tools -> observe -> repeat -> answer."""
import asyncio
import json
import logging
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, noload

from app.agent.memory_manager import LongTermMemoryStore, ShortTermMemory
from app.agent.planner import LLMClient, ToolCall
from app.agent.tools import ToolContext, dispatch_tool, get_tool_definitions, get_tool_timeout
from app.core.config import settings
from app.models.agent import AgentRun, AgentSession, RunStatus, RunStep

logger = logging.getLogger(__name__)

MAX_ITERATIONS_MESSAGE = "Max iterations reached"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


def run_channel(run_id: str) -> str:
    """Redis pub/sub channel that carries the live events of one run."""
    return f"agentcore:run:{run_id}:events"


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))


class AgentRunner:
    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
        llm: LLMClient | None = None,
        *,
        max_iterations: int | None = None,
        tool_timeout: float | None = None,
    ) -> None:
        self.db = db
        self.redis = redis
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations or settings.AGENT_MAX_ITERATIONS
        self.tool_timeout = tool_timeout or settings.TOOL_TIMEOUT_SECONDS
        self.short_term = ShortTermMemory(redis)
        self.long_term = LongTermMemoryStore(db, self.llm)

    # ------------------------------------------------------------------ public

    async def run(self, run_id: str) -> AgentRun | None:
        """Execute a run end-to-end. Raises on unexpected errors (see mark_failed)."""
        run = await self._load_run(run_id)
        if run is None:
            logger.warning("Run %s not found", run_id)
            return None
        if run.status in TERMINAL_STATUSES:
            logger.info("Run %s already %s, skipping", run_id, run.status.value)
            return run

        session = run.session
        run.status = RunStatus.RUNNING
        await self.db.commit()
        await self._publish(run.id, {"step_type": "status", "payload": {"status": "running"}})

        # Step 1: build context
        messages = await self._build_context(run, session)
        tool_defs = get_tool_definitions(session.tools_enabled)
        ctx = ToolContext(
            user_id=session.user_id,
            session_id=session.id,
            run_id=run.id,
            db=self.db,
            llm=self.llm,
        )

        final_answer = MAX_ITERATIONS_MESSAGE
        iterations = 0
        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            if await self._is_cancelled(run):
                await self.mark_cancelled(run.id)
                return run

            # Step 2: LLM call
            response = await self.llm.chat(messages, tools=tool_defs or None)
            run.tokens_used = (run.tokens_used or 0) + response.total_tokens
            await self._record_step(
                run.id,
                "llm_call",
                {
                    "iteration": iteration,
                    "model": self.llm.model,
                    "content": response.content,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in response.tool_calls
                    ],
                    "finish_reason": response.finish_reason,
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                },
            )

            # Plain text answer -> done
            if not response.tool_calls:
                final_answer = response.content or ""
                break

            # Step 3: tool dispatch
            messages.append(response.assistant_message)
            for call in response.tool_calls:
                await self._record_step(
                    run.id,
                    "tool_call",
                    {
                        "iteration": iteration,
                        "tool_call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                )
                result = await self._execute_tool(run.id, call, session, ctx)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        # Step 5: save
        run.final_answer = final_answer
        run.status = RunStatus.COMPLETED
        await self._record_step(
            run.id, "final_answer", {"content": final_answer, "iterations": iterations}
        )

        try:
            await self.short_term.append_turn(session.id, run.user_message, final_answer)
        except Exception:  # noqa: BLE001 - memory write must not fail a completed run
            logger.exception("Failed to append short-term memory for run %s", run.id)

        await self._publish(run.id, {"step_type": "done", "status": "completed"})
        return run

    async def mark_failed(self, run_id: str, error: BaseException | str) -> None:
        """Mark a run FAILED, store the error as a RunStep and notify listeners."""
        await self.db.rollback()
        run = await self.db.get(
            AgentRun, run_id, options=[noload(AgentRun.steps), noload(AgentRun.session)]
        )
        if run is None or run.status in TERMINAL_STATUSES:
            return

        run.status = RunStatus.FAILED
        await self._record_step(
            run.id,
            "error",
            {
                "error": str(error),
                "type": type(error).__name__ if isinstance(error, BaseException) else "Error",
            },
        )
        await self._publish(run.id, {"step_type": "done", "status": "failed"})

    async def mark_cancelled(self, run_id: str) -> None:
        """Mark a run CANCELLED, store a step and notify listeners."""
        await self.db.rollback()
        run = await self.db.get(
            AgentRun, run_id, options=[noload(AgentRun.steps), noload(AgentRun.session)]
        )
        if run is None or run.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            return

        run.status = RunStatus.CANCELLED
        await self._record_step(run.id, "cancelled", {"reason": "Cancelled by user"})
        await self._publish(run.id, {"step_type": "done", "status": "cancelled"})

    async def record_retry(
        self, run_id: str, attempt: int, error: BaseException, countdown: float
    ) -> None:
        """Record that the run hit a transient error and will be retried."""
        await self._record_step(
            run_id,
            "retry",
            {
                "attempt": attempt,
                "error": str(error),
                "type": type(error).__name__,
                "retry_in_seconds": round(countdown, 1),
            },
        )

    # ----------------------------------------------------------------- helpers

    async def _load_run(self, run_id: str) -> AgentRun | None:
        stmt = (
            select(AgentRun)
            .where(AgentRun.id == run_id)
            .options(
                noload(AgentRun.steps),
                joinedload(AgentRun.session).noload(AgentSession.runs),
            )
        )
        return await self.db.scalar(stmt)

    async def _build_context(self, run: AgentRun, session: AgentSession) -> list[dict[str, Any]]:
        try:
            history = await self.short_term.get_recent_messages(
                session.id, settings.SHORT_TERM_CONTEXT_MESSAGES
            )
        except Exception:  # noqa: BLE001
            logger.exception("Short-term memory unavailable for session %s", session.id)
            history = []

        memories = []
        try:
            query_embedding = await self.llm.embed(run.user_message)
        except Exception:  # noqa: BLE001
            logger.exception("Embedding failed; continuing without long-term memory")
            query_embedding = None
        if query_embedding is not None:
            memories = await self.long_term.search_by_embedding(
                session.user_id, query_embedding, settings.LONG_TERM_TOP_K
            )

        system_prompt = (session.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        if memories:
            facts = "\n".join(f"- {m.content}" for m in memories)
            system_prompt += f"\n\nRelevant facts you remember about this user:\n{facts}"

        await self._record_step(
            run.id,
            "context",
            {
                "history_messages": len(history),
                "memories": [m.content for m in memories],
                "tools": list(session.tools_enabled),
            },
        )

        return [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": run.user_message},
        ]

    async def _execute_tool(
        self, run_id: str, call: ToolCall, session: AgentSession, ctx: ToolContext
    ) -> str:
        base = {"tool_call_id": call.id, "name": call.name}

        # Guard against hallucinated or disabled tools
        if call.name not in session.tools_enabled:
            result = (
                f"Error: tool '{call.name}' is not available. "
                f"Available tools: {', '.join(session.tools_enabled) or 'none'}."
            )
            await self._record_step(run_id, "tool_result", {**base, "result": result, "error": True})
            return result

        timeout = get_tool_timeout(call.name) or self.tool_timeout
        try:
            result = await asyncio.wait_for(
                dispatch_tool(call.name, call.arguments, ctx), timeout=timeout
            )
        except TimeoutError:
            result = f"Error: tool '{call.name}' timed out after {timeout:g}s."
            await self._record_step(
                run_id, "tool_timeout", {**base, "timeout_seconds": timeout}
            )
            return result

        await self._record_step(
            run_id,
            "tool_result",
            {**base, "result": result, "error": result.startswith("Error:")},
        )
        return result

    async def _is_cancelled(self, run: AgentRun) -> bool:
        await self.db.refresh(run, attribute_names=["status"])
        return run.status == RunStatus.CANCELLED

    async def _record_step(self, run_id: str, step_type: str, payload: dict[str, Any]) -> RunStep:
        """Persist a step (own transaction) and publish it for live streaming."""
        payload = _jsonable(payload)
        step = RunStep(run_id=run_id, step_type=step_type, payload=payload)
        self.db.add(step)
        await self.db.commit()
        await self._publish(run_id, {"step_type": step_type, "payload": payload})
        return step

    async def _publish(self, run_id: str, event: dict[str, Any]) -> None:
        try:
            await self.redis.publish(run_channel(run_id), json.dumps(event, default=str))
        except Exception:  # noqa: BLE001 - streaming is best-effort; the DB is the source of truth
            logger.exception("Failed to publish event for run %s", run_id)