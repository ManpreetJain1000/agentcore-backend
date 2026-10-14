"""Thin wrapper around an OpenAI-compatible chat + embeddings API."""
import math
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

from app.core.config import settings
from app.models.memory import EMBEDDING_DIM


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string produced by the model


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message exactly as it must be appended to the conversation
    assistant_message: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class LLMClient:
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL or None,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,  # SDK retries 429/5xx with backoff
        )
        self.model = model or settings.LLM_MODEL
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**params)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for tc in message.tool_calls or []:
            function = getattr(tc, "function", None)
            if function is None:
                continue
            call_id = tc.id or f"call_{uuid4().hex[:12]}"
            tool_calls.append(
                ToolCall(id=call_id, name=function.name, arguments=function.arguments or "{}")
            )
            # model_dump keeps provider-specific extras (e.g. Gemini thought signatures)
            raw = tc.model_dump(exclude_none=True)
            raw["id"] = call_id
            raw_tool_calls.append(raw)

        assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
        if raw_tool_calls:
            assistant_message["tool_calls"] = raw_tool_calls

        usage = response.usage
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        vector = list(response.data[0].embedding)
        if len(vector) > EMBEDDING_DIM:
            # Matryoshka-style truncation, then re-normalise
            vector = _l2_normalize(vector[:EMBEDDING_DIM])
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions, expected {EMBEDDING_DIM}"
            )
        return vector