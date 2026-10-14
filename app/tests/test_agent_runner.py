"""Unit tests for the agent loop, the LLM wrapper and the tool dispatcher."""
import asyncio
import json
import math
from datetime import datetime
from types import SimpleNamespace

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from app.agent.memory_manager import LongTermMemoryStore, ShortTermMemory
from app.agent.planner import LLMClient
from app.agent.runner import MAX_ITERATIONS_MESSAGE, run_channel
from app.agent.tools import TOOL_REGISTRY, ToolContext, _build_tool, dispatch_tool
from app.models.agent import RunStatus
from app.models.memory import EMBEDDING_DIM
from app.tests.conftest import drain_events, get_steps, reload_run, text_response, tool_response


@pytest.fixture
def ctx(fake_llm) -> ToolContext:
    return ToolContext(user_id="user", session_id="session", run_id="run", db=None, llm=fake_llm)


# --- Agent loop -----------------------------------------------------------------

async def test_plain_answer_completes_run(
    db, fake_redis, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session, message="Say hello")
    fake_llm.queue(text_response("Hello!"))

    await make_runner().run(run.id)

    saved = await reload_run(db, run.id)
    assert saved.status == RunStatus.COMPLETED
    assert saved.final_answer == "Hello!"
    assert saved.tokens_used == 20
    assert [s.step_type for s in await get_steps(db, run.id)] == ["context", "llm_call", "final_answer"]
    assert fake_llm.chat_calls[0]["tools"] is None

    history = await ShortTermMemory(fake_redis).get_recent_messages(session.id, 10)
    assert history == [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello!"},
    ]


async def test_tool_result_is_fed_back_to_the_llm(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user, tools=["calculator"])
    run = await run_factory(session, message="What is 15% of 3.8 million?")
    fake_llm.queue(
        tool_response(("calculator", {"expression": "0.15 * 3800000"})),
        text_response("It is 570,000."),
    )

    await make_runner().run(run.id)

    steps = await get_steps(db, run.id)
    assert [s.step_type for s in steps] == [
        "context", "llm_call", "tool_call", "tool_result", "llm_call", "final_answer",
    ]
    assert steps[3].payload["result"] == "570000"
    assert steps[3].payload["error"] is False

    definitions = fake_llm.chat_calls[0]["tools"]
    assert [d["function"]["name"] for d in definitions] == ["calculator"]

    second_call = fake_llm.chat_calls[1]["messages"]
    assert second_call[-2]["role"] == "assistant"
    assert second_call[-2]["tool_calls"]
    assert second_call[-1] == {
        "role": "tool",
        "tool_call_id": steps[2].payload["tool_call_id"],
        "content": "570000",
    }
    assert (await reload_run(db, run.id)).final_answer == "It is 570,000."


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [("delete_database", {}), ("web_search", {"query": "x"})],
    ids=["hallucinated_tool", "tool_not_enabled"],
)
async def test_unavailable_tools_are_not_executed(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner, tool_name, arguments
):
    user, _ = await user_factory()
    session = await session_factory(user, tools=["calculator"])
    run = await run_factory(session)
    fake_llm.queue(tool_response((tool_name, arguments)), text_response("Sorry, I can't do that."))

    await make_runner().run(run.id)

    result_step = next(s for s in await get_steps(db, run.id) if s.step_type == "tool_result")
    assert result_step.payload["error"] is True
    assert "is not available" in result_step.payload["result"]
    assert "calculator" in result_step.payload["result"]
    assert (await reload_run(db, run.id)).status == RunStatus.COMPLETED


async def test_loop_stops_at_max_iterations(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user, tools=["calculator"])
    run = await run_factory(session)
    fake_llm.queue(*[tool_response(("calculator", {"expression": "1+1"})) for _ in range(3)])

    await make_runner(max_iterations=3).run(run.id)

    saved = await reload_run(db, run.id)
    assert saved.final_answer == MAX_ITERATIONS_MESSAGE
    assert saved.status == RunStatus.COMPLETED
    steps = await get_steps(db, run.id)
    assert sum(s.step_type == "llm_call" for s in steps) == 3
    assert steps[-1].payload == {"content": MAX_ITERATIONS_MESSAGE, "iterations": 3}


async def test_slow_tool_times_out_and_loop_continues(
    db, fake_llm, monkeypatch, user_factory, session_factory, run_factory, make_runner
):
    async def slow_tool() -> str:
        """Takes far too long."""
        await asyncio.sleep(1)
        return "finished"

    monkeypatch.setitem(TOOL_REGISTRY, "slow_tool", _build_tool(slow_tool, 0.05))
    user, _ = await user_factory()
    session = await session_factory(user, tools=["slow_tool"])
    run = await run_factory(session)
    fake_llm.queue(tool_response(("slow_tool", {})), text_response("The tool was too slow."))

    await make_runner().run(run.id)

    step_types = [s.step_type for s in await get_steps(db, run.id)]
    assert "tool_timeout" in step_types
    assert step_types[-1] == "final_answer"
    tool_message = fake_llm.chat_calls[1]["messages"][-1]
    assert tool_message["content"].startswith("Error: tool 'slow_tool' timed out")


async def test_finished_run_is_not_executed_again(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session, status=RunStatus.COMPLETED, final_answer="Done before")

    await make_runner().run(run.id)

    assert fake_llm.chat_calls == []
    assert await get_steps(db, run.id) == []
    assert (await reload_run(db, run.id)).final_answer == "Done before"


async def test_cancellation_is_checked_between_iterations(
    db, fake_llm, monkeypatch, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user, tools=["calculator"])
    run = await run_factory(session)
    fake_llm.queue(tool_response(("calculator", {"expression": "2+2"})), text_response("never sent"))

    runner = make_runner()
    checks = iter([False, True])

    async def fake_is_cancelled(_run):
        return next(checks)

    monkeypatch.setattr(runner, "_is_cancelled", fake_is_cancelled)
    await runner.run(run.id)

    assert (await reload_run(db, run.id)).status == RunStatus.CANCELLED
    assert len(fake_llm.chat_calls) == 1
    assert (await get_steps(db, run.id))[-1].step_type == "cancelled"


async def test_llm_failure_marks_run_failed(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session)
    fake_llm.queue(RuntimeError("provider exploded"))

    runner = make_runner()
    with pytest.raises(RuntimeError):
        await runner.run(run.id)
    await runner.mark_failed(run.id, RuntimeError("provider exploded"))

    assert (await reload_run(db, run.id)).status == RunStatus.FAILED
    error_step = (await get_steps(db, run.id))[-1]
    assert error_step.step_type == "error"
    assert error_step.payload == {"error": "provider exploded", "type": "RuntimeError"}


async def test_context_contains_history_and_only_own_memories(
    db, fake_redis, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    other_user, _ = await user_factory()
    session = await session_factory(user, system_prompt="You are Research Assistant.")
    await ShortTermMemory(fake_redis).append_turn(session.id, "Hi there", "Hello! How can I help?")
    store = LongTermMemoryStore(db, fake_llm)
    await store.add(user.id, "My favourite programming language is Python")
    await store.add(other_user.id, "My favourite programming language is Rust")
    run = await run_factory(session, message="What is my favourite programming language?")

    await make_runner().run(run.id)

    messages = fake_llm.chat_calls[0]["messages"]
    system = messages[0]["content"]
    assert system.startswith("You are Research Assistant.")
    assert "Python" in system
    assert "Rust" not in system
    assert messages[1:3] == [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    assert messages[-1] == {"role": "user", "content": "What is my favourite programming language?"}


async def test_embedding_failure_does_not_fail_the_run(
    db, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session)
    fake_llm.fail_embed = True

    await make_runner().run(run.id)

    assert (await reload_run(db, run.id)).status == RunStatus.COMPLETED
    context = (await get_steps(db, run.id))[0]
    assert context.payload["memories"] == []


async def test_every_step_is_published_to_redis(
    fake_redis, fake_llm, user_factory, session_factory, run_factory, make_runner
):
    user, _ = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session)
    pubsub = fake_redis.pubsub()
    await pubsub.subscribe(run_channel(run.id))

    await make_runner().run(run.id)

    events = await drain_events(pubsub)
    await pubsub.aclose()
    assert [e["step_type"] for e in events] == ["status", "context", "llm_call", "final_answer", "done"]
    assert events[-1] == {"step_type": "done", "status": "completed"}
    assert all("id" in e for e in events[1:-1])


# --- LLM wrapper ----------------------------------------------------------------

class FakeOpenAIClient:
    """Mimics the parts of AsyncOpenAI that LLMClient uses."""

    def __init__(self, completion=None, embedding=None) -> None:
        self.completion = completion
        self.embedding = embedding
        self.chat_kwargs: dict | None = None
        self.embed_kwargs: dict | None = None
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    async def _create(self, **kwargs):
        self.chat_kwargs = kwargs
        return self.completion

    async def _embed(self, **kwargs):
        self.embed_kwargs = kwargs
        return self.embedding

    async def close(self) -> None:
        self.closed = True


def _completion(message: dict, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
    )


def _embedding_response(values: list[float]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse.model_validate(
        {
            "object": "list",
            "model": "fake",
            "data": [{"object": "embedding", "index": 0, "embedding": values}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )


async def test_llm_client_parses_tool_calls_and_keeps_provider_extras():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
                "extra_content": {"google": {"thought_signature": "sig-123"}},
            }
        ],
    }
    fake_client = FakeOpenAIClient(completion=_completion(message, "tool_calls"))
    llm = LLMClient(client=fake_client, model="fake")

    response = await llm.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "calculator"}}],
    )

    assert fake_client.chat_kwargs["tool_choice"] == "auto"
    call = response.tool_calls[0]
    assert call.id.startswith("call_")
    assert call.name == "calculator"
    assert json.loads(call.arguments) == {"expression": "1+1"}
    raw = response.assistant_message["tool_calls"][0]
    assert raw["id"] == call.id
    assert raw["extra_content"] == {"google": {"thought_signature": "sig-123"}}
    assert response.total_tokens == 7


async def test_llm_client_plain_answer_sends_no_tools():
    fake_client = FakeOpenAIClient(completion=_completion({"role": "assistant", "content": "Hi!"}))
    llm = LLMClient(client=fake_client, model="fake")

    response = await llm.chat([{"role": "user", "content": "hi"}])

    assert "tools" not in fake_client.chat_kwargs
    assert response.content == "Hi!"
    assert response.tool_calls == []
    assert response.assistant_message == {"role": "assistant", "content": "Hi!"}


async def test_llm_client_truncates_and_normalises_large_embeddings():
    fake_client = FakeOpenAIClient(embedding=_embedding_response([1.0] * 3072))
    llm = LLMClient(client=fake_client, model="fake", embedding_model="fake-embed")

    vector = await llm.embed("hello")

    assert fake_client.embed_kwargs["dimensions"] == EMBEDDING_DIM
    assert len(vector) == EMBEDDING_DIM
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-6)


async def test_llm_client_rejects_wrong_embedding_size():
    llm = LLMClient(client=FakeOpenAIClient(embedding=_embedding_response([0.1] * 10)), model="fake")
    with pytest.raises(ValueError, match="dimensions"):
        await llm.embed("hello")


async def test_llm_client_closes_underlying_client():
    fake_client = FakeOpenAIClient()
    await LLMClient(client=fake_client, model="fake").aclose()
    assert fake_client.closed is True


# --- Tool registry and dispatcher -------------------------------------------------

def test_tool_definitions_are_generated_from_functions():
    summarise = TOOL_REGISTRY["summarise_text"].definition["function"]
    assert set(summarise["parameters"]["properties"]) == {"text", "max_words"}
    assert summarise["parameters"]["required"] == ["text"]
    assert summarise["description"].startswith("Summarise")
    assert "parameters" not in TOOL_REGISTRY["get_current_datetime"].definition["function"]
    assert '"title"' not in json.dumps(TOOL_REGISTRY["calculator"].definition)


async def test_new_tool_only_needs_a_function(monkeypatch, ctx):
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"Sunny in {city}"

    monkeypatch.setitem(TOOL_REGISTRY, "get_weather", _build_tool(get_weather, None))

    definition = TOOL_REGISTRY["get_weather"].definition["function"]
    assert definition["parameters"]["required"] == ["city"]
    assert await dispatch_tool("get_weather", '{"city": "Paris"}', ctx) == "Sunny in Paris"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.15 * 3800000", "570000"),
        ("15% * 3,800,000", "570000"),
        ("sqrt(16) + 2^3", "12"),
        ("(1 + 2) * 3", "9"),
        ("10 % 3", "1"),
        ("7 / 2", "3.5"),
        ("-pi", "-3.14159265359"),
    ],
)
async def test_calculator_evaluates_expressions(ctx, expression, expected):
    result = await dispatch_tool("calculator", json.dumps({"expression": expression}), ctx)
    assert result == expected


@pytest.mark.parametrize(
    ("expression", "error_fragment"),
    [
        ("1/0", "division by zero"),
        ("__import__('os').system('ls')", "unsupported"),
        ("9**9**9", "exponent"),
        ("2 +", "invalid expression"),
        ("1+" * 150 + "1", "longer than"),
    ],
)
async def test_calculator_rejects_unsafe_or_invalid_input(ctx, expression, error_fragment):
    result = await dispatch_tool("calculator", json.dumps({"expression": expression}), ctx)
    assert result.startswith("Error:")
    assert error_fragment in result


@pytest.mark.parametrize(
    ("name", "arguments", "error_fragment"),
    [
        ("not_a_tool", "{}", "unknown tool"),
        ("calculator", "{not json", "not valid JSON"),
        ("calculator", "[1, 2]", "must be a JSON object"),
        ("web_search", "{}", "invalid arguments"),
    ],
)
async def test_dispatcher_returns_errors_instead_of_raising(ctx, name, arguments, error_fragment):
    result = await dispatch_tool(name, arguments, ctx)
    assert result.startswith("Error:")
    assert error_fragment in result


async def test_dispatcher_catches_exceptions_inside_tools(monkeypatch, ctx):
    def broken_tool() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    monkeypatch.setitem(TOOL_REGISTRY, "broken_tool", _build_tool(broken_tool, None))
    result = await dispatch_tool("broken_tool", "{}", ctx)
    assert result == "Error: tool 'broken_tool' failed: RuntimeError: kaboom"


async def test_web_search_returns_three_snippets(ctx):
    result = await dispatch_tool("web_search", '{"query": "current population of Dubai"}', ctx)
    lines = result.splitlines()
    assert len(lines) == 4
    assert "3,800,000" in lines[1]


async def test_get_current_datetime_returns_utc_iso_timestamp(ctx):
    result = await dispatch_tool("get_current_datetime", "{}", ctx)
    assert datetime.fromisoformat(result).utcoffset().total_seconds() == 0


async def test_summarise_text_short_input_skips_llm(ctx, fake_llm):
    result = await dispatch_tool("summarise_text", '{"text": "Already short.", "max_words": 10}', ctx)
    assert result == "Already short."
    assert fake_llm.chat_calls == []


async def test_summarise_text_calls_llm_and_enforces_word_limit(ctx, fake_llm):
    fake_llm.queue(text_response("one two three four five"))
    long_text = " ".join(["word"] * 50)

    result = await dispatch_tool("summarise_text", json.dumps({"text": long_text, "max_words": 3}), ctx)

    assert result == "one two three..."
    prompt = fake_llm.chat_calls[0]["messages"][-1]["content"]
    assert "at most 3 words" in prompt