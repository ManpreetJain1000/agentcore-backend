"""Short-term (Redis) and long-term (pgvector) memory: storage, retrieval, API and isolation."""
import json

from app.agent.memory_manager import LongTermMemoryStore, ShortTermMemory
from app.agent.tools import ToolContext, dispatch_tool
from app.core.config import settings
from app.tests.conftest import auth_headers


# --- Short-term memory ----------------------------------------------------------

async def test_short_term_memory_keeps_last_turns_in_order(fake_redis):
    stm = ShortTermMemory(fake_redis, max_turns=3)
    for i in range(5):
        await stm.append_turn("session-1", f"q{i}", f"a{i}")

    assert await fake_redis.llen(stm.key("session-1")) == 3
    assert await stm.get_recent_messages("session-1", limit=4) == [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]
    assert 0 < await fake_redis.ttl(stm.key("session-1")) <= settings.SHORT_TERM_TTL_SECONDS


async def test_short_term_memory_is_isolated_per_session_and_clearable(fake_redis):
    stm = ShortTermMemory(fake_redis)
    await stm.append_turn("session-a", "hello", "hi")

    assert await stm.get_recent_messages("session-b", 10) == []
    assert await stm.get_recent_messages("session-a", 0) == []

    await stm.clear("session-a")
    assert await stm.get_recent_messages("session-a", 10) == []


# --- Long-term memory store -------------------------------------------------------

async def test_long_term_memory_deduplicates_identical_facts(db, fake_llm, user_factory):
    user, _ = await user_factory()
    store = LongTermMemoryStore(db, fake_llm)

    first, created_first = await store.add(user.id, "My favourite programming language is Python")
    second, created_second = await store.add(user.id, "my favourite programming language is python!")
    _, created_other = await store.add(user.id, "I live in Lisbon")

    assert created_first is True
    assert created_second is False
    assert second.id == first.id
    assert created_other is True
    assert len(await store.list_for_user(user.id, limit=10, offset=0)) == 2


async def test_semantic_search_ranks_by_similarity_and_isolates_users(db, fake_llm, user_factory):
    user, _ = await user_factory()
    other_user, _ = await user_factory()
    store = LongTermMemoryStore(db, fake_llm)
    await store.add(user.id, "I live in Lisbon")
    await store.add(user.id, "My favourite programming language is Python")
    await store.add(user.id, "I prefer short direct answers")
    await store.add(other_user.id, "My favourite programming language is Rust")

    results = await store.search_with_scores(user.id, "which programming language", limit=5)

    contents = [memory.content for memory, _ in results]
    assert contents[0] == "My favourite programming language is Python"
    assert "My favourite programming language is Rust" not in contents
    distances = [distance for _, distance in results]
    assert distances == sorted(distances)


async def test_delete_only_removes_own_memory(db, fake_llm, user_factory):
    user, _ = await user_factory()
    other_user, _ = await user_factory()
    store = LongTermMemoryStore(db, fake_llm)
    memory, _ = await store.add(user.id, "I live in Lisbon")

    assert await store.delete(other_user.id, memory.id) is False
    assert await store.delete(user.id, memory.id) is True
    assert await store.list_for_user(user.id, limit=10, offset=0) == []


async def test_remember_fact_tool_writes_long_term_memory(
    db, fake_llm, user_factory, session_factory, run_factory
):
    user, _ = await user_factory()
    session = await session_factory(user, tools=["remember_fact"])
    run = await run_factory(session)
    ctx = ToolContext(user_id=user.id, session_id=session.id, run_id=run.id, db=db, llm=fake_llm)

    first = await dispatch_tool("remember_fact", json.dumps({"fact": "I live in Lisbon"}), ctx)
    second = await dispatch_tool("remember_fact", json.dumps({"fact": "I live in Lisbon"}), ctx)

    assert first == "Remembered: I live in Lisbon"
    assert second == "Already remembered: I live in Lisbon"
    memories = await LongTermMemoryStore(db, fake_llm).list_for_user(user.id, limit=10, offset=0)
    assert len(memories) == 1
    assert memories[0].source_run_id == run.id


# --- Memory API -------------------------------------------------------------------

async def test_list_memories_newest_first_without_embeddings(client, db, fake_llm, user_factory):
    user, token = await user_factory()
    store = LongTermMemoryStore(db, fake_llm)
    await store.add(user.id, "I live in Lisbon")
    await store.add(user.id, "I prefer short direct answers")

    response = await client.get("/memory", headers=auth_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert [m["content"] for m in body] == ["I prefer short direct answers", "I live in Lisbon"]
    assert "embedding" not in body[0]

    paged = await client.get("/memory?limit=1&offset=1", headers=auth_headers(token))
    assert [m["content"] for m in paged.json()] == ["I live in Lisbon"]


async def test_search_endpoint_returns_scored_results(client, db, fake_llm, user_factory):
    user, token = await user_factory()
    store = LongTermMemoryStore(db, fake_llm)
    await store.add(user.id, "I live in Lisbon")
    await store.add(user.id, "My favourite programming language is Python")

    response = await client.get(
        "/memory/search", params={"q": "which programming language"}, headers=auth_headers(token)
    )
    assert response.status_code == 200
    results = response.json()
    assert results[0]["content"] == "My favourite programming language is Python"
    assert 0 < results[0]["similarity"] <= 1

    filtered = await client.get(
        "/memory/search",
        params={"q": "which programming language", "min_similarity": 0.3},
        headers=auth_headers(token),
    )
    assert [r["content"] for r in filtered.json()] == ["My favourite programming language is Python"]


async def test_search_endpoint_returns_503_when_embeddings_fail(client, fake_llm, user_factory):
    _, token = await user_factory()
    fake_llm.fail_embed = True

    response = await client.get("/memory/search", params={"q": "anything"}, headers=auth_headers(token))

    assert response.status_code == 503


async def test_memory_endpoints_are_isolated_per_user(client, db, fake_llm, user_factory):
    owner, _ = await user_factory()
    _, intruder_token = await user_factory()
    memory, _ = await LongTermMemoryStore(db, fake_llm).add(owner.id, "I live in Lisbon")
    headers = auth_headers(intruder_token)

    assert (await client.get("/memory", headers=headers)).json() == []
    assert (await client.get("/memory/search", params={"q": "Lisbon"}, headers=headers)).json() == []
    assert (await client.delete(f"/memory/{memory.id}", headers=headers)).status_code == 404


async def test_delete_memory_endpoint(client, db, fake_llm, user_factory):
    user, token = await user_factory()
    memory, _ = await LongTermMemoryStore(db, fake_llm).add(user.id, "I live in Lisbon")

    assert (await client.delete(f"/memory/{memory.id}", headers=auth_headers(token))).status_code == 204
    assert (await client.delete(f"/memory/{memory.id}", headers=auth_headers(token))).status_code == 404
    assert (await client.get("/memory", headers=auth_headers(token))).json() == []


async def test_memory_endpoints_require_auth(client):
    assert (await client.get("/memory")).status_code == 401
    assert (await client.get("/memory/search", params={"q": "x"})).status_code == 401


async def test_deleting_session_clears_short_term_but_keeps_long_term_memory(
    client, db, fake_redis, fake_llm, user_factory, session_factory, run_factory
):
    user, token = await user_factory()
    session = await session_factory(user)
    run = await run_factory(session)
    stm = ShortTermMemory(fake_redis)
    await stm.append_turn(session.id, "hello", "hi")
    await LongTermMemoryStore(db, fake_llm).add(user.id, "I live in Lisbon", source_run_id=run.id)

    response = await client.delete(f"/sessions/{session.id}", headers=auth_headers(token))

    assert response.status_code == 204
    assert await fake_redis.exists(stm.key(session.id)) == 0
    memories = (await client.get("/memory", headers=auth_headers(token))).json()
    assert len(memories) == 1
    assert memories[0]["source_run_id"] is None