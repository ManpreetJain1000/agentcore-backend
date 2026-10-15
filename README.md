# AgentCore

AgentCore is a backend API for running AI agents. A user creates an agent session with a system prompt and a set of tools, sends a message in plain English, and watches the agent work step by step: it calls the LLM, decides which tools to use, runs them, feeds the results back, and repeats until it has an answer.

Every step is saved to PostgreSQL so a run can be replayed later, and streamed to the client in real time over Server-Sent Events (SSE). Agents run in the background on a Celery worker, so the API always responds instantly.

**Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, PostgreSQL 16 + pgvector, Redis 7, Celery 5, OpenAI Python SDK, Pydantic v2, pytest.

For design decisions, the agent loop, the memory strategy and planned improvements, see [NOTES.md](NOTES.md).

---

## Quick start (Docker)

You need Docker with the Compose plugin.

1. Create your environment file and add an LLM API key:

```bash
   cp .env.example .env
```

   Then edit `.env` and set at least `JWT_SECRET_KEY` and `LLM_API_KEY` (see [Configuration](#configuration)).

2. Start everything with one command:

```bash
   docker compose up --build
```

   This starts PostgreSQL (with pgvector), Redis, a one-off migration job, the API and the Celery worker. The API and worker wait until the database is healthy and migrations have finished.

3. Open the interactive API docs at **http://localhost:8000/docs**.

| Service | Purpose | Host port |
|---|---|---|
| `api` | FastAPI application | 8000 |
| `worker` | Celery worker that executes agent runs | – |
| `db` | PostgreSQL 16 with pgvector | 5433 |
| `redis` | Celery broker, short-term memory, live events | 6380 |
| `migrate` | Runs `alembic upgrade head`, then exits | – |

Postgres and Redis are mapped to 5433 and 6380 so they don't clash with local installations.

Useful commands:

```bash
docker compose logs -f worker        # watch the agent work
docker compose down                  # stop (data is kept)
docker compose down -v               # stop and delete all data
```

---

## Local development (without Docker)

Requirements: Python 3.12, PostgreSQL 16 with the pgvector extension, Redis 7.

```bash
# 1. Virtual environment and dependencies
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Database (run once)
sudo -u postgres psql -c "CREATE USER agentcore WITH PASSWORD 'agentcore';"
sudo -u postgres psql -c "CREATE DATABASE agentcore OWNER agentcore;"
sudo -u postgres psql -d agentcore -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Configuration
cp .env.example .env                 # then fill in the values

# 4. Migrations
alembic upgrade head
```

Run the API and the worker in two terminals:

```bash
uvicorn app.main:app --reload
celery -A app.tasks.celery_app:celery_app worker --loglevel=info --concurrency=2
```

---

## Configuration

All settings are read from environment variables (or `.env`) by `app/core/config.py`.

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Async SQLAlchemy URL | `postgresql+asyncpg://agentcore:agentcore@localhost:5432/agentcore` |
| `REDIS_URL` | Redis URL (broker, memory, events) | `redis://localhost:6379/0` |
| `JWT_SECRET_KEY` | Secret used to sign access tokens | output of `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime | `30` |
| `LLM_API_KEY` | API key for the LLM provider | |
| `LLM_BASE_URL` | OpenAI-compatible endpoint. Leave empty for OpenAI | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MODEL` | Chat model | `gpt-4o-mini` or `gemini-3.1-flash-lite` |
| `EMBEDDING_MODEL` | Embedding model (1536 dimensions) | `text-embedding-3-small` or `gemini-embedding-001` |

The code uses the OpenAI Python SDK and works with any OpenAI-compatible provider. To use OpenAI as in the brief, leave `LLM_BASE_URL` empty and set `LLM_MODEL=gpt-4o-mini` and `EMBEDDING_MODEL=text-embedding-3-small`. No code changes are needed.

Agent and memory tuning (`AGENT_MAX_ITERATIONS`, `TOOL_TIMEOUT_SECONDS`, `SHORT_TERM_MAX_TURNS`, `LONG_TERM_TOP_K`, ...) have sensible defaults in `config.py`.

---

## API overview

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create a user and return an access token |
| POST | `/auth/token` | Log in (OAuth2 form: `username` = email) |
| GET | `/auth/me` | The current user |
| GET | `/sessions` | List your sessions |
| POST | `/sessions` | Create a session (`name`, `system_prompt`, `tools_enabled`) |
| GET | `/sessions/{id}` | Session detail with its most recent runs |
| DELETE | `/sessions/{id}` | Delete a session, its runs and its short-term memory |
| POST | `/sessions/{id}/run` | Submit a message. Returns `run_id` immediately (202) |
| GET | `/runs/{id}/status` | Status, tokens used, step count, final answer |
| GET | `/runs/{id}/steps` | Full step-by-step trace |
| GET | `/runs/{id}/stream` | Live SSE stream of the run's steps |
| DELETE | `/runs/{id}` | Cancel a queued or running run |
| GET | `/memory` | List your long-term memories |
| GET | `/memory/search?q=` | Semantic search over your memories (top 5, with similarity) |
| DELETE | `/memory/{id}` | Delete a memory |
| GET | `/health` | Liveness check |

All endpoints except `/auth/register`, `/auth/token` and `/health` require `Authorization: Bearer <token>`. The stream endpoint also accepts `?access_token=<token>`, because the browser's `EventSource` cannot send headers.

**Available tools:** `web_search` (mock), `calculator`, `get_current_datetime`, `summarise_text`, `remember_fact`.

### Example: the scenario from the brief

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@example.com", "password": "StrongPass123"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

SESSION_ID=$(curl -s -X POST http://localhost:8000/sessions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "Research Assistant", "system_prompt": "You are a research assistant.", "tools_enabled": ["web_search", "calculator", "remember_fact"]}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

RUN_ID=$(curl -s -X POST http://localhost:8000/sessions/$SESSION_ID/run \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "What is 15% of the current population of Dubai, and remember that fact for me?"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['run_id'])")

curl -N http://localhost:8000/runs/$RUN_ID/stream -H "Authorization: Bearer $TOKEN"
```

Each SSE frame looks like `data: {"step_type": "tool_call", "payload": {...}}`, and the stream ends with `data: {"step_type": "done", "status": "completed"}`.

---

## Watching a run in the browser (optional)

`demo/index.html` is a small, dependency-free page that uses only the public API. It shows each step of a run as a readable timeline. It is not part of the backend or of `docker compose`.

```bash
python3 -m http.server 5500 --directory demo
```

Open **http://localhost:5500** while the API is running on port 8000.

---

## Running the tests

Tests use a separate database, an in-memory fake Redis and a scripted fake LLM. No API key, worker or network access is needed.

**Locally** (one-time setup, then run):

```bash
sudo -u postgres psql -c "CREATE DATABASE agentcore_test OWNER agentcore;"
sudo -u postgres psql -d agentcore_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

pytest -v
pytest --cov=app --cov-report=term-missing
```

**In Docker** (the test database is created automatically):

```bash
docker compose run --rm api pytest --cov=app --cov-report=term-missing
```

The suite refuses to run against a database whose name does not contain `test`.

---

## Project structure

```text
app/
├── main.py                  App factory, lifespan (Redis + LLM clients), routers
├── core/
│   ├── config.py            Pydantic Settings
│   ├── security.py          Password hashing, JWT encode/decode
│   └── dependencies.py      get_db, get_redis, get_llm, get_current_user, ownership checks
├── db/
│   ├── base.py              Async engine, session factory, worker engine factory
│   └── migrations/          Alembic environment and versions
├── models/                  User, AgentSession, AgentRun, RunStep, LongTermMemory
├── schemas/                 Pydantic request/response models
├── routers/                 auth, sessions, runs, stream (SSE), memory
├── agent/
│   ├── runner.py            The agent loop
│   ├── planner.py           LLM wrapper (chat with tools, embeddings)
│   ├── tools.py             Tool registry, auto-generated schemas, dispatcher, tools
│   └── memory_manager.py    Short-term (Redis) and long-term (pgvector) memory
├── tasks/
│   ├── celery_app.py        Celery configuration
│   └── agent_tasks.py       execute_agent_run task
└── tests/                   conftest + agent runner, runs API and memory tests

demo/index.html              Optional browser viewer for live runs
docker/postgres/init.sql     Creates the pgvector extension and test database
Dockerfile, docker-compose.yml, alembic.ini, pytest.ini, .coveragerc
```