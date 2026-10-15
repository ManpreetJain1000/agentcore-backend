# AgentCore: design notes

The brief asked for three things in this document. Here's where to find each one:

1. **Agent loop design**: [section 1](#1-agent-loop-design)
2. **Memory strategy**: [section 2](#2-memory-strategy)
3. **What I would improve with more time**: [section 3](#3-what-i-would-improve-with-more-time)

After those, I explain the other parts of the system (background jobs, live streaming, privacy, testing), the demo UI I built to show runs visually, and where I changed things from the brief.

For installation and setup, see [README.md](README.md).

---

## What AgentCore does, in one minute

A user creates an AI "agent" by giving it a personality (a system prompt) and a set of tools. They then ask it something in plain English.

The agent doesn't answer straight away. It thinks about what it needs, uses tools to get information, looks at the results, and repeats until it can give a proper answer. Every step is saved, so a run can be replayed later, and streamed live, so you can watch the agent think.

Take the example from the brief: *"What is 15% of the current population of Dubai, and remember that fact for me?"* The agent:
1. searches for Dubai's population,
2. calculates 15% of it,
3. saves the result to memory,
4. replies.

---

## 1. Agent loop design

The loop lives in `app/agent/runner.py`. It follows a simple pattern:

> **Prepare → Think → Act → Observe → repeat → Answer**

1. **Prepare.** Collect what the agent needs to know:
   - the session's system prompt
   - the recent conversation
   - any facts it remembers about this user that relate to the question
2. **Think.** Send all of that to the AI model, together with the tools this session may use. The model either answers, or says *"I want to use this tool with these inputs"*.
3. **Act.** Run the tools the model asked for.
4. **Observe.** Give the tool results back to the model and go back to step 2.
5. **Answer.** When the model replies without asking for a tool, that's the final answer. It's saved, the run is marked as completed, and the conversation is remembered for next time.

### The decisions behind it

**Use the model's built-in tool calling.**
- *Why:* the model returns tool requests as structured data (a tool name plus inputs), so I don't have to pull them out of free text with fragile parsing. The model decides when it needs a tool and when it's done.
- *Trade-off:* it needs a provider that supports tool calling. All the major ones do.

**Save every step before announcing it.**
- *Why:* after each step, the agent first writes it to the database and only then publishes it for live viewers. The database is the official record. Anything someone saw live can be rebuilt exactly later, and a crash never leaves a step that was shown but not saved.
- *Trade-off:* one small database write per step, which is negligible next to an AI call.

**Stop the loop in exactly three ways.**
- *Normal finish:* the model replied without asking for tools.
- *Safety cap:* 10 rounds, after which the answer is "Max iterations reached". This protects against a model that keeps calling tools forever.
- *Cancellation:* the loop checks for it at the start of every round.
- *Why:* the loop always ends, and always for a clear, recorded reason.

**Only offer and run the tools enabled for the session.**
- *Why:* the model only sees the tools the session allows, and the agent double-checks every request. If the model makes up a tool, or asks for one that isn't enabled, it isn't run. The model gets the message *"That tool isn't available. Here are the ones you can use."* and can correct itself on the next round.
- *Trade-off:* none really. It's both safer and gives the model a chance to recover.

**Turn errors into information for the model, not crashes.**
- *Why:* if a tool fails (bad input, division by zero, a bug), the error comes back to the model as a normal result, for example *"calculator failed: division by zero"*. The model can then try another way. One broken tool never takes down a whole run.
- *How the other failure cases are handled:*
  - **Slow tools** hit a time limit, the timeout is recorded, and the agent carries on without that result.
  - **Temporary problems with the AI service** (network errors, rate limits) are retried automatically.
  - **Truly unexpected errors** mark the run as failed, and the error is saved so you can see what happened.

**Give each tool its own time limit, with 5 seconds as the default.**
- *Why:* most tools are quick. Two of them make slower external calls, though: `summarise_text` calls the AI model, and `remember_fact` calls the embedding service and writes to the database. They get 30 and 20 seconds. Cutting off a database write halfway would cause more problems than waiting a little longer.
- *Trade-off:* a slow tool can hold up a run for longer.

**Make adding a tool a five-minute job.**
- *Why:* a tool is just a Python function with a decorator:

```python
  @tool
  def get_weather(city: Annotated[str, Field(description="City name")]) -> str:
      """Return the current weather for a city."""
      return f"The weather in {city} is 31°C and sunny."
```

  From the function's name, inputs and description, the system automatically builds the instructions the AI model needs. It also uses them to check the model's inputs before the tool runs. There's no schema to write by hand and nothing else to change.

**Keep the AI provider behind a small wrapper.**
- *Why:* the rest of the code only ever calls two methods, `chat` and `embed` (in `app/agent/planner.py`). That makes it easy to:
  - **switch provider** (I developed with Gemini; switching to OpenAI only means changing settings)
  - **replace the AI with a fake one in tests**
  - **pass back provider-specific details** the model needs between rounds

**Record a clear trace.** Each saved step has a type, so a run reads like a story:

| Step | What it means |
|---|---|
| `context` | What the agent knew before starting: history, remembered facts, available tools |
| `llm_call` | What the model said or asked for, and how many tokens it used |
| `tool_call` / `tool_result` | Which tool ran, with what inputs, and what came back |
| `tool_timeout` | A tool took too long |
| `final_answer` | The answer, and how many rounds it took |
| `retry` / `error` / `cancelled` | When things didn't go to plan |

Two extra events are only sent live and never saved: `status`, when a worker starts the run, and `done`, the signal that closes the stream.

### The tools

- **web_search:** returns three search results. It's a mock, with realistic data for the Dubai example; a real version would call a search API.
- **calculator:** does maths safely. It reads the expression as data and only allows numbers, operators and a few maths functions. It never runs the input as code.
- **get_current_datetime:** the current date and time in UTC.
- **summarise_text:** asks the AI model to shorten a piece of text, so it's an AI call inside a tool.
- **remember_fact:** saves a fact about the user to long-term memory.

---

## 2. Memory strategy

The agent has two kinds of memory, much like a person: what we were *just* talking about, and facts it *knows* about you.

| | Short-term memory | Long-term memory |
|---|---|---|
| **What it holds** | The recent conversation | Facts the agent was asked to remember |
| **Belongs to** | One session | One user, across all their sessions |
| **Stored in** | Redis (fast, in-memory) | PostgreSQL with pgvector (search by meaning) |
| **Used for** | Following the conversation | Background knowledge about the user |

### The decisions behind it

**Split memory by how long it lives and who it belongs to.**
- *Why:* conversation history only matters inside one chat, and it goes stale quickly. Facts like *"I live in Lisbon"* are about the person and should follow them into every session. Mixing the two would either drag old chat into new sessions, or lose facts when a session ends.

**Keep short-term memory in Redis, as a capped list.**
- *Why:* it's read on every single run, it's always looked up by session, and it only needs the newest items. A Redis list fits perfectly:
  - each finished turn (question plus answer) is added to the front
  - anything beyond the newest **20 turns** is trimmed off in the same step
- *Trade-off:* Redis is in-memory. In Docker I turned on Redis persistence, so history survives a restart.

**Read the last 10 messages, but store 20 turns.**
- *Why:* 10 messages (5 questions and answers) gives the model enough context to follow a conversation without making every request big and expensive. The extra stored turns leave room to change that window later.
- *Trade-off:* anything older than the window is not seen by the model. Improvement 6 in section 3 addresses this.

**Clean up short-term memory automatically.**
- *Why:* a session's history expires after **7 days** of inactivity, and it's deleted along with its session. Abandoned conversations never pile up, and no cleanup job is needed.

**Keep long-term memory in the same PostgreSQL database, using pgvector.**
- *Why:* there's one database to run, and memories sit next to the users and runs they belong to. Filtering by user is just a normal `WHERE` clause.
- *How meaning search works:* each fact is turned into an embedding, a list of 1536 numbers that captures its meaning. Facts with similar meanings end up close together. For example, if you once said *"I live in Lisbon"*, asking *"Which city am I from?"* still finds it, even though the words are different.
- *Trade-off:* at very large scale, a dedicated vector database might search faster. For this project, one database is simpler and plenty fast.

**Only save facts when the agent decides to, using `remember_fact`.**
- *Why:* saving everything automatically would fill memory with noise. Saving deliberately keeps memories meaningful, and the user stays in control.
- *Trade-off:* the agent only remembers what it chooses to, or is asked to.

**Give the 3 most relevant facts to the model as background knowledge.**
- *Why:* for each new message, the agent finds the 3 facts closest in meaning to the question and adds them to the system prompt. Putting them there, rather than as fake chat messages, tells the model *"this is what you know about the user"*. Three is enough to be useful without crowding the prompt.

**Don't store duplicates.**
- *Why:* if a new fact means practically the same as one already saved for that user, it isn't stored again. Otherwise, asking the agent to remember the same thing twice would slowly fill memory with copies.

**Facts belong to the user, not the session.**
- *Why:* deleting a session doesn't delete the facts learned in it. They were about the person, not the chat.

**Let users see and manage what the agent remembers.**
- *Why:* users can list, search and delete their memories. Search results show a similarity score, because a "closest" match isn't always a good one, and the score makes weak matches obvious.

**If memory is unavailable, keep going.**
- *Why:* if Redis or the embedding service has a problem, the agent answers without that context instead of failing the whole run.

### How fast is memory search?

Every search only looks at the current user's memories. Even with 500 memories, the database compares the question against just those 500, which takes a few milliseconds.

For very large amounts of data there's also a special index (HNSW) that finds close matches without checking every row. It has one limitation with many users, explained in improvement 5 below.

---

## 3. What I would improve with more time

Two bonus items from the brief, rate limiting and structured logging, aren't done yet (see [section 7](#7-bonus-items)). They would be my very first job. Beyond those, these are the improvements I'd make to the system itself, grouped by area.

### Fix the known gaps first

These are small weaknesses I know about in the current code:

- **Cancel timing gap.** If you cancel during the very last AI call, the run can still end up marked "completed". The final save should use the same *"only if it's still running"* check that the cancel endpoint already uses.
- **Resume retries instead of restarting them.** A retried run currently starts from the beginning, so steps are repeated in the trace and an action like saving a memory could run twice. I'd rebuild the conversation from the saved steps and continue from the last one.
- **Memory search with many users.** The fast vector index looks across *everyone's* memories first and filters by user afterwards, so with lots of users someone could get fewer than 3 results. There are known fixes: tune the index, use newer pgvector features built for filtered search, or split memories per user.
- **Count every token.** The AI call made inside `summarise_text` isn't added to the run's token total yet.
- **Safer stream login.** The token goes in the URL for browser streams, where it can end up in server logs. A short-lived, single-use "stream ticket" would be safer.
- **Stopping slow synchronous tools.** When a regular, non-async tool times out, the agent moves on, but the tool keeps running in the background. Long-running tools should run in a separate process, or be written so they can be stopped.

### Make the agent faster and smarter

- **Run independent tools at the same time.** When the model asks for several tools in one round, for example two web searches, they currently run one after another. Running them in parallel would make multi-tool rounds noticeably faster.
- **Stream the answer word by word.** At the moment you see each *step* live, but the final answer arrives all at once. Streaming the model's text as it's generated would make the agent feel much more responsive.
- **Smarter conversation history.** Measure the history window in tokens instead of a fixed number of messages, and summarise older parts of a long conversation instead of dropping them, so long chats keep their context without getting expensive.
- **Real web search with sources.** Replace the mock with a real search API and have the agent cite where its facts came from, so users can check the answer.

### Make memory more useful

- **Update facts instead of piling them up.** If a user once said *"I live in Lisbon"* and later says *"I moved to Porto"*, both facts are currently kept. The agent should notice the conflict and replace the old fact, so it doesn't contradict itself.
- **Let memories age.** Track when each fact was last useful, and prefer recent and frequently used ones. Add a per-user limit that removes the least useful memories first, so memory stays relevant as it grows.
- **A "forget" tool.** Users can delete memories through the API, but they should also be able to just tell the agent *"forget where I live"*.
- **Combine meaning search with keyword search.** Search by meaning is great for paraphrases, but weaker for exact names, numbers and codes. Mixing in a normal keyword search would find those reliably.

### Keep it safe and affordable

- **Ask before sensitive actions.** Some tools, like sending an email or deleting data, shouldn't run without a person agreeing. The run would pause in a "waiting for approval" state and continue only once the user approves.
- **Protect against prompt injection.** Text from tools, especially web search results, can contain hidden instructions like *"ignore your rules and…"*. Tool results should be clearly marked as untrusted data, and risky tools restricted after the agent has read outside content.
- **Budgets per run and per user.** Set a maximum token or cost limit for each run and a daily limit per user. Stop gracefully with a clear message when a limit is reached, so one runaway run can't produce a surprise bill.
- **A backup AI provider.** If the main provider is down or rate-limited, automatically switch to a second one. Simple tasks could also go to a cheaper model to save money.

### Run it with confidence

- **A test set for answer quality.** Keep a fixed set of example questions with the expected tool usage and answers, and run them against the real model regularly. That catches regressions when the prompt, the model or a tool changes, which normal unit tests can't do.
- **End-to-end tracing and metrics.** Follow a single run across the API, the queue, the worker, the AI calls and the tools in one view. Track run duration, token usage, tool error rates and queue length on a dashboard.
- **Notify when a run finishes.** Not every client wants to keep a stream open. A webhook that calls the client back when a run completes would suit integrations and long runs.
- **Scale the workers sensibly.** Use separate queues for quick and long runs, so short tasks aren't stuck behind long ones, and add workers automatically when the queue grows.
- **API polish:**
  - versioned URLs (`/api/v1/...`)
  - one consistent error format
  - an endpoint that lists available tools (the demo UI currently keeps its own list)
  - a way to edit a session's prompt and tools
  - better pagination
- **Production readiness:**
  - keep test tools out of the production image
  - proper secret management
  - several API workers behind a proxy
  - automatic clean-up of old run traces after a set number of days

---

## 4. Other design notes

### Running agents in the background

Agents can take many seconds, so they run on a separate worker (Celery) and the API answers instantly with a run ID.

- **Save first, then queue.** Otherwise a very fast worker could look for a run that doesn't exist yet. If the queue is down, the run is marked as failed and you get a clear error instead of a run stuck "queued" forever.
- **A run never executes twice**, even if the same job arrives twice, which queues sometimes do. There are two safeguards:
  - finished runs are skipped
  - a lock ensures only one worker handles a run at a time
- **Temporary errors are retried** up to 3 times, waiting about 5, 10 and 20 seconds. Each retry shows up in the trace.
- **If a worker crashes,** its job goes back on the queue instead of being lost.
- **Cancelling** happens in three ways:
  - **Safe status change.** The status changes only if the run is still queued or running. A late cancel gets a "conflict" response and can't overwrite a finished run.
  - **Stopping the worker.** The worker receives a signal and stops mid-task.
  - **Backup check.** The loop also checks for cancellation every round.

### Watching it live (streaming)

The stream uses Server-Sent Events (SSE): one open connection over which the server keeps sending updates.

The main challenge is not missing anything. Redis only delivers messages to listeners who are already tuned in. So the stream works in this order:

1. **It starts listening first,** so nothing new can slip past.
2. **It replays every step already saved** in the database.
3. **It passes on new steps as they happen,** skipping any it already sent.
4. **Every 15 seconds it sends a small "ping"** to keep the connection alive. At the same time it double-checks the run's status, so the stream still closes properly if a worker crashed without saying "done".

It also handles a few other situations:
- **Finished runs** replay instantly.
- **Dropped connections** pick up right after the last step received.
- **Closing the page** is cleaned up quietly, without errors in the logs.

### Keeping each user's data private

- **Passwords and logins:**
  - passwords are stored as bcrypt hashes, never in plain text
  - logins return a token that expires after 30 minutes
- **Privacy is part of every database query.** Each lookup of a session, run or memory includes *"and it belongs to this user"*. Nothing is fetched first and checked afterwards.
- **Asking for someone else's data returns "not found"**, which doesn't even confirm that it exists.
- **Users running agents at the same time** can't interfere with each other. Each run has its own records, its own live channel, its own lock and its own conversation memory.

### How I tested it

The tests need no real AI service, Redis server or worker:
- **A fake AI model** gives scripted answers.
- **A fake Redis** runs in memory.
- **The queue is replaced** with something that just records what would have been sent.
- **A real PostgreSQL test database** is used, so the actual queries and memory searches are tested. It's wiped clean after every test.

The tests cover the normal flow and the things that go wrong:
- made-up tools
- the 10-round limit
- timeouts
- AI failures
- late cancels
- the queue being down
- expired logins
- dangerous calculator input
- users trying to access each other's data
- stream replay and reconnecting
- duplicate jobs and retries

---

## 5. Demo UI: watching runs in the browser

The terminal output of a live stream is accurate but hard to follow, especially in a demo. So I added a small web page, `demo/index.html`, that shows a run as a readable, step-by-step timeline.

**What you can do with it:**
- **Sign in or create an account.** You can also create a session by picking tools from a list of available ones.
- **Send a message and watch the agent work live.** Each step appears the moment it happens, as plain sentences like *"Model decided to use calculator"* or *"calculator returned a result"*. Steps are colour-coded:
  - model decisions
  - tool calls
  - tool results
  - problems
  
  You also see how long each step took and how many tokens each model call used.
- **See the final answer** in its own card when the run finishes.
- **Watch the "What the agent remembers" panel** update when the agent saves a new fact.
- **Cancel a run mid-way** and see the cancellation arrive in the stream.
- **Replay any saved run** from the database, which shows that the full trace can always be rebuilt.
- **Open "Raw event" on any step** to see the exact JSON the API sends.
- **Sign out** to clear everything on screen.

**How it fits into the project:**
- **The backend has no idea it exists.** It's a single HTML file outside the `app/` folder, with no build step and no dependencies.
- **It only uses the public API**, like any other client: login, sessions, runs, the stream and memory.
- **Nothing in the required structure or `docker compose` changed because of it.**

**How to run it:** with the API running on port 8000, run

```bash
python3 -m http.server 5500 --directory demo
```

and open **http://localhost:5500**.

---

## 6. Where I changed things from the brief, and why

- **AI provider.** I developed with Gemini instead of OpenAI, but through the official OpenAI Python SDK and Gemini's OpenAI-compatible endpoint, so the code and the tool-calling format are exactly what the brief describes. Switching to `gpt-4o-mini` only means changing three settings in `.env`, and the defaults in the code are already the OpenAI ones.
- **No refresh token.** Login returns only an access token, because Feature 1 says a refresh token isn't needed.
- **Two tools are async.** Most tools are synchronous and run in a background thread, as the brief describes. `summarise_text` and `remember_fact` talk to the AI service and the database, so they're async to avoid blocking.
- **Calculator.** The brief suggests `ast.literal_eval`, but that can't do maths like `0.15 * 3800000`. I kept the same safe idea of reading the expression without running it, and wrote a small evaluator that only allows whitelisted operations.
- **Longer limits for two tools.** These are the 30 and 20 second limits explained in section 1.
- **"Last 10 messages"** is read as 10 messages, which is 5 question-and-answer turns.
- **More step types** (`context`, `tool_timeout`, `retry`, `error`, `cancelled`), so the trace tells the full story.
- **Small additions to the database schema.** I kept the given schema and added:
  - indexes for speed
  - automatic cleanup when a session is deleted
  - a search index for memories
  
  The brief didn't define the User table, so it has just what login needs: email, password hash, active flag and creation date.
- **Extra endpoints.** `GET /auth/me` checks who you're logged in as, and `GET /health` lets Docker check the API is running. The stream also accepts the token in the URL, so browsers can use it.

## 7. Bonus items

- ✅ **Tool timeouts.** Done.
- ✅ **Real cancellation** that stops the worker mid-run. Done, with an extra check every round.
- ✅ **Detailed API docs.** Every route has response types, status codes and descriptions in Swagger.
- ❌ **Rate limiting.** Not done yet; it's improvement 1 in section 3.
- ❌ **Structured logging.** Not done yet; it's improvement 2 in section 3.