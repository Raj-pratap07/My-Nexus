# My-Nexus

A local-first AI research assistant. A natural-language request goes to a
local language model (Ollama, `qwen3:4b-instruct`), which may call a small,
allow-listed set of tools exposed through the Model Context Protocol (MCP).
Tool results come back to the model and a final answer is stored in a
persistent conversation.

**Status: early development, local use only.** There is a backend (API,
database, agent). There is no frontend yet; a web UI and a Tauri desktop app
are planned and will share this backend.

## Architecture

```
HTTP client ──► api/         thin FastAPI routes: validate input, authenticate
                  │
                  ▼
               services/     business logic, ownership checks, one agent turn
                 │     │
                 ▼     ▼
          database/   agent/runtime.py      owns the MCP connections (opened once)
        (SQLAlchemy,        │
         Alembic)           ▼
                      agent/orchestrator.py   bounded tool-calling loop
                        │            │
                        ▼            ▼
                 agent/providers.py  mcp/connection_manager.py   (single execution path)
                 (Ollama /api/chat)        │
                                           ├─ GitHub MCP server   (remote, read-only)
                                           └─ Research MCP server (local STDIO: OpenAlex, Crossref)
```

| Layer | Location | Responsibility |
|---|---|---|
| API | `backend/app/api/`, `app/main.py` | Routes, auth dependency, error mapping. Never touches MCP. |
| Services | `backend/app/services/` | Conversation logic; talks to the agent only through `AgentRunner`. |
| Database | `backend/app/database/`, `backend/alembic/` | Models, repositories, migrations. |
| Agent | `backend/app/agent/` | Ollama provider, tool adapter, orchestrator, runtime. |
| MCP | `backend/app/mcp/` | Connection manager, tool policy, argument guards. |
| MCP servers | `custom_mcp_servers/research_server/` | Local Research server (OpenAlex, Crossref). |

### Security model (what is enforced in code)
- **Default-deny tool allowlist** (`mcp/permissions.py`): only explicitly listed
  tools are visible to the model or executable.
- **Argument-level guard** for `github.get_file_contents` (`mcp/argument_policy.py`):
  only `Raj-pratap07/My-Nexus`, relative ASCII paths, no traversal, no `.git`
  segments. Sensitive servers **fail closed** if their guard is missing.
- **Single execution path**: every tool call goes through
  `MCPConnectionManager.call_tool()`, which re-checks policy and arguments.
- **Bounded loop**: at most 4 tool calls per request, sequential, results
  size-limited and marked as untrusted data.
- **Identity boundary**: every `/v1` route requires a bearer token; conversations
  and memories are always looked up by owner; messages are reachable only
  through an owned conversation.
- **Local only**: the launcher refuses non-loopback hosts.

The prompt tells the model to treat tool output as untrusted data, but that is a
request, not a guarantee: prompt injection is **not** prevented. The code-level
controls above are what limit the damage.

### Data model
`users`, `conversations`, `messages` (roles `user`/`assistant` only),
`memories` (schema only; no behaviour yet) and `tool_executions` (tool name and
status only; arguments and results are not stored).

## Prerequisites
- Python 3.13
- PostgreSQL (for running the API; the offline tests use SQLite)
- [Ollama](https://ollama.com) with the model pulled: `ollama pull qwen3:4b-instruct`
- Optional: a GitHub MCP server URL and a fine-grained token (see Configuration)

## Installation
From the repository root (Windows PowerShell shown; use `source .venv/bin/activate` on macOS/Linux):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # runtime + test dependencies
```

## Configuration
Copy `.env.example` to `.env` in the **repository root** and fill in real values.
`.env` is git-ignored; never commit it or paste secrets into chats.

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | `postgresql://USER:PASSWORD@localhost:5432/my_nexus` (URL-encode special characters). |
| `NEXUS_DEV_TOKEN` | yes | At least 32 characters. Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `GITHUB_TOKEN`, `GITHUB_MCP_URL` | optional | Set both or neither. Use a **fine-grained** token limited to the My-Nexus repository with read-only *Contents* and *Metadata*. |

Optional overrides: `NEXUS_HOST` (loopback only, default `127.0.0.1`), `NEXUS_PORT`
(`8000`), `OLLAMA_URL` (`http://localhost:11434`), `OLLAMA_MODEL`
(`qwen3:4b-instruct`), `OLLAMA_TIMEOUT` (`180`), `NEXUS_MAX_CONCURRENT_RUNS` (`1`),
`CROSSREF_MAILTO` (polite-pool contact for Crossref).

## Database migration
Create an empty PostgreSQL database and user, then from `backend/`:

```powershell
python -m alembic upgrade head
```

The API refuses to start if the schema is not at the latest migration.
(The migrations have been run on SQLite only. Their PostgreSQL SQL is checked
offline in the tests, but they have not yet been run on PostgreSQL.)

**Upgrading an existing database.** After pulling new code, run the same command
again. Migration `0002` adds `status` and `error_code` to `messages`; every
existing message becomes `completed`, and no row or tool-execution link changes.
`python -m alembic downgrade 0001` reverses it and deletes only the failed-turn
markers it introduced.

**Supabase security.** On PostgreSQL the migration secures the five application
tables and `alembic_version` against the Supabase API roles `anon` and
`authenticated`, so the Data API is fail-closed:

1. **Row Level Security with no policies.** These roles see no rows and cannot write.
2. **`REVOKE ALL` from both roles.** This also blocks `TRUNCATE`, `REFERENCES` and
   `TRIGGER`, which RLS does not govern.
3. **Verification.** The migration then checks that neither role still holds
   `SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES` or `TRIGGER` on any of
   the six tables, whether directly, via `PUBLIC` or via role membership. If one
   does, it raises an error and the whole migration rolls back. That can happen
   with a grant made by a different grantor, which `REVOKE` cannot remove.

Steps 2–3 are skipped where these roles do not exist (plain PostgreSQL).
`postgres` and `service_role` are not touched, and `FORCE ROW LEVEL SECURITY` is
not used.

The backend must therefore connect as **the same database role that ran the
migration**. That role owns the tables, so RLS does not apply to it. A different,
non-owner role would see every table as empty, including `alembic_version`. The
API would then refuse to start with "schema is not up to date" even though the
schema is current, until policies or `BYPASSRLS` are added.

*Future migrations:* every new table must also enable RLS, revoke `anon` and
`authenticated`, and be included in the privilege verification. The offline tests
enforce all three for every model table. A new `serial`/identity column also
creates a sequence, which needs its own `REVOKE ALL ON SEQUENCE … FROM anon,
authenticated`. That is not needed yet, because all keys are UUIDs.

## Running the backend
From `backend/` with the virtual environment active:

```powershell
python -m app.serve
```

It listens on `127.0.0.1:8000` only and exits with a clear message if
configuration, the database, or an MCP server is unavailable. Example:

```powershell
# create a conversation, then send a message (token from your .env)
curl.exe -X POST http://127.0.0.1:8000/v1/conversations -H "Authorization: Bearer $env:NEXUS_DEV_TOKEN" -H "Content-Type: application/json" -d "{}"
curl.exe -X POST http://127.0.0.1:8000/v1/conversations/<id>/messages -H "Authorization: Bearer $env:NEXUS_DEV_TOKEN" -H "Content-Type: application/json" -d "{\"content\": \"Find papers on retrieval augmented generation using Crossref\"}"
```

Endpoints: `GET /health`, and under `/v1`: `POST|GET /conversations`,
`GET|DELETE /conversations/{id}`, `GET|POST /conversations/{id}/messages`.
Interactive API docs are intentionally disabled.

**Messages.** Each message has `status` (`completed` or `failed`), `error_code`
and `tool_executions` (`tool_name`, `status`, `created_at`; arguments and results
are never stored). If the model fails, the user message is kept and a **failed
assistant message** is stored with a fixed description and an `error_code`
(`model_timeout`, `model_unreachable`, `model_error`, `internal_error`). The next
message works normally, and the failed exchange is left out of the model's history.

**Errors.** Every JSON error has this shape, and every response carries the same
id in an `X-Request-ID` header:

```json
{"error": {"code": "not_found", "message": "Resource not found.", "request_id": "3f9c1a2b7d4e", "details": null}}
```

| Status | `code` | `details` |
|---|---|---|
| 401 | `unauthenticated` | — |
| 404 | `not_found` | — |
| 405 | `method_not_allowed` | — |
| 409 | `conversation_busy` | — |
| 413 | `request_too_large` | — |
| 422 | `validation_error` | `[{"loc", "msg", "type"}]` (submitted values are never echoed) |
| 422 | `credential_rejected` | — |
| 503 | `agent_unavailable` | `{"reason", "user_message_id", "assistant_message_id"}` |
| 500 | `internal_error` | — |

One exception: a request with an unexpected `Host` header gets a plain-text 400
from the host check before routing.

## Testing
From `backend/` after `pip install -r requirements-dev.txt`.

**Offline** (no network, no Ollama, no PostgreSQL):

```powershell
python -m unittest app.mcp.test_argument_policy app.mcp.test_permissions app.agent.test_agent app.agent.test_github_prompt_eval app.agent.test_grounding_judge app.agent.test_research_assessment app.agent.test_runtime app.core.test_core app.database.test_database app.services.test_conversation_service app.api.test_api
cd ..
backend\.venv\Scripts\python.exe -m unittest discover -s tests
```

**PostgreSQL RLS** (needs a disposable, empty, *local* PostgreSQL database reached
as a superuser; never Supabase). This test has not yet been run against PostgreSQL.

```powershell
$env:NEXUS_TEST_PG_URL = "postgresql://postgres@localhost:5432/nexus_rls_test"
python -m app.database.postgres_rls_integration_test
```

The URL is read only from `NEXUS_TEST_PG_URL`, never from `.env`. Before changing
anything, the test refuses to run unless all of these hold:

- the URL host is loopback;
- the database name starts with `test_` or ends with `_test`;
- the URL has no query parameters except `sslmode` and `connect_timeout`
  (parameters such as `host`, `hostaddr` or `service` could redirect the connection);
- `PGHOSTADDR`, `PGSERVICE` and `PGSERVICEFILE` are unset;
- after connecting, the server address is loopback (or a Unix socket);
- the public schema is empty, and nothing else would stop the server being
  restored exactly.

Afterwards it restores default privileges exactly and drops only the roles it
created, printing each one when it creates it. It keeps `alembic_version` if a
downgrade fails. It reports the checks and the cleanup separately.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Checks passed and cleanup is complete (also used for a skip) |
| 1 | Refused, or checks failed |
| 2 | Cleanup is incomplete |

If a run is killed before its cleanup, the next run refuses. To clean up, connect
as a superuser to a *different* database on that server and run:

```sql
DROP DATABASE nexus_rls_test;    -- removes its tables and default privileges
CREATE DATABASE nexus_rls_test;  -- if you want to reuse it
DROP ROLE IF EXISTS nexus_rls_test_owner;
DROP ROLE anon;                  -- ONLY if the run printed that it created it
DROP ROLE authenticated;         -- likewise
```

**Live** (need Ollama and network; some need GitHub credentials in `.env`):

| Command (from `backend/`) | Checks |
|---|---|
| `python -m app.mcp.github_integration_test` | GitHub MCP connection and a file read |
| `python -m app.mcp.multi_server_integration_test` | GitHub + Research through one manager (includes an OpenAlex call) |
| `python -m app.agent.github_agent_integration_test` | Qwen3 reads `.gitignore` through the agent |
| `python -m app.agent.github_prompt_eval --history-file PATH` | Measures how often the model picks the GitHub tool correctly, comparing the original and revised system prompts, with and without the conversation's real introductory messages (a JSON file you export; see the module docstring), plus negative controls. **Not offline:** it always connects to the GitHub MCP endpoint (to discover tools) and to Ollama. Tool execution is stubbed: calls are checked by the real policy and guard, but nothing is fetched from GitHub and no research provider is called (a `README.md` request is answered from the local checkout's `README.md`; any other file gets a placeholder). After the exact README exchange it runs two follow-ups (the original implicit question and an explicit "fetch README.md again" question) and two no-tool controls. Follow-ups report **retrieval** (gated: `README.md` re-fetched with the exact owner, repo and path, and every quotation verbatim) and **accuracy** (advisory only: a lexical comparison of the answer with the README text that run retrieved; not semantic proof, never gated) separately. Controls report unnecessary retrieval without failing. `--seed N` (default 0) sets the Ollama seed; repeat with different seeds to measure variation. `--execute` runs real calls, which may contact GitHub, OpenAlex and Crossref. `--json OUT` writes per-case results after the summary; it never overwrites without `--force` and never creates directories. |
| `python -m app.agent.crossref_agent_integration_test` | Qwen3 searches Crossref and cites returned records |
| `python -m app.agent.ollama_integration_test` | Same for OpenAlex (see limitation) |
| `python -m app.live_app_test` | Whole application: migration, runtime, API, Ollama |
| `python readme_followup_e2e.py [--case implicit\|explicit\|both]` | Needs the API running. For each case, a fresh conversation: README summary, then a known-limitations follow-up (implicit = the original question verbatim; explicit = "fetch README.md again"). Retrieval and integrity checks set the exit code. Accuracy is an advisory comparison with the **local** `README.md`, which may differ from GitHub's default branch. The API does not expose the path that was read, so a successful GitHub call does not prove which file was used or that the answer is accurate. |

**Known limitation:** OpenAlex rate-limits unauthenticated clients (HTTP 429). The
OpenAlex tests then report a provider failure, not a success. The MCP response
hides the upstream cause; check the Research server's stderr.

## Known limitations
- Authentication is a single static local token: anyone holding it *is* the user.
  Do not expose the API to a network.
- Not yet tested against PostgreSQL.
- No rate limiting or per-user quotas; the per-conversation lock is in-memory, so
  run a single worker process.
- Research records contain titles and metadata but no abstracts, so model summaries
  of a paper's content are inference from its title.
- Long-term memory is schema only.

## Repository layout
```
backend/app/        application code (api, services, database, agent, mcp, core, schemas)
backend/alembic/    database migrations
custom_mcp_servers/ local MCP servers (research_server)
tests/              offline tests for the research providers
docs/, reports/, frontend/   placeholders
```
