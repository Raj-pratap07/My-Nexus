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
(The migration has been verified on SQLite only; it has not yet been run on PostgreSQL.)

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

## Testing
From `backend/` after `pip install -r requirements-dev.txt`.

**Offline** (no network, no Ollama, no PostgreSQL):

```powershell
python -m unittest app.mcp.test_argument_policy app.mcp.test_permissions app.agent.test_agent app.agent.test_research_assessment app.agent.test_runtime app.core.test_core app.database.test_database app.services.test_conversation_service app.api.test_api
cd ..
backend\.venv\Scripts\python.exe -m unittest discover -s tests
```

**Live** (need Ollama and network; some need GitHub credentials in `.env`):

| Command (from `backend/`) | Checks |
|---|---|
| `python -m app.mcp.github_integration_test` | GitHub MCP connection and a file read |
| `python -m app.mcp.multi_server_integration_test` | GitHub + Research through one manager (includes an OpenAlex call) |
| `python -m app.agent.github_agent_integration_test` | Qwen3 reads `.gitignore` through the agent |
| `python -m app.agent.crossref_agent_integration_test` | Qwen3 searches Crossref and cites returned records |
| `python -m app.agent.ollama_integration_test` | Same for OpenAlex (see limitation) |
| `python -m app.live_app_test` | Whole application: migration, runtime, API, Ollama |

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
