# My-Nexus frontend

React + TypeScript + Vite. Two parts:

- **Landing page** (`/`): the BLUEGOLD page with the GPU particle sphere (`src/scene/`, three.js in a lazy chunk).
- **Workspace** (`/nexus`, `/nexus/c/<id>`): conversations with the local FastAPI backend. A separate lazy chunk that never loads three.js.

Requires Node 22.12 or newer. Run all commands below from `frontend/`.

## Running against the backend

1. Start the backend as described in the project README (`python -m app.serve` in `backend/`). It listens on `http://127.0.0.1:8000`.
2. Start the frontend with the backend token supplied at runtime. Choose one:

   ```powershell
   npm run dev:secure          # asks for the token with input hidden
   ```

   or, if the token is already in this terminal's environment:

   ```powershell
   $env:NEXUS_DEV_TOKEN = "<the backend's NEXUS_DEV_TOKEN>"   # PowerShell
   npm run dev
   ```

   `dev:secure` keeps the token out of shell history and files. It also accepts the token on stdin, for example piped from a password manager.
3. Open `http://127.0.0.1:5173/` and choose **Enter Nexus**.

To use a built copy instead, run `npm run build`, then `npm run preview:secure` (or `npm run preview` with the variable set). Then open `http://127.0.0.1:4173/`.

No `.env` file is read or needed by the frontend. If the backend runs somewhere other than `http://127.0.0.1:8000`, set `NEXUS_BACKEND_URL`. It must be an `http://` URL on a loopback address.

## How the credential is kept out of the browser

```
browser ──(same origin, no credential)──▶ Vite dev/preview server ──(Authorization: Bearer …)──▶ FastAPI on 127.0.0.1
```

`dev/nexusApiProxy.ts` is a Vite plugin that serves `/api` on the dev and preview servers. Browser code only ever calls `/api/...`.

- **Token handling**
  - The proxy reads `NEXUS_DEV_TOKEN` from the server process environment and adds the `Authorization` header server-side.
  - The token is never logged, never returned in a response, and never placed in Vite config. Vite exposes only `VITE_*` variables to client code.
- **What gets forwarded**
  - Only `/api/health` and `/api/v1/*` are forwarded, with `GET`, `POST` and `DELETE`.
  - Outgoing headers are an allowlist (`Content-Type`, `Accept`). The browser's `Authorization`, `Cookie` and everything else are dropped.
  - Bodies over 64 KB are refused.
- **Exact same-origin checks**
  - The `Host` must be this loopback server.
  - `Origin`, when sent, must match it exactly.
  - `Sec-Fetch-Site`, when sent, must be `same-origin`.
  - Every request must carry `X-Nexus-Client: web`. A cross-origin page cannot add that header without a CORS preflight.
  - Preflights (`OPTIONS`) are rejected, and no CORS headers are ever sent.
- **Loopback only:**
  - the dev and preview servers must bind to loopback, or the plugin refuses to start (so never use `--host`);
  - the backend URL must be loopback;
  - the connecting peer must be loopback.
- **Errors the proxy itself returns** use the backend's envelope shape, plus an `X-Nexus-Proxy: 1` header:

  | Status | Code | Meaning |
  |---|---|---|
  | 503 | `proxy_not_configured` | no token |
  | 502 | `backend_unreachable` | connection refused: the backend is down, and nothing reached it |
  | 502 | `backend_connection_lost` | the connection closed mid-request: the backend may still be processing |
  | 504 | `proxy_timeout` | the backend was silent for 30 minutes |
  | 403 | `proxy_forbidden` | the same-origin checks failed |

`npm run check:bundle` builds with a random sentinel token and fails if the sentinel, the variable names, or an `Authorization` header literal appear anywhere in `dist/`.

### Security limits: read before using

- **Development only.** This is a single-user local development proxy, not production authentication. The backend has one fixed local identity.
- **Ambient-credential risk.** While the dev or preview server runs, any process on this computer that can reach `127.0.0.1:5173` (or `:4173`) can call the backend API without knowing the token. That includes other users' processes on a shared machine. The proxy turns the token into an ambient credential for the local machine.
  - The browser same-origin checks stop *web pages* from doing this. They cannot stop local programs, which can forge any header.
  - Stop the dev server when you are not using it, and don't run it on shared machines.
- **Token visibility.** A token set with `$env:NEXUS_DEV_TOKEN` is visible to every process started from that terminal. `dev:secure` limits it to the Vite process and its children.

## How turns behave

The backend is blocking and non-streaming. A turn can take minutes: up to 5 model calls at 180 s each by default, plus waiting for the single model slot.

- **While waiting**, the workspace shows the real elapsed time, not progress steps. Tool executions (name, status, time) appear with the reply, exactly as the backend stored them.
- **One send at a time** across the whole workspace, because the backend runs one model call at a time.
- **Failed turns.** The backend stores a failed assistant message with an `error_code`, which the workspace shows. **Send again** appears only on such a recorded failure, and only when you press it does anything get sent.
- **Uncertain turns.** A network error, a proxy timeout, a browser time limit or a reload does **not** prove the backend stopped. The workspace then:
  - never re-sends and never creates a new user message on its own;
  - re-reads the stored messages (every 5 s, then every 15 s) until a reply appears;
  - after 20 minutes (or 30 s if the message never appeared in storage), reports the outcome as **unknown** and offers **Refresh** and **Copy message**.

### Known limitation: orphaned user messages

The backend stores the user message before the model runs. If the backend stops mid-turn (crash or restart), that message stays in the conversation with no reply.

- **Indistinguishable from a slow turn.** Nothing tells the client whether such a message is still being processed or was abandoned. The workspace can only wait, then report "outcome unknown".
- **The model will see it.** If you continue the conversation, the unanswered message stays in the history sent to the model: the backend drops only *failed* pairs. The composer warns about this.

Fixing either needs a backend change (for example a stored "in progress" state), which is outside this milestone.

## Scripts

| Script | Purpose |
|---|---|
| `npm run dev` / `npm run dev:secure` | dev server on 127.0.0.1:5173 with the `/api` proxy |
| `npm run build` | production build into `dist/` |
| `npm run preview` / `npm run preview:secure` | serve `dist/` on 127.0.0.1:4173 with the same proxy |
| `npm run typecheck` | strict type check (browser code and Node-side code separately) |
| `npm test` | Vitest (jsdom; proxy tests run in Node against a local stub) |
| `npm run check:bundle` | build with a sentinel token and scan `dist/` for credential leaks |

The landing page also supports QA query parameters: `?scene=off|fail|lose`, `?bench` and `?particles=N`.
