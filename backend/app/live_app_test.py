"""
Live end-to-end test of the application (needs Ollama with qwen3:4b-instruct
and network access for Crossref; GitHub is used only if configured in .env).

Uses a throw-away SQLite database created by the real Alembic migration (so
the startup schema check is ON), the real AgentRuntime / MCP servers / Ollama,
and the real HTTP API through a TestClient. Crossref is used, not OpenAlex.

Run from backend/:
    .venv/Scripts/python.exe -m app.api.live_app_test
"""
from __future__ import annotations

import os
import sys
import uuid

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.agent.ollama_integration_test import _check, _ollama_has_model, cited_records  # noqa: F401
from app.core.config import PROJECT_ROOT, Settings
from app.database.session import BACKEND_DIR
from app.main import create_app
from app.testutil import TEST_TOKEN, TempDb

QUESTION = (
    "Use Crossref to find scholarly works about retrieval augmented "
    "generation and list the titles you found."
)


def main() -> int:
    if not _ollama_has_model():
        return 1

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    db = TempDb()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db.url)
    command.upgrade(cfg, "head")  # real migration; schema check stays enabled

    settings = Settings(
        database_url=db.url,
        dev_token=TEST_TOKEN,
        allowed_hosts=("localhost", "127.0.0.1", "[::1]", "testserver"),
        github_mcp_url=os.getenv("GITHUB_MCP_URL") or None,
        github_token=os.getenv("GITHUB_TOKEN") or None,
        crossref_mailto=os.getenv("CROSSREF_MAILTO") or None,
    )
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    failures: list[str] = []

    try:
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as client:
            runtime = app.state.runtime
            print(f"Servers: {runtime.connected_servers}; tools: {runtime.permitted_tool_names}")
            _check(client.get("/health").json() == {"status": "ok"}, "health", failures)
            _check(client.get("/v1/conversations").status_code == 401, "auth required", failures)

            conv = client.post("/v1/conversations", json={}, headers=headers).json()
            r = client.post(f"/v1/conversations/{conv['id']}/messages",
                            json={"content": QUESTION}, headers=headers)
            _check(r.status_code == 200, f"message turn returned 200 (got {r.status_code})", failures)
            if r.status_code == 200:
                body = r.json()
                print(f"\nAssistant ({body['stop_reason']}):\n{body['assistant_message']['content']}\n")
                tools = {(t["tool_name"], t["status"]) for t in body["tool_executions"]}
                print(f"  tool executions: {sorted(tools)}")
                _check(("research.search_crossref_works", "succeeded") in tools,
                       "Crossref tool ran and succeeded (recorded)", failures)
                _check(body["assistant_message"]["content"].strip() != "", "non-empty answer", failures)

                msgs = client.get(f"/v1/conversations/{conv['id']}/messages", headers=headers).json()
                _check([m["role"] for m in msgs] == ["user", "assistant"],
                       "history persisted as user, assistant", failures)
                title = client.get(f"/v1/conversations/{conv['id']}", headers=headers).json()["title"]
                _check(title != "New conversation", "title derived from first message", failures)

                r2 = client.post(f"/v1/conversations/{conv['id']}/messages",
                                 json={"content": "Which of those is the most recent?"}, headers=headers)
                _check(r2.status_code == 200, "second turn uses stored history", failures)
        _check(not app.state.runtime.started, "runtime stopped on shutdown", failures)
    except Exception as exc:  # noqa: BLE001
        print("FAIL: unexpected error:", type(exc).__name__, str(exc)[:200])
        return 1
    finally:
        db.cleanup()

    print("\nSUCCESS" if not failures else f"\nFAILED checks: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
