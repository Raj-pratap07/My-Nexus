"""
Live GitHub-through-the-agent test (needs Ollama + qwen3:4b-instruct,
network, and GitHub credentials in the project .env).

Asks Qwen3 to fetch .gitignore from Raj-pratap07/My-Nexus using only the
already permitted github.get_file_contents tool.

Run from backend/:
    .venv/Scripts/python.exe -m app.agent.github_agent_integration_test
"""
from __future__ import annotations

import asyncio
import os
import sys

from app.agent.ollama_integration_test import _check, _ollama_has_model
from app.agent.orchestrator import AgentOrchestrator, extract_result_text
from app.agent.providers import OllamaProvider
from app.mcp.connection_manager import PROJECT_ROOT, MCPConnectionManager
from app.mcp.permissions import default_policy

GH_TOOL = "github.get_file_contents"
QUESTION = (
    "Retrieve the file .gitignore from the GitHub repository "
    "Raj-pratap07/My-Nexus and show me its contents."
)
# Distinctive lines of the file; at least MIN_MATCH must reach the answer.
MARKERS = (".venv/", "__pycache__/", ".env", ".pytest_cache/", "node_modules")
MIN_MATCH = 3


async def run() -> int:
    token = os.getenv("GITHUB_TOKEN", "")
    url = os.getenv("GITHUB_MCP_URL", "")
    if not token or not url:
        print("FAIL: GITHUB_TOKEN or GITHUB_MCP_URL missing from project .env")
        return 1
    if not _ollama_has_model():
        return 1

    research_env = {}
    if os.getenv("CROSSREF_MAILTO"):
        research_env["CROSSREF_MAILTO"] = os.environ["CROSSREF_MAILTO"]

    failures: list[str] = []
    executed: list[tuple[str, dict]] = []

    try:
        async with MCPConnectionManager(default_policy()) as manager:
            await manager.connect_http_server("github", url, token, read_only=True)
            await manager.connect_stdio_server(
                "research",
                ["-m", "custom_mcp_servers.research_server.server"],
                cwd=PROJECT_ROOT,
                env=research_env or None,
            )
            permitted = set(manager.permitted_tools)
            _check(len(permitted) == 3 and GH_TOOL in permitted,
                   "permitted set is the 3 intended tools", failures)

            # Ground truth fetched directly, before the spy is installed.
            reference = extract_result_text(await manager.call_tool(
                GH_TOOL,
                {"owner": "Raj-pratap07", "repo": "My-Nexus", "path": ".gitignore"},
            ))
            present = [m for m in MARKERS if m in reference]
            _check(len(present) >= MIN_MATCH, "reference file has the markers", failures)

            real_call_tool = manager.call_tool

            async def spy(name, arguments=None):
                executed.append((name, dict(arguments or {})))
                print(f"  -> call_tool({name}, {arguments})")
                return await real_call_tool(name, arguments)

            manager.call_tool = spy  # type: ignore[method-assign]

            result = await AgentOrchestrator(manager, OllamaProvider()).run(QUESTION)
            print(f"\nAssistant ({result.stop_reason}):\n{result.answer}\n")

            names = [n for n, _ in executed]
            _check(GH_TOOL in names, "model requested github.get_file_contents", failures)
            _check(set(names) <= permitted and len(names) <= 4,
                   "only permitted tools, within the limit, via call_tool", failures)
            gh_args = next((a for n, a in executed if n == GH_TOOL), {})
            _check(gh_args.get("owner") == "Raj-pratap07"
                   and gh_args.get("repo") == "My-Nexus"
                   and gh_args.get("path") == ".gitignore",
                   "arguments target Raj-pratap07/My-Nexus .gitignore", failures)
            hits = [m for m in present if m in result.answer]
            print(f"  markers in answer: {hits} of {present}")
            _check(len(hits) >= MIN_MATCH,
                   "file content reached the final answer", failures)
    except Exception as exc:  # noqa: BLE001
        print("FAIL: unexpected error:", f"{type(exc).__name__}: {exc}".replace(token, "***"))
        return 1

    print("\nAll connections closed.")
    if failures:
        print(f"FAILED checks: {failures}")
        return 1
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
