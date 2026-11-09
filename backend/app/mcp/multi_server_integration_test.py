"""
Multi-server MCP integration test (needs network and GitHub credentials).

Connects GitHub (remote Streamable HTTP) and Research (local STDIO) through
one MCPConnectionManager and runs one permitted tool on each server.

Run from backend/:
    .venv/Scripts/python.exe -m app.mcp.multi_server_integration_test
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from app.mcp.connection_manager import PROJECT_ROOT, MCPConnectionManager
from app.mcp.permissions import default_policy

EXPECTED_TOOLS = {
    "github.get_file_contents",
    "research.search_openalex_works",
    "research.search_crossref_works",
}


def _structured(result) -> dict:
    """Structured tool output, falling back to JSON in the text content."""
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def _result_text(result) -> str:
    """Join text parts, including text carried by embedded resources."""
    parts = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            text = getattr(getattr(item, "resource", None), "text", None)
        parts.append(text or "")
    return "\n".join(parts)


def _check(condition: bool, label: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


async def run() -> int:
    token = os.getenv("GITHUB_TOKEN", "")
    url = os.getenv("GITHUB_MCP_URL", "")
    if not token or not url:
        print("FAIL: GITHUB_TOKEN or GITHUB_MCP_URL missing from project .env")
        return 1

    # Only CROSSREF_MAILTO is forwarded; GITHUB_TOKEN never reaches the child.
    research_env = {}
    if os.getenv("CROSSREF_MAILTO"):
        research_env["CROSSREF_MAILTO"] = os.environ["CROSSREF_MAILTO"]

    failures: list[str] = []

    try:
        async with MCPConnectionManager(default_policy()) as manager:
            await manager.connect_http_server(
                "github", url, token, read_only=True
            )
            await manager.connect_stdio_server(
                "research",
                ["-m", "custom_mcp_servers.research_server.server"],
                cwd=PROJECT_ROOT,
                env=research_env or None,
            )

            discovered = manager.discovered_tools
            permitted = manager.permitted_tools
            print(f"Discovered tools: {len(discovered)}")
            print(f"Permitted tools:  {len(permitted)} -> {sorted(permitted)}")

            prefixes = {name.split(".", 1)[0] for name in discovered}
            _check({"github", "research"} <= prefixes, "both namespaces present", failures)
            _check(
                any(n.startswith("github.") for n in discovered)
                and any(n.startswith("research.") for n in discovered),
                "GitHub and Research tools coexist in one catalogue",
                failures,
            )
            _check(set(permitted) == EXPECTED_TOOLS, "exactly the 3 intended tools permitted", failures)

            # 1. GitHub: read .gitignore
            gh = await manager.call_tool(
                "github.get_file_contents",
                {"owner": "Raj-pratap07", "repo": "My-Nexus", "path": ".gitignore"},
            )
            gh_text = _result_text(gh)
            _check(not gh.is_error and ".env" in gh_text, "GitHub .gitignore read", failures)

            # 2. Research providers
            for tool, source in (
                ("research.search_openalex_works", "openalex"),
                ("research.search_crossref_works", "crossref"),
            ):
                res = await manager.call_tool(
                    tool, {"query": "retrieval augmented generation", "limit": 3}
                )
                if res.is_error:
                    _check(False, f"{tool} executed", failures)
                    print("    error:", getattr(res.content[0], "text", "")[:200])
                    continue
                data = _structured(res)
                records = data["records"]
                print(f"  {tool}: {data['result_count']} records; first title: "
                      f"{(records[0]['title'] if records else None)!r}")
                _check(
                    data["source"] == source
                    and data["result_count"] == len(records) > 0
                    and all(r["source"] == source for r in records),
                    f"{source} results carry provenance",
                    failures,
                )

            # Policy still blocks everything else, end to end.
            blocked = [n for n in discovered if n not in EXPECTED_TOOLS][:1]
            for name in blocked:
                try:
                    await manager.call_tool(name, {})
                    _check(False, f"{name} denied", failures)
                except PermissionError:
                    _check(True, f"{name} denied by policy", failures)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}".replace(token, "***")
        print("FAIL: unexpected error:", msg)
        return 1

    print("\nAll connections closed.")
    if failures:
        print(f"FAILED checks: {failures}")
        return 1
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
