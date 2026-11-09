"""
Local LLM + Research integration test (needs Ollama with qwen3:4b-instruct,
network, and GitHub credentials in the project .env).

Connects GitHub (remote) and Research (local STDIO) MCP servers, gives
Qwen3 only the permitted tools and asks a research question.

Two stages are reported separately:
  CONNECTION: servers connected, tools discovered, permissions as expected.
  RETRIEVAL : Qwen3 called research.search_openalex_works THROUGH
              MCPConnectionManager.call_tool() and the call returned valid
              records. A tool error (e.g. an upstream HTTP 429) or an empty
              result is a PROVIDER FAILURE, never a retrieval success. There
              are no automatic retries.

Exit codes: 0 = both stages succeeded, 1 = test/connection failure,
            2 = connected fine but retrieval failed (provider failure).

Run from backend/:
    .venv/Scripts/python.exe -m app.agent.ollama_integration_test
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from typing import Any

from app.agent.orchestrator import AgentOrchestrator
from app.agent.providers import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaProvider,
)
from app.mcp.connection_manager import PROJECT_ROOT, MCPConnectionManager
from app.mcp.permissions import default_policy

# Model answers contain arbitrary Unicode; never let a legacy console
# encoding (e.g. cp1252 on Windows) crash a live test while printing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

RESEARCH_TOOL = "research.search_openalex_works"
EXPECTED_TOOLS = {
    "github.get_file_contents",
    "research.search_openalex_works",
    "research.search_crossref_works",
}
QUESTION = (
    "Use OpenAlex to find recent scholarly works on retrieval augmented "
    "generation, then briefly summarize what the top results are about."
)


def _check(condition: bool, label: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def _ollama_has_model() -> bool:
    try:
        with urllib.request.urlopen(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=5) as r:
            names = {m.get("name") for m in json.load(r).get("models", [])}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: Ollama not reachable at {DEFAULT_OLLAMA_URL}: {exc}")
        return False
    if DEFAULT_OLLAMA_MODEL not in names:
        print(f"FAIL: model '{DEFAULT_OLLAMA_MODEL}' not installed (ollama pull).")
        return False
    return True


def _tool_text(result: Any) -> str:
    return " ".join(
        getattr(item, "text", None) or "" for item in (result.content or [])
    )


def assess_research_result(result: Any) -> tuple[bool, str]:
    """
    Decide whether a research tool result is a real retrieval.

    Returns (ok, reason). ok only if the call did not report an error and
    carries at least one record with a title, a source and an identifier.
    """
    if getattr(result, "is_error", False):
        text = _tool_text(result)
        if "429" in text or "rate limit" in text.lower():
            return False, "provider failure: HTTP 429 rate limit"
        return False, (
            "provider failure: research tool returned an error "
            f"({text[:120]!r}). The MCP response hides the upstream cause "
            "(e.g. HTTP 429); check the research server's stderr above."
        )

    data = getattr(result, "structured_content", None)
    if not isinstance(data, dict):
        return False, "malformed result: no structured content"
    records = data.get("records")
    if not isinstance(records, list) or not records:
        return False, "empty result: zero records returned"
    if data.get("result_count") != len(records):
        return False, "inconsistent result: result_count != number of records"

    for rec in records:
        valid = (
            isinstance(rec, dict)
            and isinstance(rec.get("title"), str) and rec["title"].strip()
            and isinstance(rec.get("source"), str) and rec["source"]
            and (rec.get("doi") or rec.get("url") or rec.get("source_id"))
        )
        if not valid:
            return False, "invalid record: missing title, source or identifier"
    return True, f"{len(records)} valid records"


def _norm(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.casefold()).split())


def cited_records(answer: str, records: list[dict]) -> list[str]:
    """
    Titles of returned records that the answer actually mentions (by
    normalized title at word boundaries, or by DOI). Deterministic.

    Longest titles are matched first and their text is blanked out, so a
    short title (e.g. "Retrieval-Augmented Generation") only counts if it
    appears OUTSIDE any longer matched title.
    """
    haystack = f" {_norm(answer)} "
    found = []
    ordered = sorted(records, key=lambda r: len(_norm(r.get("title") or "")), reverse=True)
    for rec in ordered:
        title = rec.get("title") or ""
        doi = rec.get("doi") or ""
        needle = f" {_norm(title)} "
        if _norm(title) and needle in haystack:
            found.append(title)
            haystack = haystack.replace(needle, " " * len(needle))
        elif doi and doi.casefold() in answer.casefold():
            found.append(title)
    return found


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

    conn_failures: list[str] = []
    retr_failures: list[str] = []
    provider_failure: str | None = None
    executed: list[str] = []
    assessments: list[tuple[bool, str]] = []
    retrieved: list[dict] = []

    try:
        async with MCPConnectionManager(default_policy()) as manager:
            print("== CONNECTION ==")
            await manager.connect_http_server("github", url, token, read_only=True)
            await manager.connect_stdio_server(
                "research",
                ["-m", "custom_mcp_servers.research_server.server"],
                cwd=PROJECT_ROOT,
                env=research_env or None,
            )
            permitted = set(manager.permitted_tools)
            print(f"  Permitted tools: {sorted(permitted)}")
            _check(permitted == EXPECTED_TOOLS,
                   "servers connected; only the 3 intended tools permitted",
                   conn_failures)

            real_call_tool = manager.call_tool

            async def spy(name, arguments=None):
                executed.append(name)
                print(f"  -> call_tool({name}, {arguments})")
                result = await real_call_tool(name, arguments)
                if name == RESEARCH_TOOL:
                    verdict = assess_research_result(result)
                    assessments.append(verdict)
                    if verdict[0]:
                        retrieved.extend(result.structured_content["records"])
                return result

            manager.call_tool = spy  # type: ignore[method-assign]

            if not conn_failures:
                print("\n== RETRIEVAL ==")
                print(f"User: {QUESTION}\n")
                result = await AgentOrchestrator(manager, OllamaProvider()).run(QUESTION)
                print(f"\nAssistant ({result.stop_reason}):\n{result.answer}\n")

                _check(RESEARCH_TOOL in executed,
                       f"model requested {RESEARCH_TOOL} via call_tool", retr_failures)
                _check(set(executed) <= permitted and len(executed) <= 4,
                       "only permitted tools, within the limit", retr_failures)

                ok = bool(assessments) and assessments[-1][0]
                reason = assessments[-1][1] if assessments else "tool was never executed"
                print(f"  research result: {reason}")
                if assessments and not ok:
                    provider_failure = reason
                _check(ok, "valid research records were retrieved", retr_failures)
                _check(bool(result.answer.strip()), "final answer returned", retr_failures)
                if ok:
                    cited = cited_records(result.answer, retrieved)
                    print(f"  answer cites {len(cited)} of {len(retrieved)} returned records")
                    _check(bool(cited),
                           "final answer mentions at least one returned record",
                           retr_failures)
    except Exception as exc:  # noqa: BLE001
        print("FAIL: unexpected error:", f"{type(exc).__name__}: {exc}".replace(token, "***"))
        return 1

    print("\nAll connections closed.")
    print(f"CONNECTION: {'SUCCESS' if not conn_failures else 'FAILED ' + str(conn_failures)}")
    if conn_failures:
        print("RETRIEVAL:  NOT RUN")
        return 1
    if retr_failures:
        print(f"RETRIEVAL:  FAILED {retr_failures}")
        if provider_failure:
            print(f"            cause: {provider_failure}")
            return 2
        return 1
    print("RETRIEVAL:  SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
