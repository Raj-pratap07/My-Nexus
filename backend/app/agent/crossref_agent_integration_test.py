"""
Live Crossref-through-the-agent test (needs Ollama + qwen3:4b-instruct and
network access; no GitHub credentials needed).

Separate from the OpenAlex test: Crossref results are never counted as
OpenAlex results, and vice versa. Only research.search_crossref_works is
asked for. Connection and retrieval are reported separately.

Exit codes: 0 = success, 1 = test/connection failure,
            2 = connected fine but the provider failed (error/empty result).

Run from backend/:
    .venv/Scripts/python.exe -m app.agent.crossref_agent_integration_test
"""
from __future__ import annotations

import asyncio
import os
import sys

from app.agent.ollama_integration_test import (
    _check,
    _ollama_has_model,
    assess_research_result,
    cited_records,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.providers import OllamaProvider
from app.mcp.connection_manager import PROJECT_ROOT, MCPConnectionManager
from app.mcp.permissions import default_policy

TOOL = "research.search_crossref_works"
OTHER_RESEARCH_TOOL = "research.search_openalex_works"
QUESTION = (
    "Use Crossref to find scholarly works about retrieval augmented "
    "generation, then briefly list the titles you found and what they are "
    "about."
)


async def run() -> int:
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
            await manager.connect_stdio_server(
                "research",
                ["-m", "custom_mcp_servers.research_server.server"],
                cwd=PROJECT_ROOT,
                env=research_env or None,
            )
            permitted = set(manager.permitted_tools)
            print(f"  Permitted tools: {sorted(permitted)}")
            _check(TOOL in permitted, f"research server connected; {TOOL} permitted",
                   conn_failures)

            real_call_tool = manager.call_tool

            async def spy(name, arguments=None):
                executed.append(name)
                print(f"  -> call_tool({name}, {arguments})")
                result = await real_call_tool(name, arguments)
                if name == TOOL:
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

                _check(TOOL in executed,
                       f"model requested {TOOL} via call_tool", retr_failures)
                _check(OTHER_RESEARCH_TOOL not in executed,
                       "OpenAlex was not used (no cross-provider substitution)",
                       retr_failures)
                _check(set(executed) <= permitted and len(executed) <= 4,
                       "only permitted tools, within the limit", retr_failures)

                ok = bool(assessments) and assessments[-1][0]
                reason = assessments[-1][1] if assessments else "tool was never executed"
                print(f"  crossref result: {reason}")
                if assessments and not ok:
                    provider_failure = reason
                _check(ok, "at least one valid Crossref record retrieved", retr_failures)
                if ok:
                    _check(all(r.get("source") == "crossref" for r in retrieved),
                           "all records carry source=crossref", retr_failures)
                    cited = cited_records(result.answer, retrieved)
                    print(f"  answer cites {len(cited)} of {len(retrieved)} returned records")
                    for title in cited[:5]:
                        print(f"    - {title[:100]}")
                    _check(bool(cited),
                           "final answer references a returned title or DOI",
                           retr_failures)
    except Exception as exc:  # noqa: BLE001
        print("FAIL: unexpected error:", f"{type(exc).__name__}: {exc}")
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
