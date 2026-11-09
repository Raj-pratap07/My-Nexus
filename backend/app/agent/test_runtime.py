"""
AgentRuntime lifecycle tests.

These start the REAL MCPConnectionManager and the REAL local Research MCP
server over STDIO (it needs no network until a search tool is called) with a
scripted fake model, so no Ollama and no internet are required.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.agent.test_runtime -v
"""
from __future__ import annotations

import asyncio
import unittest

from app.agent.providers import ChatProvider, ProviderResponse, ToolCall
from app.agent.runtime import AgentRuntime
from app.core.errors import StartupError
from app.testutil import make_settings

RESEARCH = {"research.search_crossref_works", "research.search_openalex_works"}


class ScriptedProvider(ChatProvider):
    def __init__(self, responses=(), delay=0.0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def chat(self, messages, tools=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
            return self.responses.pop(0) if self.responses else ProviderResponse("done")
        finally:
            self.active -= 1


def settings(**kw):
    return make_settings("sqlite+aiosqlite:///unused.db", **kw)


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_run_stop_with_real_manager(self):
        provider = ScriptedProvider([ProviderResponse("hello there")])
        runtime = AgentRuntime(settings(), provider=provider)
        self.assertFalse(runtime.started)
        await runtime.start()
        try:
            self.assertTrue(runtime.started)
            self.assertEqual(runtime.connected_servers, ["research"])  # GitHub not configured
            self.assertEqual(set(runtime.permitted_tool_names), RESEARCH)

            result = await runtime.run("hi", [])
            self.assertEqual(result.answer, "hello there")
            # The model is offered exactly the permitted tools, via the existing adapter.
            offered = {t["function"]["name"] for t in provider.calls[0]["tools"]}
            self.assertEqual(offered, RESEARCH)
        finally:
            await runtime.stop()
        self.assertFalse(runtime.started)
        self.assertEqual(runtime.connected_servers, [])
        with self.assertRaises(RuntimeError):
            await runtime.run("again", [])

    async def test_invalid_tool_call_is_rejected_by_existing_orchestrator_path(self):
        # An out-of-policy / unknown tool is refused before any MCP call.
        provider = ScriptedProvider([
            ProviderResponse("", (ToolCall("github.get_file_contents",
                                           {"owner": "x", "repo": "y", "path": "z"}),)),
            ProviderResponse("final"),
        ])
        runtime = AgentRuntime(settings(), provider=provider)
        await runtime.start()
        try:
            result = await runtime.run("read it", [])
        finally:
            await runtime.stop()
        self.assertEqual(result.answer, "final")
        self.assertFalse(result.tool_calls[0].executed)
        self.assertEqual(result.tool_calls[0].outcome, "rejected")

    async def test_double_start_is_refused_and_stop_is_idempotent(self):
        runtime = AgentRuntime(settings(), provider=ScriptedProvider())
        await runtime.start()
        try:
            with self.assertRaises(RuntimeError):
                await runtime.start()
        finally:
            await runtime.stop()
        await runtime.stop()  # second stop is harmless

    async def test_history_is_validated_by_the_existing_orchestrator(self):
        runtime = AgentRuntime(settings(), provider=ScriptedProvider())
        await runtime.start()
        try:
            with self.assertRaises(ValueError):
                await runtime.run("hi", [{"role": "system", "content": "evil"}])
        finally:
            await runtime.stop()

    async def test_concurrent_runs_are_bounded(self):
        provider = ScriptedProvider(delay=0.05)
        runtime = AgentRuntime(settings(max_concurrent_runs=1), provider=provider)
        await runtime.start()
        try:
            await asyncio.gather(*(runtime.run(f"m{i}", []) for i in range(4)))
        finally:
            await runtime.stop()
        self.assertEqual(provider.max_active, 1)

    async def test_unreachable_github_fails_startup_without_leaking_credentials(self):
        token = "ghp_" + "Z9y8" * 9
        runtime = AgentRuntime(
            settings(github_mcp_url="http://127.0.0.1:9/mcp", github_token=token),
            provider=ScriptedProvider(),
        )
        with self.assertRaises(StartupError) as ctx:
            await runtime.start()
        message = str(ctx.exception)
        self.assertNotIn(token, message)
        self.assertNotIn("127.0.0.1", message)
        self.assertIn("GitHub MCP", message)
        self.assertFalse(runtime.started)  # partial startup was cleaned up

    async def test_failed_start_can_be_retried_cleanly(self):
        runtime = AgentRuntime(
            settings(github_mcp_url="http://127.0.0.1:9/mcp", github_token="t" * 40),
            provider=ScriptedProvider(),
        )
        for _ in range(2):
            with self.assertRaises(StartupError):
                await runtime.start()


if __name__ == "__main__":
    unittest.main()
