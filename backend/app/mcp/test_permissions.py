"""
Offline unit tests for MCP tool permissions.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.mcp.test_permissions -v
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.permissions import ToolPolicy

GITHUB_FILE = "github.get_file_contents"
GITHUB_SEARCH = "github.search_repositories"
SLACK_SEARCH = "slack.search_messages"


def make_manager(policy: ToolPolicy | None) -> MCPConnectionManager:
    """Manager whose SDK group is replaced by an offline fake."""
    manager = MCPConnectionManager(policy)
    manager._group = SimpleNamespace(
        tools={
            name: SimpleNamespace(name=name)
            for name in (GITHUB_FILE, GITHUB_SEARCH, SLACK_SEARCH)
        },
        call_tool=AsyncMock(return_value="ok"),
    )
    return manager


class ToolPolicyTests(unittest.TestCase):
    def test_empty_policy_denies_everything(self) -> None:
        policy = ToolPolicy()
        self.assertFalse(policy.is_allowed(GITHUB_FILE))
        with self.assertRaises(PermissionError):
            policy.check(GITHUB_FILE)

    def test_namespace_mismatch_rejected_at_configuration(self) -> None:
        with self.assertRaises(ValueError):
            ToolPolicy({"github": [SLACK_SEARCH]})

    def test_unqualified_name_is_denied(self) -> None:
        self.assertFalse(ToolPolicy({"github": [GITHUB_FILE]}).is_allowed("get_file_contents"))

    def test_independent_per_server_allowlists(self) -> None:
        policy = ToolPolicy({"github": [GITHUB_FILE], "slack": [SLACK_SEARCH]})
        self.assertTrue(policy.is_allowed(GITHUB_FILE))
        self.assertTrue(policy.is_allowed(SLACK_SEARCH))
        self.assertFalse(policy.is_allowed(GITHUB_SEARCH))


class ConnectionManagerPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_approved_tool_reaches_routing_layer(self) -> None:
        manager = make_manager(ToolPolicy({"github": [GITHUB_FILE]}))
        args = {"owner": "o", "repo": "r", "path": "p"}

        result = await manager.call_tool(GITHUB_FILE, args)

        self.assertEqual(result, "ok")
        manager._group.call_tool.assert_awaited_once_with(GITHUB_FILE, arguments=args)

    async def test_disallowed_tool_rejected_without_sdk_call(self) -> None:
        manager = make_manager(ToolPolicy({"github": [GITHUB_FILE]}))

        with self.assertRaises(PermissionError):
            await manager.call_tool(GITHUB_SEARCH, {})

        manager._group.call_tool.assert_not_awaited()

    async def test_unknown_tool_rejected_without_sdk_call(self) -> None:
        manager = make_manager(ToolPolicy({"github": ["github.delete_repo"]}))

        with self.assertRaises(KeyError):
            await manager.call_tool("github.delete_repo", {})

        manager._group.call_tool.assert_not_awaited()

    async def test_other_server_cannot_bypass_namespace(self) -> None:
        # Only GitHub is approved; the identically named-style Slack tool is not.
        manager = make_manager(ToolPolicy({"github": [GITHUB_FILE]}))

        with self.assertRaises(PermissionError):
            await manager.call_tool(SLACK_SEARCH, {})

        manager._group.call_tool.assert_not_awaited()

    async def test_empty_allowlist_denies_every_tool(self) -> None:
        for policy in (None, ToolPolicy(), ToolPolicy({"github": []})):
            manager = make_manager(policy)
            for name in manager.discovered_tools:
                with self.assertRaises(PermissionError):
                    await manager.call_tool(name, {})
            manager._group.call_tool.assert_not_awaited()

    async def test_discovered_vs_permitted_catalogues(self) -> None:
        manager = make_manager(ToolPolicy({"github": [GITHUB_FILE]}))

        self.assertEqual(len(manager.discovered_tools), 3)
        self.assertEqual(list(manager.permitted_tools), [GITHUB_FILE])


if __name__ == "__main__":
    unittest.main()
