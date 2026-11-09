"""
Offline tests for GitHub argument-level authorization.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.mcp.test_argument_policy -v

Every rejected call must fail before reaching the (fake) MCP server.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent.orchestrator import AgentOrchestrator
from app.agent.providers import ChatProvider, ProviderResponse, ToolCall
from app.mcp.argument_policy import (
    ArgumentPolicyError,
    github_file_guard,
    normalize_repo_path,
)
from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.permissions import ToolPolicy, default_policy

TOOL = "github.get_file_contents"
OWNER, REPO = "Raj-pratap07", "My-Nexus"
GOOD = {"owner": OWNER, "repo": REPO, "path": ".gitignore"}

# Mirrors the real schema's shape (no additionalProperties).
GITHUB_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "path": {"type": "string"},
        "ref": {"type": "string"},
        "sha": {"type": "string"},
    },
    "required": ["owner", "repo"],
}


def with_(**changes):
    return {**GOOD, **changes}


def make_manager():
    manager = MCPConnectionManager(default_policy())
    group_call = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(text="file")], structured_content=None, is_error=False))
    manager._group = SimpleNamespace(
        tools={TOOL: SimpleNamespace(description="d", input_schema=GITHUB_SCHEMA)},
        call_tool=group_call,
    )
    return manager, group_call


class GuardTests(unittest.TestCase):
    def test_valid_call_is_accepted_and_normalized(self):
        self.assertEqual(github_file_guard(GOOD), GOOD)
        self.assertEqual(
            github_file_guard(with_(path="backend/app/")),
            with_(path="backend/app"),
        )
        out = github_file_guard(with_(ref="refs/heads/main", sha="a" * 40,
                                      fields=["name", "type"]))
        self.assertEqual(out["fields"], ["name", "type"])

    def test_other_owner_and_repo_rejected(self):
        for bad in (with_(owner="someone-else"), with_(repo="Other-Repo"),
                    with_(owner="raj-pratap07"), with_(owner=None), with_(repo=5)):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    ArgumentPolicyError, "repository not authorized"):
                github_file_guard(bad)

    def test_missing_owner_or_repo_rejected(self):
        for key in ("owner", "repo"):
            args = dict(GOOD)
            del args[key]
            with self.subTest(key=key), self.assertRaises(ArgumentPolicyError):
                github_file_guard(args)

    def test_bad_paths_rejected(self):
        bad_paths = [
            "/etc/passwd", "/", "..", "../secret", "a/../../b", "a/..",
            "./a", "a/./b", "a//b", "..\\..\\x", "a\\b",
            "%2e%2e/secret", "a/%2E%2E/b", "..%2fsecret", "%252e%252e/x", "100%",
            "．．/x",  # full-width dots
            "a\x00b", "a\nb", " a", "a /b",
            "", None, 5, ["a"], "x" * 600,
        ]
        for path in bad_paths:
            with self.subTest(path=path), self.assertRaises(ArgumentPolicyError):
                github_file_guard(with_(path=path))

    def test_missing_path_rejected(self):
        args = dict(GOOD)
        del args["path"]
        with self.assertRaisesRegex(ArgumentPolicyError, "path is required"):
            github_file_guard(args)

    def test_unexpected_arguments_rejected(self):
        for extra in ({"token": "x"}, {"url": "http://evil"}, {"owner2": "x"}):
            with self.subTest(extra=extra), self.assertRaisesRegex(
                    ArgumentPolicyError, "unexpected arguments"):
                github_file_guard(with_(**extra))

    def test_bad_optional_arguments_rejected(self):
        for bad in (with_(ref="../x"), with_(ref="a b"), with_(ref="-flag"),
                    with_(ref=5), with_(sha="zzzz"), with_(sha="abc"),
                    with_(fields="name"), with_(fields=["secret"]),
                    with_(fields=[1])):
            with self.subTest(bad=bad), self.assertRaises(ArgumentPolicyError):
                github_file_guard(bad)

    def test_normalize_repo_path_accepts_ordinary_paths(self):
        for ok in (".gitignore", "README.md", "backend/app/mcp/permissions.py",
                   "dir.with.dots/file-name_1.txt"):
            self.assertEqual(normalize_repo_path(ok), ok)

    def test_non_mapping_arguments_rejected(self):
        with self.assertRaises(ArgumentPolicyError):
            github_file_guard(["owner"])  # type: ignore[arg-type]


ADVERSARIAL_PATHS = {
    # .git as any segment, any case or variant
    "dot-git": [".git", ".git/config", "backend/.git/HEAD", ".git/", "a/.git",
                ".GIT", ".Git/config", "a/.gIt/HEAD", ".git.", ".git/..",
                "GIT~1/config", "x/git~1"],
    # Unicode normalization variants and look-alike separators
    "unicode": [
        "．．/secret",        # full-width dots
        "．git/config",           # full-width dot + git
        "a／b", "..／x",      # full-width solidus
        "a∕b", "a⁄b",       # division slash, fraction slash
        "a⧸b", "a＼b",       # big solidus, full-width backslash
        "K",                     # Kelvin sign (NFKC changes it)
        "é.txt",                # decomposed accent (not NFKC-stable)
        "a​b", "a‮b",       # zero-width space, bidi override
        ".​git/config",          # zero-width inside .git
    ],
    # Null bytes and control characters
    "control": ["a\x00b", "a\x00", "\x00", "a\nb", "a\rb", "a\tb", "a\x1bb",
                "a\x7fb", "a\x85b"],
    # Percent encoding, repeated and mixed
    "percent": ["%2e%2e/x", "%2E%2E/x", "..%2fx", "%2e%2e%2fx", "a%2fb",
                "%252e%252e/x", "%25252e%25252e/x", "%2500", "%00", "%5c..",
                "%2egit/config", "a%", "%", "%zz", "%E0%A4%A"],
    # Slashes, backslashes, repeated separators, whitespace
    "separators": ["a\\b", "..\\..\\x", "a/..\\b", "a\\../b", "\\a", "a\\",
                   "a//b", "a///b", "//a", "///", "/a", "a/./b",
                   "./a", "a/.", "a/../b"],
    "whitespace": [" a", "a ", " ", "a ", " a", "a/ b/c ", "a/ /b",
                   "a/.. /b", " ../x"],
    # URL meaning downstream
    "url-meta": ["a?ref=other", "a#frag", "?x", "#", "a/b?x=1"],
}

# Substring uses of ".git" that are NOT a ".git" path segment.
LEGITIMATE_PATHS = [
    ".gitignore", ".gitattributes", ".gitmodules", "backend/.gitkeep",
    "docs/.github/workflows/ci.yml", ".github/CODEOWNERS", "a.git", "x.git/y",
    "my.git.txt", "digit.py", "legit/gitfile", "git", "GIT.md", "src/.gitignore",
    "git~2", "dir with space/file name.txt",
]


class AdversarialPathTests(unittest.TestCase):
    def test_adversarial_paths_rejected_by_normalizer(self):
        for group, paths in ADVERSARIAL_PATHS.items():
            for path in paths:
                with self.subTest(group=group, path=path), self.assertRaises(
                        ArgumentPolicyError):
                    normalize_repo_path(path)

    def test_adversarial_paths_rejected_by_guard(self):
        for group, paths in ADVERSARIAL_PATHS.items():
            for path in paths:
                with self.subTest(group=group, path=path), self.assertRaises(
                        ArgumentPolicyError):
                    github_file_guard(with_(path=path))

    def test_git_error_message_is_specific(self):
        for path in (".git", ".git/config", "backend/.git/HEAD", ".GIT"):
            with self.subTest(path=path), self.assertRaisesRegex(
                    ArgumentPolicyError, r"\.git"):
                normalize_repo_path(path)

    def test_legitimate_git_substrings_accepted_unchanged(self):
        for path in LEGITIMATE_PATHS:
            with self.subTest(path=path):
                self.assertEqual(normalize_repo_path(path), path)
                self.assertEqual(github_file_guard(with_(path=path))["path"], path)

    def test_nothing_unsafe_survives_normalization(self):
        # Property check: whatever is accepted is a plain relative path.
        candidates = [p for ps in ADVERSARIAL_PATHS.values() for p in ps] + LEGITIMATE_PATHS
        for path in candidates:
            try:
                out = normalize_repo_path(path)
            except ArgumentPolicyError:
                continue
            with self.subTest(path=path):
                self.assertFalse(out.startswith("/") or "\\" in out or "%" in out)
                self.assertTrue(out.isprintable())
                for seg in out.split("/"):
                    self.assertNotIn(seg.casefold().rstrip("."), ("", ".", "..", ".git", "git~1"))


class PolicyWiringTests(unittest.TestCase):
    def test_default_policy_guards_github_only(self):
        policy = default_policy()
        with self.assertRaises(PermissionError):
            policy.check_arguments(TOOL, with_(owner="x"))
        # Research tools have no guard and pass through unchanged.
        args = {"query": "rag", "limit": 3}
        self.assertEqual(policy.check_arguments("research.search_openalex_works", args), args)

    def test_policy_without_guards_is_unchanged(self):
        policy = ToolPolicy({"github": [TOOL]})
        self.assertEqual(policy.check_arguments(TOOL, {"owner": "o"}), {"owner": "o"})

    def test_guard_name_must_be_qualified(self):
        with self.assertRaises(ValueError):
            ToolPolicy(argument_guards={"get_file_contents": github_file_guard})


class FailClosedTests(unittest.IsolatedAsyncioTestCase):
    """A sensitive server's tool is unusable while its guard is missing."""

    def unguarded_manager(self):
        policy = ToolPolicy({"github": [TOOL]}, require_guards=("github",))  # no guard
        manager = MCPConnectionManager(policy)
        group_call = AsyncMock()
        manager._group = SimpleNamespace(
            tools={TOOL: SimpleNamespace(description="d", input_schema=GITHUB_SCHEMA)},
            call_tool=group_call,
        )
        return manager, policy, group_call

    async def test_tool_without_required_guard_is_never_permitted_or_executed(self):
        manager, policy, group_call = self.unguarded_manager()
        self.assertFalse(policy.is_allowed(TOOL))
        self.assertEqual(manager.permitted_tools, {})
        with self.assertRaisesRegex(PermissionError, "guard is not configured"):
            await manager.call_tool(TOOL, GOOD)
        group_call.assert_not_awaited()

    def test_check_arguments_also_fails_closed(self):
        policy = ToolPolicy({"github": [TOOL]}, require_guards=("github",))
        with self.assertRaises(ArgumentPolicyError):
            policy.check_arguments(TOOL, GOOD)

    async def test_unguarded_orchestrator_run_cannot_reach_the_tool(self):
        manager, _, group_call = self.unguarded_manager()
        provider = ScriptedProvider([ProviderResponse("", (ToolCall(TOOL, GOOD),))])
        result = await AgentOrchestrator(manager, provider).run("x")
        # Never offered to the model, and a call for it is not executed.
        self.assertEqual(provider.tools_seen, [[]])
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.stop_reason, "tool_limit")
        group_call.assert_not_awaited()

    def test_default_policy_requires_a_guard_for_every_github_tool(self):
        policy = default_policy()
        self.assertTrue(policy.is_allowed(TOOL))
        policy.allow("github", [TOOL, "github.search_code"])  # allowed but no guard
        self.assertTrue(policy.is_allowed(TOOL))
        self.assertFalse(policy.is_allowed("github.search_code"))
        with self.assertRaises(PermissionError):
            policy.check_arguments("github.search_code", {"q": "x"})

    def test_servers_not_marked_sensitive_are_unaffected(self):
        policy = default_policy()
        self.assertTrue(policy.is_allowed("research.search_openalex_works"))
        self.assertEqual(policy.check_arguments("research.search_openalex_works", {"query": "x"}),
                         {"query": "x"})


class DirectInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_call_reaches_server_with_normalized_args(self):
        manager, group_call = make_manager()
        await manager.call_tool(TOOL, with_(path="backend/"))
        group_call.assert_awaited_once_with(TOOL, arguments=with_(path="backend"))

    async def test_rejected_direct_calls_never_reach_server(self):
        manager, group_call = make_manager()
        attempts = [
            with_(owner="torvalds", repo="linux"),
            with_(repo="Other"),
            with_(path="/etc/passwd"),
            with_(path="../../x"),
            with_(path="%2e%2e/x"),
            with_(path=""),
            with_(extra="x"),
        ]
        for args in attempts:
            with self.subTest(args=args), self.assertRaises(PermissionError):
                await manager.call_tool(TOOL, args)
        group_call.assert_not_awaited()

    async def test_adversarial_paths_never_reach_server(self):
        manager, group_call = make_manager()
        for paths in ADVERSARIAL_PATHS.values():
            for path in paths:
                with self.subTest(path=path), self.assertRaises(PermissionError):
                    await manager.call_tool(TOOL, with_(path=path))
        group_call.assert_not_awaited()

    async def test_legitimate_git_substrings_reach_server(self):
        manager, group_call = make_manager()
        for path in LEGITIMATE_PATHS:
            await manager.call_tool(TOOL, with_(path=path))
        self.assertEqual(group_call.await_count, len(LEGITIMATE_PATHS))


class ScriptedProvider(ChatProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.tools_seen = []

    async def chat(self, messages, tools=None):
        self.tools_seen.append(tools)
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


class OrchestratorEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_policy_calls_blocked_before_server(self):
        manager, group_call = make_manager()
        bad_calls = [
            with_(owner="torvalds", repo="linux"),
            with_(path="../../etc/passwd"),
            with_(path="%2e%2e/x"),
            with_(token="x"),
        ]
        provider = ScriptedProvider([
            ProviderResponse("", tuple(ToolCall(TOOL, a) for a in bad_calls)),
            ProviderResponse("done"),
        ])
        result = await AgentOrchestrator(manager, provider).run("read it")

        group_call.assert_not_awaited()
        self.assertEqual([r.executed for r in result.tool_calls], [False] * 4)
        tool_msgs = [json.loads(m["content"]) for m in provider.calls[1]
                     if m["role"] == "tool"]
        self.assertTrue(all(m["is_error"] for m in tool_msgs))
        self.assertIn("rejected by policy", tool_msgs[0]["content"])

    async def test_valid_call_succeeds_through_orchestrator(self):
        manager, group_call = make_manager()
        provider = ScriptedProvider([
            ProviderResponse("", (ToolCall(TOOL, GOOD),)),
            ProviderResponse("done"),
        ])
        result = await AgentOrchestrator(manager, provider).run("read it")
        group_call.assert_awaited_once_with(TOOL, arguments=GOOD)
        self.assertTrue(result.tool_calls[0].executed)


if __name__ == "__main__":
    unittest.main()
