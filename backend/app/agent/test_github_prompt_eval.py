"""
Offline tests for the manual GitHub prompt evaluation script. No Ollama, no
MCP server, no network: the provider is scripted and the MCP session group
is faked.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.agent.test_github_prompt_eval -v
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app.agent.github_prompt_eval as ev
from app.agent.orchestrator import (
    GITHUB_FILE_GUIDANCE,
    TOOL_RESULT_RULES,
    AgentResult,
    ToolCallRecord,
)
from app.agent.providers import ChatProvider, ProviderResponse, ToolCall
from app.mcp.argument_policy import GITHUB_FILE_TOOL, github_file_guard
from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.permissions import ToolPolicy

GH = GITHUB_FILE_TOOL
OPENALEX = "research.search_openalex_works"
GH_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("owner", "repo", "path", "ref", "sha")},
    "required": ["owner", "repo"],
}
SEARCH_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}},
                 "required": ["query"]}
INTRO = [
    {"role": "user", "content": "  Hello — exact text, kept verbatim. "},
    {"role": "assistant", "content": "Hi! Line one.\nLine two."},
]
EXPECTED_PATH = {q: sorted(p)[0] for _, q, p in ev.POSITIVE_CASES}


def quiet(coro):
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(coro)


def record(function, tool, outcome, **args):
    return ToolCallRecord(function, tool, args, outcome != "rejected", None, outcome)


def agent_result(*records):
    return AgentResult("answer", "complete", list(records), [])


def case(variant="revised", history=ev.NO_HISTORY, kind="positive", passed=True):
    flags = dict(selected=passed, args_correct=passed, accepted=passed) \
        if kind == "positive" else dict(selected=None, args_correct=None, accepted=None)
    return ev.CaseResult(variant, history, "c", kind, passed, "d", calls=[],
                         stop_reason="complete", **flags)


def full_results(variant="revised", passes=None, negatives_ok=True):
    """10 positives + 3 negatives per condition; passes = {condition: n}."""
    passes = passes or {}
    out = []
    for condition in ev.CONDITIONS:
        n = passes.get(condition, 10)
        out += [case(variant, condition, "positive", i < n) for i in range(10)]
        out += [case(variant, condition, "negative", negatives_ok) for _ in range(3)]
    return out


class ScriptedProvider(ChatProvider):
    """Calls the GitHub tool correctly for positive cases; records every request."""

    def __init__(self, github_for_negatives=False):
        self.requests = []
        self.github_for_negatives = github_for_negatives

    async def chat(self, messages, tools=None):
        self.requests.append([dict(m) for m in messages])
        if messages[-1]["role"] == "tool":
            return ProviderResponse("final answer")
        question = messages[-1]["content"]
        path = EXPECTED_PATH.get(question)
        if path is None and self.github_for_negatives:
            path = "README.md"
        if path is None:
            return ProviderResponse("plain answer")
        return ProviderResponse("", (ToolCall(GH, {"owner": "Raj-pratap07",
                                                   "repo": "My-Nexus", "path": path}),))


def make_manager(policy):
    manager = MCPConnectionManager(policy)
    manager._group = SimpleNamespace(
        tools={GH: SimpleNamespace(description="d", input_schema=GH_SCHEMA),
               OPENALEX: SimpleNamespace(description="d", input_schema=SEARCH_SCHEMA)},
        call_tool=AsyncMock(return_value=ev.stub_result(GH)),
    )
    return manager


def recording_policy():
    seen = []

    def guard(arguments):
        seen.append(dict(arguments))
        return github_file_guard(arguments)

    policy = ToolPolicy({"github": [GH], "research": [OPENALEX]},
                        argument_guards={GH: guard}, require_guards=("github",))
    return policy, seen


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_new_file_in_existing_directory_accepted_and_not_created(self):
        target = self.dir / "out.json"
        self.assertEqual(ev.validate_output_path(str(target), force=False), target.resolve())
        self.assertFalse(target.exists())

    def test_existing_file_refused_without_force(self):
        target = self.dir / "out.json"
        target.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ev.SetupError, "already exists"):
            ev.validate_output_path(str(target), force=False)
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_existing_file_accepted_with_force_and_left_untouched(self):
        target = self.dir / "out.json"
        target.write_text("keep", encoding="utf-8")
        ev.validate_output_path(str(target), force=True)
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_missing_directory_refused_and_not_created(self):
        missing = self.dir / "no" / "such"
        with self.assertRaisesRegex(ev.SetupError, "does not exist"):
            ev.validate_output_path(str(missing / "out.json"), force=True)
        self.assertFalse((self.dir / "no").exists())

    def test_directory_target_refused(self):
        with self.assertRaisesRegex(ev.SetupError, "is a directory"):
            ev.validate_output_path(str(self.dir), force=True)

    def test_invalid_paths_refused(self):
        bad = ["", "   ", str(self.dir / "a\0b.json")]
        if os.name == "nt":
            bad += [str(self.dir / "bad|name.json"), str(self.dir / "what?.json"),
                    str(self.dir / "CON.json"), str(self.dir / "nul")]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ev.SetupError):
                ev.validate_output_path(raw, force=True)


class WriteResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / "out.json"
        self.results = [case()]

    def test_writes_results(self):
        self.assertIsNone(ev.write_results_json(self.target, self.results, force=False))
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["case"], "c")
        self.assertIn("selected", data[0])

    def test_file_created_after_validation_is_not_overwritten(self):
        self.target.write_text("keep", encoding="utf-8")
        error = ev.write_results_json(self.target, self.results, force=False)
        self.assertIn("already exists", error)
        self.assertEqual(self.target.read_text(encoding="utf-8"), "keep")

    def test_force_overwrites(self):
        self.target.write_text("old", encoding="utf-8")
        self.assertIsNone(ev.write_results_json(self.target, self.results, force=True))
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8"))[0]["case"], "c")

    def test_filesystem_error_is_returned_not_raised(self):
        gone = Path(self.tmp.name) / "removed" / "out.json"
        error = ev.write_results_json(gone, self.results, force=False)
        self.assertIsNotNone(error)
        self.assertFalse(gone.parent.exists())


class FinishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def finish(self, results, variants, json_path, force=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ev.finish(results, variants, json_path, force=force)
        return code, out.getvalue()

    def test_verdict_printed_even_if_json_write_fails(self):
        bad = Path(self.tmp.name) / "vanished" / "out.json"
        code, out = self.finish(full_results(), ["revised"], bad)
        self.assertIn("=== Summary", out)
        self.assertIn("RESULT: revised prompt meets the acceptance bar", out)
        self.assertIn("JSON NOT WRITTEN", out)
        self.assertEqual(code, 2)

    def test_gate_failure_takes_precedence_over_json_failure(self):
        bad = Path(self.tmp.name) / "vanished" / "out.json"
        code, out = self.finish(full_results(passes={ev.NO_HISTORY: 5}), ["revised"], bad)
        self.assertIn("BELOW the acceptance bar", out)
        self.assertEqual(code, 1)

    def test_json_written_only_after_summary_and_verdict(self):
        target = Path(self.tmp.name) / "out.json"
        code, out = self.finish(full_results(), ["revised"], target)
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())
        self.assertLess(out.index("=== Summary"), out.index("RESULT:"))
        self.assertLess(out.index("RESULT:"), out.index("Per-case results written"))

    def test_original_only_is_not_gated(self):
        code, out = self.finish(full_results("original", passes={ev.NO_HISTORY: 0}),
                                ["original"], None)
        self.assertEqual(code, 0)
        self.assertNotIn("RESULT:", out)


class GateTests(unittest.TestCase):
    def ok(self, results):
        with contextlib.redirect_stdout(io.StringIO()):
            return ev.summarize(results, ["revised"])

    def test_both_conditions_at_threshold_pass(self):
        self.assertTrue(self.ok(full_results(passes={ev.NO_HISTORY: 9, ev.INTRO_HISTORY: 9})))

    def test_each_condition_is_judged_independently(self):
        self.assertFalse(self.ok(full_results(passes={ev.NO_HISTORY: 10, ev.INTRO_HISTORY: 8})))
        self.assertFalse(self.ok(full_results(passes={ev.NO_HISTORY: 8, ev.INTRO_HISTORY: 10})))

    def test_any_negative_failure_fails_the_gate(self):
        self.assertFalse(self.ok(full_results(negatives_ok=False)))

    def test_missing_condition_fails_the_gate(self):
        only_one = [r for r in full_results() if r.history == ev.NO_HISTORY]
        self.assertFalse(self.ok(only_one))

    def test_summary_reports_separate_measurements(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ev.summarize(full_results(), ["revised"])
        header = next(line for line in out.getvalue().splitlines()
                      if line.startswith("variant"))
        for column in ("selected", "args ok", "accepted", "PASSED", "negatives"):
            self.assertIn(column, header)


class JudgeTests(unittest.TestCase):
    PATHS = frozenset({"README.md"})
    GOOD = dict(owner="Raj-pratap07", repo="My-Nexus", path="README.md")

    def judge(self, *records):
        return ev.judge_positive(agent_result(*records), self.PATHS)

    def flags(self, verdict):
        return (verdict["selected"], verdict["args_correct"], verdict["accepted"],
                verdict["passed"])

    def test_no_call(self):
        self.assertEqual(self.flags(self.judge()), (False, False, False, False))

    def test_wrong_tool_name_with_correct_arguments(self):
        v = self.judge(record("get_file_contents", None, "rejected", **self.GOOD))
        self.assertEqual(self.flags(v), (False, True, False, False))
        self.assertIn("wrong tool name", v["detail"])

    def test_right_tool_wrong_but_valid_path(self):
        v = self.judge(record(GH, GH, "succeeded", **{**self.GOOD, "path": "docs/x.md"}))
        self.assertEqual(self.flags(v), (True, False, True, False))
        self.assertIn("wrong path", v["detail"])

    def test_right_tool_rejected_by_guard(self):
        v = self.judge(record(GH, GH, "rejected",
                              **{**self.GOOD, "owner": "Raj-pratap07/My-Nexus"}))
        self.assertEqual(self.flags(v), (True, False, False, False))

    def test_correct_call(self):
        self.assertEqual(self.flags(self.judge(record(GH, GH, "succeeded", **self.GOOD))),
                         (True, True, True, True))

    def test_accepted_but_tool_failed_is_not_a_pass(self):
        v = self.judge(record(GH, GH, "failed", **self.GOOD))
        self.assertEqual(self.flags(v), (True, True, True, False))

    def test_only_first_github_attempt_is_judged(self):
        v = self.judge(record(GH, GH, "rejected", **{**self.GOOD, "path": "/README.md"}),
                       record(GH, GH, "succeeded", **self.GOOD))
        self.assertFalse(v["passed"])

    def test_negative_control(self):
        self.assertTrue(ev.judge_negative(agent_result())["passed"])
        self.assertTrue(ev.judge_negative(agent_result(
            record(OPENALEX, OPENALEX, "succeeded", query="x")))["passed"])
        self.assertFalse(ev.judge_negative(agent_result(
            record(GH, GH, "succeeded", **self.GOOD)))["passed"])


class HistoryFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, payload, *, bom=False, raw=None):
        path = Path(self.tmp.name) / "intro.json"
        text = raw if raw is not None else json.dumps(payload, ensure_ascii=False)
        path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
        return str(path)

    def test_messages_loaded_verbatim(self):
        for bom in (False, True):  # PowerShell 5.1 writes UTF-8 with a BOM
            with self.subTest(bom=bom):
                self.assertEqual(ev.load_history_fixture(self.write(INTRO, bom=bom)), INTRO)

    def test_invalid_fixtures_refused(self):
        bad = {
            "not json": None,
            "object": {"role": "user", "content": "x"},
            "empty": [],
            "extra key": [{**INTRO[0], "id": 1}, INTRO[1]],
            "starts with assistant": [INTRO[1], INTRO[0]],
            "not alternating": [INTRO[0], INTRO[0], INTRO[1]],
            "ends with user": [INTRO[0], INTRO[1], INTRO[0]],
            "empty content": [{"role": "user", "content": ""}, INTRO[1]],
            "non-string content": [{"role": "user", "content": 5}, INTRO[1]],
            "system role": [{"role": "system", "content": "x"}, INTRO[1]],
            "too many": (INTRO * 11),
        }
        for label, payload in bad.items():
            with self.subTest(label):
                path = self.write(payload, raw="{not json" if payload is None else None)
                with self.assertRaises(ev.SetupError):
                    ev.load_history_fixture(path)

    def test_missing_file_refused(self):
        with self.assertRaises(ev.SetupError):
            ev.load_history_fixture(str(Path(self.tmp.name) / "missing.json"))


class EvaluateTests(unittest.TestCase):
    def test_case_counts_and_run_count(self):
        self.assertEqual((len(ev.POSITIVE_CASES), len(ev.NEGATIVE_CASES)), (10, 3))
        policy, _ = recording_policy()
        results = quiet(ev.evaluate(make_manager(policy), policy, ScriptedProvider(),
                                    ["original", "revised"], INTRO, execute=False))
        self.assertEqual(len(results), 52)

    def test_stub_uses_the_given_policy_and_never_dispatches(self):
        policy, seen = recording_policy()
        manager = make_manager(policy)
        results = quiet(ev.evaluate(manager, policy, ScriptedProvider(), ["revised"], INTRO,
                                    execute=False))
        self.assertEqual(len(seen), 20)  # 10 positives x 2 conditions, through the guard
        manager._group.call_tool.assert_not_awaited()
        self.assertTrue(all(r.passed for r in results))

    def test_stub_rejections_come_from_the_policy(self):
        policy = ToolPolicy({"github": [GH]}, require_guards=("github",))  # guard missing
        manager = make_manager(policy)
        results = quiet(ev.evaluate(manager, policy, ScriptedProvider(), ["revised"], INTRO,
                                    execute=False))
        # The tool is not even offered under this policy, so nothing can pass.
        self.assertFalse(any(r.passed for r in results if r.kind == "positive"))
        manager._group.call_tool.assert_not_awaited()

    def test_execute_mode_dispatches_through_the_manager(self):
        policy, seen = recording_policy()
        manager = make_manager(policy)
        quiet(ev.evaluate(manager, policy, ScriptedProvider(), ["revised"], INTRO,
                          execute=True))
        self.assertEqual(manager._group.call_tool.await_count, 20)
        self.assertEqual(len(seen), 20)

    def test_history_conditions_are_isolated(self):
        policy, _ = recording_policy()
        provider = ScriptedProvider()
        intro = [dict(m) for m in INTRO]
        results = quiet(ev.evaluate(make_manager(policy), policy, provider, ["revised"],
                                    intro, execute=False))
        self.assertEqual(intro, INTRO)  # fixture never mutated
        firsts = [r for r in provider.requests if r[-1]["role"] == "user"]
        self.assertEqual(len(firsts), len(results))
        for request, result in zip(firsts, results):
            body = request[1:-1]  # between the system message and the question
            expected = [] if result.history == ev.NO_HISTORY else INTRO
            self.assertEqual(body, expected, result.history)

    def test_variants_use_exact_prompts(self):
        policy, _ = recording_policy()
        provider = ScriptedProvider()
        results = quiet(ev.evaluate(make_manager(policy), policy, provider,
                                    ["original", "revised"], INTRO, execute=False))
        firsts = [r for r in provider.requests if r[-1]["role"] == "user"]
        for request, result in zip(firsts, results):
            system = request[0]["content"]
            if result.variant == "original":
                self.assertEqual(system, ev.ORIGINAL_SYSTEM_PROMPT)
            else:
                self.assertIn(TOOL_RESULT_RULES, system)
                self.assertIn(GITHUB_FILE_GUIDANCE.format(tool=GH), system)

    def test_negative_controls_fail_when_github_is_called(self):
        policy, _ = recording_policy()
        results = quiet(ev.evaluate(make_manager(policy), policy,
                                    ScriptedProvider(github_for_negatives=True), ["revised"],
                                    INTRO, execute=False))
        self.assertFalse(any(r.passed for r in results if r.kind == "negative"))


class MainPreflightTests(unittest.TestCase):
    """main() must refuse before any connection, and share one policy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.history = Path(self.tmp.name) / "intro.json"
        self.history.write_text(json.dumps(INTRO), encoding="utf-8")
        self.env = patch.dict(os.environ, {"GITHUB_TOKEN": "dummy", "GITHUB_MCP_URL":
                                           "http://127.0.0.1:9/never-contacted"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_main(self, *argv):
        def forbidden(*a, **k):
            raise AssertionError("must not be constructed during preflight refusal")

        out = io.StringIO()
        with patch.object(ev, "MCPConnectionManager", forbidden), \
                patch.object(ev, "OllamaProvider", forbidden), \
                contextlib.redirect_stdout(out):
            code = asyncio.run(ev.main(list(argv)))
        return code, out.getvalue()

    def test_refuses_missing_json_directory_before_connecting(self):
        code, out = self.run_main("--history-file", str(self.history),
                                  "--json", str(Path(self.tmp.name) / "no" / "out.json"))
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", out)

    def test_refuses_existing_json_without_force_before_connecting(self):
        existing = Path(self.tmp.name) / "out.json"
        existing.write_text("keep", encoding="utf-8")
        code, _ = self.run_main("--history-file", str(self.history), "--json", str(existing))
        self.assertEqual(code, 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_refuses_bad_history_before_connecting(self):
        self.history.write_text("[]", encoding="utf-8")
        code, _ = self.run_main("--history-file", str(self.history))
        self.assertEqual(code, 1)

    def test_refuses_missing_github_settings_before_connecting(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "", "GITHUB_MCP_URL": ""}):
            code, out = self.run_main("--history-file", str(self.history))
        self.assertEqual(code, 1)
        self.assertIn("GITHUB_TOKEN", out)

    def test_history_file_is_required(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ev.parse_args([])

    def test_one_policy_instance_is_shared(self):
        captured = {}
        sentinel_policy = object()

        class FakeManager:
            def __init__(self, policy):
                captured["manager_policy"] = policy
                self.permitted_tools = {GH: object()}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def connect_http_server(self, *a, **k):
                return None

            async def connect_stdio_server(self, *a, **k):
                return None

        async def fake_evaluate(manager, policy, provider, variants, intro, *, execute,
                                followup=None, seed=None, on_result=None,
                                continue_after_timeout=False):
            captured["evaluate_policy"] = policy
            captured["intro"] = intro
            captured["followup"] = followup
            captured["evaluate_seed"] = seed
            captured["on_result"] = on_result
            captured["continue_after_timeout"] = continue_after_timeout
            return []

        def fake_provider(**kwargs):
            captured["provider_kwargs"] = kwargs
            return object()

        with patch.object(ev, "default_policy", return_value=sentinel_policy) as factory, \
                patch.object(ev, "MCPConnectionManager", FakeManager), \
                patch.object(ev, "OllamaProvider", fake_provider), \
                patch.object(ev, "evaluate", fake_evaluate), \
                patch.object(ev, "finish", return_value=0), \
                contextlib.redirect_stdout(io.StringIO()):
            code = asyncio.run(ev.main(["--history-file", str(self.history)]))
        self.assertEqual(code, 0)
        factory.assert_called_once()
        self.assertIs(captured["manager_policy"], sentinel_policy)
        self.assertIs(captured["evaluate_policy"], sentinel_policy)
        self.assertEqual(captured["intro"], INTRO)
        # The exact follow-up fixture is always passed to the evaluation.
        self.assertEqual(captured["followup"], ev.load_followup_fixture())
        # Default seed 0 (the provider's previous fixed seed), to both places.
        self.assertEqual(captured["provider_kwargs"]["seed"], 0)
        self.assertEqual(captured["evaluate_seed"], 0)
        # Defaults: stop after a timeout; no --json, so no incremental writer.
        self.assertFalse(captured["continue_after_timeout"])
        self.assertIsNone(captured["on_result"])

    def test_refuses_missing_followup_fixture_before_connecting(self):
        with patch.object(ev, "FOLLOWUP_FIXTURE", Path(self.tmp.name) / "missing.json"):
            code, out = self.run_main("--history-file", str(self.history))
        self.assertEqual(code, 1)
        self.assertIn("follow-up fixture", out)


# ----- follow-up grounding -----------------------------------------------------
README_TEXT = ev.LOCAL_README.read_text(encoding="utf-8")
FOLLOWUP = ev.load_followup_fixture()
# The five passages the failed turn-2 answer presented as README quotations.
TURN2_MISQUOTES = [
    "Tool outputs are treated as untrusted data in the prompt, but prompt injection "
    "is not prevented—code-level controls are the primary defense.",
    "a web UI and Tauri desktop app are planned to share this backend",
    "memories (schema only, no behavior yet)",
    "exits with clear error messages",
    "The project is in early development and intended for local use only.",
]


def envelope(text, *, tool=GH, is_error=False, truncated=False):
    return {"role": "tool", "tool_name": tool, "content": json.dumps(
        {"tool": tool, "is_error": is_error, "untrusted_data": True,
         "truncated": truncated, "content": text})}


def followup_result(answer, *records, messages=()):
    return AgentResult(answer, "complete", list(records), list(messages))


class FollowupProvider(ScriptedProvider):
    """Scripted behaviour for the follow-up question only."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode  # verbatim | misquote | no_refetch

    async def chat(self, messages, tools=None):
        last = messages[-1]
        # Both source-grounding follow-ups (implicit and explicit) are scripted.
        is_followup = any(m.get("content") in (FOLLOWUP["question"],
                                               ev.EXPLICIT_FOLLOWUP_QUESTION)
                          for m in messages)
        if not is_followup:
            return await super().chat(messages, tools)
        self.requests.append([dict(m) for m in messages])
        if last["role"] == "tool":
            source = json.loads(last["content"])["content"]
            snippet = " ".join(source.split()[2:10])
            quote = snippet if self.mode == "verbatim" else TURN2_MISQUOTES[0]
            return ProviderResponse(f'The README states: "{quote}".')
        if self.mode == "no_refetch":
            return ProviderResponse(f'As the README says, "{TURN2_MISQUOTES[4]}"')
        return ProviderResponse("", (ToolCall(GH, {"owner": "Raj-pratap07",
                                                   "repo": "My-Nexus",
                                                   "path": "README.md"}),))


class QuoteFidelityTests(unittest.TestCase):
    def test_verbatim_quote_matches(self):
        q = ev.extract_quotes('It says "A local-first AI research assistant."')
        self.assertEqual(q, ["A local-first AI research assistant"])
        self.assertEqual(ev.unmatched_quotes(q, [README_TEXT]), [])

    def test_every_turn2_misquote_is_detected(self):
        answer = " ".join(f'"{q}"' for q in TURN2_MISQUOTES)
        quotes = ev.extract_quotes(answer)
        self.assertEqual(len(quotes), 5)
        self.assertEqual(ev.unmatched_quotes(quotes, [README_TEXT]), quotes)

    def test_markdown_emphasis_and_whitespace_are_normalized(self):
        # README text is "prompt injection is **not** prevented"; the quote has
        # no emphasis markers and irregular spacing.
        quotes = ev.extract_quotes('"prompt   injection is  not prevented"')
        self.assertEqual(ev.unmatched_quotes(quotes, [README_TEXT]), [])

    def test_curly_quotation_marks_are_extracted(self):
        quotes = ev.extract_quotes("It says “early development, local use only”.")
        self.assertEqual(quotes, ["early development, local use only"])
        self.assertEqual(ev.unmatched_quotes(quotes, [README_TEXT]), [])

    def test_ellipsis_segments_must_each_be_verbatim(self):
        good = ev.extract_quotes('"The prompt tells the model ... prompt injection is not prevented"')
        bad = ev.extract_quotes('"The prompt tells the model ... injection is fully prevented"')
        self.assertEqual(ev.unmatched_quotes(good, [README_TEXT]), [])
        self.assertEqual(len(ev.unmatched_quotes(bad, [README_TEXT])), 1)

    def test_short_quoted_tokens_are_ignored(self):
        self.assertEqual(ev.extract_quotes('roles "user" and "assistant"'), [])

    def test_tool_sources_use_only_successful_github_results_of_this_request(self):
        result = followup_result("x", messages=[
            {"role": "user", "content": "q"},
            envelope("README body", truncated=True),
            envelope("an error", is_error=True),
            envelope('{"results": []}', tool=OPENALEX),
        ])
        self.assertEqual(ev.tool_sources(result), (["README body"], True))


class JudgeFollowupTests(unittest.TestCase):
    GOOD = dict(owner="Raj-pratap07", repo="My-Nexus", path="README.md")
    Q = FOLLOWUP["question"]

    def judge(self, answer, *records, messages=()):
        return ev.judge_followup(followup_result(answer, *records, messages=messages), self.Q)

    def test_answer_from_history_without_refetch_fails(self):
        v = self.judge(f'The README says "{TURN2_MISQUOTES[4]}"')
        self.assertFalse(v["passed"])
        self.assertIn("no GitHub call", v["detail"])
        self.assertEqual(v["quotes_unmatched"], [TURN2_MISQUOTES[4].rstrip(".")])

    def test_refetch_with_verbatim_quotes_passes(self):
        v = self.judge('It is "A local-first AI research assistant".',
                       record(GH, GH, "succeeded", **self.GOOD),
                       messages=[envelope(README_TEXT[:6000], truncated=True)])
        self.assertTrue(v["passed"])
        self.assertEqual((v["quotes_checked"], v["quotes_unmatched"]), (1, []))
        self.assertTrue(v["source_truncated"])

    def test_refetch_with_a_paraphrase_presented_as_quote_fails(self):
        v = self.judge(f'The README: "{TURN2_MISQUOTES[0]}"',
                       record(GH, GH, "succeeded", **self.GOOD),
                       messages=[envelope(README_TEXT[:6000], truncated=True)])
        self.assertFalse(v["passed"])
        self.assertIn("not verbatim", v["detail"])

    def test_wrong_file_fails(self):
        v = self.judge("No quotes here.", record(GH, GH, "succeeded",
                                                 **{**self.GOOD, "path": "docs/README.md"}),
                       messages=[envelope("other")])
        self.assertFalse(v["passed"])

    def test_quoting_the_user_question_is_allowed(self):
        v = self.judge('You asked: "Do not invent details, and distinguish implemented '
                       'functionality from planned functionality".',
                       record(GH, GH, "succeeded", **self.GOOD),
                       messages=[envelope(README_TEXT[:6000])])
        self.assertTrue(v["passed"])

    def test_truncation_acknowledgement_is_reported(self):
        msgs = [envelope(README_TEXT[:6000], truncated=True)]
        rec = record(GH, GH, "succeeded", **self.GOOD)
        self.assertTrue(self.judge("The README I received was truncated.", rec,
                                   messages=msgs)["truncation_acknowledged"])
        self.assertFalse(self.judge("Here is the full list.", rec,
                                    messages=msgs)["truncation_acknowledged"])
        self.assertIsNone(self.judge("ok", rec, messages=[envelope("short")])
                          ["truncation_acknowledged"])


class StubSourceTests(unittest.TestCase):
    def text(self, result):
        return result.content[0].text

    def test_readme_is_served_from_the_local_checkout(self):
        result = ev.stub_result(GH, {"owner": "Raj-pratap07", "repo": "My-Nexus",
                                     "path": "README.md"})
        self.assertEqual(self.text(result), README_TEXT)
        self.assertFalse(result.is_error)

    def test_no_other_local_file_is_ever_served(self):
        for path in (".env", ".env.example", "backend/app/main.py", "README"):
            with self.subTest(path=path):
                text = self.text(ev.stub_result(GH, {"path": path}))
                self.assertEqual(text, "(evaluation stub: the file was not fetched)")

    def test_guarded_stub_decides_on_guard_normalized_arguments(self):
        policy, _ = recording_policy()
        stub = ev.make_guarded_stub(policy)
        result = asyncio.run(stub(GH, {"owner": "Raj-pratap07", "repo": "My-Nexus",
                                       "path": "README.md/"}))
        self.assertEqual(self.text(result), README_TEXT)

    def test_unreadable_local_readme_is_an_error_result(self):
        with patch.object(ev, "LOCAL_README", Path(tempfile.gettempdir()) / "no-such-readme.md"):
            result = ev.stub_result(GH, {"path": "README.md"})
        self.assertTrue(result.is_error)


class FollowupFixtureTests(unittest.TestCase):
    def test_fixture_is_the_exact_failed_exchange(self):
        history = FOLLOWUP["history"]
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        # Turn 1 is the original failing request, verbatim.
        self.assertEqual(history[0]["content"], ev.POSITIVE_CASES[0][1])
        self.assertIn("### Architecture Overview", history[1]["content"])
        self.assertTrue(FOLLOWUP["question"].startswith(
            "According to the README you just retrieved, what are the known limitations"))

    def test_invalid_followup_fixtures_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = {
                "no question": {"history": INTRO},
                "blank question": {"history": INTRO, "question": "  "},
                "bad history": {"history": [INTRO[1]], "question": "q"},
                "not an object": [INTRO],
            }
            for label, payload in bad.items():
                with self.subTest(label):
                    path = Path(tmp) / "f.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ev.SetupError):
                        ev.load_followup_fixture(path)


class FollowupEvaluationTests(unittest.TestCase):
    def run_eval(self, provider, variants=("revised",)):
        policy, _ = recording_policy()
        manager = make_manager(policy)
        results = quiet(ev.evaluate(manager, policy, provider, list(variants), INTRO,
                                    execute=False, followup=FOLLOWUP))
        return results, manager

    def followups(self, results):
        # The original, implicit follow-up (the explicit one is tested below).
        return [r for r in results if r.case == ev.FOLLOWUP_CASE]

    def test_one_followup_run_per_variant(self):
        results, _ = self.run_eval(FollowupProvider("verbatim"), ("original", "revised"))
        # 52 positives/negatives + per variant: 2 follow-ups and 2 no-tool controls.
        self.assertEqual(len(results), 60)
        self.assertEqual([r.variant for r in self.followups(results)], ["original", "revised"])

    def test_followup_request_uses_exact_history_and_refetches(self):
        provider = FollowupProvider("verbatim")
        results, manager = self.run_eval(provider)
        first = next(r for r in provider.requests
                     if r[-1]["role"] == "user" and r[-1]["content"] == FOLLOWUP["question"])
        self.assertEqual(first[1:-1], FOLLOWUP["history"])  # verbatim, nothing else
        (followup,) = self.followups(results)
        self.assertTrue(followup.passed, followup.detail)
        self.assertEqual(followup.calls[0]["tool"], GH)
        self.assertEqual(followup.calls[0]["arguments"]["path"], "README.md")
        manager._group.call_tool.assert_not_awaited()  # stubbed: nothing dispatched

    def test_followup_source_is_the_whole_local_readme(self):
        # Previously cut at 6,000 chars, before the Known limitations section.
        provider = FollowupProvider("verbatim")
        self.run_eval(provider)
        seen = next(json.loads(r[-1]["content"]) for r in provider.requests
                    if r[-1]["role"] == "tool" and FOLLOWUP["question"] in
                    [m.get("content") for m in r])
        self.assertEqual(seen["content"], README_TEXT)
        self.assertFalse(seen["truncated"])
        self.assertIn("## Known limitations", seen["content"])

    def test_misquote_after_refetch_fails(self):
        results, _ = self.run_eval(FollowupProvider("misquote"))
        (followup,) = self.followups(results)
        self.assertFalse(followup.passed)
        self.assertEqual(len(followup.quotes_unmatched), 1)

    def test_answer_without_refetch_fails(self):
        results, _ = self.run_eval(FollowupProvider("no_refetch"))
        (followup,) = self.followups(results)
        self.assertFalse(followup.passed)
        self.assertFalse(followup.selected)

    def test_revised_followup_failure_fails_the_gate(self):
        results = full_results() + [ev.CaseResult(
            "revised", ev.FOLLOWUP_HISTORY, ev.FOLLOWUP_CASE, "followup", False, "d",
            selected=False, args_correct=False, accepted=False, calls=[],
            stop_reason="complete", quotes_checked=1, quotes_unmatched=["x" * 12])]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ok = ev.summarize(results, ["revised"])
        self.assertFalse(ok)
        self.assertIn("re-fetched README.md: no", out.getvalue())

    def test_original_followup_failure_does_not_affect_the_gate(self):
        results = full_results() + full_results("original") + [ev.CaseResult(
            "original", ev.FOLLOWUP_HISTORY, ev.FOLLOWUP_CASE, "followup", False, "d",
            selected=False, args_correct=False, accepted=False, calls=[],
            stop_reason="complete")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(ev.summarize(results, ["original", "revised"]))

    def test_negative_controls_unchanged_by_followup(self):
        results, _ = self.run_eval(FollowupProvider("verbatim"))
        negatives = [r for r in results if r.kind == "negative"]
        self.assertEqual(len(negatives), 6)
        self.assertTrue(all(r.passed for r in negatives))


# ----- Phase 3: explicit follow-up, controls, retrieval vs accuracy, seed -----
GRADER_CASES = {c["id"]: c for c in json.loads(
    (Path(ev.__file__).with_name("eval_fixtures") / "limitations_answers.json")
    .read_text(encoding="utf-8"))["cases"]}
ACCURATE = GRADER_CASES["complete-paraphrased"]["answer"]       # all five items
MISSING_ONE = GRADER_CASES["missing-limitation"]["answer"]      # omits item 4
README_ARGS = dict(owner="Raj-pratap07", repo="My-Nexus", path="README.md")
APPROVED_EXPLICIT = (
    "Use the GitHub tools to fetch README.md from the Raj-pratap07/My-Nexus repository "
    "again now. Do not rely on your previous answer; it was only a summary. Using only "
    "the file you retrieve in this request, list the items in its '## Known limitations' "
    "section. Explain each item accurately, do not add limitations that section does not "
    "list, and for each one describe its implementation status only when the README "
    "explicitly supports that classification. If the README does not establish whether "
    "something is planned or implemented, say that its status is not specified.")


def followup_case(variant="revised", case=ev.FOLLOWUP_CASE, passed=True, **kw):
    return ev.CaseResult(variant, ev.FOLLOWUP_HISTORY, case, "followup", passed, "d",
                         selected=passed, args_correct=passed, accepted=passed, calls=[],
                         stop_reason="complete", retrieval_passed=passed, **kw)


def control_case(variant="revised", retrieved=False):
    return ev.CaseResult(variant, ev.FOLLOWUP_HISTORY, "ctl", "control", not retrieved,
                         "d", selected=None, args_correct=None, accepted=None, calls=[],
                         stop_reason="complete", gated=False,
                         unnecessary_retrieval=retrieved)


class CaseDefinitionTests(unittest.TestCase):
    def test_explicit_question_is_the_approved_wording(self):
        self.assertEqual(ev.EXPLICIT_FOLLOWUP_QUESTION, APPROVED_EXPLICIT)
        self.assertIn("If the README does not establish whether something is planned or "
                      "implemented, say that its status is not specified.",
                      ev.EXPLICIT_FOLLOWUP_QUESTION)

    def test_implicit_question_is_still_the_verbatim_fixture(self):
        self.assertEqual(FOLLOWUP["question"],
                         "According to the README you just retrieved, what are the known "
                         "limitations of My-Nexus? Explain each limitation accurately. Do "
                         "not invent details, and distinguish implemented functionality "
                         "from planned functionality.")

    def test_no_tool_controls_are_the_approved_questions(self):
        self.assertEqual([q for _, q in ev.NO_TOOL_CONTROLS],
                         ["Rephrase your previous answer in two sentences.",
                          "In general, what is a README file for?"])

    def test_followup_cases_order_and_kinds(self):
        self.assertEqual(
            [(k, n) for k, n, _ in ev.followup_cases(FOLLOWUP)],
            [("followup", ev.FOLLOWUP_CASE), ("followup", ev.FOLLOWUP_EXPLICIT_CASE),
             ("control", "rephrase-previous-answer"), ("control", "readme-in-general")])


class RetrievedReadmeTests(unittest.TestCase):
    """Accuracy must be graded against the README THIS run retrieved, from an
    exact owner/repo/README.md call, paired call-by-call."""

    def result(self, pairs, answer="x"):
        records = [record(GH, GH, outcome, **args) for args, outcome, _ in pairs]
        messages = [{"role": "user", "content": "q"}]
        messages += [envelope(text, is_error=(outcome == "failed")) for _, outcome, text in pairs]
        return followup_result(answer, *records, messages=messages)

    def test_exact_readme_call_is_selected(self):
        text, truncated, _ = ev.retrieved_readme(self.result([(README_ARGS, "succeeded", "R")]))
        self.assertEqual((text, truncated), ("R", False))

    def test_other_files_and_arguments_are_ignored(self):
        for args in ({**README_ARGS, "path": "docs/README.md"},
                     {**README_ARGS, "owner": "someone"}, {**README_ARGS, "repo": "Other"},
                     {**README_ARGS, "path": "/README.md"}):
            with self.subTest(args=args):
                text, _, note = ev.retrieved_readme(self.result([(args, "succeeded", "R")]))
                self.assertIsNone(text)
                self.assertIn("no README.md retrieved", note)

    def test_failed_or_rejected_calls_are_ignored(self):
        for outcome in ("failed", "rejected"):
            with self.subTest(outcome=outcome):
                text, _, _ = ev.retrieved_readme(self.result([(README_ARGS, outcome, "R")]))
                self.assertIsNone(text)

    def test_readme_is_found_after_another_file(self):
        pairs = [({**README_ARGS, "path": ".gitignore"}, "succeeded", "G"),
                 (README_ARGS, "succeeded", "R")]
        self.assertEqual(ev.retrieved_readme(self.result(pairs))[0], "R")

    def test_unpairable_messages_are_not_guessed(self):
        result = followup_result("x", record(GH, GH, "succeeded", **README_ARGS),
                                 messages=[envelope("R"), envelope("S")])
        text, _, note = ev.retrieved_readme(result)
        self.assertIsNone(text)
        self.assertIn("cannot pair", note)


class RetrievalAccuracyIndependenceTests(unittest.TestCase):
    """The four quadrants: retrieval and accuracy never decide each other."""
    Q = ev.EXPLICIT_FOLLOWUP_QUESTION

    def judge(self, answer, source=README_TEXT, args=README_ARGS, refetch=True):
        if not refetch:
            return ev.judge_followup(followup_result(answer), self.Q)
        return ev.judge_followup(followup_result(
            answer, record(GH, GH, "succeeded", **args),
            messages=[envelope(source)]), self.Q)

    def test_refetched_and_accurate(self):
        v = self.judge(ACCURATE)
        self.assertTrue(v["passed"])
        self.assertTrue(v["retrieval_passed"])
        self.assertEqual(v["accuracy"], "pass", v["accuracy_report"])

    def test_refetched_but_inaccurate_still_passes_retrieval(self):
        v = self.judge(MISSING_ONE)
        self.assertTrue(v["passed"])  # accuracy is advisory: it never fails the case
        self.assertEqual(v["accuracy"], "fail")
        self.assertEqual(v["accuracy_failed_checks"], ["completeness"])
        self.assertEqual(len(v["accuracy_missing"]), 1)
        self.assertIn("abstracts", v["accuracy_missing"][0])

    def test_accurate_without_refetch_fails_and_is_ungradeable(self):
        v = self.judge(ACCURATE, refetch=False)
        self.assertFalse(v["passed"])
        self.assertEqual(v["accuracy"], "ungradeable")
        self.assertIn("no README.md retrieved", v["accuracy_detail"])

    def test_wrong_file_fails_and_is_ungradeable_even_if_answer_is_right(self):
        v = self.judge(ACCURATE, args={**README_ARGS, "path": "docs/README.md"})
        self.assertFalse(v["passed"])
        self.assertFalse(v["args_correct"])
        self.assertEqual(v["accuracy"], "ungradeable")

    def test_accuracy_uses_the_retrieved_text_not_the_local_readme(self):
        retrieved = ("# My-Nexus\n## Known limitations\n- Sessions expire after one hour.\n"
                     "- Exports are limited to CSV files.\n")
        on_retrieved = self.judge("- Sessions expire after one hour.\n"
                                  "- Exports are limited to CSV files.", source=retrieved)
        self.assertEqual(on_retrieved["accuracy"], "pass", on_retrieved["accuracy_report"])
        local_answer = self.judge(ACCURATE, source=retrieved)
        self.assertEqual(local_answer["accuracy"], "fail")
        self.assertIn("completeness", local_answer["accuracy_failed_checks"])

    def test_retrieved_text_without_the_section_is_ungradeable_not_failed(self):
        v = self.judge(ACCURATE, source=README_TEXT[:6000])
        self.assertTrue(v["passed"])
        self.assertEqual(v["accuracy"], "ungradeable")
        self.assertIn("no '## Known limitations' section", v["accuracy_detail"])

    def test_full_per_claim_report_is_kept_for_review(self):
        v = self.judge(ACCURATE)
        self.assertIn("not semantic proof", v["accuracy_report"])
        self.assertIsInstance(v["accuracy_review"], list)


class ControlTests(unittest.TestCase):
    def test_no_call_is_reported_clean(self):
        v = ev.judge_control(agent_result())
        self.assertEqual((v["passed"], v["gated"], v["unnecessary_retrieval"]),
                         (True, False, False))

    def test_github_call_is_unnecessary_retrieval_not_a_gated_failure(self):
        v = ev.judge_control(agent_result(record(GH, GH, "succeeded", **README_ARGS)))
        self.assertTrue(v["unnecessary_retrieval"])
        self.assertFalse(v["gated"])
        self.assertIn("report-only", v["detail"])

    def test_controls_never_affect_the_gate_and_are_not_listed_as_failures(self):
        results = full_results() + [followup_case(), control_case(retrieved=True)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(ev.summarize(results, ["revised"]))
        text = out.getvalue()
        self.assertIn("No-tool controls (report-only; never gated)", text)
        self.assertIn("unnecessary retrieval: yes", text)
        self.assertNotIn("Failures:", text)


class FollowupGateTests(unittest.TestCase):
    def ok(self, extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return ev.summarize(full_results() + extra, ["revised"]), out.getvalue()

    def test_explicit_retrieval_failure_fails_the_gate(self):
        ok, _ = self.ok([followup_case(), followup_case(case=ev.FOLLOWUP_EXPLICIT_CASE,
                                                        passed=False)])
        self.assertFalse(ok)

    def test_accuracy_failure_alone_does_not_fail_the_gate(self):
        ok, out = self.ok([followup_case(accuracy="fail", accuracy_detail="lexical checks "
                                         "failed: ['completeness']",
                                         accuracy_missing=["Long-term memory is schema only."])])
        self.assertTrue(ok)
        self.assertIn("accuracy (advisory): fail", out)
        self.assertIn("missing item: Long-term memory is schema only.", out)

    def test_summary_states_that_accuracy_is_advisory_and_not_proof(self):
        _, out = self.ok([followup_case(accuracy="pass", accuracy_detail="ok")])
        self.assertIn("ACCURACY is an ADVISORY lexical diagnostic", out)
        self.assertIn("does not prove the answer is accurate", out)
        self.assertIn("re-fetched README.md: yes", out)


class Phase3EvaluateTests(unittest.TestCase):
    def run_eval(self, provider, seed=None):
        policy, _ = recording_policy()
        return quiet(ev.evaluate(make_manager(policy), policy, provider, ["revised"], INTRO,
                                 execute=False, followup=FOLLOWUP, seed=seed))

    def test_explicit_followup_refetches_and_is_graded(self):
        results = self.run_eval(FollowupProvider("verbatim"))
        [explicit] = [r for r in results if r.case == ev.FOLLOWUP_EXPLICIT_CASE]
        self.assertTrue(explicit.passed, explicit.detail)
        self.assertEqual(explicit.calls[0]["arguments"], README_ARGS)
        # The scripted answer is not a limitations list: graded, and it fails.
        self.assertEqual(explicit.accuracy, "fail")

    def test_explicit_followup_uses_the_exact_fixture_history(self):
        provider = FollowupProvider("verbatim")
        self.run_eval(provider)
        first = next(r for r in provider.requests if r[-1]["role"] == "user"
                     and r[-1]["content"] == ev.EXPLICIT_FOLLOWUP_QUESTION)
        self.assertEqual(first[1:-1], FOLLOWUP["history"])

    def test_controls_run_after_the_history_and_report_no_retrieval(self):
        provider = FollowupProvider("verbatim")
        results = self.run_eval(provider)
        controls = [r for r in results if r.kind == "control"]
        self.assertEqual([r.case for r in controls],
                         ["rephrase-previous-answer", "readme-in-general"])
        self.assertTrue(all(r.unnecessary_retrieval is False and not r.gated
                            for r in controls))
        for _, question in ev.NO_TOOL_CONTROLS:
            first = next(r for r in provider.requests if r[-1]["content"] == question)
            self.assertEqual(first[1:-1], FOLLOWUP["history"])

    def test_retrieving_controls_are_reported_without_failing(self):
        results = self.run_eval(ScriptedProvider(github_for_negatives=True))
        controls = [r for r in results if r.kind == "control"]
        self.assertTrue(all(r.unnecessary_retrieval for r in controls))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ev.summarize(results, ["revised"])
        text = out.getvalue()
        # This provider also fails the negatives, so a Failures section exists;
        # the controls must be in the report-only section and not in Failures.
        failures = text.split("\nFailures:\n", 1)[1]
        self.assertIn("negative", " ".join(r.kind for r in results if not r.passed))
        for name, _ in ev.NO_TOOL_CONTROLS:
            self.assertNotIn(name, failures)
            self.assertIn(f"{name:30} unnecessary retrieval: yes", text)

    def test_every_result_carries_the_seed(self):
        results = self.run_eval(FollowupProvider("verbatim"), seed=7)
        self.assertEqual({r.seed for r in results}, {7})

    def test_json_records_seed_and_both_axes(self):
        results = self.run_eval(FollowupProvider("verbatim"), seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            self.assertIsNone(ev.write_results_json(path, results, force=False))
            rows = json.loads(path.read_text(encoding="utf-8"))
        [row] = [r for r in rows if r["case"] == ev.FOLLOWUP_EXPLICIT_CASE]
        self.assertEqual(row["seed"], 3)
        self.assertTrue(row["retrieval_passed"])
        self.assertIn(row["accuracy"], ("pass", "fail", "ungradeable"))
        self.assertIn("not semantic proof", row["accuracy_report"])


class SeedOptionTests(unittest.TestCase):
    def test_default_seed_is_zero(self):
        self.assertEqual(ev.parse_args(["--history-file", "h"]).seed, 0)

    def test_seed_is_parsed_as_an_integer(self):
        self.assertEqual(ev.parse_args(["--history-file", "h", "--seed", "4"]).seed, 4)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ev.parse_args(["--history-file", "h", "--seed", "x"])


# ----- readme_followup_e2e.py: pure check logic (no API, no .env) -------------
import readme_followup_e2e as e2e  # noqa: E402  (backend/ is the working directory)


def turn(executions=(("github.get_file_contents", "succeeded"),), answer="a",
         stop="complete"):
    return {"tool_executions": [{"tool_name": n, "status": s} for n, s in executions],
            "stop_reason": stop, "assistant_message": {"content": answer},
            "_elapsed_s": 1.0}


def stored_for(t2):
    return [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": t2["assistant_message"]["content"]}]


class E2ECheckTests(unittest.TestCase):
    def report(self, t2, t1=None, stored=None, readme=README_TEXT, question="q"):
        t1 = t1 or turn()
        return e2e.build_report("explicit", "conv", t1, t2,
                                stored if stored is not None else stored_for(t2),
                                question, readme)

    def test_questions_implicit_verbatim_and_explicit_shared(self):
        questions = e2e.load_questions(FOLLOWUP)
        self.assertEqual(questions["implicit"],
                         (FOLLOWUP["history"][0]["content"], FOLLOWUP["question"]))
        self.assertEqual(questions["explicit"],
                         (FOLLOWUP["history"][0]["content"], ev.EXPLICIT_FOLLOWUP_QUESTION))

    def test_default_runs_both_cases_and_each_can_run_alone(self):
        self.assertEqual(e2e.parse_args([]).case, "both")
        self.assertEqual(e2e.CASES, ("implicit", "explicit"))
        for case in e2e.CASES:
            self.assertEqual(e2e.parse_args(["--case", case]).case, case)

    def test_all_good(self):
        r = self.report(turn(answer=ACCURATE))
        self.assertTrue(r.gated_ok)
        self.assertEqual(r.accuracy, "pass", r.accuracy_report)

    def test_no_turn2_retrieval_fails_retrieval_only(self):
        r = self.report(turn(executions=(), answer=ACCURATE))
        self.assertFalse(r.gated_ok)
        self.assertEqual([ok for _, ok in r.retrieval], [True, False])
        self.assertTrue(all(ok for _, ok in r.integrity))
        # Accurate wording without retrieval is explicitly marked as not grounded.
        self.assertEqual(r.accuracy, "pass")
        self.assertIn("NOT GROUNDED", r.accuracy_detail)

    def test_inaccurate_answer_never_changes_the_gated_result(self):
        r = self.report(turn(answer=MISSING_ONE))
        self.assertTrue(r.gated_ok)
        self.assertEqual(r.accuracy, "fail")

    def test_integrity_failures(self):
        cases = {
            "research tool": turn(executions=(("github.get_file_contents", "succeeded"),
                                              ("research.search_openalex_works", "succeeded"))),
            "rejected call": turn(executions=(("github.get_file_contents", "succeeded"),
                                              ("github.get_file_contents", "rejected"))),
            "not complete": turn(stop="tool_limit"),
        }
        for label, t2 in cases.items():
            with self.subTest(label):
                r = self.report(t2)
                self.assertFalse(all(ok for _, ok in r.integrity))
        r = self.report(turn(), stored=stored_for(turn())[:3])
        self.assertFalse(all(ok for _, ok in r.integrity))

    def test_unreadable_local_readme_is_ungradeable(self):
        r = self.report(turn(answer=ACCURATE), readme=None)
        self.assertEqual(r.accuracy, "ungradeable")
        self.assertTrue(r.gated_ok)

    def test_printed_report_separates_groups_and_warns(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            e2e.print_report(self.report(turn(answer=ACCURATE)))
        text = out.getvalue()
        for heading in ("RETRIEVAL (gated):", "INTEGRITY (gated):",
                        "ACCURACY (ADVISORY, not gated):"):
            self.assertIn(heading, text)
        self.assertIn("may differ from GitHub's default branch", text)
        self.assertIn("does not prove the answer is accurate", text)
        self.assertIn("not the path", text)

    def test_main_refuses_without_token_before_any_request(self):
        def forbidden(*a, **k):
            raise AssertionError("no HTTP request may be made without a token")

        with patch.object(e2e, "load_dotenv", lambda *a, **k: None), \
                patch.dict(os.environ, {"NEXUS_DEV_TOKEN": ""}), \
                patch.object(e2e, "urlopen", forbidden), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = e2e.main([])
        self.assertEqual(code, 2)
        self.assertIn("NEXUS_DEV_TOKEN is not set", out.getvalue())


# ----- Phase 3.1: per-case errors, per-call timing, atomic incremental JSON ----
from app.agent.providers import (  # noqa: E402
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)

SHOW_README_Q = ev.POSITIVE_CASES[1][1]          # case 2: revised / no-history / show-readme
NEGATIVE_Q = ev.NEGATIVE_CASES[0][1]
REVISED_TOTAL = 30                               # 26 positives/negatives + 4 follow-up cases
PRE_31_KEYS = {
    "variant", "history", "case", "kind", "passed", "detail", "selected", "args_correct",
    "accepted", "calls", "stop_reason", "quotes_checked", "quotes_unmatched",
    "source_truncated", "truncation_acknowledged", "gated", "seed", "retrieval_passed",
    "accuracy", "accuracy_detail", "accuracy_failed_checks", "accuracy_missing",
    "accuracy_unsupported", "accuracy_review", "accuracy_report", "unnecessary_retrieval",
}
MODEL_CALL_KEYS = {"round", "phase", "request_messages", "tool_result_chars", "duration_s",
                   "outcome", "reply_tool_calls", "reply_chars"}


class FaultyProvider(FollowupProvider):
    """FollowupProvider that raises ``exc`` ONCE: the first time ``question``
    reaches ``round_`` (1 = tool selection, 2 = after the tool result).
    Questions that run in both history conditions succeed the second time.
    Counts every request."""

    def __init__(self, question, round_, exc, mode="verbatim"):
        super().__init__(mode)
        self.question, self.round, self.exc = question, round_, exc
        self.fired = False
        self.all_requests = []

    async def chat(self, messages, tools=None):
        self.all_requests.append(messages[-1].get("role"))
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        round_ = 2 if messages[-1]["role"] == "tool" else 1
        if not self.fired and user == self.question and round_ == self.round:
            self.fired = True
            raise self.exc
        return await super().chat(messages, tools)


def run_with(provider, **kw):
    policy, _ = recording_policy()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        results = asyncio.run(ev.evaluate(make_manager(policy), policy, provider, ["revised"],
                                          INTRO, execute=False, followup=FOLLOWUP, **kw))
    return results, out.getvalue()


def errored(kind="positive", history=ev.NO_HISTORY, **kw):
    fields = dict(status="error", error_type="ProviderTimeoutError", error_round=2,
                  error_phase="post_tool_result")
    fields.update(kw)
    return ev.CaseResult("revised", history, "c", kind, False, "UNGRADED", selected=None,
                         args_correct=None, accepted=None, calls=[], stop_reason="error",
                         **fields)


class TimeoutHandlingTests(unittest.TestCase):
    def test_timeout_after_tool_result_is_ungraded_and_stops_the_run(self):
        provider = FaultyProvider(SHOW_README_Q, 2, ProviderTimeoutError(
            "Ollama did not respond within 600.0s."))
        results, out = run_with(provider)
        self.assertEqual(len(results), 2)                 # stopped after case 2
        first, second = results
        self.assertEqual((first.status, first.passed), ("completed", True))
        self.assertEqual(second.status, "error")
        self.assertFalse(second.passed)
        self.assertEqual((second.error_type, second.error_round, second.error_phase),
                         ("ProviderTimeoutError", 2, "post_tool_result"))
        self.assertEqual(second.error_message, "Ollama did not respond within 600.0s.")
        self.assertEqual([c["outcome"] for c in second.model_calls], ["completed", "timeout"])
        self.assertEqual((second.case_index, second.cases_total), (2, REVISED_TOTAL))
        self.assertIn("UNGRADED: ProviderTimeoutError in round 2 (post_tool_result)",
                      second.detail)
        # Exactly 2 + 2 requests: no retry of the timed-out call, nothing after it.
        self.assertEqual(len(provider.all_requests), 4)
        self.assertIn("STOPPED after ProviderTimeoutError", out)
        self.assertIn(f"{REVISED_TOTAL - 2} remaining case(s) were not run", out)

    def test_timeout_during_tool_selection_is_identified(self):
        provider = FaultyProvider(SHOW_README_Q, 1, ProviderTimeoutError("t"))
        results, _ = run_with(provider)
        r = results[-1]
        self.assertEqual((r.error_round, r.error_phase), (1, "tool_selection"))
        self.assertEqual([c["outcome"] for c in r.model_calls], ["timeout"])

    def test_continue_after_timeout_runs_the_rest_without_retrying(self):
        provider = FaultyProvider(SHOW_README_Q, 2, ProviderTimeoutError("t"))
        results, out = run_with(provider, continue_after_timeout=True)
        self.assertEqual(len(results), REVISED_TOTAL)
        self.assertEqual([r.after_timeout for r in results[:2]], [False, False])
        self.assertTrue(all(r.after_timeout for r in results[2:]))
        self.assertNotIn("STOPPED", out)
        # The timed-out case contributed exactly its two calls; it was not re-sent.
        self.assertEqual(sum(1 for r in results if r.status == "error"), 1)
        self.assertEqual(len(provider.all_requests),
                         sum(len(r.model_calls) for r in results))

    def test_connection_error_also_stops_by_default(self):
        results, out = run_with(FaultyProvider(SHOW_README_Q, 1, ProviderConnectionError("x")))
        self.assertEqual(len(results), 2)
        self.assertIn("STOPPED after ProviderConnectionError", out)

    def test_response_error_continues_because_the_server_answered(self):
        results, out = run_with(FaultyProvider(SHOW_README_Q, 1, ProviderResponseError("bad")))
        self.assertEqual(len(results), REVISED_TOTAL)
        self.assertEqual([r.status for r in results].count("error"), 1)
        self.assertFalse(any(r.after_timeout for r in results))
        self.assertNotIn("STOPPED", out)

    def test_non_provider_errors_still_abort_after_saving_completed_cases(self):
        saved = []
        with self.assertRaises(RuntimeError):
            run_with(FaultyProvider(SHOW_README_Q, 1, RuntimeError("bug")),
                     on_result=lambda res: saved.append(len(res)))
        self.assertEqual(saved, [1])                      # case 1 was persisted first

    def test_error_messages_are_sanitized(self):
        exc = ProviderConnectionError("Cannot reach Ollama at http://10.0.0.5:11434/api: "
                                      "refused " + "x" * 500)
        message = ev.sanitize_error(exc)
        self.assertNotIn("10.0.0.5", message)
        self.assertIn("<url>", message)
        self.assertLessEqual(len(message), ev.MAX_ERROR_CHARS)

    def test_ungraded_followup_and_control_fields(self):
        timeout = ProviderTimeoutError("t")
        results, _ = run_with(FaultyProvider(ev.EXPLICIT_FOLLOWUP_QUESTION, 2, timeout),
                              continue_after_timeout=True)
        [f] = [r for r in results if r.case == ev.FOLLOWUP_EXPLICIT_CASE]
        self.assertEqual((f.status, f.passed, f.retrieval_passed, f.accuracy),
                         ("error", False, None, "ungradeable"))
        control_q = ev.NO_TOOL_CONTROLS[0][1]
        results, _ = run_with(FaultyProvider(control_q, 1, timeout))
        c = results[-1]
        self.assertEqual((c.kind, c.status, c.gated, c.unnecessary_retrieval, c.passed),
                         ("control", "error", False, None, False))


class GateIntegrityTests(unittest.TestCase):
    def gate(self, results, variants=("revised",)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return ev.summarize(list(results), list(variants)), out.getvalue()

    def test_timed_out_negative_is_never_passed(self):
        # The trap: judging an empty result as a negative would PASS it.
        self.assertTrue(ev.judge_negative(agent_result())["passed"])
        results, _ = run_with(FaultyProvider(NEGATIVE_Q, 1, ProviderTimeoutError("t")),
                              continue_after_timeout=True)
        [neg] = [r for r in results if r.case == ev.NEGATIVE_CASES[0][0]
                 and r.history == ev.NO_HISTORY]
        self.assertEqual((neg.status, neg.passed), ("error", False))
        self.assertFalse(self.gate(results)[0])

    def test_timed_out_positive_cannot_satisfy_the_positive_gate(self):
        base = full_results()
        # Control: one ordinary (graded) failure is within the 9/10 threshold.
        graded_fail = list(base)
        graded_fail[0] = case("revised", ev.NO_HISTORY, "positive", False)
        self.assertTrue(self.gate(graded_fail)[0])
        # The same position UNGRADED: the run is incomplete, so the gate fails.
        ungraded = list(base)
        ungraded[0] = errored()
        ok, out = self.gate(ungraded)
        self.assertFalse(ok)
        self.assertIn("Run incomplete", out)
        self.assertIn("Ungraded (errors; never counted as passed)", out)

    def test_errored_negative_fails_the_gate(self):
        results = full_results()
        results[10] = errored(kind="negative")
        self.assertFalse(self.gate(results)[0])

    def test_unrun_cases_fail_the_gate_and_finish_reports_incomplete(self):
        results = full_results()
        for r in results:
            r.cases_total = len(results) + 4          # 4 planned cases never ran
        ok, out = self.gate(results)
        self.assertFalse(ok)
        self.assertIn("4 not run", out)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = ev.finish(results, ["revised"], None, force=False)
        self.assertEqual(code, 1)
        self.assertIn("BELOW the acceptance bar (INCOMPLETE: 0 gated case(s) ungraded, "
                      "4 not run)", buf.getvalue())

    def test_ungraded_controls_do_not_make_the_gate_incomplete(self):
        results = full_results() + [ev.CaseResult(
            "revised", ev.FOLLOWUP_HISTORY, "ctl", "control", False, "UNGRADED",
            selected=None, args_correct=None, accepted=None, calls=[], stop_reason="error",
            gated=False, status="error")]
        self.assertTrue(self.gate(results)[0])

    def test_ungraded_cases_are_not_listed_as_graded_failures(self):
        results = full_results()
        results[0] = errored()
        _, out = self.gate(results)
        failures = out.split("\nFailures:\n")[1] if "\nFailures:\n" in out else ""
        self.assertNotIn("UNGRADED", failures)


class InstrumentationTests(unittest.TestCase):
    def test_positive_case_records_selection_then_post_result_generation(self):
        results, out = run_with(FollowupProvider("verbatim"))
        calls = results[0].model_calls
        self.assertEqual([(c["round"], c["phase"], c["outcome"]) for c in calls],
                         [(1, "tool_selection", "completed"),
                          (2, "post_tool_result", "completed")])
        self.assertEqual([c["reply_tool_calls"] for c in calls], [1, 0])
        self.assertTrue(all(isinstance(c["duration_s"], float) and c["duration_s"] >= 0
                            for c in calls))
        self.assertEqual(calls[0]["tool_result_chars"], 0)
        self.assertGreaterEqual(calls[1]["tool_result_chars"], len(README_TEXT))
        self.assertIn("calls: r1 tool_selection", out)

    def test_negative_case_makes_one_selection_call(self):
        results, _ = run_with(FollowupProvider("verbatim"))
        [neg] = [r for r in results if r.kind == "negative"][:1]
        self.assertEqual([c["phase"] for c in neg.model_calls], ["tool_selection"])

    def test_call_records_hold_sizes_only_never_content(self):
        results, _ = run_with(FollowupProvider("verbatim"))
        for r in results:
            for c in r.model_calls:
                self.assertEqual(set(c), MODEL_CALL_KEYS)
        dump = json.dumps([r.model_calls for r in results])
        for fragment in (README_TEXT[:120], FOLLOWUP["question"], SHOW_README_Q):
            self.assertNotIn(fragment, dump)

    def test_timed_provider_reraises_the_same_exception(self):
        exc = ProviderTimeoutError("t")

        class Raising(ChatProvider):
            async def chat(self, messages, tools=None):
                raise exc

        timed = ev.TimedProvider(Raising())
        with self.assertRaises(ProviderTimeoutError) as caught:
            asyncio.run(timed.chat([{"role": "user", "content": "q"}]))
        self.assertIs(caught.exception, exc)
        self.assertEqual(timed.calls[0]["outcome"], "timeout")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "out.json"

    def records(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def leftovers(self):
        return [p.name for p in self.dir.iterdir() if p.name != self.path.name]

    def test_report_is_saved_after_every_case(self):
        writer = ev.ReportWriter(self.path, force=False)
        sizes = []

        def save(results):
            self.assertIsNone(writer.save(results))
            sizes.append(len(self.records()))

        results, _ = run_with(FollowupProvider("verbatim"), on_result=save)
        self.assertEqual(sizes, list(range(1, REVISED_TOTAL + 1)))
        self.assertEqual(len(self.records()), len(results))
        self.assertEqual(self.leftovers(), [])

    def test_interruption_keeps_every_completed_case(self):
        writer = ev.ReportWriter(self.path, force=False)
        with self.assertRaises(KeyboardInterrupt):
            run_with(FaultyProvider(ev.POSITIVE_CASES[2][1], 1, KeyboardInterrupt()),
                     on_result=writer.save)
        rows = self.records()
        self.assertEqual([r["case_index"] for r in rows], [1, 2])
        self.assertTrue(all(r["status"] == "completed" for r in rows))

    def test_timeout_case_itself_is_persisted(self):
        writer = ev.ReportWriter(self.path, force=False)
        run_with(FaultyProvider(SHOW_README_Q, 2, ProviderTimeoutError("t")),
                 on_result=writer.save)
        rows = self.records()
        self.assertEqual([r["status"] for r in rows], ["completed", "error"])
        self.assertEqual(rows[1]["cases_total"], REVISED_TOTAL)

    def test_failed_replace_keeps_the_previous_file_and_no_temp_files(self):
        writer = ev.ReportWriter(self.path, force=False)
        self.assertIsNone(writer.save([case()]))
        before = self.path.read_text(encoding="utf-8")
        with patch.object(ev.os, "replace", side_effect=PermissionError(13, "Access denied")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            error = writer.save([case(), case()])
        self.assertIn("PermissionError", error)
        self.assertIn("JSON SAVE FAILED after 2 case(s)", out.getvalue())
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.leftovers(), [])
        self.assertEqual((writer.saves, writer.failures), (1, 1))

    def test_failed_first_write_leaves_no_placeholder(self):
        with patch.object(ev.os, "replace", side_effect=PermissionError(13, "denied")):
            error = ev.write_results_json(self.path, [case()], force=False)
        self.assertIsNotNone(error)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.leftovers(), [])

    def test_unserializable_results_never_touch_the_file(self):
        self.assertIsNone(ev.write_results_json(self.path, [case()], force=True))
        before = self.path.read_text(encoding="utf-8")
        bad = case()
        bad.calls = [{"arguments": object()}]
        error = ev.write_results_json(self.path, [bad], force=True)
        self.assertIn("could not be serialized", error)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_file_appearing_after_validation_is_refused_then_force_replaces(self):
        self.path.write_text("keep", encoding="utf-8")
        writer = ev.ReportWriter(self.path, force=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIn("already exists", writer.save([case()]))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "keep")
        self.assertIsNone(ev.ReportWriter(self.path, force=True).save([case()]))
        self.assertEqual(len(self.records()), 1)

    def test_later_saves_replace_the_runs_own_file(self):
        writer = ev.ReportWriter(self.path, force=False)
        for n in (1, 2, 3):
            self.assertIsNone(writer.save([case()] * n))
        self.assertEqual(len(self.records()), 3)

    def test_finish_final_save_after_incremental_saves(self):
        writer = ev.ReportWriter(self.path, force=False)
        results = full_results()
        for i in range(1, len(results) + 1):
            writer.save(results[:i])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ev.finish(results, ["revised"], self.path, force=False, writer=writer)
        self.assertEqual(code, 0)
        self.assertIn(f"Per-case results written to {self.path}", out.getvalue())
        self.assertEqual(len(self.records()), len(results))

    def test_an_earlier_failed_save_is_reflected_in_the_exit_status(self):
        writer = ev.ReportWriter(self.path, force=False)
        results = full_results()
        with patch.object(ev.os, "replace", side_effect=PermissionError(13, "denied")), \
                contextlib.redirect_stdout(io.StringIO()):
            writer.save(results[:1])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ev.finish(results, ["revised"], self.path, force=False, writer=writer)
        self.assertEqual(code, 2)                          # final save OK, earlier one not
        self.assertIn("1 save(s) failed during this run", out.getvalue())
        self.assertEqual(len(self.records()), len(results))

    def test_gate_failure_still_takes_precedence_over_persistence_failure(self):
        writer = ev.ReportWriter(self.path, force=False)
        results = full_results(passes={ev.NO_HISTORY: 5})
        with patch.object(ev.os, "replace", side_effect=PermissionError(13, "denied")), \
                contextlib.redirect_stdout(io.StringIO()):
            code = ev.finish(results, ["revised"], self.path, force=False, writer=writer)
        self.assertEqual(code, 1)

    def test_report_format_is_backward_compatible(self):
        writer = ev.ReportWriter(self.path, force=False)
        results, _ = run_with(FollowupProvider("verbatim"), on_result=writer.save)
        rows = self.records()
        self.assertIsInstance(rows, list)
        for row in rows:
            self.assertLessEqual(PRE_31_KEYS, set(row))
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["cases_total"], REVISED_TOTAL)
        self.assertEqual([r["case_index"] for r in rows], list(range(1, REVISED_TOTAL + 1)))

    def test_main_saves_incrementally_when_json_is_requested(self):
        history = self.dir / "intro.json"
        history.write_text(json.dumps(INTRO), encoding="utf-8")
        seen = {}

        class Manager:
            def __init__(self, policy):
                self.permitted_tools = {GH: object()}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def connect_http_server(self, *a, **k):
                return None

            async def connect_stdio_server(self, *a, **k):
                return None

        async def fake_evaluate(*a, on_result=None, continue_after_timeout=False, **k):
            seen["continue"] = continue_after_timeout
            results = [case()]
            on_result(results)
            seen["after_first_save"] = len(self.records())
            return results

        env = {"GITHUB_TOKEN": "dummy", "GITHUB_MCP_URL": "http://127.0.0.1:9/never"}
        with patch.dict(os.environ, env), \
                patch.object(ev, "MCPConnectionManager", Manager), \
                patch.object(ev, "OllamaProvider", lambda **k: object()), \
                patch.object(ev, "evaluate", fake_evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(ev.main(["--history-file", str(history), "--json", str(self.path),
                                 "--continue-after-timeout"]))
        self.assertEqual(seen, {"continue": True, "after_first_save": 1})
        self.assertEqual(len(self.records()), 1)


class ContinueFlagTests(unittest.TestCase):
    def test_default_is_stop(self):
        self.assertFalse(ev.parse_args(["--history-file", "h"]).continue_after_timeout)

    def test_flag_enables_continuation(self):
        self.assertTrue(ev.parse_args(["--history-file", "h", "--continue-after-timeout"])
                        .continue_after_timeout)


if __name__ == "__main__":
    unittest.main()
