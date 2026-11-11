"""
Offline tests for ConversationService (SQLite + a fake agent; no MCP, no LLM).

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.services.test_conversation_service -v
"""
from __future__ import annotations

import asyncio
import unittest
import uuid
from types import SimpleNamespace

from unittest.mock import patch

from app.agent.orchestrator import AgentResult, ToolCallRecord
from app.agent.providers import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.core.errors import (
    AgentUnavailableError,
    ConflictError,
    ConversationBusyError,
    CredentialRejectedError,
    NotFoundError,
)
from app.database.models import Message, ToolExecution
from app.database.repositories import MessageRepository, ToolExecutionRepository
from app.database.session import create_session_factory
from app.services.conversation_service import (
    TURN_FAILURES,
    ConversationService,
    model_history,
)
from app.testutil import TempDb, make_engine
from sqlalchemy import select

SECRET = "super-secret-value-123456"


class FakeAgent:
    def __init__(self, answer="ok", tool_calls=(), error=None, gate=None):
        self.answer, self.tool_calls, self.error, self.gate = answer, tool_calls, error, gate
        self.calls: list[tuple[str, list]] = []

    async def run(self, message, history):
        self.calls.append((message, list(history)))
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return AgentResult(self.answer, "complete", list(self.tool_calls))


def record(function, tool, outcome, executed=True):
    return ToolCallRecord(function, tool, {}, executed, None, outcome)


class ServiceCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = TempDb()
        self.engine = await make_engine(self.db.url)
        self.sessions = create_session_factory(self.engine)
        self.alice, self.bob = uuid.uuid4(), uuid.uuid4()

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.db.cleanup()

    def service(self, agent, **kw):
        return ConversationService(self.sessions, agent, secrets=(SECRET,), **kw)


class OwnershipTests(ServiceCase):
    async def test_other_user_cannot_read_write_or_delete(self):
        agent = FakeAgent()
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        await svc.send_message(self.alice, conv.id, "hello")

        for call in (
            svc.get_conversation(self.bob, conv.id),
            svc.list_messages(self.bob, conv.id),
            svc.delete_conversation(self.bob, conv.id),
            svc.send_message(self.bob, conv.id, "intrude"),
        ):
            with self.assertRaises(NotFoundError):
                await call
        self.assertEqual(len(agent.calls), 1)  # the agent never ran for Bob
        self.assertEqual(len(await svc.list_messages(self.alice, conv.id)), 2)
        self.assertEqual(await svc.list_conversations(self.bob), [])

    async def test_unknown_conversation_is_indistinguishable_from_foreign(self):
        svc = self.service(FakeAgent())
        with self.assertRaises(NotFoundError):
            await svc.get_conversation(self.alice, uuid.uuid4())


class TurnTests(ServiceCase):
    async def test_turn_persists_messages_history_and_title(self):
        agent = FakeAgent(answer="first answer")
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)

        turn = await svc.send_message(self.alice, conv.id, "  What is RAG?  ")
        self.assertEqual((turn.user_message.role, turn.assistant_message.role), ("user", "assistant"))
        self.assertEqual(agent.calls[0], ("  What is RAG?  ", []))
        self.assertEqual((turn.user_message.position, turn.assistant_message.position), (1, 2))
        self.assertEqual((await svc.get_conversation(self.alice, conv.id)).title, "What is RAG?")

        await svc.send_message(self.alice, conv.id, "and more?")
        _, history = agent.calls[1]
        self.assertEqual(
            history,
            [{"role": "user", "content": "  What is RAG?  "},
             {"role": "assistant", "content": "first answer"}],
        )

    async def test_history_is_bounded(self):
        agent = FakeAgent()
        svc = self.service(agent, history_messages=3)
        conv = await svc.create_conversation(self.alice)
        for i in range(4):
            await svc.send_message(self.alice, conv.id, f"m{i}")
        self.assertEqual(len(agent.calls[-1][1]), 3)

    async def test_stored_history_only_has_user_and_assistant_roles(self):
        svc = self.service(FakeAgent())
        conv = await svc.create_conversation(self.alice)
        await svc.send_message(self.alice, conv.id, "hi")
        roles = {m.role for m in await svc.list_messages(self.alice, conv.id)}
        self.assertEqual(roles, {"user", "assistant"})

    async def test_credential_in_user_message_rejected_and_not_stored(self):
        agent = FakeAgent()
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        for text in (f"my key is {SECRET}", "token: ghp_" + "a" * 36,
                     "Authorization: Bearer " + "x" * 30):
            with self.subTest(text=text[:20]), self.assertRaises(CredentialRejectedError):
                await svc.send_message(self.alice, conv.id, text)
        self.assertEqual(agent.calls, [])
        self.assertEqual(await svc.list_messages(self.alice, conv.id), [])

    async def test_assistant_output_is_redacted_before_storage(self):
        svc = self.service(FakeAgent(answer=f"leak {SECRET} and ghp_{'b' * 36}"))
        conv = await svc.create_conversation(self.alice)
        turn = await svc.send_message(self.alice, conv.id, "hi")
        self.assertNotIn(SECRET, turn.assistant_message.content)
        self.assertNotIn("ghp_", turn.assistant_message.content)
        self.assertIn("[REDACTED]", turn.assistant_message.content)


class ToolExecutionTests(ServiceCase):
    async def test_tool_executions_recorded_with_name_and_status_only(self):
        calls = [
            record("research.search_crossref_works", "research.search_crossref_works", "succeeded"),
            record("github.get_file_contents", "github.get_file_contents", "failed"),
            record("rm -rf /;\n", None, "rejected", executed=False),
        ]
        svc = self.service(FakeAgent(tool_calls=calls))
        conv = await svc.create_conversation(self.alice)
        turn = await svc.send_message(self.alice, conv.id, "go")

        got = [(t.tool_name, t.status) for t in turn.tool_executions]
        self.assertEqual(got[0], ("research.search_crossref_works", "succeeded"))
        self.assertEqual(got[1], ("github.get_file_contents", "failed"))
        self.assertEqual(got[2][1], "rejected")
        self.assertRegex(got[2][0], r"^unknown:[A-Za-z0-9_.:?-]+$")  # sanitized model text
        self.assertTrue(all(t.message_id == turn.assistant_message.id for t in turn.tool_executions))
        self.assertFalse(hasattr(ToolExecution, "arguments"))
        self.assertFalse(hasattr(ToolExecution, "result"))


class FailureAndConcurrencyTests(ServiceCase):
    async def test_provider_failure_maps_to_unavailable_and_records_the_failed_turn(self):
        svc = self.service(FakeAgent(error=ProviderConnectionError("http://localhost:11434 down")))
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(AgentUnavailableError) as ctx:
            await svc.send_message(self.alice, conv.id, "hello")
        self.assertNotIn("11434", ctx.exception.public_message)
        msgs = await svc.list_messages(self.alice, conv.id)
        # The user message is kept and followed by an explicit failed marker.
        self.assertEqual([(m.role, m.status, m.error_code) for m in msgs],
                         [("user", "completed", None),
                          ("assistant", "failed", "model_unreachable")])
        self.assertEqual(msgs[0].content, "hello")

    async def test_second_concurrent_turn_on_same_conversation_is_rejected(self):
        gate = asyncio.Event()
        agent = FakeAgent(gate=gate)
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)

        first = asyncio.create_task(svc.send_message(self.alice, conv.id, "one"))
        await asyncio.sleep(0.2)  # first turn is now inside the agent
        with self.assertRaises(ConflictError):
            await svc.send_message(self.alice, conv.id, "two")
        gate.set()
        await first
        self.assertEqual(len(agent.calls), 1)
        # The slot is released afterwards.
        await svc.send_message(self.alice, conv.id, "three")

    async def test_slot_released_after_failure(self):
        agent = FakeAgent(error=ProviderConnectionError("x"))
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        for _ in range(2):
            with self.assertRaises(AgentUnavailableError):
                await svc.send_message(self.alice, conv.id, "hello")

    async def test_conversation_deleted_while_agent_runs(self):
        gate = asyncio.Event()
        svc = self.service(FakeAgent(gate=gate))
        conv = await svc.create_conversation(self.alice)
        task = asyncio.create_task(svc.send_message(self.alice, conv.id, "hi"))
        await asyncio.sleep(0.2)
        await svc.delete_conversation(self.alice, conv.id)
        gate.set()
        with self.assertRaises(NotFoundError):
            await task
        async with self.sessions() as s:
            self.assertEqual((await s.execute(select(Message))).scalars().all(), [])


# ----- Phase 4 M1: failed turns, recovery, tool history ------------------------
class ScriptedAgent:
    """Returns/raises the scripted outcomes in order (an Exception = raise)."""

    def __init__(self, *outcomes, gate=None):
        self.outcomes, self.gate = list(outcomes), gate
        self.calls: list[tuple[str, list]] = []

    async def run(self, message, history):
        self.calls.append((message, list(history)))
        if self.gate is not None:
            await self.gate.wait()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def answer(text="ok", *tool_calls):
    return AgentResult(text, "complete", list(tool_calls))


LEAKY = f"http://10.1.2.3:11434/api/chat refused; token={SECRET} at C:/srv/app.py"


class FailedTurnTests(ServiceCase):
    async def failed_turn(self, exc):
        svc = self.service(ScriptedAgent(exc))
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(Exception) as ctx:
            await svc.send_message(self.alice, conv.id, "question")
        return svc, conv, ctx.exception, await svc.list_messages(self.alice, conv.id)

    async def test_each_failure_kind_has_its_code_and_fixed_safe_text(self):
        cases = ((ProviderTimeoutError(LEAKY), "model_timeout"),
                 (ProviderConnectionError(LEAKY), "model_unreachable"),
                 (ProviderResponseError(LEAKY), "model_error"))
        for exc, code in cases:
            with self.subTest(code=code):
                _, _, raised, msgs = await self.failed_turn(exc)
                self.assertIsInstance(raised, AgentUnavailableError)
                failed = msgs[-1]
                self.assertEqual((failed.role, failed.status, failed.error_code),
                                 ("assistant", "failed", code))
                self.assertEqual(failed.content, TURN_FAILURES[code])
                for leaked in ("10.1.2.3", "11434", SECRET, "app.py", "refused"):
                    self.assertNotIn(leaked, failed.content)
                    self.assertNotIn(leaked, str(raised.details))

    async def test_unavailable_error_points_at_both_messages(self):
        _, _, raised, msgs = await self.failed_turn(ProviderTimeoutError("t"))
        self.assertEqual(raised.details, {
            "reason": "model_timeout",
            "user_message_id": str(msgs[0].id),
            "assistant_message_id": str(msgs[1].id),
        })

    async def test_unexpected_error_is_recorded_and_the_original_is_reraised(self):
        boom = RuntimeError("password=hunter2")
        _, _, raised, msgs = await self.failed_turn(boom)
        self.assertIs(raised, boom)                    # 500 mapping preserved
        self.assertEqual((msgs[-1].status, msgs[-1].error_code), ("failed", "internal_error"))
        self.assertNotIn("hunter2", msgs[-1].content)

    async def test_next_message_succeeds_and_model_never_sees_the_failed_pair(self):
        agent = ScriptedAgent(answer("first"), ProviderTimeoutError("t"), answer("third"))
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        await svc.send_message(self.alice, conv.id, "one")
        with self.assertRaises(AgentUnavailableError):
            await svc.send_message(self.alice, conv.id, "two")
        turn = await svc.send_message(self.alice, conv.id, "two again")
        self.assertEqual((turn.assistant_message.content, turn.assistant_message.status),
                         ("third", "completed"))
        self.assertEqual(agent.calls[2][1], [{"role": "user", "content": "one"},
                                             {"role": "assistant", "content": "first"}])
        msgs = await svc.list_messages(self.alice, conv.id)
        self.assertEqual([(m.content, m.status) for m in msgs], [
            ("one", "completed"), ("first", "completed"),
            ("two", "completed"), (TURN_FAILURES["model_timeout"], "failed"),
            ("two again", "completed"), ("third", "completed")])

    async def test_title_still_comes_from_the_first_message_after_a_failure(self):
        svc = self.service(ScriptedAgent(ProviderTimeoutError("t"), answer()))
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(AgentUnavailableError):
            await svc.send_message(self.alice, conv.id, "first question")
        await svc.send_message(self.alice, conv.id, "second question")
        self.assertEqual((await svc.get_conversation(self.alice, conv.id)).title,
                         "first question")

    async def test_lock_is_released_after_every_failure_kind(self):
        agent = ScriptedAgent(ProviderTimeoutError("t"), RuntimeError("x"), answer())
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(AgentUnavailableError):
            await svc.send_message(self.alice, conv.id, "a")
        with self.assertRaises(RuntimeError):
            await svc.send_message(self.alice, conv.id, "b")
        await svc.send_message(self.alice, conv.id, "c")  # no ConversationBusyError

    async def test_busy_error_is_a_conflict_with_its_own_code(self):
        gate = asyncio.Event()
        svc = self.service(ScriptedAgent(answer(), gate=gate))
        conv = await svc.create_conversation(self.alice)
        first = asyncio.create_task(svc.send_message(self.alice, conv.id, "one"))
        await asyncio.sleep(0.2)
        with self.assertRaises(ConversationBusyError) as ctx:
            await svc.send_message(self.alice, conv.id, "two")
        self.assertIsInstance(ctx.exception, ConflictError)
        self.assertEqual(ctx.exception.code, "conversation_busy")
        gate.set()
        await first


class FailureRecordingSafeguardTests(ServiceCase):
    """If storing the failed marker itself fails, the ORIGINAL failure (and
    its HTTP mapping) must still reach the caller."""

    async def test_deleted_conversation_keeps_the_agent_failure(self):
        gate = asyncio.Event()
        svc = self.service(ScriptedAgent(ProviderTimeoutError("t"), gate=gate))
        conv = await svc.create_conversation(self.alice)
        task = asyncio.create_task(svc.send_message(self.alice, conv.id, "hi"))
        await asyncio.sleep(0.2)
        await svc.delete_conversation(self.alice, conv.id)
        gate.set()
        with self.assertLogs("app.services.conversation_service", "WARNING") as logs:
            with self.assertRaises(AgentUnavailableError) as ctx:   # not NotFoundError
                await task
        self.assertIsNone(ctx.exception.details["assistant_message_id"])
        self.assertIn("Could not record failed turn (NotFoundError)", "".join(logs.output))

    async def test_storage_failure_never_masks_a_provider_failure(self):
        svc = self.service(ScriptedAgent(ProviderConnectionError("x")))
        conv = await svc.create_conversation(self.alice)
        real_add = MessageRepository.add

        async def add(self_, conversation, role, content, **kw):
            if kw.get("status") == "failed":
                raise OSError(f"disk full at C:/data {SECRET}")
            return await real_add(self_, conversation, role, content, **kw)

        with patch.object(MessageRepository, "add", add), \
                self.assertLogs("app.services.conversation_service", "WARNING") as logs:
            with self.assertRaises(AgentUnavailableError) as ctx:
                await svc.send_message(self.alice, conv.id, "hi")
        self.assertEqual(ctx.exception.details["reason"], "model_unreachable")
        text = "".join(logs.output)
        self.assertIn("Could not record failed turn (OSError)", text)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("disk full", text)

    async def test_storage_failure_never_masks_an_unexpected_failure(self):
        boom = RuntimeError("original")
        svc = self.service(ScriptedAgent(boom))
        conv = await svc.create_conversation(self.alice)

        async def add(self_, conversation, role, content, **kw):
            raise OSError("secondary")

        # The user message is stored before the run, so patch only afterwards.
        real_run = svc._agent.run

        async def run_then_break_storage(message, history):
            patcher.start()
            return await real_run(message, history)

        patcher = patch.object(MessageRepository, "add", add)
        self.addCleanup(patch.stopall)
        svc._agent.run = run_then_break_storage
        with self.assertLogs("app.services.conversation_service", "WARNING"):
            with self.assertRaises(RuntimeError) as ctx:
                await svc.send_message(self.alice, conv.id, "hi")
        self.assertIs(ctx.exception, boom)


class ModelHistoryTests(unittest.TestCase):
    def msg(self, role, content, status="completed"):
        return SimpleNamespace(role=role, content=content, status=status)

    def test_drops_each_failed_marker_and_its_user_message(self):
        msgs = [self.msg("user", "a"), self.msg("assistant", "A"),
                self.msg("user", "b"), self.msg("assistant", "x", "failed"),
                self.msg("user", "c"), self.msg("assistant", "y", "failed")]
        self.assertEqual(model_history(msgs), [{"role": "user", "content": "a"},
                                               {"role": "assistant", "content": "A"}])

    def test_failed_marker_at_the_start_of_the_window(self):
        msgs = [self.msg("assistant", "x", "failed"), self.msg("user", "b"),
                self.msg("assistant", "B")]
        self.assertEqual([m["content"] for m in model_history(msgs)], ["b", "B"])

    def test_no_failures_is_unchanged(self):
        msgs = [self.msg("user", "a"), self.msg("assistant", "A")]
        self.assertEqual(model_history(msgs), [{"role": "user", "content": "a"},
                                               {"role": "assistant", "content": "A"}])


class ToolHistoryTests(ServiceCase):
    async def test_views_attach_each_turns_tools_to_its_assistant_message(self):
        agent = ScriptedAgent(
            answer("no tools"),
            answer("with tools",
                   record("research.search_crossref_works", "research.search_crossref_works",
                          "succeeded"),
                   record("github.get_file_contents", "github.get_file_contents", "failed"),
                   record("bad;name", None, "rejected", executed=False)))
        svc = self.service(agent)
        conv = await svc.create_conversation(self.alice)
        await svc.send_message(self.alice, conv.id, "q1")
        await svc.send_message(self.alice, conv.id, "q2")
        views = await svc.list_message_views(self.alice, conv.id)
        self.assertEqual([(v.message.role, [t.status for t in v.tool_executions])
                          for v in views],
                         [("user", []), ("assistant", []), ("user", []),
                          ("assistant", ["succeeded", "failed", "rejected"])])
        names = [t.tool_name for t in views[3].tool_executions]
        self.assertEqual(names[:2], ["research.search_crossref_works",
                                     "github.get_file_contents"])
        self.assertTrue(names[2].startswith("unknown:"))

    async def test_failed_turn_view_has_no_tools(self):
        svc = self.service(ScriptedAgent(ProviderTimeoutError("t")))
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(AgentUnavailableError):
            await svc.send_message(self.alice, conv.id, "q")
        views = await svc.list_message_views(self.alice, conv.id)
        self.assertEqual([(v.message.status, v.tool_executions) for v in views],
                         [("completed", []), ("failed", [])])

    async def test_views_are_paginated_and_owner_scoped(self):
        tool = record("research.x", "research.x", "succeeded")
        svc = self.service(ScriptedAgent(answer("a", tool), answer("b", tool)))
        conv = await svc.create_conversation(self.alice)
        await svc.send_message(self.alice, conv.id, "q1")
        await svc.send_message(self.alice, conv.id, "q2")
        page = await svc.list_message_views(self.alice, conv.id, limit=1, offset=3)
        self.assertEqual([(v.message.content, len(v.tool_executions)) for v in page], [("b", 1)])
        with self.assertRaises(NotFoundError):
            await svc.list_message_views(self.bob, conv.id)


if __name__ == "__main__":
    unittest.main()
