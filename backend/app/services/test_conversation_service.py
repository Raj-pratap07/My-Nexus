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

from app.agent.orchestrator import AgentResult, ToolCallRecord
from app.agent.providers import ProviderConnectionError
from app.core.errors import (
    AgentUnavailableError,
    ConflictError,
    CredentialRejectedError,
    NotFoundError,
)
from app.database.models import Message, ToolExecution
from app.database.repositories import ToolExecutionRepository
from app.database.session import create_session_factory
from app.services.conversation_service import ConversationService
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
    async def test_provider_failure_maps_to_unavailable_and_keeps_user_message(self):
        svc = self.service(FakeAgent(error=ProviderConnectionError("http://localhost:11434 down")))
        conv = await svc.create_conversation(self.alice)
        with self.assertRaises(AgentUnavailableError) as ctx:
            await svc.send_message(self.alice, conv.id, "hello")
        self.assertNotIn("11434", ctx.exception.public_message)
        msgs = await svc.list_messages(self.alice, conv.id)
        self.assertEqual([m.role for m in msgs], ["user"])

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


if __name__ == "__main__":
    unittest.main()
