"""
Conversation business logic.

The service layer is the only place that combines the database with the
agent. It talks to the agent exclusively through ``AgentRunner`` (satisfied
by AgentRuntime, which wraps the existing AgentOrchestrator); it never sees
MCP sessions, tools or the connection manager.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.orchestrator import AgentResult
from app.agent.providers import ProviderError
from app.core.errors import (
    AgentUnavailableError,
    ConflictError,
    CredentialRejectedError,
    NotFoundError,
)
from app.core.security import contains_credential, redact
from app.database.models import Conversation, Message, ToolExecution
from app.database.repositories import (
    ConversationRepository,
    MessageRepository,
    ToolExecutionRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New conversation"
_SAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_.:-]")


class AgentRunner(Protocol):
    async def run(
        self, message: str, history: Sequence[dict[str, Any]]
    ) -> AgentResult: ...


@dataclass
class TurnResult:
    user_message: Message
    assistant_message: Message
    tool_executions: list[ToolExecution]
    stop_reason: str


def _title_from(content: str) -> str:
    line = " ".join(content.split())
    return (line[:77] + "...") if len(line) > 80 else line


class ConversationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        agent: AgentRunner,
        *,
        history_messages: int = 20,
        secrets: Sequence[str] = (),
    ) -> None:
        self._sessions = session_factory
        self._agent = agent
        self._history_messages = history_messages
        self._secrets = tuple(secrets)
        self._in_flight: dict[uuid.UUID, bool] = {}

    # ----- plain CRUD, always owner-scoped -------------------------------
    async def create_conversation(
        self, user_id: uuid.UUID, title: str | None = None
    ) -> Conversation:
        async with self._sessions() as session, session.begin():
            await UserRepository(session).get_or_create(user_id)
            return await ConversationRepository(session).create(user_id, title)

    async def list_conversations(
        self, user_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Conversation]:
        async with self._sessions() as session:
            return await ConversationRepository(session).list_owned(
                user_id, limit=limit, offset=offset
            )

    async def get_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation:
        async with self._sessions() as session:
            return await self._owned(session, user_id, conversation_id)

    async def delete_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> None:
        async with self._sessions() as session, session.begin():
            if not await ConversationRepository(session).delete_owned(
                user_id, conversation_id
            ):
                raise NotFoundError()

    async def list_messages(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        async with self._sessions() as session:
            conv = await self._owned(session, user_id, conversation_id)
            return await MessageRepository(session).list_for(
                conv, limit=limit, offset=offset
            )

    @staticmethod
    async def _owned(
        session: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation:
        conv = await ConversationRepository(session).get_owned(user_id, conversation_id)
        if conv is None:
            raise NotFoundError()
        return conv

    # ----- one agent turn --------------------------------------------------
    async def send_message(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, content: str
    ) -> TurnResult:
        if contains_credential(content, self._secrets):
            raise CredentialRejectedError()

        # One in-flight turn per conversation (check+set has no await in
        # between, so it is atomic on the event loop).
        if self._in_flight.get(conversation_id):
            raise ConflictError("This conversation is already processing a message.")
        self._in_flight[conversation_id] = True
        try:
            return await self._turn(user_id, conversation_id, content)
        finally:
            self._in_flight.pop(conversation_id, None)

    async def _turn(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, content: str
    ) -> TurnResult:
        # Transaction 1: authorize, load history, persist the user message.
        async with self._sessions() as session, session.begin():
            await UserRepository(session).get_or_create(user_id)
            conv = await self._owned(session, user_id, conversation_id)
            messages = MessageRepository(session)
            history = [
                {"role": m.role, "content": m.content}
                for m in await messages.recent_for(conv, self._history_messages)
            ]
            user_msg = await messages.add(conv, "user", content)
            if conv.title == DEFAULT_TITLE and not history:
                conv.title = _title_from(content)
            await ConversationRepository(session).touch(conv)

        # No database transaction is held while the model runs.
        try:
            result = await self._agent.run(content, history)
        except ProviderError as exc:
            logger.warning("Agent run failed: %s", type(exc).__name__)
            raise AgentUnavailableError() from None

        answer = redact(result.answer, self._secrets) or "(empty response)"

        # Transaction 2: authorize again (it may have been deleted meanwhile).
        async with self._sessions() as session, session.begin():
            conv = await self._owned(session, user_id, conversation_id)
            assistant_msg = await MessageRepository(session).add(
                conv, "assistant", answer
            )
            executions = [
                await ToolExecutionRepository(session).add(
                    conv, self._tool_name(rec), rec.outcome, assistant_msg
                )
                for rec in result.tool_calls
            ]
            await ConversationRepository(session).touch(conv)

        return TurnResult(user_msg, assistant_msg, executions, result.stop_reason)

    @staticmethod
    def _tool_name(record) -> str:
        if record.tool:
            return record.tool
        return "unknown:" + _SAFE_TOOL_NAME.sub("?", record.function)[:100]
