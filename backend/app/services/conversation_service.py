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
from app.agent.providers import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from app.core.errors import (
    AgentUnavailableError,
    ConversationBusyError,
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

# A failed turn is stored as an assistant message with status 'failed', one
# of these codes, and the matching FIXED description as its content. Never
# exception text: it can carry URLs, paths or credentials.
TURN_FAILURES: dict[str, str] = {
    "model_timeout": "The language model did not respond in time.",
    "model_unreachable": "The language model could not be reached.",
    "model_error": "The language model returned an error.",
    "internal_error": "The request failed because of an internal error.",
}


def failure_code(exc: BaseException) -> str:
    if isinstance(exc, ProviderTimeoutError):
        return "model_timeout"
    if isinstance(exc, ProviderConnectionError):
        return "model_unreachable"
    if isinstance(exc, ProviderError):
        return "model_error"
    return "internal_error"


def model_history(messages: Sequence[Message]) -> list[dict[str, str]]:
    """History for the model: drop each failed assistant message AND the user
    message directly before it. The model never answered that turn, so if the
    user re-sends it the model must not see the question twice. Both stay in
    the database and the API."""
    dropped: set[int] = set()
    for i, m in enumerate(messages):
        if m.status == "failed":
            dropped.add(i)
            if i > 0 and messages[i - 1].role == "user":
                dropped.add(i - 1)
    return [{"role": m.role, "content": m.content}
            for i, m in enumerate(messages) if i not in dropped]


@dataclass
class MessageView:
    """A message with the tool executions recorded for it (read model)."""

    message: Message
    tool_executions: list[ToolExecution]


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

    async def list_message_views(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MessageView]:
        """One page of messages, each with its tool executions (2 queries)."""
        async with self._sessions() as session:
            conv = await self._owned(session, user_id, conversation_id)
            page = await MessageRepository(session).list_for(
                conv, limit=limit, offset=offset
            )
            tools = await ToolExecutionRepository(session).list_for_messages(
                conv, [m.id for m in page]
            )
            return [MessageView(m, tools.get(m.id, [])) for m in page]

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
            raise ConversationBusyError()
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
            recent = await messages.recent_for(conv, self._history_messages)
            history = model_history(recent)
            user_msg = await messages.add(conv, "user", content)
            if conv.title == DEFAULT_TITLE and not recent:
                conv.title = _title_from(content)
            await ConversationRepository(session).touch(conv)

        # No database transaction is held while the model runs.
        try:
            result = await self._agent.run(content, history)
        except ProviderError as exc:
            code = failure_code(exc)
            logger.warning("Agent run failed: %s", type(exc).__name__)
            failed = await self._record_failure(user_id, conversation_id, code)
            raise AgentUnavailableError(details={
                "reason": code,
                "user_message_id": str(user_msg.id),
                "assistant_message_id": str(failed.id) if failed else None,
            }) from None
        except Exception:
            # Not a provider error: record the failed turn, then re-raise the
            # ORIGINAL exception so its HTTP mapping (500) is unchanged.
            await self._record_failure(user_id, conversation_id, "internal_error")
            raise

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

    async def _record_failure(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID, code: str
    ) -> Message | None:
        """Store the failed assistant message. Never raises: a failure here is
        logged by type only and must not mask the agent failure the caller is
        about to raise (including NotFound if the conversation was deleted)."""
        try:
            async with self._sessions() as session, session.begin():
                conv = await self._owned(session, user_id, conversation_id)
                failed = await MessageRepository(session).add(
                    conv, "assistant", TURN_FAILURES[code],
                    status="failed", error_code=code,
                )
                await ConversationRepository(session).touch(conv)
            return failed
        except Exception as exc:  # noqa: BLE001 - secondary failure, reported below
            logger.warning(
                "Could not record failed turn (%s) after agent failure %s.",
                type(exc).__name__, code,
            )
            return None

    @staticmethod
    def _tool_name(record) -> str:
        if record.tool:
            return record.tool
        return "unknown:" + _SAFE_TOOL_NAME.sub("?", record.function)[:100]
