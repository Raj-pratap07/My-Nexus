"""
Repositories. Every access to user-owned data requires the owner.

  * Conversations and memories are looked up by (id AND user_id); a row
    owned by someone else is indistinguishable from a missing one.
  * Messages and tool executions can only be reached by passing an
    already-authorized ``Conversation`` object, never a bare id.
Repositories flush but never commit; the caller owns the transaction.
"""
from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Conversation,
    Memory,
    Message,
    ToolExecution,
    User,
    utcnow,
)


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_or_create(self, user_id: uuid.UUID) -> User:
        user = await self._s.get(User, user_id)
        if user is None:
            user = User(id=user_id)
            self._s.add(user)
            await self._s.flush()
        return user


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, user_id: uuid.UUID, title: str | None = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        if title:
            conv.title = title
        self._s.add(conv)
        await self._s.flush()
        return conv

    async def get_owned(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list_owned(
        self, user_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Conversation]:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def delete_owned(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> bool:
        conv = await self.get_owned(user_id, conversation_id)
        if conv is None:
            return False
        await self._s.delete(conv)
        await self._s.flush()
        return True

    async def touch(self, conversation: Conversation) -> None:
        conversation.updated_at = utcnow()
        await self._s.flush()


class MessageRepository:
    """Messages are reached only through an authorized Conversation."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        *,
        status: str = "completed",
        error_code: str | None = None,
    ) -> Message:
        next_pos = (
            await self._s.execute(
                select(func.coalesce(func.max(Message.position), 0) + 1).where(
                    Message.conversation_id == conversation.id
                )
            )
        ).scalar_one()
        msg = Message(
            conversation_id=conversation.id,
            position=next_pos,
            role=role,
            content=content,
            status=status,
            error_code=error_code,
        )
        self._s.add(msg)
        await self._s.flush()
        return msg

    async def list_for(
        self, conversation: Conversation, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.position)
            .limit(limit)
            .offset(offset)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def recent_for(self, conversation: Conversation, count: int) -> list[Message]:
        if count <= 0:
            return []
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.position.desc())
            .limit(count)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return list(reversed(rows))


class ToolExecutionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(
        self,
        conversation: Conversation,
        tool_name: str,
        status: str,
        message: Message | None = None,
    ) -> ToolExecution:
        row = ToolExecution(
            conversation_id=conversation.id,
            message_id=message.id if message else None,
            tool_name=tool_name[:128],
            status=status,
        )
        self._s.add(row)
        await self._s.flush()
        return row

    async def list_for(self, conversation: Conversation) -> Sequence[ToolExecution]:
        stmt = (
            select(ToolExecution)
            .where(ToolExecution.conversation_id == conversation.id)
            .order_by(ToolExecution.created_at, ToolExecution.id)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def list_for_messages(
        self, conversation: Conversation, message_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, list[ToolExecution]]:
        """Executions of the given messages (one query), grouped by message,
        each group in execution order. Scoped to the authorized conversation."""
        if not message_ids:
            return {}
        stmt = (
            select(ToolExecution)
            .where(
                ToolExecution.conversation_id == conversation.id,
                ToolExecution.message_id.in_(list(message_ids)),
            )
            .order_by(ToolExecution.created_at, ToolExecution.id)
        )
        grouped: dict[uuid.UUID, list[ToolExecution]] = {}
        for row in (await self._s.execute(stmt)).scalars().all():
            grouped.setdefault(row.message_id, []).append(row)
        return grouped


class MemoryRepository:
    """Ownership-scoped storage only; no memory behaviour in this milestone."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self,
        user_id: uuid.UUID,
        content: str,
        category: str = "general",
        status: str = "pending",
    ) -> Memory:
        mem = Memory(user_id=user_id, content=content, category=category, status=status)
        self._s.add(mem)
        await self._s.flush()
        return mem

    async def get_owned(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> Memory | None:
        stmt = select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list_owned(
        self, user_id: uuid.UUID, *, status: str | None = None
    ) -> Sequence[Memory]:
        stmt = select(Memory).where(Memory.user_id == user_id)
        if status:
            stmt = stmt.where(Memory.status == status)
        return (await self._s.execute(stmt.order_by(Memory.created_at, Memory.id))).scalars().all()

    async def delete_owned(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        result = await self._s.execute(
            delete(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        )
        return result.rowcount > 0
