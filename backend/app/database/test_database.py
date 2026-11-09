"""
Offline database tests (SQLite; Postgres is not exercised here).

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.database.test_database -v
"""
from __future__ import annotations

import unittest
import uuid

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import sync_database_url
from app.database.models import (
    Base,
    Conversation,
    Memory,
    Message,
    ToolExecution,
    User,
)
from app.database.repositories import (
    ConversationRepository,
    MemoryRepository,
    MessageRepository,
    ToolExecutionRepository,
    UserRepository,
)
from app.database.session import (
    BACKEND_DIR,
    alembic_head,
    create_session_factory,
    verify_database,
)
from app.core.errors import StartupError
from app.testutil import TempDb, make_engine, create_schema
from app.database.session import create_engine


class DbCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = TempDb()
        self.engine: AsyncEngine = await make_engine(self.db.url)
        self.sessions = create_session_factory(self.engine)
        self.alice, self.bob = uuid.uuid4(), uuid.uuid4()
        async with self.sessions() as s, s.begin():
            users = UserRepository(s)
            await users.get_or_create(self.alice)
            await users.get_or_create(self.bob)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        self.db.cleanup()


class InitializationTests(DbCase):
    async def test_expected_tables_exist(self):
        async with self.engine.connect() as conn:
            tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
        self.assertEqual(
            tables, {"users", "conversations", "messages", "memories", "tool_executions"}
        )

    async def test_get_or_create_user_is_idempotent(self):
        async with self.sessions() as s, s.begin():
            await UserRepository(s).get_or_create(self.alice)
        async with self.sessions() as s:
            count = len((await s.execute(select(User))).scalars().all())
        self.assertEqual(count, 2)

    async def test_verify_database_reports_unreachable_without_secrets(self):
        bad = create_engine("sqlite+aiosqlite:///Z:/no/such/dir/secret-name.db")
        try:
            with self.assertRaises(StartupError) as ctx:
                await verify_database(bad, check_schema=False)
            self.assertNotIn("secret-name", str(ctx.exception))
        finally:
            await bad.dispose()

    async def test_verify_database_detects_missing_migration(self):
        with self.assertRaisesRegex(StartupError, "alembic upgrade head"):
            await verify_database(self.engine, check_schema=True)


class MigrationTests(unittest.TestCase):
    def _config(self, url: str) -> Config:
        cfg = Config(str(BACKEND_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
        cfg.set_main_option("sqlalchemy.url", url)
        return cfg

    def test_migration_matches_models_and_downgrades(self):
        db = TempDb()
        self.addCleanup(db.cleanup)
        cfg = self._config(db.url)
        command.upgrade(cfg, "head")

        engine = create_sync_engine(sync_database_url(db.url))
        try:
            with engine.connect() as conn:
                diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
                self.assertEqual(diff, [], f"models and migration differ: {diff}")
                version = conn.exec_driver_sql("SELECT version_num FROM alembic_version").scalar()
            self.assertEqual(version, alembic_head())

            command.downgrade(cfg, "base")
            with engine.connect() as conn:
                tables = set(inspect(conn).get_table_names())
            self.assertEqual(tables - {"alembic_version"}, set())
        finally:
            engine.dispose()


class OwnershipTests(DbCase):
    async def test_conversation_is_visible_only_to_its_owner(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice, "mine")
        async with self.sessions() as s:
            repo = ConversationRepository(s)
            self.assertIsNotNone(await repo.get_owned(self.alice, conv.id))
            self.assertIsNone(await repo.get_owned(self.bob, conv.id))
            self.assertEqual(await repo.list_owned(self.bob), [])
            self.assertEqual([c.id for c in await repo.list_owned(self.alice)], [conv.id])

    async def test_delete_by_non_owner_is_refused_and_row_survives(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
        async with self.sessions() as s, s.begin():
            self.assertFalse(await ConversationRepository(s).delete_owned(self.bob, conv.id))
        async with self.sessions() as s:
            self.assertIsNotNone(await ConversationRepository(s).get_owned(self.alice, conv.id))

    async def test_conversation_requires_existing_user(self):
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(Conversation(user_id=uuid.uuid4()))

    async def test_messages_are_reached_only_through_an_owned_conversation(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
            await MessageRepository(s).add(conv, "user", "hi")
            conv_id = conv.id
        async with self.sessions() as s:
            # Bob cannot obtain the Conversation object, hence no messages.
            self.assertIsNone(await ConversationRepository(s).get_owned(self.bob, conv_id))
            owned = await ConversationRepository(s).get_owned(self.alice, conv_id)
            msgs = await MessageRepository(s).list_for(owned)
            self.assertEqual([m.content for m in msgs], ["hi"])

    async def test_message_requires_existing_conversation(self):
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(Message(conversation_id=uuid.uuid4(), position=1, role="user", content="x"))

    async def test_memory_ownership(self):
        async with self.sessions() as s, s.begin():
            mem = await MemoryRepository(s).create(self.alice, "likes tea", "preference")
        async with self.sessions() as s, s.begin():
            repo = MemoryRepository(s)
            self.assertEqual(mem.status, "pending")  # user-controlled: not active by default
            self.assertIsNone(await repo.get_owned(self.bob, mem.id))
            self.assertEqual(await repo.list_owned(self.bob), [])
            self.assertFalse(await repo.delete_owned(self.bob, mem.id))
            self.assertEqual(len(await repo.list_owned(self.alice)), 1)
            self.assertTrue(await repo.delete_owned(self.alice, mem.id))

    async def test_memory_requires_existing_user(self):
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(Memory(user_id=uuid.uuid4(), content="x"))


class ConstraintTests(DbCase):
    async def _conv(self):
        async with self.sessions() as s, s.begin():
            return (await ConversationRepository(s).create(self.alice)).id

    async def test_system_and_tool_roles_cannot_be_stored(self):
        conv_id = await self._conv()
        for role in ("system", "tool", "developer", ""):
            with self.subTest(role=role), self.assertRaises(IntegrityError):
                async with self.sessions() as s, s.begin():
                    s.add(Message(conversation_id=conv_id, position=1, role=role, content="x"))

    async def test_empty_content_and_bad_position_rejected(self):
        conv_id = await self._conv()
        for kwargs in ({"content": ""}, {"position": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(IntegrityError):
                async with self.sessions() as s, s.begin():
                    base = dict(conversation_id=conv_id, position=1, role="user", content="x")
                    s.add(Message(**{**base, **kwargs}))

    async def test_message_position_is_unique_and_ordered(self):
        conv_id = await self._conv()
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).get_owned(self.alice, conv_id)
            repo = MessageRepository(s)
            for text in ("a", "b", "c"):
                await repo.add(conv, "user", text)
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(Message(conversation_id=conv_id, position=2, role="user", content="dup"))
        async with self.sessions() as s:
            conv = await ConversationRepository(s).get_owned(self.alice, conv_id)
            repo = MessageRepository(s)
            self.assertEqual([m.position for m in await repo.list_for(conv)], [1, 2, 3])
            self.assertEqual([m.content for m in await repo.recent_for(conv, 2)], ["b", "c"])

    async def test_memory_and_tool_status_constraints(self):
        conv_id = await self._conv()
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(Memory(user_id=self.alice, content="x", status="deleted"))
        with self.assertRaises(IntegrityError):
            async with self.sessions() as s, s.begin():
                s.add(ToolExecution(conversation_id=conv_id, tool_name="t", status="bogus"))


class RelationshipTests(DbCase):
    async def test_cascade_delete_from_user(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
            msg = await MessageRepository(s).add(conv, "user", "hi")
            await ToolExecutionRepository(s).add(conv, "research.x", "succeeded", msg)
            await MemoryRepository(s).create(self.alice, "m")
        async with self.sessions() as s, s.begin():
            await s.delete(await s.get(User, self.alice))
        async with self.sessions() as s:
            for model in (Conversation, Message, ToolExecution, Memory):
                self.assertEqual((await s.execute(select(model))).scalars().all(), [], model)
            self.assertIsNotNone(await s.get(User, self.bob))

    async def test_deleting_conversation_removes_messages_and_tool_rows(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
            msg = await MessageRepository(s).add(conv, "assistant", "a")
            await ToolExecutionRepository(s).add(conv, "research.x", "failed", msg)
            conv_id = conv.id
        async with self.sessions() as s, s.begin():
            self.assertTrue(await ConversationRepository(s).delete_owned(self.alice, conv_id))
        async with self.sessions() as s:
            self.assertEqual((await s.execute(select(Message))).scalars().all(), [])
            self.assertEqual((await s.execute(select(ToolExecution))).scalars().all(), [])

    async def test_tool_execution_survives_message_deletion(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
            msg = await MessageRepository(s).add(conv, "assistant", "a")
            row = await ToolExecutionRepository(s).add(conv, "research.x", "succeeded", msg)
            row_id, msg_id = row.id, msg.id
        async with self.sessions() as s, s.begin():
            await s.delete(await s.get(Message, msg_id))
        async with self.sessions() as s:
            kept = await s.get(ToolExecution, row_id)
            self.assertIsNotNone(kept)
            self.assertIsNone(kept.message_id)

    async def test_orm_relationships_navigate(self):
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).create(self.alice)
            await MessageRepository(s).add(conv, "user", "q")
            conv_id = conv.id
        async with self.sessions() as s:
            user = await s.get(User, self.alice)
            await s.refresh(user, ["conversations"])
            self.assertEqual([c.id for c in user.conversations], [conv_id])
            conv = await s.get(Conversation, conv_id)
            await s.refresh(conv, ["messages", "user"])
            self.assertEqual([m.content for m in conv.messages], ["q"])
            self.assertEqual(conv.user.id, self.alice)


if __name__ == "__main__":
    unittest.main()
