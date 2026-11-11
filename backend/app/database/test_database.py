"""
Offline database tests (SQLite; no PostgreSQL server is used).

PostgreSQL-only migration DDL (RLS, privileges) is checked by rendering the
migration as SQL offline; that proves what is emitted, not that PostgreSQL
accepts it -- see app.database.postgres_rls_integration_test for that.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.database.test_database -v
"""
from __future__ import annotations

import io
import re
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
from app.database.postgres_rls_integration_test import (
    Refused,
    check_target,
    is_disposable_name,
    is_loopback_address,
)
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

    def test_0002_preserves_existing_data_and_round_trips(self):
        """SQLite (batch recreate of messages): existing rows, the tool ->
        message link and every constraint survive; down -> up works."""
        db = TempDb()
        self.addCleanup(db.cleanup)
        cfg = self._config(db.url)
        command.upgrade(cfg, "0001")
        engine = create_sync_engine(sync_database_url(db.url))
        self.addCleanup(engine.dispose)
        ids = {k: uuid.uuid4().hex for k in ("u", "c", "m1", "m2", "t")}
        ts = "2026-01-01 00:00:00"
        with engine.begin() as conn:
            conn.exec_driver_sql("INSERT INTO users (id, created_at, updated_at) "
                                 "VALUES (?, ?, ?)", (ids["u"], ts, ts))
            conn.exec_driver_sql("INSERT INTO conversations (id, user_id, title, created_at, "
                                 "updated_at) VALUES (?, ?, 't', ?, ?)",
                                 (ids["c"], ids["u"], ts, ts))
            for key, pos, role, text in (("m1", 1, "user", "question"),
                                         ("m2", 2, "assistant", "answer")):
                conn.exec_driver_sql("INSERT INTO messages (id, conversation_id, position, role, "
                                     "content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                                     (ids[key], ids["c"], pos, role, text, ts))
            conn.exec_driver_sql("INSERT INTO tool_executions (id, conversation_id, message_id, "
                                 "tool_name, status, created_at) VALUES (?, ?, ?, "
                                 "'github.get_file_contents', 'succeeded', ?)",
                                 (ids["t"], ids["c"], ids["m2"], ts))

        def messages(conn):
            return conn.exec_driver_sql(
                "SELECT role, content, status, error_code FROM messages ORDER BY position").all()

        def tool_link(conn):
            return conn.exec_driver_sql("SELECT message_id FROM tool_executions").scalar()

        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            self.assertEqual(messages(conn), [("user", "question", "completed", None),
                                              ("assistant", "answer", "completed", None)])
            self.assertEqual(tool_link(conn), ids["m2"])
            insp = inspect(conn)
            self.assertEqual(
                {c["name"] for c in insp.get_check_constraints("messages")},
                {"ck_messages_role_allowed", "ck_messages_content_not_empty",
                 "ck_messages_position_positive", "ck_messages_status_allowed",
                 "ck_messages_failed_only_assistant", "ck_messages_error_code_iff_failed"})
            self.assertEqual([u["name"] for u in insp.get_unique_constraints("messages")],
                             ["uq_messages_conversation_id_position"])
            self.assertEqual([i["name"] for i in insp.get_indexes("messages")],
                             ["ix_messages_conversation_id_created_at"])
            self.assertEqual([(f["referred_table"], f["options"].get("ondelete"))
                              for f in insp.get_foreign_keys("messages")],
                             [("conversations", "CASCADE")])
            self.assertEqual(compare_metadata(MigrationContext.configure(conn), Base.metadata), [])
        # Old constraints still enforced after the table copy.
        for sql in ("INSERT INTO messages (id, conversation_id, position, role, content, "
                    "created_at) VALUES ('x', ?, 1, 'user', 'dup', ?)",            # unique
                    "INSERT INTO messages (id, conversation_id, position, role, content, "
                    "created_at) VALUES ('y', ?, 9, 'system', 'z', ?)"):           # role
            with self.subTest(sql=sql[-40:]), self.assertRaises(IntegrityError):
                with engine.begin() as conn:
                    conn.exec_driver_sql(sql, (ids["c"], ts))

        # A failed turn exists; downgrade removes only that marker.
        with engine.begin() as conn:
            conn.exec_driver_sql("INSERT INTO messages (id, conversation_id, position, role, "
                                 "content, status, error_code, created_at) VALUES "
                                 "(?, ?, 3, 'assistant', 'safe', 'failed', 'model_error', ?)",
                                 (uuid.uuid4().hex, ids["c"], ts))
        command.downgrade(cfg, "0001")
        with engine.connect() as conn:
            self.assertEqual(conn.exec_driver_sql(
                "SELECT role, content FROM messages ORDER BY position").all(),
                [("user", "question"), ("assistant", "answer")])
            self.assertEqual(tool_link(conn), ids["m2"])
            self.assertNotIn("status", {c["name"] for c in inspect(conn).get_columns("messages")})
        command.upgrade(cfg, "head")
        with engine.connect() as conn:
            self.assertEqual([r[2] for r in messages(conn)], ["completed", "completed"])
            self.assertEqual(tool_link(conn), ids["m2"])
            self.assertEqual(conn.exec_driver_sql(
                "SELECT version_num FROM alembic_version").scalar(), alembic_head())


# Never connected to: offline (--sql) mode only builds the dialect.
POSTGRES_OFFLINE_URL = "postgresql+psycopg://offline@localhost/offline"
VERSION_TABLE = "alembic_version"
API_ROLES = ("anon", "authenticated")
TABLE_PRIVILEGES = {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"}
RLS_ENABLE_RE = re.compile(r'ALTER TABLE "([^"]+)" ENABLE ROW LEVEL SECURITY')
REVOKE_RE = re.compile(r'REVOKE ALL ON TABLE ([^;]+) FROM "([^"]+)"')
VERIFY_RE = re.compile(
    r"rolname = '([^']+)'\) THEN FOREACH t IN ARRAY ARRAY\[([^\]]+)\] LOOP "
    r"IF has_table_privilege\('([^']+)', quote_ident\(t\), '([^']+)'\) THEN "
    r"RAISE EXCEPTION"
)
QUOTED_NAME_RE = re.compile(r"""["']([^"']+)["']""")


def render_postgres_sql(revisions: str, *, downgrade: bool = False) -> str:
    buf = io.StringIO()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"), output_buffer=buf)
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", POSTGRES_OFFLINE_URL)
    (command.downgrade if downgrade else command.upgrade)(cfg, revisions, sql=True)
    return buf.getvalue()


class PostgresMigrationSqlTests(unittest.TestCase):
    """Render the PostgreSQL migration offline and check its security DDL."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sql = render_postgres_sql("head")
        # Accumulate across every statement: each future migration adds its own.
        cls.revokes: dict[str, set[str]] = {}
        for tables, role in REVOKE_RE.findall(cls.sql):
            cls.revokes.setdefault(role, set()).update(QUOTED_NAME_RE.findall(tables))
        cls.verified: dict[str, set[str]] = {}
        cls.verify_matches = VERIFY_RE.findall(cls.sql)
        for guard_role, tables, checked_role, _privs in cls.verify_matches:
            cls.verified.setdefault(checked_role, set()).update(QUOTED_NAME_RE.findall(tables))

    def test_rls_enabled_on_every_model_table(self):
        # Iterates the models, so a future table without RLS fails here.
        for table in Base.metadata.tables:
            self.assertIn(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY', self.sql)

    def test_rls_enabled_on_alembic_version(self):
        self.assertIn(f'ALTER TABLE "{VERSION_TABLE}" ENABLE ROW LEVEL SECURITY', self.sql)

    def test_rls_set_is_exactly_model_tables_plus_alembic_version(self):
        self.assertEqual(
            set(RLS_ENABLE_RE.findall(self.sql)),
            set(Base.metadata.tables) | {VERSION_TABLE},
        )

    def test_fail_closed_no_policies_and_owner_not_forced(self):
        upper = self.sql.upper()
        self.assertNotIn("CREATE POLICY", upper)
        self.assertNotIn("FORCE ROW LEVEL SECURITY", upper)

    def test_revokes_target_only_api_roles(self):
        # postgres (owner/backend) and service_role must never be revoked.
        self.assertEqual(set(self.revokes), set(API_ROLES))
        for role in ("postgres", "service_role"):
            self.assertNotIn(f'FROM "{role}"', self.sql)
        self.assertNotIn("GRANT ", self.sql.upper())

    def test_api_roles_revoked_on_every_model_table(self):
        for role in API_ROLES:
            for table in Base.metadata.tables:
                self.assertIn(table, self.revokes[role])

    def test_api_roles_revoked_on_alembic_version(self):
        for role in API_ROLES:
            self.assertIn(VERSION_TABLE, self.revokes[role])

    def test_privilege_verification_guarded_per_api_role_only(self):
        self.assertTrue(self.verify_matches, "no privilege-verification block rendered")
        for guard_role, _tables, checked_role, _privs in self.verify_matches:
            # The existence guard and the checked role must be the same role.
            self.assertEqual(guard_role, checked_role)
        # Exactly anon and authenticated: never postgres or service_role.
        self.assertEqual(set(self.verified), set(API_ROLES))

    def test_privilege_verification_covers_every_model_table(self):
        for role in API_ROLES:
            for table in Base.metadata.tables:
                self.assertIn(table, self.verified[role])

    def test_privilege_verification_covers_alembic_version(self):
        for role in API_ROLES:
            self.assertIn(VERSION_TABLE, self.verified[role])

    def test_privilege_verification_checks_every_table_privilege(self):
        for *_rest, privs in self.verify_matches:
            self.assertEqual(set(privs.split(",")), TABLE_PRIVILEGES)

    def test_privilege_verification_runs_after_all_revokes(self):
        last_revoke = max(m.start() for m in REVOKE_RE.finditer(self.sql))
        first_verify = min(m.start() for m in VERIFY_RE.finditer(self.sql))
        self.assertGreater(first_verify, last_revoke)

    def test_security_statements_contain_no_percent_sign(self):
        # '%' is a placeholder for pyformat drivers such as psycopg.
        for line in self.sql.splitlines():
            if "ROW LEVEL SECURITY" in line or "REVOKE" in line or "has_table_privilege" in line:
                self.assertNotIn("%", line)

    def test_revoke_is_guarded_for_plain_postgresql(self):
        # The roles exist only on Supabase; a bare REVOKE would fail elsewhere.
        for role in API_ROLES:
            self.assertIn(
                f"IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}')", self.sql
            )

    def test_downgrade_drops_every_model_table_and_keeps_version_table(self):
        sql = render_postgres_sql("head:base", downgrade=True)
        for table in Base.metadata.tables:
            self.assertRegex(sql, rf'DROP TABLE "?{table}"?;')
        self.assertNotRegex(sql, rf'DROP TABLE "?{VERSION_TABLE}"?')
        self.assertIn(f"DELETE FROM {VERSION_TABLE}", sql)


class PostgresTurnStatusMigrationTests(unittest.TestCase):
    """0002 on PostgreSQL, rendered offline (nothing connects): only ALTERs on
    messages, backfill by DEFAULT, no table recreate, no FK or RLS change,
    and a downgrade that removes failed markers before dropping columns."""

    @classmethod
    def setUpClass(cls) -> None:
        def statements(sql):
            # Drop "-- Running upgrade ..." comment LINES first: Alembic puts
            # them directly before a statement, in the same ';'-chunk.
            code = "\n".join(line for line in sql.splitlines()
                             if not line.lstrip().startswith("--"))
            return [" ".join(s.split()) for s in code.split(";") if s.strip()]

        cls.up = statements(render_postgres_sql("0001:0002"))
        cls.down = statements(render_postgres_sql("0002:0001", downgrade=True))

    def body(self, stmts):
        return [s for s in stmts if s not in ("BEGIN", "COMMIT")
                and not s.startswith("UPDATE alembic_version")]

    def test_upgrade_is_alter_table_messages_only_in_one_transaction(self):
        self.assertEqual((self.up[0], self.up[-1]), ("BEGIN", "COMMIT"))
        for stmt in self.body(self.up):
            self.assertTrue(stmt.startswith("ALTER TABLE messages "), stmt)

    def test_existing_rows_are_backfilled_as_completed(self):
        self.assertIn("ALTER TABLE messages ADD COLUMN status VARCHAR(16) "
                      "DEFAULT 'completed' NOT NULL", self.up)
        self.assertIn("ALTER TABLE messages ADD COLUMN error_code VARCHAR(32)", self.up)

    def test_constraints_match_the_models(self):
        model = {c.name: str(c.sqltext) for c in Message.__table__.constraints
                 if c.__class__.__name__ == "CheckConstraint"}
        for name in ("ck_messages_status_allowed", "ck_messages_failed_only_assistant",
                     "ck_messages_error_code_iff_failed"):
            with self.subTest(name=name):
                self.assertIn(f"ALTER TABLE messages ADD CONSTRAINT {name} "
                              f"CHECK ({model[name]})", self.up)

    def test_no_recreate_no_fk_and_no_rls_change(self):
        sql = " ".join(self.up)
        for forbidden in ("CREATE TABLE", "DROP TABLE", "_alembic_tmp", "tool_executions",
                          "FOREIGN KEY", "ROW LEVEL SECURITY", "REVOKE", "GRANT", "%"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, sql)

    def test_downgrade_deletes_failed_markers_first_and_keeps_tables(self):
        body = self.body(self.down)
        self.assertEqual(body[0], "DELETE FROM messages WHERE status = 'failed'")
        self.assertEqual(body[-2:], ["ALTER TABLE messages DROP COLUMN error_code",
                                     "ALTER TABLE messages DROP COLUMN status"])
        self.assertNotIn("DROP TABLE", " ".join(self.down))


class IntegrationGuardTests(unittest.TestCase):
    """The PostgreSQL integration test's refusal logic (pure; nothing connects)."""

    # Dummy URLs only; none of these hosts is ever contacted.
    OK = "postgresql+psycopg://u@localhost:5432/nexus_rls_test"

    def assert_refused(self, url: str, environ: dict | None = None) -> None:
        with self.assertRaises(Refused):
            check_target(url, environ or {})

    def test_local_disposable_url_is_accepted(self):
        check_target(self.OK, {})
        check_target(self.OK + "?sslmode=disable&connect_timeout=5", {})
        check_target("postgresql+psycopg://u@127.0.0.1/test_nexus", {})
        check_target("postgresql+psycopg://u@[::1]/nexus_test", {})

    def test_redirecting_query_parameters_are_refused(self):
        for param in ("host=remote.example.invalid", "hostaddr=203.0.113.9",
                      "service=prod", "port=6543", "options=-csearch_path"):
            self.assert_refused(f"{self.OK}?{param}")
        self.assert_refused(f"{self.OK}?sslmode=require&host=remote.example.invalid")

    def test_libpq_redirect_environment_variables_are_refused(self):
        for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
            self.assert_refused(self.OK, {name: "x"})

    def test_remote_or_ambiguous_hosts_are_refused(self):
        for host in ("db.example.invalid", "aws-0-x.pooler.supabase.com", "10.0.0.5",
                     "localhost,remote.example.invalid"):
            self.assert_refused(f"postgresql+psycopg://u@{host}/nexus_test")
        self.assert_refused("postgresql+psycopg:///nexus_test")  # no host: libpq defaults
        # An unencoded '@' in a password makes the parsed host bogus -> refused.
        self.assert_refused("postgresql+psycopg://u:p@ss@localhost/nexus_test")

    def test_database_name_must_be_marked_disposable(self):
        for name in ("my_nexus", "latest", "contest", "testdb", "nexus_tests", "postgres"):
            self.assert_refused(f"postgresql+psycopg://u@localhost/{name}")
        for name in ("test_nexus", "nexus_test", "TEST_X", "x_TEST"):
            self.assertTrue(is_disposable_name(name), name)

    def test_server_address_check(self):
        for addr in (None, "127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(is_loopback_address(addr), addr)
        for addr in ("10.0.0.5", "203.0.113.9", "::ffff:203.0.113.9", "2001:db8::1", "", "bogus"):
            self.assertFalse(is_loopback_address(addr), addr)


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

    # Failed-turn tracking (0002).
    async def test_messages_default_to_completed(self):
        conv_id = await self._conv()
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).get_owned(self.alice, conv_id)
            msg = await MessageRepository(s).add(conv, "user", "hi")
        self.assertEqual((msg.status, msg.error_code), ("completed", None))

    async def test_failed_assistant_message_is_accepted(self):
        conv_id = await self._conv()
        async with self.sessions() as s, s.begin():
            conv = await ConversationRepository(s).get_owned(self.alice, conv_id)
            msg = await MessageRepository(s).add(conv, "assistant", "safe text",
                                                 status="failed", error_code="model_error")
        self.assertEqual((msg.role, msg.status, msg.error_code),
                         ("assistant", "failed", "model_error"))

    async def test_inconsistent_turn_status_is_rejected(self):
        conv_id = await self._conv()
        bad = (
            {"role": "user", "status": "failed", "error_code": "model_error"},  # only assistant
            {"role": "assistant", "status": "failed", "error_code": None},     # code required
            {"role": "assistant", "status": "completed", "error_code": "x"},   # code only if failed
            {"role": "assistant", "status": "pending", "error_code": None},    # unknown status
        )
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(IntegrityError):
                async with self.sessions() as s, s.begin():
                    s.add(Message(conversation_id=conv_id, position=1, content="x", **kwargs))

    async def test_tool_executions_grouped_by_message_and_scoped(self):
        async with self.sessions() as s, s.begin():
            repo, tools = ConversationRepository(s), ToolExecutionRepository(s)
            conv, other = await repo.create(self.alice), await repo.create(self.alice)
            m1 = await MessageRepository(s).add(conv, "assistant", "a")
            m2 = await MessageRepository(s).add(conv, "assistant", "b")
            mo = await MessageRepository(s).add(other, "assistant", "c")
            await tools.add(conv, "t.first", "succeeded", m1)
            await tools.add(conv, "t.second", "failed", m1)
            await tools.add(other, "t.other", "succeeded", mo)
            ids = (conv, m1.id, m2.id, mo.id)
        async with self.sessions() as s:
            conv, m1_id, m2_id, mo_id = ids
            conv = await ConversationRepository(s).get_owned(self.alice, conv.id)
            got = await ToolExecutionRepository(s).list_for_messages(conv, [m1_id, m2_id, mo_id])
            self.assertEqual([t.tool_name for t in got[m1_id]], ["t.first", "t.second"])
            self.assertNotIn(m2_id, got)   # message without tools
            self.assertNotIn(mo_id, got)   # another conversation's message is never returned
            self.assertEqual(await ToolExecutionRepository(s).list_for_messages(conv, []), {})


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
