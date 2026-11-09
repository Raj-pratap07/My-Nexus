"""Async engine / session-factory construction and schema checks."""
from __future__ import annotations

import logging
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.errors import StartupError

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]


def create_engine(database_url: str) -> AsyncEngine:
    engine = create_async_engine(database_url, pool_pre_ping=True)

    if engine.dialect.name == "sqlite":
        # SQLite ignores foreign keys unless asked; ownership/cascade
        # guarantees depend on them.
        @event.listens_for(engine.sync_engine, "connect")
        def _enable_fk(dbapi_connection, _record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def alembic_head() -> str:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return ScriptDirectory.from_config(cfg).get_current_head()


async def verify_database(engine: AsyncEngine, *, check_schema: bool = True) -> None:
    """
    Fail fast with a clear, secret-free message if the database is
    unreachable or its schema is not at the Alembic head.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            if check_schema:
                try:
                    row = (
                        await conn.execute(text("SELECT version_num FROM alembic_version"))
                    ).first()
                except SQLAlchemyError:
                    row = None
    except SQLAlchemyError as exc:
        # Type name only: driver messages can embed connection details.
        raise StartupError(
            f"Cannot connect to the database ({type(exc).__name__}). "
            "Check DATABASE_URL and that the server is running."
        ) from None

    if check_schema:
        head = alembic_head()
        if row is None or row[0] != head:
            raise StartupError(
                "Database schema is not up to date. Run "
                "'alembic upgrade head' from the backend directory."
            )
