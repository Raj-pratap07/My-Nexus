"""Helpers shared by offline tests (SQLite-backed; no network, no Postgres)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.database.models import Base
from app.database.session import create_engine

TEST_TOKEN = "t" * 40


class TempDb:
    """A throw-away SQLite file database (a file, so it works across event loops)."""

    def __init__(self) -> None:
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self._dir.name) / "test.db"
        self.url = f"sqlite+aiosqlite:///{self.path.as_posix()}"

    def cleanup(self) -> None:
        self._dir.cleanup()


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def make_engine(url: str) -> AsyncEngine:
    engine = create_engine(url)
    await create_schema(engine)
    return engine


def make_settings(db_url: str, **overrides) -> Settings:
    values = dict(
        database_url=db_url,
        dev_token=TEST_TOKEN,
        allowed_hosts=("localhost", "127.0.0.1", "[::1]", "testserver"),
        skip_schema_check=True,
    )
    values.update(overrides)
    return Settings(**values)
