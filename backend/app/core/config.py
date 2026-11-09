"""
Application settings, read from the environment / project .env.

Secrets are excluded from repr() and never included in error messages.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.core.errors import StartupError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOOPBACK_NAMES = frozenset({"localhost"})
MIN_TOKEN_LENGTH = 32


def is_loopback_host(host: str) -> bool:
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def normalize_database_url(url: str) -> str:
    """Accept plain postgresql:// URLs and select the psycopg 3 driver."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url


def sync_database_url(url: str) -> str:
    """Driver URL for Alembic (synchronous)."""
    url = normalize_database_url(url)
    return url.replace("sqlite+aiosqlite://", "sqlite://", 1)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    dev_token: str = field(repr=False)
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_timeout: float = 180.0
    github_mcp_url: str | None = None
    github_token: str | None = field(default=None, repr=False)
    crossref_mailto: str | None = None
    max_concurrent_runs: int = 1
    history_messages: int = 20
    max_message_chars: int = 8000
    # Skip the "alembic head applied" check (tests that use create_all).
    skip_schema_check: bool = False

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.dev_token, self.github_token) if s)

    def validate(self) -> None:
        if not is_loopback_host(self.host):
            raise StartupError(
                f"Refusing to bind to non-loopback host '{self.host}'. This "
                "build has only a local development identity; it must not "
                "be exposed to a network."
            )
        if len(self.dev_token) < MIN_TOKEN_LENGTH:
            raise StartupError(
                f"NEXUS_DEV_TOKEN must be at least {MIN_TOKEN_LENGTH} characters."
            )
        if not self.database_url:
            raise StartupError("DATABASE_URL is not set.")
        if not (self.database_url.startswith("postgresql+psycopg://")
                or self.database_url.startswith("sqlite+aiosqlite://")):
            raise StartupError(
                "DATABASE_URL must start with postgresql:// or "
                "sqlite+aiosqlite:// (the latter is for tests)."
            )
        if self.max_concurrent_runs < 1 or self.history_messages < 0:
            raise StartupError("Invalid concurrency/history settings.")
        if bool(self.github_mcp_url) != bool(self.github_token):
            raise StartupError(
                "GITHUB_MCP_URL and GITHUB_TOKEN must be set together "
                "(or both left unset to disable the GitHub MCP server)."
            )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ

        def num(name: str, default, cast):
            raw = env.get(name)
            if raw in (None, ""):
                return default
            try:
                return cast(raw)
            except ValueError:
                raise StartupError(f"{name} has an invalid value.") from None

        settings = cls(
            database_url=normalize_database_url(env.get("DATABASE_URL", "")),
            dev_token=env.get("NEXUS_DEV_TOKEN", ""),
            host=env.get("NEXUS_HOST", "127.0.0.1"),
            port=num("NEXUS_PORT", 8000, int),
            ollama_url=env.get("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=env.get("OLLAMA_MODEL", "qwen3:4b-instruct"),
            ollama_timeout=num("OLLAMA_TIMEOUT", 180.0, float),
            github_mcp_url=env.get("GITHUB_MCP_URL") or None,
            github_token=env.get("GITHUB_TOKEN") or None,
            crossref_mailto=env.get("CROSSREF_MAILTO") or None,
            max_concurrent_runs=num("NEXUS_MAX_CONCURRENT_RUNS", 1, int),
        )
        settings.validate()
        return settings
