"""
Offline tests for settings, credential handling and local authentication.

Run from backend/:
    .venv/Scripts/python.exe -m unittest app.core.test_core -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.core.auth import AuthenticationError, LocalTokenAuth, LOCAL_USER_ID
from app.core.config import (
    Settings,
    is_loopback_host,
    normalize_database_url,
    sync_database_url,
)
from app.core.errors import StartupError
from app.core.security import REDACTED, contains_credential, redact

TOKEN = "k" * 40
GOOD_ENV = {
    "DATABASE_URL": "postgresql://nexus:pw@localhost:5432/nexus",
    "NEXUS_DEV_TOKEN": TOKEN,
}


def from_env(env):
    with patch.dict(os.environ, env, clear=True), patch("app.core.config.load_dotenv"):
        return Settings.from_env()


class SettingsTests(unittest.TestCase):
    def test_valid_environment(self):
        s = from_env(GOOD_ENV)
        self.assertEqual(s.database_url, "postgresql+psycopg://nexus:pw@localhost:5432/nexus")
        self.assertEqual(s.host, "127.0.0.1")
        self.assertIsNone(s.github_token)

    def test_secrets_are_not_in_repr_or_errors(self):
        s = from_env({**GOOD_ENV, "GITHUB_TOKEN": "ghp_" + "q" * 36, "GITHUB_MCP_URL": "https://x/mcp"})
        text = repr(s)
        for secret in (TOKEN, "pw@", "ghp_"):
            self.assertNotIn(secret, text)
        with self.assertRaises(StartupError) as ctx:
            from_env({**GOOD_ENV, "NEXUS_DEV_TOKEN": "short-secret-1"})
        self.assertNotIn("short-secret-1", str(ctx.exception))

    def test_missing_or_bad_values_fail_clearly(self):
        cases = [
            ({"NEXUS_DEV_TOKEN": TOKEN}, "DATABASE_URL"),
            ({"DATABASE_URL": GOOD_ENV["DATABASE_URL"]}, "NEXUS_DEV_TOKEN"),
            ({**GOOD_ENV, "DATABASE_URL": "mysql://x"}, "DATABASE_URL"),
            ({**GOOD_ENV, "NEXUS_PORT": "abc"}, "NEXUS_PORT"),
            ({**GOOD_ENV, "GITHUB_TOKEN": "t" * 20}, "GITHUB_MCP_URL"),
            ({**GOOD_ENV, "GITHUB_MCP_URL": "https://x"}, "GITHUB_MCP_URL"),
        ]
        for env, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(StartupError, needle):
                from_env(env)

    def test_only_loopback_hosts_allowed(self):
        for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5"):
            self.assertTrue(is_loopback_host(host), host)
        for host in ("0.0.0.0", "::", "192.168.0.2", "10.0.0.1", "my-nexus.local", ""):
            self.assertFalse(is_loopback_host(host), host)
        with self.assertRaisesRegex(StartupError, "non-loopback"):
            from_env({**GOOD_ENV, "NEXUS_HOST": "0.0.0.0"})

    def test_url_normalization(self):
        self.assertEqual(normalize_database_url("postgres://u@h/d"), "postgresql+psycopg://u@h/d")
        self.assertEqual(sync_database_url("sqlite+aiosqlite:///x.db"), "sqlite:///x.db")


class SecurityTests(unittest.TestCase):
    def test_credential_detection(self):
        for text in (
            "github_pat_" + "A" * 30, "ghp_" + "a" * 36, "Bearer " + "x" * 30,
            "sk-" + "a" * 30, "AKIA" + "A" * 16, "-----BEGIN RSA PRIVATE KEY-----",
            "password: hunter2hunter2", "api_key=abcdefgh12345", "mine is s3cr3t-value!",
        ):
            if text.startswith("mine"):
                self.assertTrue(contains_credential(text, ["s3cr3t-value!"]))
            else:
                self.assertTrue(contains_credential(text), text)

    def test_ordinary_text_is_not_flagged(self):
        for text in ("Find papers on retrieval augmented generation",
                     "What does the token limit of a transformer mean?",
                     "Summarize the .gitignore of my repo", "password policy research"):
            self.assertFalse(contains_credential(text), text)

    def test_redaction(self):
        out = redact("a SECRETVALUE1 b ghp_" + "z" * 36, ["SECRETVALUE1"])
        self.assertEqual(out, f"a {REDACTED} b {REDACTED}")
        self.assertEqual(redact("nothing here", ["", "short"]), "nothing here")


class LocalAuthTests(unittest.TestCase):
    def test_valid_token_maps_to_the_local_user(self):
        auth = LocalTokenAuth(TOKEN)
        self.assertEqual(auth.authenticate(f"Bearer {TOKEN}").user_id, LOCAL_USER_ID)
        self.assertEqual(auth.authenticate(f"bearer {TOKEN}").user_id, LOCAL_USER_ID)

    def test_invalid_credentials_rejected(self):
        auth = LocalTokenAuth(TOKEN)
        for header in (None, "", "Bearer", "Bearer ", f"Basic {TOKEN}", "Bearer wrong",
                       f"Bearer {TOKEN}x", TOKEN):
            with self.subTest(header=header), self.assertRaises(AuthenticationError):
                auth.authenticate(header)

    def test_empty_token_cannot_be_configured(self):
        with self.assertRaises(ValueError):
            LocalTokenAuth("")


if __name__ == "__main__":
    unittest.main()
