"""
Argument-level authorization for MCP tools.

``ToolPolicy`` decides *which tool* may run. A guard here decides *with
what arguments*. Guards are plain functions that either return the
normalized arguments to send, or raise ``ArgumentPolicyError``. They are
deliberately independent of JSON Schema validation: a schema says what is
well-formed, a guard says what is authorized.

Guards run inside ``MCPConnectionManager.call_tool`` immediately before
routing, so they apply to every caller, not just the LLM orchestrator.
The normalized result (not the original input) is what gets sent.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping


class ArgumentPolicyError(PermissionError):
    """Arguments are not authorized. Messages never include secrets."""


ArgumentGuard = Callable[[Mapping[str, Any]], dict[str, Any]]

GITHUB_FILE_TOOL = "github.get_file_contents"
ALLOWED_GITHUB_REPOS = frozenset({("Raj-pratap07", "My-Nexus")})

_ALLOWED_KEYS = frozenset({"owner", "repo", "path", "ref", "sha", "fields"})
_MAX_PATH = 512
_REF = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_FIELD_NAMES = frozenset(
    {"type", "name", "path", "size", "sha", "url", "git_url",
     "html_url", "download_url"}
)


def normalize_repo_path(value: Any) -> str:
    """
    Validate a repository-relative path and return it normalized.

    Rejected: non-strings, empty, absolute, backslashes, any '%' (so no
    percent-encoded traversal or malformed encodings), '?' and '#',
    non-ASCII or control characters (look-alike dots/slashes), padded
    segments, the segments '', '.', '..', and any '.git' segment. A single
    trailing '/' (directory) is stripped.
    """
    if not isinstance(value, str) or not value:
        raise ArgumentPolicyError("path must be a non-empty string.")
    if len(value) > _MAX_PATH:
        raise ArgumentPolicyError("path is too long.")
    if value.startswith("/"):
        raise ArgumentPolicyError("path must be relative, not absolute.")
    if "\\" in value:
        raise ArgumentPolicyError("path must not contain backslashes.")
    if "%" in value:
        raise ArgumentPolicyError(
            "path must not contain '%' (percent-encoding is not allowed)."
        )
    # '?' and '#' would change meaning if the downstream client places the
    # path into a request URL unescaped (query/fragment injection).
    if "?" in value or "#" in value:
        raise ArgumentPolicyError("path must not contain '?' or '#'.")
    # ASCII-only allowlist: closes every Unicode look-alike for '.', '/',
    # '\' and '%' (full-width forms, division/fraction slash, ...) without
    # depending on how a downstream layer normalizes them. Non-ASCII
    # filenames are therefore not readable through this tool.
    if not value.isascii() or not value.isprintable():
        raise ArgumentPolicyError(
            "path must be printable ASCII (no control or non-ASCII characters)."
        )

    if value.endswith("/"):
        value = value[:-1]
    segments = value.split("/")
    for seg in segments:
        if seg in ("", ".", "..") or seg != seg.strip():
            raise ArgumentPolicyError(
                "path must not contain empty, '.', '..' or padded segments."
            )
        # Checked on the normalized segments, before anything is sent.
        # Case-insensitive, and tolerant of trailing dots and the 8.3
        # short name, which some systems treat as the same entry.
        folded = seg.casefold().rstrip(".")
        if folded in (".git", "git~1"):
            raise ArgumentPolicyError("path must not contain a '.git' segment.")
    return "/".join(segments)


def github_file_guard(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Authorize ``github.get_file_contents`` for the approved repositories."""
    if not isinstance(arguments, Mapping):
        raise ArgumentPolicyError("arguments must be an object.")

    unexpected = sorted(str(k)[:40] for k in set(arguments) - _ALLOWED_KEYS)
    if unexpected:
        raise ArgumentPolicyError(
            f"unexpected arguments: {unexpected[:5]}. "
            f"Allowed: {sorted(_ALLOWED_KEYS)}."
        )

    owner, repo = arguments.get("owner"), arguments.get("repo")
    if (owner, repo) not in ALLOWED_GITHUB_REPOS:
        allowed = ", ".join(f"{o}/{r}" for o, r in sorted(ALLOWED_GITHUB_REPOS))
        raise ArgumentPolicyError(
            f"repository not authorized. Only {allowed} may be read."
        )

    if "path" not in arguments:
        raise ArgumentPolicyError("path is required (a repository-relative path).")
    normalized: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "path": normalize_repo_path(arguments["path"]),
    }

    if "ref" in arguments:
        ref = arguments["ref"]
        if (
            not isinstance(ref, str)
            or not _REF.match(ref)
            or ".." in ref
            or ref.startswith(("/", "-"))
        ):
            raise ArgumentPolicyError("ref is malformed.")
        normalized["ref"] = ref

    if "sha" in arguments:
        sha = arguments["sha"]
        if not isinstance(sha, str) or not _SHA.match(sha):
            raise ArgumentPolicyError("sha must be 7-64 hex characters.")
        normalized["sha"] = sha

    if "fields" in arguments:
        fields = arguments["fields"]
        if (
            not isinstance(fields, list)
            or len(fields) > len(_FIELD_NAMES)
            or not all(isinstance(f, str) and f in _FIELD_NAMES for f in fields)
        ):
            raise ArgumentPolicyError("fields contains unsupported values.")
        normalized["fields"] = list(fields)

    return normalized


DEFAULT_ARGUMENT_GUARDS: dict[str, ArgumentGuard] = {
    GITHUB_FILE_TOOL: github_file_guard,
}
