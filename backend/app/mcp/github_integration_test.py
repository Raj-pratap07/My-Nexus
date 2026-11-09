"""
Read-only GitHub MCP integration test (needs network and credentials).

Run from backend/:
    .venv/Scripts/python.exe -m app.mcp.github_integration_test
"""
from __future__ import annotations

import asyncio
import os
import sys

from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.permissions import ToolPolicy

TOOL_NAME = "github.get_file_contents"
ARGUMENTS = {
    "owner": "Raj-pratap07",
    "repo": "My-Nexus",
    "path": ".gitignore",
}


def _redact(message: str, secret: str) -> str:
    """Make sure the token can never leak through an error message."""
    return message.replace(secret, "***") if secret else message


def _result_text(result) -> str:
    """Join the text parts of an MCP tool result.

    Text content is shown as-is; embedded resources carry their text in
    `resource.text`.
    """
    parts: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            text = getattr(getattr(item, "resource", None), "text", None)
        parts.append(text if text is not None else f"<{item.type} content>")
    return "\n".join(parts)


def _explain_connection_error(exc: BaseException) -> str:
    """Flatten (possibly nested) exception groups into a hint."""
    leaves: list[BaseException] = []

    def walk(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                walk(sub)
        else:
            leaves.append(e)

    walk(exc)
    text = "; ".join(f"{type(e).__name__}: {e}" for e in leaves)
    lowered = text.lower()
    if "401" in lowered or "403" in lowered or "unauthorized" in lowered:
        return f"Authentication failed (check GITHUB_TOKEN): {text}"
    return f"Connection failed: {text}"


async def run() -> int:
    token = os.getenv("GITHUB_TOKEN", "")
    url = os.getenv("GITHUB_MCP_URL", "")

    if not token or not url:
        print("FAIL: GITHUB_TOKEN or GITHUB_MCP_URL missing from project .env")
        return 1

    try:
        policy = ToolPolicy({"github": [TOOL_NAME]})

        async with MCPConnectionManager(policy) as manager:
            print("Connecting to GitHub MCP...")
            try:
                await manager.connect_http_server(
                    server_key="github", url=url, token=token, read_only=True
                )
            except Exception as exc:  # noqa: BLE001
                print("FAIL:", _redact(_explain_connection_error(exc), token))
                return 1

            print(f"Connected. Discovered {len(manager.tools)} tools.")

            if TOOL_NAME not in manager.tools:
                print(f"FAIL: {TOOL_NAME} not found. Available:")
                for name in sorted(manager.tools):
                    print(f"  {name}")
                return 1
            print(f"Tool available: {TOOL_NAME}")

            print(f"\nCalling {TOOL_NAME} with {ARGUMENTS} ...")
            try:
                result = await manager.call_tool(TOOL_NAME, ARGUMENTS)
            except Exception as exc:  # noqa: BLE001
                print("FAIL: tool call raised:",
                      _redact(f"{type(exc).__name__}: {exc}", token))
                return 1

            text = _result_text(result)
            print("\n--- RESULT ---")
            print(text)
            print("--- END RESULT ---\n")

            if result.is_error:
                print("FAIL: MCP tool reported an error (see result above).")
                return 1

            # The repo's .gitignore protects .env and virtual environments.
            if ".env" not in text:
                print("WARN: result does not look like the expected .gitignore "
                      "(no '.env' entry found).")
                return 1

            print("SUCCESS: tool executed and content matches .gitignore.")
            return 0
    except Exception as exc:  # noqa: BLE001 - cleanup/teardown errors
        print("FAIL: unexpected error:",
              _redact(_explain_connection_error(exc), token))
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
