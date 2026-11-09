
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.client.session_group import (
    ClientSessionGroup,
    StdioServerParameters,
    StreamableHttpParameters,
)

from app.mcp.permissions import ToolPolicy



# Find the My-Nexus project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load secrets from the project-root .env file.
load_dotenv(PROJECT_ROOT / ".env")


class MCPConnectionManager:
    """
    Central manager for multiple MCP server connections.

    Uses ClientSessionGroup to aggregate tools and route
    tool calls to the appropriate MCP server.
    """

    def __init__(self, policy: ToolPolicy | None = None) -> None:
        # Default policy is empty, so every tool is denied.
        self._policy = policy or ToolPolicy()
        self._connecting_server_key: str | None = None
        self._sessions: dict[str, Any] = {}
        self._entered = False

        self._group = ClientSessionGroup(
            component_name_hook=self._namespace_component
        )

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Convert a server key into a safe namespace."""
        normalized = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            value,
        ).strip("_").lower()

        if not normalized:
            raise ValueError("Server key cannot be empty.")

        return normalized

    def _namespace_component(
        self,
        component_name: str,
        server_info: Any,
    ) -> str:
        """
        Give every server's tools a unique namespace.

        Example:
            github.search_repositories
            slack.search_messages
        """
        prefix = (
            self._connecting_server_key
            or self._normalize_key(server_info.name)
        )

        return f"{prefix}.{component_name}"

    async def __aenter__(self) -> MCPConnectionManager:
        await self._group.__aenter__()
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self._entered = False
        await self._group.__aexit__(
            exc_type,
            exc_value,
            traceback,
        )

    async def connect_http_server(
        self,
        server_key: str,
        url: str,
        token: str,
        *,
        read_only: bool = True,
    ) -> None:
        """Connect to an MCP server using Streamable HTTP."""

        if not self._entered:
            raise RuntimeError(
                "Use MCPConnectionManager inside an async with block."
            )

        key = self._normalize_key(server_key)

        if key in self._sessions:
            raise ValueError(
                f"MCP server '{key}' is already connected."
            )

        if not token.strip():
            raise ValueError("Authentication token is empty.")

        headers = {
            "Authorization": f"Bearer {token.strip()}",
        }

        if read_only:
            headers["X-MCP-Readonly"] = "true"

        parameters = StreamableHttpParameters(
            url=url,
            headers=headers,
            timeout=30.0,
            sse_read_timeout=300.0,
        )

        # Set the namespace while the SDK discovers server tools.
        self._connecting_server_key = key

        try:
            session = await self._group.connect_to_server(
                parameters
            )
            self._sessions[key] = session
        finally:
            self._connecting_server_key = None

    async def connect_stdio_server(
        self,
        server_key: str,
        args: list[str],
        *,
        command: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """
        Launch a local MCP server as a subprocess and connect over STDIO.

        `command` defaults to the Python interpreter running this process
        (the virtual environment's Python). `env` is passed as the only
        extra environment: the SDK adds just its small safe default set
        (PATH, etc.), so secrets such as GITHUB_TOKEN are not inherited.
        """

        if not self._entered:
            raise RuntimeError(
                "Use MCPConnectionManager inside an async with block."
            )

        key = self._normalize_key(server_key)

        if key in self._sessions:
            raise ValueError(
                f"MCP server '{key}' is already connected."
            )

        parameters = StdioServerParameters(
            command=command or sys.executable,
            args=list(args),
            env=env,
            cwd=cwd,
        )

        self._connecting_server_key = key

        try:
            session = await self._group.connect_to_server(
                parameters
            )
            self._sessions[key] = session
        finally:
            self._connecting_server_key = None

    @property
    def discovered_tools(self) -> dict:
        """Every tool the connected servers advertise (not necessarily usable)."""
        return self._group.tools

    @property
    def tools(self) -> dict:
        """Alias for discovered_tools."""
        return self.discovered_tools

    @property
    def permitted_tools(self) -> dict:
        """Discovered tools that policy allows; the only set an agent may see."""
        return {
            name: tool
            for name, tool in self._group.tools.items()
            if self._policy.is_allowed(name)
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Enforce policy, then route a tool call to its owning MCP server."""

        if name not in self._group.tools:
            raise KeyError(f"Unknown MCP tool: {name}")

        self._policy.check(name)

        # Argument-level authorization; the normalized arguments are what
        # is sent, so the server never sees the unvalidated originals.
        safe_arguments = self._policy.check_arguments(name, arguments)

        return await self._group.call_tool(
            name,
            arguments=safe_arguments,
        )


async def main() -> None:
    """Initial GitHub connection and tool-discovery test."""

    token = os.getenv("GITHUB_TOKEN")
    url = os.getenv("GITHUB_MCP_URL")

    if not token or not url:
        raise RuntimeError(
            "GITHUB_TOKEN or GITHUB_MCP_URL is missing from .env"
        )

    async with MCPConnectionManager() as manager:
        print("Connecting to GitHub MCP...")

        await manager.connect_http_server(
            server_key="github",
            url=url,
            token=token,
            read_only=True,
        )

        print("\nConnection established.")
        print(f"Discovered {len(manager.tools)} tools.\n")

        
        # Display all discovered tools
        for name, tool in sorted(manager.tools.items()):
            description = tool.description or "No description"
            print(f"  {name}")
            print(f"    {description[:180]}\n")

        # Inspect only the tools we intend to use.
        print("\n--- SEARCH REPOSITORIES SCHEMA ---")
        print(
            manager.tools[
                "github.search_repositories"
            ].input_schema
        )

        print("\n--- GET FILE CONTENTS SCHEMA ---")
        print(
            manager.tools[
                "github.get_file_contents"
            ].input_schema
        )



if __name__ == "__main__":
    asyncio.run(main())

