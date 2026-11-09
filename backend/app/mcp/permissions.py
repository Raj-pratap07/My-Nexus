from __future__ import annotations

from typing import Iterable, Mapping


class ToolPolicy:
    """
    Default-deny allowlist for MCP tool execution.

    Permissions are kept per server and use fully qualified tool names,
    for example ``github.get_file_contents``. A tool runs only if it is
    explicitly listed under the server that owns its namespace.

    The policy knows nothing about MCP sessions or LLM providers.
    """

    def __init__(
        self,
        allowed: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._allowed: dict[str, frozenset[str]] = {}

        for server, tools in (allowed or {}).items():
            self.allow(server, tools)

    @staticmethod
    def server_of(tool_name: str) -> str:
        """Return the server namespace of a qualified tool name."""
        server, sep, tool = tool_name.partition(".")
        if not sep or not server or not tool:
            raise ValueError(
                f"'{tool_name}' is not a fully qualified tool name "
                "(expected '<server>.<tool>')."
            )
        return server

    def allow(self, server: str, tools: Iterable[str]) -> None:
        """Approve tools for one server, replacing its previous list."""
        if not server or "." in server:
            raise ValueError(f"Invalid server key: '{server}'")

        approved = frozenset(tools)

        for name in approved:
            if self.server_of(name) != server:
                raise ValueError(
                    f"Tool '{name}' cannot be approved for server "
                    f"'{server}': namespace mismatch."
                )

        self._allowed[server] = approved

    def allowed_for(self, server: str) -> frozenset[str]:
        """Return the approved tools for a server (empty if none)."""
        return self._allowed.get(server, frozenset())

    def is_allowed(self, tool_name: str) -> bool:
        try:
            server = self.server_of(tool_name)
        except ValueError:
            return False
        return tool_name in self.allowed_for(server)

    def check(self, tool_name: str) -> None:
        """Raise PermissionError unless the tool is explicitly approved."""
        if not self.is_allowed(tool_name):
            raise PermissionError(
                f"MCP tool '{tool_name}' is not permitted by policy."
            )


def default_policy() -> ToolPolicy:
    """The application's approved tools. Everything else stays denied."""
    return ToolPolicy(
        {
            "github": ["github.get_file_contents"],
            "research": [
                "research.search_openalex_works",
                "research.search_crossref_works",
            ],
        }
    )
