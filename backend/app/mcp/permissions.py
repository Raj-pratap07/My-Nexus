from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.mcp.argument_policy import (
    DEFAULT_ARGUMENT_GUARDS,
    ArgumentGuard,
    ArgumentPolicyError,
)


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
        argument_guards: Mapping[str, ArgumentGuard] | None = None,
        require_guards: Iterable[str] = (),
    ) -> None:
        self._allowed: dict[str, frozenset[str]] = {}
        # Optional per-tool argument authorization (default: none).
        self._guards: dict[str, ArgumentGuard] = {}
        # Servers whose tools are sensitive: a tool there is usable only
        # while an argument guard is registered for it (fail closed).
        self._guard_required: frozenset[str] = frozenset(require_guards)

        for name, guard in (argument_guards or {}).items():
            self.server_of(name)  # must be fully qualified
            self._guards[name] = guard

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
        if tool_name not in self.allowed_for(server):
            return False
        return not self._missing_required_guard(tool_name)

    def _missing_required_guard(self, tool_name: str) -> bool:
        server = self.server_of(tool_name)
        return server in self._guard_required and tool_name not in self._guards

    def check(self, tool_name: str) -> None:
        """Raise PermissionError unless the tool is explicitly approved."""
        try:
            listed = tool_name in self.allowed_for(self.server_of(tool_name))
        except ValueError:
            listed = False
        if listed and self._missing_required_guard(tool_name):
            raise ArgumentPolicyError(
                f"MCP tool '{tool_name}' is blocked: its required argument "
                "guard is not configured."
            )
        if not self.is_allowed(tool_name):
            raise PermissionError(
                f"MCP tool '{tool_name}' is not permitted by policy."
            )

    def check_arguments(
        self, tool_name: str, arguments: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """
        Authorize arguments for a tool and return the normalized arguments
        to send. Tools without a guard pass through unchanged. Raises
        ArgumentPolicyError (a PermissionError) if not authorized.
        """
        guard = self._guards.get(tool_name)
        if guard is None:
            if self.server_of(tool_name) in self._guard_required:
                raise ArgumentPolicyError(
                    f"MCP tool '{tool_name}' is blocked: its required "
                    "argument guard is not configured."
                )
            return dict(arguments or {})
        return guard(arguments or {})


def default_policy() -> ToolPolicy:
    """The application's approved tools. Everything else stays denied."""
    return ToolPolicy(
        {
            "github": ["github.get_file_contents"],
            "research": [
                "research.search_openalex_works",
                "research.search_crossref_works",
            ],
        },
        argument_guards=DEFAULT_ARGUMENT_GUARDS,
        require_guards=("github",),
    )
