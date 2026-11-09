"""
Application-level owner of the agent stack.

One AgentRuntime exists per application process. It opens the MCP
connections ONCE at startup (never per request), builds the existing
OllamaProvider + AgentOrchestrator on top of the existing
MCPConnectionManager, bounds concurrent runs, and closes everything on
shutdown. It is the only application code that touches MCP.

Why a supervisor task
  The MCP SDK enters anyio cancel scopes while connecting and requires them
  to be exited by the same task. Worse, a failed HTTP connect (verified with
  mcp 2.3.0) leaves the *calling* task with a pending cancellation, so every
  later await in it raises CancelledError. To keep that from ever reaching the
  application lifespan or a request task, the manager lives entirely inside
  one dedicated task: enter -> connect -> signal ready -> wait for stop ->
  exit. Request handlers only call it (cross-task calls are normal).
  GitHub is connected first, so if it fails nothing else is open to clean up.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from app.agent.orchestrator import AgentOrchestrator, AgentResult
from app.agent.providers import ChatProvider, OllamaProvider
from app.core.config import PROJECT_ROOT, Settings
from app.core.errors import StartupError
from app.mcp.connection_manager import MCPConnectionManager
from app.mcp.permissions import ToolPolicy, default_policy

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 15.0


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: ChatProvider | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._policy = policy
        self._manager: MCPConnectionManager | None = None
        self._orchestrator: AgentOrchestrator | None = None
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None
        self._stop_event: asyncio.Event | None = None
        self._slots = asyncio.Semaphore(settings.max_concurrent_runs)
        self.connected_servers: list[str] = []

    @property
    def started(self) -> bool:
        return self._orchestrator is not None

    @property
    def permitted_tool_names(self) -> list[str]:
        return sorted(self._manager.permitted_tools) if self._manager else []

    # ----- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("AgentRuntime is already started.")

        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._supervise(), name="mcp-supervisor")
        try:
            await self._ready  # StartupError if connecting failed
        except BaseException:
            await self._reap(cancel=True)
            raise

        provider = self._provider or OllamaProvider(
            model=self._settings.ollama_model,
            base_url=self._settings.ollama_url,
            timeout=self._settings.ollama_timeout,
        )
        self._orchestrator = AgentOrchestrator(self._manager, provider)
        logger.info(
            "Agent runtime started: servers=%s permitted_tools=%d",
            self.connected_servers,
            len(self._manager.permitted_tools),
        )

    async def _supervise(self) -> None:
        """Owns the MCPConnectionManager for its whole life (single task)."""
        try:
            async with MCPConnectionManager(self._policy or default_policy()) as manager:
                try:
                    await self._connect_servers(manager)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:  # incl. CancelledError leaked by the SDK
                    self._fail(exc)
                    return
                self._manager = manager
                if not self._ready.done():
                    self._ready.set_result(None)
                await self._stop_event.wait()
        except BaseException as exc:  # noqa: BLE001
            if not self._ready.done():
                self._fail(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("MCP shutdown raised %s.", type(exc).__name__)

    def _fail(self, exc: BaseException) -> None:
        self.connected_servers = []
        if self._ready is None or self._ready.done():
            return
        if isinstance(exc, StartupError):
            self._ready.set_exception(exc)
        else:  # type only: never forward text that could echo credentials
            self._ready.set_exception(
                StartupError(f"Agent startup failed ({type(exc).__name__}).")
            )

    async def _connect_servers(self, manager: MCPConnectionManager) -> None:
        s = self._settings

        # GitHub first: a failed HTTP connect must not leave other servers open.
        if s.github_mcp_url and s.github_token:
            try:
                await manager.connect_http_server(
                    "github", s.github_mcp_url, s.github_token, read_only=True
                )
                self.connected_servers.append("github")
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise StartupError(
                    f"Could not connect to the GitHub MCP server ({type(exc).__name__}). "
                    "Check GITHUB_MCP_URL / GITHUB_TOKEN and network access."
                ) from None
        else:
            logger.warning("GitHub MCP is not configured; running without it.")

        research_env = {"CROSSREF_MAILTO": s.crossref_mailto} if s.crossref_mailto else None
        try:
            await manager.connect_stdio_server(
                "research",
                ["-m", "custom_mcp_servers.research_server.server"],
                cwd=PROJECT_ROOT,
                env=research_env,
            )
            self.connected_servers.append("research")
        except Exception as exc:  # noqa: BLE001
            raise StartupError(
                f"Could not start the Research MCP server ({type(exc).__name__})."
            ) from None

    async def _reap(self, *, cancel: bool = False) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        if cancel and not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), STOP_TIMEOUT_SECONDS)
        except BaseException:  # noqa: BLE001 - supervisor already logged/handled
            if not task.done():
                task.cancel()

    async def stop(self) -> None:
        self._orchestrator = None
        if self._stop_event is not None:
            self._stop_event.set()
        await self._reap()
        self._manager = None
        self.connected_servers = []
        self._stop_event = self._ready = None

    # ----- use -------------------------------------------------------------
    async def run(
        self, message: str, history: Sequence[dict[str, Any]]
    ) -> AgentResult:
        if self._orchestrator is None:
            raise RuntimeError("AgentRuntime is not started.")
        async with self._slots:
            return await self._orchestrator.run(message, list(history))
