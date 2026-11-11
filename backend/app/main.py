"""
FastAPI application factory and lifecycle.

Startup order: validate settings -> open DB engine and verify schema ->
start the AgentRuntime (opens the MCP connections once) -> serve.
Shutdown reverses it. Startup failures are logged as one clear,
secret-free message and re-raised so the process exits non-zero.

Errors: every JSON error response has ONE shape,
    {"error": {"code": str, "message": str, "request_id": str, "details": ...}}
``details`` is null unless documented (validation errors: [{loc, msg, type}];
agent_unavailable: {reason, user_message_id, assistant_message_id}). Every
response carries an X-Request-ID header with the same id that is logged.
Documented exception: TrustedHostMiddleware answers a bad Host header with
its own plain-text 400 before routing (it still gets X-Request-ID).
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.agent.runtime import AgentRuntime
from app.api.routes import health_router, router
from app.core.auth import AuthProvider, LocalTokenAuth
from app.core.config import Settings
from app.core.errors import DomainError, StartupError
from app.database.session import create_engine, create_session_factory, verify_database
from app.services.conversation_service import ConversationService

logger = logging.getLogger("app.main")

MAX_REQUEST_BYTES = 64 * 1024
REQUEST_ID_HEADER = "X-Request-ID"

# Status -> (code, message) for errors raised by routing and auth.
_HTTP_ERRORS: dict[int, tuple[str, str]] = {
    401: ("unauthenticated", "Authentication required."),
    404: ("not_found", "Resource not found."),
    405: ("method_not_allowed", "Method not allowed."),
}


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _scope_request_id(scope) -> str:
    state = scope.setdefault("state", {})
    if "request_id" not in state:
        state["request_id"] = _new_request_id()
    return state["request_id"]


def error_response(scope, status_code: int, code: str, message: str, *,
                   details: Any = None, headers: dict[str, str] | None = None
                   ) -> JSONResponse:
    """The single error envelope. Only fixed text, never exception text."""
    request_id = _scope_request_id(scope)
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request_id,
                   "details": details}},
        status_code=status_code,
        headers={**(headers or {}), REQUEST_ID_HEADER: request_id},
    )


class _RequestIdMiddleware:
    """Gives every HTTP request an id (scope state) and returns it as
    X-Request-ID on every response that passes through this middleware.
    The 500 handler runs outside all user middleware and sets it itself."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = _scope_request_id(scope)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if REQUEST_ID_HEADER not in headers:
                    headers.append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_id)


class _BodyLimitMiddleware:
    """Reject requests that declare a body larger than MAX_REQUEST_BYTES."""

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name == b"content-length":
                    try:
                        too_big = int(value) > self.max_bytes
                    except ValueError:
                        too_big = True
                    if too_big:
                        response = error_response(
                            scope, 413, "request_too_large", "Request body too large."
                        )
                        return await response(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(
    settings: Settings,
    *,
    auth: AuthProvider | None = None,
    runtime_factory: Callable[[Settings], AgentRuntime] = AgentRuntime,
) -> FastAPI:
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = runtime = None
        try:
            engine = create_engine(settings.database_url)
            await verify_database(engine, check_schema=not settings.skip_schema_check)
            runtime = runtime_factory(settings)
            await runtime.start()
            app.state.engine = engine
            app.state.runtime = runtime
            app.state.conversation_service = ConversationService(
                create_session_factory(engine),
                runtime,
                history_messages=settings.history_messages,
                secrets=settings.secrets,
            )
        except StartupError as exc:
            logger.error("Startup failed: %s", exc)
            await _shutdown(runtime, engine)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Startup failed unexpectedly (%s).", type(exc).__name__)
            await _shutdown(runtime, engine)
            raise StartupError(f"Unexpected startup failure ({type(exc).__name__}).") from None

        try:
            yield
        finally:
            await _shutdown(runtime, engine)

    app = FastAPI(
        title="My-Nexus",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.auth = auth or LocalTokenAuth(settings.dev_token)

    app.add_middleware(_BodyLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(_RequestIdMiddleware)  # last added = outermost
    app.include_router(health_router)
    app.include_router(router)

    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError):
        return error_response(request.scope, exc.status_code, exc.code,
                              exc.public_message, details=exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # Auth (401) and routing (404 unknown path, 405). Fixed messages;
        # headers such as WWW-Authenticate and Allow are preserved.
        code, message = _HTTP_ERRORS.get(exc.status_code, ("http_error", "Request failed."))
        return error_response(request.scope, exc.status_code, code, message,
                              headers=dict(exc.headers or {}))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Field location, message and error type only; never echo submitted values.
        errors = [
            {"loc": list(e.get("loc", [])), "msg": str(e.get("msg", "")),
             "type": str(e.get("type", ""))}
            for e in exc.errors()
        ]
        return error_response(request.scope, 422, "validation_error",
                              "Request validation failed.", details=errors)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        response = error_response(request.scope, 500, "internal_error",
                                  "Internal server error.")
        # Type + id only: no traceback or message that could carry secrets.
        logger.error("Unhandled %s (request_id=%s)", type(exc).__name__,
                     response.headers[REQUEST_ID_HEADER])
        return response

    return app


async def _shutdown(runtime, engine) -> None:
    if runtime is not None:
        try:
            await runtime.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error while stopping agent runtime (%s).", type(exc).__name__)
    if engine is not None:
        await engine.dispose()


def create_app_from_env() -> FastAPI:
    """Factory used by the launcher (uvicorn --factory)."""
    return create_app(Settings.from_env())
