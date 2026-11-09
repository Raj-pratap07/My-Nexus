"""
FastAPI application factory and lifecycle.

Startup order: validate settings -> open DB engine and verify schema ->
start the AgentRuntime (opens the MCP connections once) -> serve.
Shutdown reverses it. Startup failures are logged as one clear,
secret-free message and re-raised so the process exits non-zero.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
                        response = JSONResponse(
                            {"detail": "Request body too large."}, status_code=413
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
    app.include_router(health_router)
    app.include_router(router)

    @app.exception_handler(DomainError)
    async def _domain_error(_: Request, exc: DomainError):
        return JSONResponse({"detail": exc.public_message}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # Field locations and messages only; never echo submitted values.
        errors = [
            {"loc": list(e.get("loc", [])), "msg": str(e.get("msg", ""))}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        request_id = uuid.uuid4().hex[:12]
        # Type + id only: no traceback or message that could carry secrets.
        logger.error("Unhandled %s (request_id=%s)", type(exc).__name__, request_id)
        return JSONResponse(
            {"detail": "Internal server error.", "request_id": request_id},
            status_code=500,
        )

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
