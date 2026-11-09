"""FastAPI dependencies: authentication and service access (no MCP here)."""
from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.core.auth import AuthenticationError, Principal
from app.services.conversation_service import ConversationService


def get_principal(request: Request) -> Principal:
    """Authenticate every request; routes never trust a client-supplied user id."""
    try:
        return request.app.state.auth.authenticate(request.headers.get("authorization"))
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service
