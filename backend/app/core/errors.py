"""Application-level exceptions shared across layers.

Every ``DomainError`` maps to one HTTP status and one stable, machine-readable
``code``; ``public_message`` and ``details`` are the only client-visible text
and must never carry exception text, credentials or internals. The API turns
them into the error envelope (see app/main.py).
"""
from __future__ import annotations

from typing import Any


class StartupError(Exception):
    """Configuration or startup failure. Messages must never contain secrets."""


class DomainError(Exception):
    """Base for errors that map to a client-visible HTTP status."""

    status_code = 400
    code = "bad_request"
    public_message = "Request failed."

    def __init__(self, message: str | None = None, *,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.public_message)
        self.public_message = message or self.public_message
        self.details = details


class NotFoundError(DomainError):
    """Missing OR not owned by the caller (the two are indistinguishable)."""

    status_code = 404
    code = "not_found"
    public_message = "Resource not found."


class ConflictError(DomainError):
    status_code = 409
    code = "conflict"
    public_message = "Conflict."


class ConversationBusyError(ConflictError):
    code = "conversation_busy"
    public_message = "This conversation is already processing a message."


class CredentialRejectedError(DomainError):
    status_code = 422
    code = "credential_rejected"
    public_message = (
        "The message appears to contain a credential or secret and was "
        "not processed or stored."
    )


class AgentUnavailableError(DomainError):
    status_code = 503
    code = "agent_unavailable"
    public_message = "The language model is currently unavailable."
