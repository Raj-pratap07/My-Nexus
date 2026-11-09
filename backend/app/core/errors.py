"""Application-level exceptions shared across layers."""
from __future__ import annotations


class StartupError(Exception):
    """Configuration or startup failure. Messages must never contain secrets."""


class DomainError(Exception):
    """Base for errors that map to a client-visible HTTP status."""

    status_code = 400
    public_message = "Request failed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)
        self.public_message = message or self.public_message


class NotFoundError(DomainError):
    """Missing OR not owned by the caller (the two are indistinguishable)."""

    status_code = 404
    public_message = "Resource not found."


class ConflictError(DomainError):
    status_code = 409
    public_message = "Conflict."


class CredentialRejectedError(DomainError):
    status_code = 422
    public_message = (
        "The message appears to contain a credential or secret and was "
        "not processed or stored."
    )


class AgentUnavailableError(DomainError):
    status_code = 503
    public_message = "The language model is currently unavailable."
