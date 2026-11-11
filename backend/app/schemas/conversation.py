"""API request/response models. Requests are strict: unknown fields rejected."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_MESSAGE_CHARS = 8000

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreate(_Strict):
    title: Title | None = None


class MessageCreate(_Strict):
    # Only user content is accepted. There is deliberately no 'role' field:
    # clients cannot submit system, tool or assistant messages.
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be blank")
        if "\x00" in v:
            raise ValueError("content must not contain NUL characters")
        return v


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ToolExecutionOut(BaseModel):
    """Name, status and time only: arguments and results are never stored."""

    model_config = ConfigDict(from_attributes=True)

    tool_name: str
    status: str            # succeeded | failed | rejected
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    position: int
    created_at: datetime
    # completed | failed. A failed turn is an assistant message whose content
    # is a fixed, safe description and whose error_code says why.
    status: str = "completed"
    error_code: str | None = None
    # Tool executions recorded for this message ([] for user messages).
    tool_executions: list[ToolExecutionOut] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
    # Same executions as assistant_message.tool_executions; kept for
    # compatibility with clients written before per-message history.
    tool_executions: list[ToolExecutionOut]
    stop_reason: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: Any = None


class ErrorResponse(BaseModel):
    """The one JSON error shape of the API: {"error": {...}}."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
