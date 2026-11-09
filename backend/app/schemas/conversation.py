"""API request/response models. Requests are strict: unknown fields rejected."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

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


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    position: int
    created_at: datetime


class ToolExecutionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tool_name: str
    status: str


class SendMessageResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
    tool_executions: list[ToolExecutionOut]
    stop_reason: str
