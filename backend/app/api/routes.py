"""HTTP routes. Thin: validate input, authenticate, delegate to services."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.deps import get_conversation_service, get_principal
from app.core.auth import Principal
from app.schemas.conversation import (
    ConversationCreate,
    ConversationOut,
    MessageCreate,
    MessageOut,
    SendMessageResponse,
    ToolExecutionOut,
)
from app.services.conversation_service import ConversationService

health_router = APIRouter()
router = APIRouter(prefix="/v1", dependencies=[Depends(get_principal)])


def message_out(message, tool_executions=()) -> MessageOut:
    """A stored message plus the tool executions recorded for it."""
    return MessageOut.model_validate(message).model_copy(update={
        "tool_executions": [ToolExecutionOut.model_validate(t) for t in tool_executions],
    })


@health_router.get("/health")
async def health() -> dict[str, str]:
    """Liveness only; deliberately reveals nothing about internals."""
    return {"status": "ok"}


@router.post(
    "/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    body: ConversationCreate,
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.create_conversation(principal.user_id, body.title)


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.list_conversations(principal.user_id, limit=limit, offset=offset)


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: uuid.UUID,
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.get_conversation(principal.user_id, conversation_id)


@router.delete(
    "/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    await service.delete_conversation(principal.user_id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0, le=100_000),
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    views = await service.list_message_views(
        principal.user_id, conversation_id, limit=limit, offset=offset
    )
    return [message_out(v.message, v.tool_executions) for v in views]


@router.post(
    "/conversations/{conversation_id}/messages", response_model=SendMessageResponse
)
async def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    principal: Principal = Depends(get_principal),
    service: ConversationService = Depends(get_conversation_service),
):
    turn = await service.send_message(principal.user_id, conversation_id, body.content)
    assistant = message_out(turn.assistant_message, turn.tool_executions)
    return SendMessageResponse(
        user_message=message_out(turn.user_message),
        assistant_message=assistant,
        tool_executions=assistant.tool_executions,
        stop_reason=turn.stop_reason,
    )
