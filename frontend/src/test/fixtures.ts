/**
 * Response fixtures shaped exactly like the FastAPI models in
 * backend/app/schemas/conversation.py (ConversationOut, MessageOut,
 * ToolExecutionOut, SendMessageResponse, ErrorResponse).
 */
import type { Conversation, Message, SendMessageResult } from "../api/types";

export const CONV_A = "3f1c1e9a-7d1b-4c3e-9a51-0d6b9f1e2a01";
export const CONV_B = "8a2d4b6c-1e3f-4a5b-8c7d-9e0f1a2b3c02";

export function conversationFixture(over: Partial<Conversation> = {}): Conversation {
  return {
    id: CONV_A,
    title: "New conversation",
    created_at: "2026-11-11T09:00:00Z",
    updated_at: "2026-11-11T09:00:00Z",
    ...over,
  };
}

export function messageFixture(over: Partial<Message> = {}): Message {
  return {
    id: "6b0f8d7e-0000-4000-8000-000000000001",
    role: "user",
    content: "Hello",
    position: 0,
    created_at: "2026-11-11T09:00:00Z",
    status: "completed",
    error_code: null,
    tool_executions: [],
    ...over,
  };
}

export function sendResultFixture(over: Partial<SendMessageResult> = {}): SendMessageResult {
  const tools = [{ tool_name: "research:search_works", status: "succeeded", created_at: "2026-11-11T09:00:05Z" }];
  return {
    user_message: messageFixture({ id: "u-1", position: 0 }),
    assistant_message: messageFixture({
      id: "a-1",
      role: "assistant",
      content: "Here is what I found.",
      position: 1,
      created_at: "2026-11-11T09:00:09Z",
      tool_executions: tools,
    }),
    tool_executions: tools,
    stop_reason: "complete",
    ...over,
  };
}

export function errorFixture(code: string, message: string, details: unknown = null, request_id = "req123") {
  return { error: { code, message, request_id, details } };
}
