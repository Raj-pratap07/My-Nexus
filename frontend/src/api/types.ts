/**
 * TypeScript mirror of the FastAPI response models in
 * backend/app/schemas/conversation.py. Field names and meanings are copied
 * exactly; nothing here is invented. Datetimes arrive as ISO-8601 strings.
 */

/** ConversationOut */
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

/** ToolExecutionOut: name, status and time only (arguments/results are never stored). */
export interface ToolExecution {
  tool_name: string;
  /** "succeeded" | "failed" | "rejected" today; shown verbatim if the backend adds one. */
  status: string;
  created_at: string;
}

export type MessageRole = "user" | "assistant" | (string & {});
export type MessageStatus = "completed" | "failed" | (string & {});

/** MessageOut */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  position: number;
  created_at: string;
  status: MessageStatus;
  /** model_timeout | model_unreachable | model_error | internal_error, for failed turns. */
  error_code: string | null;
  tool_executions: ToolExecution[];
}

/** SendMessageResponse */
export interface SendMessageResult {
  user_message: Message;
  assistant_message: Message;
  /** Same records as assistant_message.tool_executions (kept by the backend for old clients). */
  tool_executions: ToolExecution[];
  /** complete | tool_limit | empty_response. Not persisted by the backend. */
  stop_reason: string;
}

/** ErrorBody inside {"error": ...} */
export interface ErrorBody {
  code: string;
  message: string;
  request_id: string;
  details: unknown;
}

/** Limits copied from the backend (schemas/conversation.py, routes.py, main.py). */
export const LIMITS = {
  messageChars: 8000,
  titleChars: 200,
  conversationsPage: 50,
  conversationsPageMax: 100,
  messagesPage: 200,
  requestBytes: 64 * 1024,
} as const;
