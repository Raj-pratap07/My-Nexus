/**
 * Runtime validation of backend responses. A response that does not match
 * the verified schema is reported as a contract error instead of being
 * rendered half-understood.
 */
import type { Conversation, ErrorBody, Message, SendMessageResult, ToolExecution } from "./types";

export class ContractError extends Error {
  constructor(what: string) {
    super(`Unexpected response from the backend (${what}).`);
    this.name = "ContractError";
  }
}

type Obj = Record<string, unknown>;

function obj(v: unknown, what: string): Obj {
  if (typeof v !== "object" || v === null || Array.isArray(v)) throw new ContractError(what);
  return v as Obj;
}
function str(o: Obj, key: string, what: string): string {
  const v = o[key];
  if (typeof v !== "string") throw new ContractError(`${what}.${key}`);
  return v;
}
function date(o: Obj, key: string, what: string): string {
  const v = str(o, key, what);
  if (Number.isNaN(Date.parse(v))) throw new ContractError(`${what}.${key}`);
  return v;
}
function list<T>(v: unknown, what: string, item: (x: unknown) => T): T[] {
  if (!Array.isArray(v)) throw new ContractError(what);
  return v.map(item);
}

export function decodeConversation(v: unknown): Conversation {
  const o = obj(v, "conversation");
  return {
    id: str(o, "id", "conversation"),
    title: str(o, "title", "conversation"),
    created_at: date(o, "created_at", "conversation"),
    updated_at: date(o, "updated_at", "conversation"),
  };
}

export function decodeToolExecution(v: unknown): ToolExecution {
  const o = obj(v, "tool_execution");
  return {
    tool_name: str(o, "tool_name", "tool_execution"),
    status: str(o, "status", "tool_execution"),
    created_at: date(o, "created_at", "tool_execution"),
  };
}

export function decodeMessage(v: unknown): Message {
  const o = obj(v, "message");
  const position = o.position;
  if (typeof position !== "number" || !Number.isInteger(position)) {
    throw new ContractError("message.position");
  }
  const errorCode = o.error_code ?? null;
  if (errorCode !== null && typeof errorCode !== "string") throw new ContractError("message.error_code");
  return {
    id: str(o, "id", "message"),
    role: str(o, "role", "message"),
    content: str(o, "content", "message"),
    position,
    created_at: date(o, "created_at", "message"),
    // MessageOut defaults: status "completed", tool_executions [].
    status: o.status === undefined ? "completed" : str(o, "status", "message"),
    error_code: errorCode,
    tool_executions:
      o.tool_executions === undefined ? [] : list(o.tool_executions, "message.tool_executions", decodeToolExecution),
  };
}

export const decodeConversations = (v: unknown) => list(v, "conversations", decodeConversation);
export const decodeMessages = (v: unknown) => list(v, "messages", decodeMessage);

export function decodeSendResult(v: unknown): SendMessageResult {
  const o = obj(v, "send_result");
  return {
    user_message: decodeMessage(o.user_message),
    assistant_message: decodeMessage(o.assistant_message),
    tool_executions: list(o.tool_executions, "send_result.tool_executions", decodeToolExecution),
    stop_reason: str(o, "stop_reason", "send_result"),
  };
}

/** The error envelope {"error": {...}}, or null if the body is not one. */
export function decodeErrorEnvelope(v: unknown): ErrorBody | null {
  if (typeof v !== "object" || v === null) return null;
  const e = (v as Obj).error;
  if (typeof e !== "object" || e === null) return null;
  const b = e as Obj;
  if (typeof b.code !== "string" || typeof b.message !== "string") return null;
  return {
    code: b.code,
    message: b.message,
    request_id: typeof b.request_id === "string" ? b.request_id : "",
    details: b.details ?? null,
  };
}

/**
 * Parses an ISO timestamp. The backend stores timezone-aware values; a
 * string without an offset is treated as UTC rather than local time.
 */
export function parseTimestamp(iso: string): Date {
  const hasZone = /(Z|[+-]\d{2}:?\d{2})$/i.test(iso);
  return new Date(hasZone ? iso : `${iso}Z`);
}
