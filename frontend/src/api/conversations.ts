/** One function per verified backend endpoint (backend/app/api/routes.py). */
import { request } from "./client";
import {
  decodeConversation,
  decodeConversations,
  decodeMessages,
  decodeSendResult,
} from "./decode";
import { LIMITS, type Message } from "./types";

const enc = encodeURIComponent;

export const api = {
  health: () =>
    request("/health", (v) => {
      if (typeof v !== "object" || v === null || (v as { status?: unknown }).status !== "ok") {
        throw new Error("unexpected health body");
      }
      return true as const;
    }),

  /** POST /v1/conversations -> 201 ConversationOut. No title: the backend titles it on first message. */
  createConversation: () =>
    request("/v1/conversations", decodeConversation, { method: "POST", body: {} }),

  /** GET /v1/conversations?limit&offset -> ConversationOut[], most recently updated first. */
  listConversations: (offset = 0, limit: number = LIMITS.conversationsPage) =>
    request(`/v1/conversations?limit=${limit}&offset=${offset}`, decodeConversations),

  /** GET /v1/conversations/{id} -> ConversationOut */
  getConversation: (id: string) => request(`/v1/conversations/${enc(id)}`, decodeConversation),

  /** DELETE /v1/conversations/{id} -> 204 */
  deleteConversation: (id: string) =>
    request(`/v1/conversations/${enc(id)}`, () => undefined, { method: "DELETE" }),

  /** GET /v1/conversations/{id}/messages?limit&offset -> MessageOut[] ordered by position. */
  listMessagesPage: (id: string, offset: number, limit: number = LIMITS.messagesPage) =>
    request(`/v1/conversations/${enc(id)}/messages?limit=${limit}&offset=${offset}`, decodeMessages),

  /** POST /v1/conversations/{id}/messages {content} -> SendMessageResponse. Blocking; can take minutes. */
  sendMessage: (id: string, content: string) =>
    request(`/v1/conversations/${enc(id)}/messages`, decodeSendResult, {
      method: "POST",
      body: { content },
    }),
};

/** The backend has no total count, so pages are read until a short page arrives. */
export const MAX_MESSAGE_PAGES = 10;

export async function listAllMessages(id: string): Promise<{ messages: Message[]; truncated: boolean }> {
  const messages: Message[] = [];
  for (let page = 0; page < MAX_MESSAGE_PAGES; page++) {
    const batch = await api.listMessagesPage(id, page * LIMITS.messagesPage);
    messages.push(...batch);
    if (batch.length < LIMITS.messagesPage) return { messages, truncated: false };
  }
  return { messages, truncated: true };
}

/** Length as Pydantic counts it (Unicode code points, not UTF-16 units). */
export const codePoints = (s: string) => [...s].length;

/**
 * Client-side check mirroring MessageCreate: 1..8000 characters, not blank,
 * no NUL. Callers send the trimmed text, so the backend's length limit
 * (checked before its own strip) applies to exactly what was validated.
 */
export function validateMessage(content: string): string | null {
  const trimmed = content.trim();
  if (!trimmed) return "Write a message first.";
  if (trimmed.includes("\u0000")) return "The message contains a character that cannot be sent.";
  if (codePoints(trimmed) > LIMITS.messageChars) {
    return `Messages are limited to ${LIMITS.messageChars.toLocaleString("en")} characters.`;
  }
  return null;
}
