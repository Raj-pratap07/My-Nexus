/**
 * Workspace state. Pure reducer; side effects live in useWorkspace.ts.
 *
 * Everything shown comes from the backend except two clearly marked local
 * things: the text of the message currently being sent (shown as
 * "Sending…" until the server returns it) and per-conversation drafts.
 */
import type { ApiError } from "../../../api/client";
import type { Conversation, Message } from "../../../api/types";

/** Draft key for a conversation that does not exist yet (created on first send, D2). */
export const NEW = "new";
export type Key = string;

export interface ConversationsState {
  status: "idle" | "loading" | "ready" | "error";
  items: Conversation[];
  hasMore: boolean;
  loadingMore: boolean;
  error: ApiError | null;
}

export interface Thread {
  status: "loading" | "ready" | "error" | "missing";
  messages: Message[];
  /** More than MAX_MESSAGE_PAGES pages exist; only the first ones are shown. */
  truncated: boolean;
  error: ApiError | null;
}

/** The one send request in flight across the whole workspace (D6). */
export interface ActiveTurn {
  key: Key;
  conversationId: string | null;
  content: string;
  startedAt: number;
}

/**
 * A conversation whose latest turn has an UNKNOWN outcome: the browser lost
 * the response (network error, proxy timeout, disconnect), the backend said
 * it is busy, or the page was opened while the last message is an
 * unanswered user message. Resolved only by reading stored messages.
 */
export interface Awaiting {
  /** Highest message position known before the send; null = "trailing user message" rule. */
  baseline: number | null;
  /** Whether a user message for this turn is stored on the backend. */
  userStored: boolean;
  /** The text, for "Copy message". */
  content: string;
  /** When the turn started (client clock) or, after a reload, when the user message was stored. */
  since: number;
  /** When checking began; bounds the shorter "was it stored at all?" window. */
  checkingSince: number;
  /** Polling stops here and the outcome is reported as unknown. */
  deadline: number;
  phase: "polling" | "expired";
}

export interface Notice {
  error: ApiError;
}

export interface WorkspaceState {
  conversations: ConversationsState;
  threads: Record<string, Thread>;
  turn: ActiveTurn | null;
  awaiting: Record<string, Awaiting>;
  notices: Record<Key, Notice | undefined>;
  drafts: Record<Key, string>;
  /** Proxy/auth problem affecting every request (401, proxy_*, backend_unreachable). */
  setup: ApiError | null;
}

export const initialState: WorkspaceState = {
  conversations: { status: "idle", items: [], hasMore: false, loadingMore: false, error: null },
  threads: {},
  turn: null,
  awaiting: {},
  notices: {},
  drafts: {},
  setup: null,
};

export type Action =
  | { type: "list/loading"; more: boolean }
  | { type: "list/loaded"; items: Conversation[]; append: boolean; hasMore: boolean }
  | { type: "list/failed"; error: ApiError }
  | { type: "conversation/upsert"; conversation: Conversation; toTop: boolean }
  | { type: "conversation/remove"; id: string }
  | { type: "thread/loading"; id: string }
  | { type: "thread/loaded"; id: string; messages: Message[]; truncated: boolean }
  | { type: "thread/failed"; id: string; error: ApiError }
  | { type: "thread/missing"; id: string }
  | { type: "thread/append"; id: string; messages: Message[] }
  | { type: "turn/start"; key: Key; content: string; startedAt: number }
  | { type: "turn/conversation"; id: string }
  | { type: "turn/end" }
  | { type: "draft/set"; key: Key; text: string }
  | { type: "notice/set"; key: Key; notice: Notice | null }
  | { type: "awaiting/set"; id: string; awaiting: Awaiting | null }
  | { type: "setup/set"; error: ApiError | null };

const without = <T>(rec: Record<string, T>, key: string): Record<string, T> => {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
};

export function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case "list/loading":
      return {
        ...state,
        conversations: action.more
          ? { ...state.conversations, loadingMore: true }
          : { ...state.conversations, status: state.conversations.status === "ready" ? "ready" : "loading", error: null },
      };
    case "list/loaded": {
      const items = action.append ? [...state.conversations.items] : [];
      const seen = new Set(items.map((c) => c.id));
      for (const c of action.items) if (!seen.has(c.id)) items.push(c);
      return {
        ...state,
        conversations: { status: "ready", items, hasMore: action.hasMore, loadingMore: false, error: null },
      };
    }
    case "list/failed":
      return {
        ...state,
        conversations: {
          ...state.conversations,
          status: state.conversations.items.length ? "ready" : "error",
          loadingMore: false,
          error: action.error,
        },
      };
    case "conversation/upsert": {
      const rest = state.conversations.items.filter((c) => c.id !== action.conversation.id);
      const existingIndex = state.conversations.items.findIndex((c) => c.id === action.conversation.id);
      let items: Conversation[];
      if (action.toTop) items = [action.conversation, ...rest];
      else if (existingIndex < 0) items = [...rest, action.conversation];
      else {
        items = [...state.conversations.items];
        items[existingIndex] = action.conversation;
      }
      return { ...state, conversations: { ...state.conversations, items } };
    }
    case "conversation/remove":
      return {
        ...state,
        conversations: {
          ...state.conversations,
          items: state.conversations.items.filter((c) => c.id !== action.id),
        },
        threads: without(state.threads, action.id),
        awaiting: without(state.awaiting, action.id),
        drafts: without(state.drafts, action.id),
        notices: without(state.notices, action.id),
      };
    case "thread/loading": {
      const prev = state.threads[action.id];
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.id]: prev && prev.status === "ready"
            ? prev // keep showing messages while refreshing
            : { status: "loading", messages: [], truncated: false, error: null },
        },
      };
    }
    case "thread/loaded":
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.id]: { status: "ready", messages: action.messages, truncated: action.truncated, error: null },
        },
      };
    case "thread/failed": {
      const prev = state.threads[action.id];
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.id]: prev && prev.status === "ready"
            ? { ...prev, error: action.error }
            : { status: "error", messages: [], truncated: false, error: action.error },
        },
      };
    }
    case "thread/missing":
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.id]: { status: "missing", messages: [], truncated: false, error: null },
        },
      };
    case "thread/append": {
      const prev = state.threads[action.id] ?? { status: "ready" as const, messages: [], truncated: false, error: null };
      const byId = new Map(prev.messages.map((m) => [m.id, m]));
      for (const m of action.messages) byId.set(m.id, m);
      const messages = [...byId.values()].sort((a, b) => a.position - b.position);
      return { ...state, threads: { ...state.threads, [action.id]: { ...prev, status: "ready", messages } } };
    }
    case "turn/start":
      return {
        ...state,
        turn: { key: action.key, conversationId: action.key === NEW ? null : action.key, content: action.content, startedAt: action.startedAt },
        drafts: { ...state.drafts, [action.key]: "" },
        notices: { ...state.notices, [action.key]: undefined },
      };
    case "turn/conversation":
      return state.turn
        ? {
            ...state,
            turn: { ...state.turn, key: action.id, conversationId: action.id },
            drafts: { ...without(state.drafts, NEW), [action.id]: "" },
          }
        : state;
    case "turn/end":
      return { ...state, turn: null };
    case "draft/set":
      return { ...state, drafts: { ...state.drafts, [action.key]: action.text } };
    case "notice/set":
      return { ...state, notices: { ...state.notices, [action.key]: action.notice ?? undefined } };
    case "awaiting/set":
      return {
        ...state,
        awaiting: action.awaiting
          ? { ...state.awaiting, [action.id]: action.awaiting }
          : without(state.awaiting, action.id),
      };
    case "setup/set":
      return { ...state, setup: action.error };
  }
}

/** Result of comparing stored messages with what was known before a send. */
export type TurnOutcome =
  | { kind: "replied"; assistant: Message }
  | { kind: "awaiting"; user: Message }
  | { kind: "not-stored" };

/**
 * baseline = highest position known before the send (null: judge only by
 * whether the last stored message is an unanswered user message).
 */
export function analyzeTurn(messages: readonly Message[], baseline: number | null): TurnOutcome {
  const fresh = baseline === null ? messages : messages.filter((m) => m.position > baseline);
  const last = messages[messages.length - 1];
  if (baseline === null) {
    if (last && last.role === "user") return { kind: "awaiting", user: last };
    return last ? { kind: "replied", assistant: last } : { kind: "not-stored" };
  }
  const user = fresh.find((m) => m.role === "user");
  if (!user) return { kind: "not-stored" };
  const assistant = fresh.find((m) => m.role === "assistant" && m.position > user.position);
  if (assistant) return { kind: "replied", assistant };
  return { kind: "awaiting", user };
}

export const maxPosition = (messages: readonly Message[]) =>
  messages.reduce((max, m) => Math.max(max, m.position), -1);

/** The trailing user message with no reply after it, if any. */
export function unansweredTail(messages: readonly Message[]): Message | null {
  const last = messages[messages.length - 1];
  return last && last.role === "user" ? last : null;
}
