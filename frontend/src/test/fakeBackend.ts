/**
 * In-memory stand-in for the FastAPI backend behind the /api proxy, used by
 * component tests through a stubbed fetch. It follows the verified contract
 * (routes.py, schemas/conversation.py, conversation_service.py):
 *  - POST /v1/conversations creates "New conversation"; the first message
 *    retitles it from the content.
 *  - A turn stores the user message first, then the assistant message.
 *  - agent failures store a failed assistant message and answer 503.
 * Each send can be scripted to reply, fail, hang or lose its connection.
 */
import type { Conversation, Message, ToolExecution } from "../api/types";

export type SendScript =
  | { kind: "reply"; content?: string; tools?: Array<Pick<ToolExecution, "tool_name" | "status">> }
  | { kind: "agent-failure"; code: "model_timeout" | "model_unreachable" | "model_error" }
  | { kind: "internal-error" }
  | { kind: "error"; status: number; code: string; message?: string; stores?: false }
  | { kind: "network"; stores: "user" | "nothing" }
  /** An error produced by the dev proxy (X-Nexus-Proxy), optionally after the backend stored the user message. */
  | { kind: "proxy"; status: number; code: string; storesUser: boolean }
  | { kind: "hang" };

const FAILURE_TEXT: Record<string, string> = {
  model_timeout: "The language model did not respond in time.",
  model_unreachable: "The language model could not be reached.",
  model_error: "The language model returned an error.",
  internal_error: "The request failed because of an internal error.",
};

let seq = 0;
const uuid = () => {
  seq++;
  const hex = seq.toString(16).padStart(12, "0");
  return `00000000-0000-4000-8000-${hex}`;
};

export class FakeBackend {
  conversations: Conversation[] = [];
  messages = new Map<string, Message[]>();
  sends: SendScript[] = [];
  calls: Array<{ method: string; path: string; body?: unknown }> = [];
  /** Resolvers for "hang" sends, so tests can finish them later. */
  hanging: Array<(r: Response) => void> = [];
  /** When set, every request answers with this (e.g. proxy_not_configured). */
  override: (() => Response) | null = null;
  clock = Date.parse("2026-11-11T09:00:00Z");

  private now() {
    this.clock += 1000;
    return new Date(this.clock).toISOString().replace(".000Z", "Z");
  }

  addConversation(title = "New conversation", messages: Array<Partial<Message>> = []): Conversation {
    const at = this.now();
    const c: Conversation = { id: uuid(), title, created_at: at, updated_at: at };
    this.conversations.push(c);
    this.messages.set(
      c.id,
      messages.map((m, i) => ({
        id: uuid(),
        role: "user",
        content: "",
        position: i,
        created_at: this.now(),
        status: "completed",
        error_code: null,
        tool_executions: [],
        ...m,
      })),
    );
    return c;
  }

  storeMessage(convId: string, m: Partial<Message> & Pick<Message, "role" | "content">): Message {
    const list = this.messages.get(convId)!;
    const msg: Message = {
      id: uuid(),
      position: list.length,
      created_at: this.now(),
      status: "completed",
      error_code: null,
      tool_executions: [],
      ...m,
    };
    list.push(msg);
    const c = this.conversations.find((x) => x.id === convId)!;
    c.updated_at = msg.created_at;
    return msg;
  }

  count(method: string, pattern: RegExp) {
    return this.calls.filter((c) => c.method === method && pattern.test(c.path)).length;
  }

  fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(String(input), "http://127.0.0.1:5173");
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    const path = url.pathname.replace(/^\/api/, "");
    this.calls.push({ method, path: path + url.search, body });
    if (new Headers(init?.headers).get("X-Nexus-Client") !== "web") return err(403, "proxy_forbidden", "refused", true);
    if (this.override) return this.override();

    if (path === "/health") return json(200, { status: "ok" });
    if (path === "/v1/conversations" && method === "POST") {
      return json(201, this.addConversation());
    }
    if (path === "/v1/conversations" && method === "GET") {
      const limit = Number(url.searchParams.get("limit") ?? 50);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const sorted = [...this.conversations].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
      return json(200, sorted.slice(offset, offset + limit));
    }
    const m = /^\/v1\/conversations\/([^/]+)(\/messages)?$/.exec(path);
    if (!m) return err(404, "not_found", "Resource not found.");
    const id = m[1]!;
    const conv = this.conversations.find((c) => c.id === id);
    if (!conv) return err(404, "not_found", "Resource not found.");

    if (!m[2]) {
      if (method === "GET") return json(200, conv);
      if (method === "DELETE") {
        this.conversations = this.conversations.filter((c) => c.id !== id);
        this.messages.delete(id);
        return new Response(null, { status: 204 });
      }
      return err(405, "method_not_allowed", "Method not allowed.");
    }
    if (method === "GET") {
      const limit = Number(url.searchParams.get("limit") ?? 100);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      return json(200, this.messages.get(id)!.slice(offset, offset + limit));
    }
    return this.send(conv, String(body?.content ?? ""));
  };

  private async send(conv: Conversation, content: string): Promise<Response> {
    const script = this.sends.shift() ?? { kind: "reply" };
    if (script.kind === "error") return err(script.status, script.code, script.message ?? script.code);
    if (script.kind === "network" && script.stores === "nothing") throw new TypeError("Failed to fetch");
    if (script.kind === "proxy" && !script.storesUser) return err(script.status, script.code, script.code, true);

    const list = this.messages.get(conv.id)!;
    if (conv.title === "New conversation" && list.length === 0) conv.title = content.slice(0, 80);
    const user = this.storeMessage(conv.id, { role: "user", content });

    switch (script.kind) {
      case "network":
        throw new TypeError("Failed to fetch");
      case "proxy":
        return err(script.status, script.code, script.code, true);
      case "hang":
        return new Promise<Response>((resolve) => this.hanging.push(resolve));
      case "agent-failure": {
        const failed = this.storeMessage(conv.id, {
          role: "assistant",
          content: FAILURE_TEXT[script.code]!,
          status: "failed",
          error_code: script.code,
        });
        return err(503, "agent_unavailable", "The language model is currently unavailable.", false, {
          reason: script.code,
          user_message_id: user.id,
          assistant_message_id: failed.id,
        });
      }
      case "internal-error":
        this.storeMessage(conv.id, {
          role: "assistant",
          content: FAILURE_TEXT.internal_error!,
          status: "failed",
          error_code: "internal_error",
        });
        return err(500, "internal_error", "Internal server error.");
      case "reply": {
        const at = this.now();
        const tools = (script.tools ?? []).map((t) => ({ ...t, created_at: at }));
        const assistant = this.storeMessage(conv.id, {
          role: "assistant",
          content: script.content ?? `Answer to: ${content}`,
          tool_executions: tools,
        });
        return json(200, { user_message: user, assistant_message: assistant, tool_executions: tools, stop_reason: "complete" });
      }
    }
  }

  /** Completes the oldest hanging send with a stored reply. */
  finishHanging(convId: string, content = "Late answer.") {
    const resolve = this.hanging.shift()!;
    const list = this.messages.get(convId)!;
    const user = list[list.length - 1]!;
    const assistant = this.storeMessage(convId, { role: "assistant", content });
    resolve(json(200, { user_message: user, assistant_message: assistant, tool_executions: [], stop_reason: "complete" }));
  }
}

export function json(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "x-request-id": `req${status}`, ...headers },
  });
}

export function err(status: number, code: string, message: string, proxy = false, details: unknown = null) {
  return json(
    status,
    { error: { code, message, request_id: proxy ? "proxy-test" : `req${status}`, details: proxy ? { source: "proxy" } : details } },
    proxy ? { "X-Nexus-Proxy": "1" } : {},
  );
}
