import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./client";
import { api, codePoints, listAllMessages, validateMessage } from "./conversations";
import { decodeMessage, decodeSendResult, parseTimestamp } from "./decode";
import { conversationFixture, messageFixture, sendResultFixture } from "../test/fixtures";

function stubFetch(...responses: Array<Response | Error | DOMException>) {
  const fn = vi.fn<typeof fetch>();
  for (const r of responses) {
    if (r instanceof Response) fn.mockResolvedValueOnce(r);
    else fn.mockRejectedValueOnce(r);
  }
  vi.stubGlobal("fetch", fn);
  return fn;
}
const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

afterEach(() => vi.unstubAllGlobals());

describe("decoders follow the Pydantic schemas", () => {
  it("decodes MessageOut, including failed turns and tool executions", () => {
    const failed = messageFixture({
      role: "assistant",
      status: "failed",
      error_code: "model_timeout",
      content: "The language model did not respond in time.",
      tool_executions: [{ tool_name: "github:get_file_contents", status: "rejected", created_at: "2026-11-11T09:00:01Z" }],
    });
    expect(decodeMessage(failed)).toEqual(failed);
  });

  it("applies MessageOut defaults when status/error_code/tool_executions are absent", () => {
    const { status: _s, error_code: _e, tool_executions: _t, ...bare } = messageFixture();
    const m = decodeMessage(bare);
    expect(m.status).toBe("completed");
    expect(m.error_code).toBeNull();
    expect(m.tool_executions).toEqual([]);
  });

  it("decodes SendMessageResponse", () => {
    const body = sendResultFixture();
    expect(decodeSendResult(body)).toEqual(body);
  });

  it.each([
    ["missing id", { ...messageFixture(), id: undefined }],
    ["non-integer position", { ...messageFixture(), position: 1.5 }],
    ["bad timestamp", { ...messageFixture(), created_at: "yesterday" }],
    ["tool_executions not a list", { ...messageFixture(), tool_executions: {} }],
  ])("rejects a message with %s", (_label, body) => {
    expect(() => decodeMessage(body)).toThrow(/Unexpected response/);
  });

  it("treats timestamps without an offset as UTC", () => {
    expect(parseTimestamp("2026-11-11T09:00:00").toISOString()).toBe("2026-11-11T09:00:00.000Z");
    expect(parseTimestamp("2026-11-11T09:00:00+00:00").toISOString()).toBe("2026-11-11T09:00:00.000Z");
    expect(parseTimestamp("2026-11-11T10:00:00+01:00").toISOString()).toBe("2026-11-11T09:00:00.000Z");
  });
});

describe("HTTP client", () => {
  it("sends the client header to the same-origin /api proxy and never an Authorization header", async () => {
    const fetchFn = stubFetch(json(201, conversationFixture()));
    await api.createConversation();
    const [url, init] = fetchFn.mock.calls[0]!;
    expect(url).toBe("/api/v1/conversations");
    const headers = new Headers(init!.headers);
    expect(headers.get("X-Nexus-Client")).toBe("web");
    expect(headers.get("Authorization")).toBeNull();
    expect(init!.method).toBe("POST");
    expect(init!.body).toBe("{}");
    expect(init!.credentials).toBe("same-origin");
  });

  it("maps the backend error envelope onto ApiError", async () => {
    stubFetch(
      json(503, {
        error: {
          code: "agent_unavailable",
          message: "The language model is currently unavailable.",
          request_id: "abc123",
          details: { reason: "model_timeout", user_message_id: "u", assistant_message_id: "a" },
        },
      }),
    );
    const err = await api.sendMessage("c", "hi").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({
      kind: "http",
      status: 503,
      code: "agent_unavailable",
      requestId: "abc123",
      source: "backend",
      details: { reason: "model_timeout" },
    });
  });

  it("marks errors produced by the dev proxy", async () => {
    stubFetch(
      json(503, { error: { code: "proxy_not_configured", message: "m", request_id: "proxy-1", details: { source: "proxy" } } }, { "X-Nexus-Proxy": "1" }),
    );
    await expect(api.listConversations()).rejects.toMatchObject({ code: "proxy_not_configured", source: "proxy" });
  });

  it("handles non-JSON error bodies (TrustedHost plain-text 400)", async () => {
    stubFetch(new Response("Invalid host header", { status: 400, headers: { "X-Request-ID": "r9" } }));
    await expect(api.listConversations()).rejects.toMatchObject({ kind: "http", status: 400, code: "http_400", requestId: "r9" });
  });

  it("reports a network failure as 'network', which does not prove the request was not processed", async () => {
    stubFetch(new TypeError("Failed to fetch"));
    await expect(api.sendMessage("c", "hi")).rejects.toMatchObject({ kind: "network", code: "network_error" });
  });

  it("accepts 204 with no body for DELETE", async () => {
    const fetchFn = stubFetch(new Response(null, { status: 204 }));
    await expect(api.deleteConversation("abc")).resolves.toBeUndefined();
    expect(fetchFn.mock.calls[0]![1]!.method).toBe("DELETE");
  });

  it("reports a malformed success body as a contract error", async () => {
    stubFetch(json(200, [{ id: 1 }]));
    await expect(api.listConversations()).rejects.toMatchObject({ kind: "contract", code: "contract_error" });
  });

  it("propagates aborts unchanged", async () => {
    stubFetch(new DOMException("aborted", "AbortError"));
    await expect(request("/x", () => 1)).rejects.toMatchObject({ name: "AbortError" });
  });

  it("URL-encodes ids and uses the documented pagination parameters", async () => {
    const fetchFn = stubFetch(json(200, []));
    await api.listMessagesPage("a/b", 400);
    expect(fetchFn.mock.calls[0]![0]).toBe("/api/v1/conversations/a%2Fb/messages?limit=200&offset=400");
  });
});

describe("listAllMessages", () => {
  it("reads 200-message pages until a short page arrives", async () => {
    const page = (n: number, start: number) =>
      json(200, Array.from({ length: n }, (_, i) => messageFixture({ id: `m${start + i}`, position: start + i })));
    const fetchFn = stubFetch(page(200, 0), page(3, 200));
    const result = await listAllMessages("c");
    expect(result.messages).toHaveLength(203);
    expect(result.truncated).toBe(false);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });
});

describe("validateMessage mirrors MessageCreate", () => {
  it("rejects blank, NUL and over-long input; counts code points like Python", () => {
    expect(validateMessage("   \n ")).toMatch(/Write a message/);
    expect(validateMessage("a\u0000b")).toMatch(/cannot be sent/);
    expect(validateMessage("x".repeat(8000))).toBeNull();
    expect(validateMessage("x".repeat(8001))).toMatch(/8,000/);
    const emoji = "😀".repeat(8000); // 16000 UTF-16 units, 8000 code points
    expect(codePoints(emoji)).toBe(8000);
    expect(validateMessage(emoji)).toBeNull();
    expect(validateMessage(`  ${"x".repeat(8000)}  `)).toBeNull();
  });
});
