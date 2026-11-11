// @vitest-environment node
/**
 * The proxy is tested against a real local HTTP "backend" stub that records
 * what it receives. No real backend, token or model is involved.
 */
import http from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createServer as createViteServer } from "vite";
import {
  assertLoopbackBind,
  createApiMiddleware,
  nexusApiProxy,
  resolveBackendUrl,
  upstreamPath,
  type ProxyOptions,
} from "./nexusApiProxy";

const TOKEN = "test-token-0123456789-abcdefghijklmnopqrstuvwxyz";

interface Seen {
  method: string;
  url: string;
  headers: http.IncomingHttpHeaders;
  body: string;
}

let seen: Seen[] = [];
let upstreamReply: (req: http.IncomingMessage, res: http.ServerResponse) => void;
let upstream: http.Server;
let upstreamUrl: string;

function listen(server: http.Server): Promise<number> {
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => resolve((server.address() as AddressInfo).port)),
  );
}

beforeAll(async () => {
  upstream = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      seen.push({
        method: req.method ?? "",
        url: req.url ?? "",
        headers: req.headers,
        body: Buffer.concat(chunks).toString(),
      });
      upstreamReply(req, res);
    });
  });
  upstreamUrl = `http://127.0.0.1:${await listen(upstream)}`;
});
afterAll(() => new Promise<void>((r) => upstream.close(() => r())));
afterEach(() => {
  seen = [];
});

function defaultReply(_req: http.IncomingMessage, res: http.ServerResponse) {
  res.writeHead(200, {
    "content-type": "application/json",
    "x-request-id": "abc123",
    "set-cookie": "session=leak",
    "x-internal": "nope",
  });
  res.end(JSON.stringify({ ok: true }));
}

async function startProxy(options: ProxyOptions = {}) {
  const middleware = createApiMiddleware({
    readToken: () => TOKEN,
    backendUrl: upstreamUrl,
    ...options,
  });
  const server = http.createServer((req, res) =>
    middleware(req, res, () => {
      res.statusCode = 299;
      res.end("passed-through");
    }),
  );
  const port = await listen(server);
  return {
    port,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

interface Reply {
  status: number;
  headers: http.IncomingHttpHeaders;
  body: string;
}

function request(port: number, opts: {
  method?: string;
  path: string;
  headers?: Record<string, string>;
  body?: string;
  host?: string;
}): Promise<Reply> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: "127.0.0.1",
        port,
        method: opts.method ?? "GET",
        path: opts.path,
        headers: {
          host: opts.host ?? `127.0.0.1:${port}`,
          ...(opts.headers ?? { "x-nexus-client": "web" }),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () =>
          resolve({ status: res.statusCode ?? 0, headers: res.headers, body: Buffer.concat(chunks).toString() }),
        );
      },
    );
    req.on("error", reject);
    req.end(opts.body);
  });
}

const sameOrigin = (port: number, extra: Record<string, string> = {}) => ({
  "x-nexus-client": "web",
  origin: `http://127.0.0.1:${port}`,
  "sec-fetch-site": "same-origin",
  ...extra,
});

describe("path mapping", () => {
  it("maps only /api/health and /api/v1/*", () => {
    expect(upstreamPath("/api/health")).toBe("/health");
    expect(upstreamPath("/api/v1/conversations?limit=5")).toBe("/v1/conversations?limit=5");
    expect(upstreamPath("/api/v2/x")).toBeNull();
    expect(upstreamPath("/api/v1/../health")).toBeNull();
    expect(upstreamPath("/api/v1/%2e%2e/x")).toBeNull();
    expect(upstreamPath("/api/v1/a%2Fb")).toBeNull();
    expect(upstreamPath("/api")).toBeNull();
  });
});

describe("token handling", () => {
  it("adds the bearer token server-side and drops browser Authorization and Cookie", async () => {
    upstreamReply = defaultReply;
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      method: "POST",
      path: "/api/v1/conversations",
      headers: sameOrigin(proxy.port, {
        authorization: "Bearer attacker-supplied",
        cookie: "session=browser",
        "content-type": "application/json",
        "x-forwarded-for": "1.2.3.4",
      }),
      body: "{}",
    });
    await proxy.close();

    expect(reply.status).toBe(200);
    expect(seen).toHaveLength(1);
    const got = seen[0]!;
    expect(got.method).toBe("POST");
    expect(got.url).toBe("/v1/conversations");
    expect(got.body).toBe("{}");
    expect(got.headers.authorization).toBe(`Bearer ${TOKEN}`);
    expect(got.headers.cookie).toBeUndefined();
    expect(got.headers["x-forwarded-for"]).toBeUndefined();
    expect(got.headers["x-nexus-client"]).toBeUndefined();
    expect(got.headers.origin).toBeUndefined();
    expect(got.headers["content-type"]).toBe("application/json");
    expect(got.headers.host).toBe(new URL(upstreamUrl).host);
  });

  it("never returns the token, Set-Cookie or unlisted upstream headers to the browser", async () => {
    upstreamReply = defaultReply;
    const proxy = await startProxy();
    const reply = await request(proxy.port, { path: "/api/v1/conversations", headers: sameOrigin(proxy.port) });
    await proxy.close();
    expect(reply.headers["x-request-id"]).toBe("abc123");
    expect(reply.headers["set-cookie"]).toBeUndefined();
    expect(reply.headers["x-internal"]).toBeUndefined();
    expect(reply.headers["access-control-allow-origin"]).toBeUndefined();
    expect(JSON.stringify(reply)).not.toContain(TOKEN);
  });

  it("answers proxy_not_configured when no token is set, without contacting the backend", async () => {
    const proxy = await startProxy({ readToken: () => undefined });
    const reply = await request(proxy.port, { path: "/api/v1/conversations", headers: sameOrigin(proxy.port) });
    await proxy.close();
    expect(reply.status).toBe(503);
    expect(reply.headers["x-nexus-proxy"]).toBe("1");
    const body = JSON.parse(reply.body);
    expect(body.error.code).toBe("proxy_not_configured");
    expect(body.error.details).toEqual({ source: "proxy" });
    expect(body.error.request_id).toMatch(/^proxy-/);
    expect(seen).toHaveLength(0);
  });

  it("does not log the token", async () => {
    const spies = (["log", "info", "warn", "error", "debug"] as const).map((m) =>
      vi.spyOn(console, m).mockImplementation(() => {}),
    );
    upstreamReply = defaultReply;
    const lines: string[] = [];
    const plugin = nexusApiProxy({ readToken: () => TOKEN, log: (m) => lines.push(m) });
    (plugin.configResolved as (c: unknown) => void)({ server: { host: "127.0.0.1" }, preview: { host: "127.0.0.1" } });
    (plugin.configureServer as (s: unknown) => void)({ middlewares: { use: () => {} } });
    const proxy = await startProxy();
    await request(proxy.port, { path: "/api/v1/conversations", headers: sameOrigin(proxy.port) });
    await proxy.close();
    const logged = [...lines, ...spies.flatMap((s) => s.mock.calls.flat().map(String))].join("\n");
    expect(lines.join("\n")).toContain("NEXUS_DEV_TOKEN");
    expect(logged).not.toContain(TOKEN);
  });
});

describe("same-origin enforcement", () => {
  it.each([
    ["missing X-Nexus-Client", { origin: "" }],
    ["a cross-origin Origin", { origin: "http://evil.example" }],
    ["an Origin on another port", { origin: "http://127.0.0.1:1" }],
    ["Sec-Fetch-Site cross-site", { "sec-fetch-site": "cross-site" }],
    ["Sec-Fetch-Site same-site", { "sec-fetch-site": "same-site" }],
  ])("rejects %s", async (_label, change) => {
    upstreamReply = defaultReply;
    const proxy = await startProxy();
    const headers: Record<string, string> = sameOrigin(proxy.port);
    for (const [k, v] of Object.entries(change)) headers[k] = v;
    if (_label === "missing X-Nexus-Client") {
      delete headers["x-nexus-client"];
      delete headers.origin;
    }
    const reply = await request(proxy.port, { path: "/api/v1/conversations", headers });
    await proxy.close();
    expect(reply.status).toBe(403);
    expect(JSON.parse(reply.body).error.code).toBe("proxy_forbidden");
    expect(seen).toHaveLength(0);
  });

  it("rejects a foreign Host header (DNS rebinding)", async () => {
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      path: "/api/v1/conversations",
      host: `attacker.example:${proxy.port}`,
      headers: { "x-nexus-client": "web" },
    });
    await proxy.close();
    expect(reply.status).toBe(403);
    expect(seen).toHaveLength(0);
  });

  it("rejects CORS preflights and sends no CORS headers", async () => {
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      method: "OPTIONS",
      path: "/api/v1/conversations",
      headers: {
        origin: "http://evil.example",
        "access-control-request-method": "DELETE",
        "access-control-request-headers": "x-nexus-client",
      },
    });
    await proxy.close();
    expect(reply.status).toBe(403);
    expect(reply.headers["access-control-allow-origin"]).toBeUndefined();
    expect(reply.headers["access-control-allow-headers"]).toBeUndefined();
    expect(seen).toHaveLength(0);
  });

  it("accepts a same-origin GET without Origin (browsers omit it)", async () => {
    upstreamReply = defaultReply;
    const proxy = await startProxy();
    const reply = await request(proxy.port, { path: "/api/health", headers: { "x-nexus-client": "web" } });
    await proxy.close();
    expect(reply.status).toBe(200);
    expect(seen[0]!.url).toBe("/health");
  });
});

describe("routing and limits", () => {
  it("passes non-/api requests to the next middleware untouched", async () => {
    const proxy = await startProxy();
    const reply = await request(proxy.port, { path: "/apiary", headers: {} });
    await proxy.close();
    expect(reply.status).toBe(299);
  });

  it("404s unknown /api paths and 405s other methods without contacting the backend", async () => {
    const proxy = await startProxy();
    const unknown = await request(proxy.port, { path: "/api/admin", headers: sameOrigin(proxy.port) });
    const put = await request(proxy.port, {
      method: "PUT",
      path: "/api/v1/conversations",
      headers: sameOrigin(proxy.port),
    });
    await proxy.close();
    expect(unknown.status).toBe(404);
    expect(put.status).toBe(405);
    expect(seen).toHaveLength(0);
  });

  it("refuses bodies over 64 KB", async () => {
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      method: "POST",
      path: "/api/v1/conversations",
      headers: sameOrigin(proxy.port, { "content-type": "application/json" }),
      body: "x".repeat(64 * 1024 + 1),
    });
    await proxy.close();
    expect(reply.status).toBe(413);
    expect(seen).toHaveLength(0);
  });

  it("relays backend error envelopes and status codes unchanged", async () => {
    upstreamReply = (_req, res) => {
      res.writeHead(409, { "content-type": "application/json", "x-request-id": "r1" });
      res.end(JSON.stringify({ error: { code: "conversation_busy", message: "m", request_id: "r1", details: null } }));
    };
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      method: "POST",
      path: "/api/v1/conversations/x/messages",
      headers: sameOrigin(proxy.port, { "content-type": "application/json" }),
      body: '{"content":"hi"}',
    });
    await proxy.close();
    expect(reply.status).toBe(409);
    expect(reply.headers["x-nexus-proxy"]).toBeUndefined();
    expect(JSON.parse(reply.body).error.code).toBe("conversation_busy");
  });

  it("answers 502 backend_unreachable when the backend is down", async () => {
    const dead = http.createServer();
    const deadPort = await listen(dead);
    await new Promise<void>((r) => dead.close(() => r()));
    const proxy = await startProxy({ backendUrl: `http://127.0.0.1:${deadPort}` });
    const reply = await request(proxy.port, { path: "/api/v1/conversations", headers: sameOrigin(proxy.port) });
    await proxy.close();
    expect(reply.status).toBe(502);
    expect(JSON.parse(reply.body).error.code).toBe("backend_unreachable");
    expect(reply.body).not.toContain(String(deadPort));
  });

  it("answers 502 backend_connection_lost when the backend drops a request it received", async () => {
    upstreamReply = (req) => req.socket.destroy();
    const proxy = await startProxy();
    const reply = await request(proxy.port, {
      method: "POST",
      path: "/api/v1/conversations/x/messages",
      headers: sameOrigin(proxy.port, { "content-type": "application/json" }),
      body: '{"content":"hi"}',
    });
    await proxy.close();
    expect(reply.status).toBe(502);
    expect(JSON.parse(reply.body).error.code).toBe("backend_connection_lost");
    expect(seen).toHaveLength(1);
  });

  it("answers 504 proxy_timeout when the backend is silent past the limit", async () => {
    upstreamReply = () => {}; // never answers
    const proxy = await startProxy({ timeoutMs: 150 });
    const reply = await request(proxy.port, { path: "/api/v1/conversations", headers: sameOrigin(proxy.port) });
    await proxy.close();
    expect(reply.status).toBe(504);
    expect(JSON.parse(reply.body).error.code).toBe("proxy_timeout");
  });
});

describe("configuration safety", () => {
  it("refuses non-loopback binds", () => {
    expect(() => assertLoopbackBind("server", undefined)).not.toThrow();
    expect(() => assertLoopbackBind("server", "127.0.0.1")).not.toThrow();
    expect(() => assertLoopbackBind("server", "localhost")).not.toThrow();
    expect(() => assertLoopbackBind("server", "::1")).not.toThrow();
    expect(() => assertLoopbackBind("server", true)).toThrow(/loopback/);
    expect(() => assertLoopbackBind("server", "0.0.0.0")).toThrow(/loopback/);
    expect(() => assertLoopbackBind("server", "192.168.1.5")).toThrow(/loopback/);
  });

  it("refuses a non-loopback or non-http backend URL", () => {
    expect(resolveBackendUrl(undefined).href).toBe("http://127.0.0.1:8000/");
    expect(() => resolveBackendUrl("http://example.com:8000")).toThrow(/loopback/);
    expect(() => resolveBackendUrl("https://127.0.0.1:8000")).toThrow(/http:\/\//);
    expect(() => resolveBackendUrl("not a url")).toThrow(/valid URL/);
  });

  it("is installed ahead of Vite's own middleware in a real dev server", async () => {
    upstreamReply = defaultReply;
    vi.spyOn(console, "info").mockImplementation(() => {});
    const server = await createViteServer({
      configFile: false,
      logLevel: "silent",
      appType: "spa",
      plugins: [nexusApiProxy({ readToken: () => TOKEN, backendUrl: upstreamUrl })],
      server: { host: "127.0.0.1", port: 0, strictPort: false, ws: false },
    });
    await server.listen();
    const port = (server.httpServer!.address() as AddressInfo).port;
    try {
      const ok = await request(port, { path: "/api/v1/conversations", headers: sameOrigin(port) });
      expect(ok.status).toBe(200);
      expect(seen[0]!.headers.authorization).toBe(`Bearer ${TOKEN}`);
      // Without the client header the SPA fallback must NOT serve index.html for /api.
      const bare = await request(port, { path: "/api/v1/conversations", headers: { accept: "text/html" } });
      expect(bare.status).toBe(403);
    } finally {
      await server.close();
    }
  });

  it("refuses to start a dev server bound to all interfaces", async () => {
    await expect(
      createViteServer({
        configFile: false,
        logLevel: "silent",
        plugins: [nexusApiProxy({ readToken: () => TOKEN })],
        server: { host: true, port: 0, ws: false },
      }),
    ).rejects.toThrow(/loopback/);
  });
});
