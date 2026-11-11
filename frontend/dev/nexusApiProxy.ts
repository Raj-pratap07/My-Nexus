/**
 * Single-user LOCAL DEVELOPMENT proxy from the browser to the FastAPI backend.
 *
 * This is NOT production authentication. It exists so that the backend's
 * static development bearer token never reaches browser JavaScript:
 *
 *   browser --(same origin, no credentials)--> Vite dev/preview server
 *           --(Authorization: Bearer <token>, added here)--> FastAPI on loopback
 *
 * Guarantees (see nexusApiProxy.test.ts):
 *  - The token is read from the server process environment (NEXUS_DEV_TOKEN)
 *    and lives only in this module's closure. It is never logged, never put
 *    in a response, and never placed in Vite config (so it cannot reach the
 *    client bundle, which only receives VITE_* variables).
 *  - Outgoing headers are an allowlist. Browser-supplied Authorization,
 *    Cookie and every other header are dropped.
 *  - Only exact same-origin requests are accepted: the Host header must be
 *    this loopback server, Origin (when sent) must equal it, Sec-Fetch-Site
 *    (when sent) must be same-origin, and the X-Nexus-Client header must be
 *    present, which a cross-origin page cannot add without a CORS preflight.
 *    Preflights (OPTIONS) are rejected. No CORS headers are ever sent.
 *    The Host check is done here because Vite's own host check runs after
 *    plugin middleware.
 *  - The dev/preview server must bind to loopback, the backend URL must be
 *    loopback, and the connecting peer must be loopback.
 *
 * Residual risk (documented in frontend/README.md): while the server runs,
 * any local non-browser process that can reach it can call the API without
 * knowing the token. The proxy makes the token an ambient credential for
 * the local machine.
 */
import { randomBytes } from "node:crypto";
import http from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Connect, Plugin } from "vite";

export const CLIENT_HEADER = "x-nexus-client";
export const CLIENT_HEADER_VALUE = "web";
export const PROXY_HEADER = "X-Nexus-Proxy";
export const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
/** Above the worst-case agent turn (5 model calls x 180 s default timeout). */
export const UPSTREAM_TIMEOUT_MS = 30 * 60 * 1000;
/** Same limit as the backend's body-size middleware. */
export const MAX_BODY_BYTES = 64 * 1024;

const FORWARDED_REQUEST_HEADERS = ["content-type", "accept"] as const;
const FORWARDED_RESPONSE_HEADERS = ["content-type", "x-request-id", "allow", "www-authenticate"] as const;
const ALLOWED_METHODS = new Set(["GET", "POST", "DELETE"]);
/** Connect-time failures: the request never reached the backend. */
const REFUSED_CODES = new Set(["ECONNREFUSED", "EHOSTUNREACH", "ENETUNREACH", "EADDRNOTAVAIL"]);

export interface ProxyOptions {
  /** Reads the token on each request; defaults to process.env.NEXUS_DEV_TOKEN. */
  readToken?: () => string | undefined;
  /** Defaults to NEXUS_BACKEND_URL or http://127.0.0.1:8000; must be loopback. */
  backendUrl?: string;
  timeoutMs?: number;
  /** Logger for secret-free status lines; defaults to console. */
  log?: (message: string) => void;
}

export function isLoopbackHostname(host: string): boolean {
  const h = host.replace(/^\[|\]$/g, "").toLowerCase();
  return h === "localhost" || h === "::1" || /^127(\.\d{1,3}){3}$/.test(h);
}

function isLoopbackAddress(address: string | undefined): boolean {
  if (!address) return false;
  const a = address.replace(/^::ffff:/, "");
  return a === "::1" || /^127(\.\d{1,3}){3}$/.test(a);
}

/** Vite's `host` option: undefined means "localhost"; true or "0.0.0.0" means all interfaces. */
export function assertLoopbackBind(kind: string, host: string | boolean | undefined): void {
  if (host === undefined) return;
  if (host === true || typeof host !== "string" || !isLoopbackHostname(host)) {
    throw new Error(
      `nexus-api-proxy: refusing to start because ${kind}.host is not a loopback address. ` +
        "The development proxy adds the backend credential to requests and must never be " +
        "reachable from a network. Do not use --host.",
    );
  }
}

export function resolveBackendUrl(raw: string | undefined): URL {
  let url: URL;
  try {
    url = new URL(raw || DEFAULT_BACKEND_URL);
  } catch {
    throw new Error("nexus-api-proxy: NEXUS_BACKEND_URL is not a valid URL.");
  }
  if (url.protocol !== "http:" || !isLoopbackHostname(url.hostname)) {
    throw new Error(
      "nexus-api-proxy: NEXUS_BACKEND_URL must be an http:// URL on a loopback host, " +
        "so the credential is never sent off this machine.",
    );
  }
  return url;
}

/** Maps a browser path under /api to a backend path, or null if not proxied. */
export function upstreamPath(url: string): string | null {
  const [path = "", query] = url.split("?", 2);
  if (/\.\.|%2e|%2f|%5c|\\/i.test(path)) return null;
  let mapped: string | null = null;
  if (path === "/api/health") mapped = "/health";
  else if (path.startsWith("/api/v1/")) mapped = path.slice("/api".length);
  if (mapped === null) return null;
  return query === undefined ? mapped : `${mapped}?${query}`;
}

function sendError(res: ServerResponse, status: number, code: string, message: string,
                   headers: Record<string, string> = {}): void {
  if (res.headersSent) {
    res.destroy();
    return;
  }
  const requestId = `proxy-${randomBytes(5).toString("hex")}`;
  const body = JSON.stringify({
    error: { code, message, request_id: requestId, details: { source: "proxy" } },
  });
  res.writeHead(status, {
    ...headers,
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
    "X-Request-ID": requestId,
    [PROXY_HEADER]: "1",
  });
  res.end(body);
}

/** Exact same-origin check. Returns a reason when the request must be refused. */
export function originProblem(req: IncomingMessage): string | null {
  if (!isLoopbackAddress(req.socket.remoteAddress)) return "peer is not loopback";
  const host = req.headers.host;
  const port = req.socket.localPort;
  if (!host || port === undefined) return "missing host";
  const allowedHosts = ["127.0.0.1", "localhost", "[::1]"].map((h) => `${h}:${port}`);
  if (!allowedHosts.includes(host.toLowerCase())) return "host is not this server";
  const origin = req.headers.origin;
  if (origin !== undefined && origin.toLowerCase() !== `http://${host.toLowerCase()}`) {
    return "cross-origin";
  }
  const site = req.headers["sec-fetch-site"];
  if (site !== undefined && site !== "same-origin") return "cross-site";
  if (req.headers[CLIENT_HEADER] !== CLIENT_HEADER_VALUE) return "missing client header";
  return null;
}

function readBody(req: IncomingMessage): Promise<Buffer | "too-large"> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        resolve("too-large");
        req.removeAllListeners("data");
        req.resume();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

export function createApiMiddleware(options: ProxyOptions = {}): Connect.NextHandleFunction {
  const readToken = options.readToken ?? (() => process.env.NEXUS_DEV_TOKEN);
  const backend = resolveBackendUrl(options.backendUrl ?? process.env.NEXUS_BACKEND_URL);
  const timeoutMs = options.timeoutMs ?? UPSTREAM_TIMEOUT_MS;

  return (req, res, next) => {
    const url = req.url ?? "";
    if (url !== "/api" && !url.startsWith("/api/") && !url.startsWith("/api?")) return next();

    // Never answer CORS preflights: a cross-origin page gets no permission.
    if (req.method === "OPTIONS") {
      return sendError(res, 403, "proxy_forbidden", "Cross-origin requests are not allowed.");
    }
    const problem = originProblem(req);
    if (problem) {
      return sendError(res, 403, "proxy_forbidden", "Request refused by the development proxy.");
    }
    const path = upstreamPath(url);
    if (path === null) return sendError(res, 404, "not_found", "Resource not found.");
    if (!ALLOWED_METHODS.has(req.method ?? "")) {
      return sendError(res, 405, "method_not_allowed", "Method not allowed.", {
        Allow: [...ALLOWED_METHODS].join(", "),
      });
    }
    const token = readToken();
    if (!token) {
      return sendError(res, 503, "proxy_not_configured",
        "The development proxy has no backend credential. Start it with NEXUS_DEV_TOKEN set " +
          "(see frontend/README.md).");
    }

    void readBody(req).then(
      (body) => {
        if (body === "too-large") {
          return sendError(res, 413, "request_too_large", "Request body too large.");
        }
        const headers: Record<string, string | number> = {
          host: backend.host,
          authorization: `Bearer ${token}`,
          "content-length": body.length,
        };
        for (const name of FORWARDED_REQUEST_HEADERS) {
          const value = req.headers[name];
          if (typeof value === "string") headers[name] = value;
        }
        const upstream = http.request(
          {
            protocol: backend.protocol,
            hostname: backend.hostname,
            port: backend.port || 80,
            method: req.method,
            path,
            headers,
            timeout: timeoutMs,
          },
          (upRes) => {
            const out: Record<string, string | string[]> = { "Cache-Control": "no-store" };
            for (const name of FORWARDED_RESPONSE_HEADERS) {
              const value = upRes.headers[name];
              if (value !== undefined) out[name] = value;
            }
            const length = upRes.headers["content-length"];
            if (length !== undefined) out["content-length"] = length;
            res.writeHead(upRes.statusCode ?? 502, out);
            upRes.pipe(res);
          },
        );
        upstream.on("timeout", () => {
          upstream.destroy();
          sendError(res, 504, "proxy_timeout",
            "The backend did not answer within the proxy's time limit. It may still be processing.");
        });
        upstream.on("error", (error: NodeJS.ErrnoException) => {
          // Error details are deliberately not echoed (they can include addresses).
          if (res.headersSent) return;
          if (REFUSED_CODES.has(error.code ?? "")) {
            // No connection was made, so the backend cannot have processed the request.
            sendError(res, 502, "backend_unreachable", "The backend could not be reached.");
          } else {
            // The request may have reached the backend, which may still be processing it.
            sendError(res, 502, "backend_connection_lost",
              "The connection to the backend was lost before it answered. It may still be processing.");
          }
        });
        upstream.end(body);
      },
      () => sendError(res, 400, "bad_request", "The request body could not be read."),
    );
  };
}

/**
 * Vite plugin. Installs the /api middleware first (before Vite's own
 * middleware) on both the dev server and `vite preview`.
 */
export function nexusApiProxy(options: ProxyOptions = {}): Plugin {
  // Silent under Vitest, which starts its own Vite server for the test run.
  const log = options.log ?? ((m: string) => (process.env.VITEST ? undefined : console.info(m)));
  let announce = () => {};
  return {
    name: "nexus-api-proxy",
    configResolved(config) {
      assertLoopbackBind("server", config.server.host);
      assertLoopbackBind("preview", config.preview.host);
      const readToken = options.readToken ?? (() => process.env.NEXUS_DEV_TOKEN);
      announce = () => {
        // Presence only; the value is never printed.
        log(readToken()
          ? "[nexus-api-proxy] Backend credential found in NEXUS_DEV_TOKEN; /api is proxied to the local backend."
          : "[nexus-api-proxy] NEXUS_DEV_TOKEN is not set; /api requests will fail with proxy_not_configured.");
      };
    },
    configureServer(server) {
      announce();
      server.middlewares.use(createApiMiddleware(options));
    },
    configurePreviewServer(server) {
      announce();
      server.middlewares.use(createApiMiddleware(options));
    },
  };
}
