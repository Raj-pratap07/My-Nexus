/**
 * The only module that talks HTTP. Every request goes to the same-origin
 * development proxy under /api, which adds the backend credential
 * server-side. This file never sets Authorization and never sees a token.
 */
import { ContractError, decodeErrorEnvelope } from "./decode";

export const API_BASE = "/api";
export const CLIENT_HEADER = { "X-Nexus-Client": "web" } as const;

/**
 * Error kinds:
 *  - http: the server answered with an error status (envelope or not).
 *  - network: no answer (connection refused, dropped, browser timeout).
 *    The request MAY still have been processed.
 *  - contract: a success response that does not match the schema.
 */
export type ApiErrorKind = "http" | "network" | "contract";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number;
  readonly code: string;
  readonly requestId: string;
  readonly details: unknown;
  /** "proxy" when the dev proxy produced the error rather than FastAPI. */
  readonly source: "backend" | "proxy" | "client";

  constructor(init: {
    kind: ApiErrorKind;
    status?: number;
    code: string;
    message: string;
    requestId?: string;
    details?: unknown;
    source?: "backend" | "proxy" | "client";
  }) {
    super(init.message);
    this.name = "ApiError";
    this.kind = init.kind;
    this.status = init.status ?? 0;
    this.code = init.code;
    this.requestId = init.requestId ?? "";
    this.details = init.details ?? null;
    this.source = init.source ?? "backend";
  }
}

export interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

async function readJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export async function request<T>(
  path: string,
  decode: (v: unknown) => T,
  options: RequestOptions = {},
): Promise<T> {
  const headers: Record<string, string> = { ...CLIENT_HEADER, Accept: "application/json" };
  let body: string | undefined;
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: options.method ?? "GET",
      headers,
      body,
      signal: options.signal,
      credentials: "same-origin",
      cache: "no-store",
    });
  } catch (err) {
    if ((err as { name?: unknown } | null)?.name === "AbortError") throw err;
    throw new ApiError({
      kind: "network",
      code: "network_error",
      message: "The request did not complete. The server may or may not have processed it.",
      source: "client",
    });
  }

  const requestId = res.headers.get("X-Request-ID") ?? "";
  if (!res.ok) {
    const envelope = decodeErrorEnvelope(await readJson(res));
    const fromProxy = res.headers.get("X-Nexus-Proxy") === "1";
    throw new ApiError({
      kind: "http",
      status: res.status,
      code: envelope?.code ?? `http_${res.status}`,
      message: envelope?.message ?? `The server answered with status ${res.status}.`,
      requestId: envelope?.request_id || requestId,
      details: envelope?.details ?? null,
      source: fromProxy ? "proxy" : "backend",
    });
  }

  const data = res.status === 204 ? undefined : await readJson(res);
  try {
    return decode(data);
  } catch (err) {
    if (err instanceof ContractError) {
      throw new ApiError({ kind: "contract", status: res.status, code: "contract_error", message: err.message, requestId });
    }
    throw err;
  }
}
