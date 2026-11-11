/** Human text for every error the workspace can meet. Codes come from the backend or the dev proxy. */
import type { ApiError } from "../../api/client";

export interface Described {
  title: string;
  body: string;
}

const REASONS: Record<string, string> = {
  model_timeout: "The language model did not respond in time.",
  model_unreachable: "The language model could not be reached. Is Ollama running?",
  model_error: "The language model returned an error.",
  internal_error: "The backend hit an internal error.",
};

export const FAILURE_LABELS: Record<string, string> = {
  model_timeout: "Timed out",
  model_unreachable: "Model unreachable",
  model_error: "Model error",
  internal_error: "Internal error",
};

export function isSetupError(e: ApiError): boolean {
  if (e.kind !== "http") return false;
  if (e.status === 401) return true;
  return e.source === "proxy" && ["proxy_not_configured", "proxy_forbidden", "backend_unreachable"].includes(e.code);
}

export function describeError(e: ApiError): Described {
  if (e.kind === "network") {
    return {
      title: "Connection lost",
      body: "The browser did not receive an answer. The backend may still have processed the request.",
    };
  }
  if (e.kind === "contract") {
    return { title: "Unexpected response", body: e.message };
  }
  switch (e.code) {
    case "unauthenticated":
      return {
        title: "The backend rejected the proxy's credential",
        body: "The token the dev server was started with does not match the backend's. Restart the dev server with the right token (see frontend/README.md).",
      };
    case "proxy_not_configured":
      return {
        title: "The dev proxy has no backend credential",
        body: "Stop the dev server and start it with `npm run dev:secure`, which asks for the backend token without echoing it. See frontend/README.md.",
      };
    case "proxy_forbidden":
      return { title: "Request refused by the dev proxy", body: "Open the app at the dev server's own address (http://127.0.0.1:5173)." };
    case "backend_unreachable":
      return {
        title: "The backend is not reachable",
        body: "Nothing was sent to it. Start it with `python -m app.serve` in backend/ and try again.",
      };
    case "backend_connection_lost":
      return {
        title: "Connection to the backend lost",
        body: "The connection closed before the backend answered. It may still be processing the message.",
      };
    case "proxy_timeout":
      return {
        title: "The proxy stopped waiting",
        body: "The backend took longer than the proxy's 30-minute limit. It may still be processing.",
      };
    case "not_found":
      return { title: "Not found", body: "This conversation does not exist or was deleted." };
    case "conversation_busy":
      return {
        title: "Still processing an earlier message",
        body: "This conversation already has a message in progress (perhaps from another tab). Your new message was not stored.",
      };
    case "credential_rejected":
      return {
        title: "Message looks like it contains a secret",
        body: "It was not processed or stored. Remove the credential and send it again.",
      };
    case "validation_error":
      return { title: "The backend rejected the message", body: "It was not stored. Check its length and content." };
    case "request_too_large":
      return { title: "Message too large", body: "It was not stored. Shorten it and try again." };
    case "agent_unavailable": {
      const reason = (e.details as { reason?: unknown } | null)?.reason;
      return {
        title: "No answer from the model",
        body: `${typeof reason === "string" && REASONS[reason] ? REASONS[reason] : "The language model is currently unavailable."} The failed turn is recorded below.`,
      };
    }
    case "internal_error":
      return { title: "The backend hit an internal error", body: "See the backend log for this request ID." };
    default:
      return { title: "Request failed", body: e.message || `Status ${e.status}.` };
  }
}
