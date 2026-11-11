/**
 * Minimal History-API router for three routes:
 *   /                 landing page
 *   /nexus            workspace, nothing selected
 *   /nexus/c/:id      workspace with a conversation (id must be a UUID)
 * Vite's SPA fallback serves index.html for these paths in dev and preview.
 */
import { useSyncExternalStore } from "react";

export type Route =
  | { name: "landing" }
  | { name: "workspace"; conversationId: string | null; invalidId?: boolean }
  | { name: "not-found" };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const NAV_EVENT = "nexus:navigate";

export const paths = {
  landing: "/",
  workspace: "/nexus",
  conversation: (id: string) => `/nexus/c/${encodeURIComponent(id)}`,
};

export function parseRoute(pathname: string): Route {
  const p = pathname.replace(/\/+$/, "") || "/";
  if (p === "/") return { name: "landing" };
  if (p === "/nexus") return { name: "workspace", conversationId: null };
  const m = /^\/nexus\/c\/([^/]+)$/.exec(p);
  if (m) {
    const id = decodeURIComponent(m[1]!);
    return UUID.test(id)
      ? { name: "workspace", conversationId: id.toLowerCase() }
      : { name: "workspace", conversationId: null, invalidId: true };
  }
  return { name: "not-found" };
}

export function navigate(path: string, { replace = false } = {}): void {
  if (path === window.location.pathname + window.location.search) return;
  if (replace) window.history.replaceState(null, "", path);
  else window.history.pushState(null, "", path);
  window.dispatchEvent(new Event(NAV_EVENT));
}

function subscribe(onChange: () => void) {
  window.addEventListener("popstate", onChange);
  window.addEventListener(NAV_EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(NAV_EVENT, onChange);
  };
}

const getPath = () => window.location.pathname;

export function usePathname(): string {
  return useSyncExternalStore(subscribe, getPath, () => "/");
}

/** True for a plain left click that the router should handle itself. */
export function isPlainClick(e: { button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean; defaultPrevented: boolean }): boolean {
  return !e.defaultPrevented && e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;
}
