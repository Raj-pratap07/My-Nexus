import { useEffect, useState } from "react";
import { ApiError } from "../../../api/client";
import { api } from "../../../api/conversations";

type Status = "checking" | "ok" | "proxy" | "down";

/** GET /api/health through the proxy. Liveness only, as the backend defines it. */
export function BackendStatus({ refreshKey }: { refreshKey: unknown }) {
  const [status, setStatus] = useState<Status>("checking");
  useEffect(() => {
    let live = true;
    api.health().then(
      () => live && setStatus("ok"),
      (e: unknown) => {
        if (!live) return;
        const proxyProblem = e instanceof ApiError && e.source === "proxy" && e.code !== "backend_unreachable";
        setStatus(proxyProblem ? "proxy" : "down");
      },
    );
    return () => {
      live = false;
    };
  }, [refreshKey]);

  const label = {
    checking: "Checking backend…",
    ok: "Backend connected",
    proxy: "Proxy not configured",
    down: "Backend unreachable",
  }[status];
  const dot = {
    checking: "bg-ink-muted",
    ok: "bg-electric-glow",
    proxy: "bg-danger",
    down: "bg-danger",
  }[status];
  return (
    <p className="flex items-center gap-2 text-xs text-ink-muted" role="status" data-status={status}>
      <span aria-hidden="true" className={`h-2 w-2 rounded-full ${dot}`} />
      {label}
    </p>
  );
}
