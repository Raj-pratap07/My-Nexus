import type { ApiError } from "../../../api/client";
import { describeError } from "../errors";

/** Inline error for one conversation, with the request ID for the logs. */
export function ErrorNotice({ error, onDismiss }: { error: ApiError; onDismiss?: () => void }) {
  const d = describeError(error);
  return (
    <div role="alert" className="rounded-xl border border-danger/40 bg-navy/80 px-4 py-3 text-sm" data-code={error.code}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="font-semibold text-danger">{d.title}</p>
          <p className="mt-1 text-ink-muted">{d.body}</p>
          <p className="mt-1.5 text-xs text-ink-muted">
            {error.status ? `HTTP ${error.status} · ` : ""}
            {error.code}
            {error.requestId ? ` · request ${error.requestId}` : ""}
            {error.source === "proxy" ? " · from the dev proxy" : ""}
          </p>
        </div>
        {onDismiss && (
          <button type="button" onClick={onDismiss} className="rounded-md px-2 py-1 text-xs text-ink-muted hover:text-ink" aria-label="Dismiss">
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

/** Workspace-wide banner for proxy/credential/backend problems. */
export function SetupBanner({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  const d = describeError(error);
  return (
    <div role="alert" className="border-b border-danger/40 bg-navy px-5 py-3 text-sm" data-testid="setup-banner">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-4 gap-y-2">
        <p className="min-w-0 flex-1">
          <span className="font-semibold text-danger">{d.title}.</span> <span className="text-ink-muted">{d.body}</span>
        </p>
        <button
          type="button"
          onClick={onRetry}
          className="rounded-full border border-electric/50 px-3.5 py-1.5 text-xs font-semibold text-electric-glow hover:bg-electric/10"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
