import type { ToolExecution } from "../../../api/types";
import { parseTimestamp } from "../../../api/decode";
import { formatTime } from "../format";

const STATUS_STYLE: Record<string, string> = {
  succeeded: "border-electric/40 text-electric-glow",
  failed: "border-danger/50 text-danger",
  rejected: "border-champagne-deep/60 text-champagne",
};

/** Genuine tool execution records: name, status and time as stored by the backend. */
export function ToolExecutions({ items }: { items: readonly ToolExecution[] }) {
  if (!items.length) return null;
  return (
    <div className="mt-3 border-t border-navy-2 pt-3">
      <h3 className="text-[11px] font-semibold tracking-[0.12em] text-ink-muted uppercase">
        Tools used ({items.length})
      </h3>
      <ul className="mt-2 flex flex-col gap-1.5" aria-label="Tool executions">
        {items.map((t, i) => (
          <li key={`${t.tool_name}-${t.created_at}-${i}`} className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs">
            <code className="rounded bg-abyss/60 px-1.5 py-0.5 font-mono text-ink break-all">{t.tool_name}</code>
            <span
              className={`rounded-full border px-2 py-px text-[11px] font-medium ${STATUS_STYLE[t.status] ?? "border-navy-2 text-ink-muted"}`}
            >
              {t.status}
            </span>
            <time dateTime={t.created_at} title={parseTimestamp(t.created_at).toISOString()} className="text-ink-muted tabular-nums">
              {formatTime(t.created_at)}
            </time>
          </li>
        ))}
      </ul>
    </div>
  );
}
