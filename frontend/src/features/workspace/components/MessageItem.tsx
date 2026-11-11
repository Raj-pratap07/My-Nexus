import type { Message } from "../../../api/types";
import { parseTimestamp } from "../../../api/decode";
import { FAILURE_LABELS } from "../errors";
import { formatTime } from "../format";
import { ToolExecutions } from "./ToolExecutions";

/**
 * One stored message. Content is rendered as plain text (D3): no Markdown or
 * HTML interpretation, so model output can never inject markup.
 */
export function MessageItem({
  message,
  onSendAgain,
  sendAgainDisabled,
}: {
  message: Message;
  /** Present only for the latest turn when the backend recorded it as failed. */
  onSendAgain?: () => void;
  sendAgainDisabled?: boolean;
}) {
  const isUser = message.role === "user";
  const failed = message.status === "failed";
  const who = isUser ? "You" : message.role === "assistant" ? "Nexus" : message.role;

  return (
    <li
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      data-role={message.role}
      data-status={message.status}
    >
      <article
        aria-label={`${who}${failed ? ", failed" : ""}`}
        className={[
          "max-w-[min(42rem,100%)] rounded-2xl px-4 py-3",
          isUser
            ? "bg-navy-2 text-ink"
            : failed
              ? "border border-danger/45 bg-navy/80"
              : "border border-navy-2 bg-navy/70",
        ].join(" ")}
      >
        <header className="mb-1.5 flex items-center gap-2 text-xs">
          <span className={`font-semibold ${isUser ? "text-champagne" : "text-electric-glow"}`}>{who}</span>
          {failed && (
            <span className="rounded-full border border-danger/50 px-2 py-px text-[11px] font-medium text-danger">
              Failed{message.error_code ? ` · ${FAILURE_LABELS[message.error_code] ?? message.error_code}` : ""}
            </span>
          )}
          <time dateTime={message.created_at} title={parseTimestamp(message.created_at).toISOString()} className="ml-auto text-ink-muted tabular-nums">
            {formatTime(message.created_at)}
          </time>
        </header>
        <p className={`text-[15px] leading-relaxed break-words whitespace-pre-wrap ${failed ? "text-danger" : ""}`}>
          {message.content}
        </p>
        <ToolExecutions items={message.tool_executions} />
        {failed && onSendAgain && (
          <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-navy-2 pt-3">
            <button
              type="button"
              onClick={onSendAgain}
              disabled={sendAgainDisabled}
              className="rounded-full border border-champagne/60 px-3.5 py-1.5 text-xs font-semibold text-champagne transition-colors hover:bg-champagne/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send again
            </button>
            <span className="text-xs text-ink-muted">Sends your previous message once more. Nothing is retried automatically.</span>
          </div>
        )}
      </article>
    </li>
  );
}
