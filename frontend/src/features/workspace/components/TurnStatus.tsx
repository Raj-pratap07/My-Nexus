import { useState } from "react";
import type { ActiveTurn, Awaiting } from "../state/reducer";
import { copyText, formatElapsed, useNow } from "../format";
import { TURN_WINDOW_MS } from "../state/useWorkspace";

/** The user's own message while its request is in flight (not yet confirmed stored). */
export function SendingMessage({ content }: { content: string }) {
  return (
    <li className="flex justify-end" data-role="user" data-status="sending">
      <article aria-label="You, sending" className="max-w-[min(42rem,100%)] rounded-2xl border border-dashed border-navy-2 bg-navy-2/60 px-4 py-3">
        <header className="mb-1.5 flex items-center gap-2 text-xs">
          <span className="font-semibold text-champagne">You</span>
          <span className="ml-auto text-ink-muted">Sending…</span>
        </header>
        <p className="text-[15px] leading-relaxed break-words whitespace-pre-wrap">{content}</p>
      </article>
    </li>
  );
}

/** Real elapsed time since the request was sent. No invented progress steps. */
export function PendingReply({ turn }: { turn: ActiveTurn }) {
  const now = useNow(true);
  return (
    <li className="flex justify-start" data-testid="pending-reply">
      <div className="max-w-[min(42rem,100%)] rounded-2xl border border-electric/30 bg-navy/70 px-4 py-3" role="status">
        <div className="flex items-center gap-2.5 text-sm text-ink">
          <span aria-hidden="true" className="h-2 w-2 animate-pulse rounded-full bg-electric-glow" />
          Waiting for the model
          <span className="font-mono text-ink-muted tabular-nums" aria-label="Elapsed time">
            {formatElapsed(now - turn.startedAt)}
          </span>
        </div>
        <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
          The backend answers in one piece when the model finishes; tools it uses are listed with the reply.
          This can take several minutes on a local model.
        </p>
      </div>
    </li>
  );
}

function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  return (
    <button
      type="button"
      onClick={async () => setState((await copyText(text)) ? "copied" : "failed")}
      className="rounded-full border border-navy-2 px-3.5 py-1.5 text-xs font-semibold text-ink transition-colors hover:border-ink-muted"
    >
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy message"}
    </button>
  );
}

/**
 * Shown while a turn's outcome is unknown. Never offers "Send again": the
 * backend may still be processing, or may have stored the message.
 */
export function AwaitingPanel({ awaiting, onRefresh }: { awaiting: Awaiting; onRefresh: () => void }) {
  const polling = awaiting.phase === "polling";
  const now = useNow(polling);
  const minutes = Math.round(TURN_WINDOW_MS / 60000);

  let title: string;
  let body: string;
  if (polling && awaiting.userStored) {
    title = "Waiting for a stored reply";
    body =
      "The connection for this message was lost or the page was reloaded, but the backend may still be working on it. " +
      "Stored messages are checked every few seconds.";
  } else if (polling) {
    title = "Checking whether your message reached the backend";
    body = "The browser lost the response. Nothing is re-sent; stored messages are being checked.";
  } else if (awaiting.userStored) {
    title = "Outcome unknown";
    body =
      `No reply has been recorded for the last message after ${minutes} minutes. The backend may have stopped ` +
      "(for example, it was restarted) or may still be working. Refresh to check again. If you send a new message, " +
      "the unanswered one stays in the conversation and the model will see it in the history.";
  } else {
    title = "Outcome unknown";
    body =
      "The message does not appear to have been stored, but this cannot be confirmed. It was not re-sent. " +
      "Copy it if you want to send it yourself.";
  }

  return (
    <div
      className="rounded-2xl border border-champagne-deep/50 bg-navy/80 px-4 py-3"
      role="status"
      data-testid="awaiting-panel"
      data-phase={awaiting.phase}
    >
      <div className="flex flex-wrap items-center gap-2.5 text-sm font-semibold text-champagne">
        {polling && <span aria-hidden="true" className="h-2 w-2 animate-pulse rounded-full bg-champagne" />}
        {title}
        {polling && (
          <span className="font-mono font-normal text-ink-muted tabular-nums" aria-label="Elapsed time">
            {formatElapsed(now - awaiting.since)}
          </span>
        )}
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{body}</p>
      {!awaiting.userStored && (
        <blockquote className="mt-2 border-l-2 border-champagne-deep/60 pl-3 text-sm break-words whitespace-pre-wrap text-ink">
          {awaiting.content}
        </blockquote>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onRefresh}
          className="rounded-full border border-electric/50 px-3.5 py-1.5 text-xs font-semibold text-electric-glow transition-colors hover:bg-electric/10"
        >
          Refresh
        </button>
        <CopyButton text={awaiting.content} />
      </div>
    </div>
  );
}
