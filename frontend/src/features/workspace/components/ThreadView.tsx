import { useEffect, useRef } from "react";
import type { Conversation } from "../../../api/types";
import { Link } from "../../../app/Link";
import { paths } from "../../../app/router";
import type { ActiveTurn, Awaiting, Notice, Thread } from "../state/reducer";
import { MAX_MESSAGE_PAGES } from "../../../api/conversations";
import { MessageItem } from "./MessageItem";
import { ErrorNotice } from "./Notices";
import { AwaitingPanel, PendingReply, SendingMessage } from "./TurnStatus";

export function ThreadView({
  conversation,
  thread,
  turn,
  awaiting,
  notice,
  isNew,
  invalidId,
  sendAgainDisabled,
  onSendAgain,
  onRefresh,
  onDismissNotice,
}: {
  conversation: Conversation | undefined;
  thread: Thread | undefined;
  /** The active turn if it belongs to this view. */
  turn: ActiveTurn | null;
  awaiting: Awaiting | undefined;
  notice: Notice | undefined;
  isNew: boolean;
  invalidId: boolean;
  sendAgainDisabled: boolean;
  onSendAgain: (failedMessageId: string) => void;
  onRefresh: () => void;
  onDismissNotice: () => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const messages = thread?.messages ?? [];
  const last = messages[messages.length - 1];

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages.length, turn?.startedAt, awaiting?.phase]);

  if (invalidId || thread?.status === "missing") {
    return (
      <div className="grid flex-1 place-items-center px-6 text-center">
        <div>
          <h2 className="font-display text-xl font-semibold text-ink">Conversation not found</h2>
          <p className="mt-2 text-sm text-ink-muted">It does not exist or was deleted.</p>
          <Link href={paths.workspace} className="mt-5 inline-block text-sm font-semibold text-electric-glow hover:underline">
            Start a new conversation
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {isNew && !turn && (
            <div className="py-10 text-center">
              <h2 className="font-display text-2xl font-semibold tracking-tight text-ink">Start a conversation</h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-ink-muted">
                Ask about files in the My-Nexus repository or search OpenAlex and Crossref. The conversation is
                created when you send the first message.
              </p>
            </div>
          )}
          {thread?.status === "loading" && <p role="status" className="text-sm text-ink-muted">Loading messages…</p>}
          {thread?.status === "error" && thread.error && (
            <div>
              <ErrorNotice error={thread.error} />
              <button type="button" onClick={onRefresh} className="mt-2 text-xs font-semibold text-electric-glow hover:underline">
                Retry
              </button>
            </div>
          )}
          {thread?.truncated && (
            <p className="text-xs text-ink-muted">
              Only the first {MAX_MESSAGE_PAGES * 200} messages are shown.
            </p>
          )}
          {conversation && !isNew && (
            <h2 className="sr-only">{conversation.title}</h2>
          )}
          <ol className="flex flex-col gap-4" aria-label="Messages">
            {messages.map((m) => (
              <MessageItem
                key={m.id}
                message={m}
                onSendAgain={m.status === "failed" && m === last && !awaiting ? () => onSendAgain(m.id) : undefined}
                sendAgainDisabled={sendAgainDisabled}
              />
            ))}
            {/* The thread gains this turn's messages only from the server's
                response, so while a turn is active its text is shown locally. */}
            {turn && <SendingMessage content={turn.content} />}
            {turn && <PendingReply turn={turn} />}
          </ol>
          {awaiting && <AwaitingPanel awaiting={awaiting} onRefresh={onRefresh} />}
          {notice && <ErrorNotice error={notice.error} onDismiss={onDismissNotice} />}
          {thread?.status === "ready" && thread.error && <ErrorNotice error={thread.error} />}
          <div ref={endRef} />
        </div>
      </div>
    </div>
  );
}
