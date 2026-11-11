import { Link } from "../../../app/Link";
import { paths } from "../../../app/router";
import type { Conversation } from "../../../api/types";
import type { ConversationsState } from "../state/reducer";
import { formatWhen } from "../format";
import { ErrorNotice } from "./Notices";

export function Sidebar({
  conversations,
  selectedId,
  busyIds,
  onNavigate,
  onDelete,
  onLoadMore,
  onRetry,
}: {
  conversations: ConversationsState;
  selectedId: string | null;
  /** Conversations with a request in flight or an unknown outcome. */
  busyIds: ReadonlySet<string>;
  onNavigate: () => void;
  onDelete: (c: Conversation) => void;
  onLoadMore: () => void;
  onRetry: () => void;
}) {
  const { status, items, hasMore, loadingMore, error } = conversations;
  return (
    <nav aria-label="Conversations" className="flex h-full flex-col">
      <div className="p-4">
        <Link
          href={paths.workspace}
          onClick={onNavigate}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-champagne px-4 py-2.5 text-sm font-semibold text-abyss transition-colors hover:bg-[#EFD7A6]"
        >
          <span aria-hidden="true">+</span> New conversation
        </Link>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {status === "loading" && <p className="px-3 py-2 text-sm text-ink-muted" role="status">Loading conversations…</p>}
        {status === "error" && error && (
          <div className="px-2">
            <ErrorNotice error={error} />
            <button type="button" onClick={onRetry} className="mt-2 px-1 text-xs font-semibold text-electric-glow hover:underline">
              Retry
            </button>
          </div>
        )}
        {status === "ready" && items.length === 0 && (
          <p className="px-3 py-2 text-sm text-ink-muted">No conversations yet.</p>
        )}
        <ul className="flex flex-col gap-0.5">
          {items.map((c) => {
            const selected = c.id === selectedId;
            return (
              <li key={c.id} className="group relative">
                <Link
                  href={paths.conversation(c.id)}
                  onClick={onNavigate}
                  aria-current={selected ? "page" : undefined}
                  className={[
                    "block rounded-lg py-2 pr-10 pl-3 text-sm transition-colors",
                    selected ? "bg-navy-2 text-ink" : "text-ink-muted hover:bg-navy hover:text-ink",
                  ].join(" ")}
                >
                  <span className="block truncate">{c.title}</span>
                  <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-muted">
                    <time dateTime={c.updated_at}>{formatWhen(c.updated_at)}</time>
                    {busyIds.has(c.id) && (
                      <span className="text-electric-glow">· waiting</span>
                    )}
                  </span>
                </Link>
                <button
                  type="button"
                  onClick={() => onDelete(c)}
                  aria-label={`Delete conversation: ${c.title}`}
                  className="absolute top-1/2 right-1.5 -translate-y-1/2 rounded-md px-2 py-1 text-xs text-ink-muted opacity-100 transition-opacity hover:text-danger focus-visible:opacity-100 lg:opacity-0 lg:group-hover:opacity-100"
                >
                  Delete
                </button>
              </li>
            );
          })}
        </ul>
        {status === "ready" && error && items.length > 0 && (
          <div className="mt-2 px-2">
            <ErrorNotice error={error} />
          </div>
        )}
        {hasMore && (
          <button
            type="button"
            onClick={onLoadMore}
            disabled={loadingMore}
            className="mt-2 w-full rounded-lg px-3 py-2 text-xs font-semibold text-electric-glow hover:bg-navy disabled:opacity-60"
          >
            {loadingMore ? "Loading…" : "Load more"}
          </button>
        )}
      </div>
    </nav>
  );
}
