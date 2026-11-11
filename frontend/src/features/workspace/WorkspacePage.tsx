/**
 * The Nexus workspace (lazy chunk). It never imports src/scene: no three.js
 * is downloaded or started here (enforced by src/architecture.test.ts).
 */
import { useEffect, useRef, useState } from "react";
import type { Conversation } from "../../api/types";
import { Link } from "../../app/Link";
import { paths } from "../../app/router";
import { brand } from "../landing/content";
import { BackendStatus } from "./components/BackendStatus";
import { Composer } from "./components/Composer";
import { DeleteDialog } from "./components/DeleteDialog";
import { SetupBanner } from "./components/Notices";
import { Sidebar } from "./components/Sidebar";
import { ThreadView } from "./components/ThreadView";
import { NEW } from "./state/reducer";
import { useWorkspace } from "./state/useWorkspace";

export default function WorkspacePage({
  conversationId,
  invalidId = false,
}: {
  conversationId: string | null;
  invalidId?: boolean;
}) {
  const ws = useWorkspace(invalidId ? null : conversationId);
  const { state, selectedKey } = ws;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [healthKey, setHealthKey] = useState(0);
  const mainRef = useRef<HTMLElement>(null);
  const firstRender = useRef(true);

  // Move focus to the thread when the selected conversation changes.
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    mainRef.current?.focus({ preventScroll: true });
  }, [conversationId]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const isNew = selectedKey === NEW && !invalidId;
  const conversation = isNew ? undefined : state.conversations.items.find((c) => c.id === selectedKey);
  const thread = isNew ? undefined : state.threads[selectedKey];
  const turn = state.turn && state.turn.key === selectedKey ? state.turn : null;
  const awaiting = isNew ? undefined : state.awaiting[selectedKey];
  const showComposer = !invalidId && thread?.status !== "missing";

  const busyIds = new Set<string>();
  if (state.turn?.conversationId) busyIds.add(state.turn.conversationId);
  for (const [id, a] of Object.entries(state.awaiting)) if (a.phase === "polling") busyIds.add(id);

  const sidebar = (
    <Sidebar
      conversations={state.conversations}
      selectedId={isNew ? null : selectedKey}
      busyIds={busyIds}
      onNavigate={() => setDrawerOpen(false)}
      onDelete={(c) => setDeleteTarget(c)}
      onLoadMore={ws.loadMore}
      onRetry={ws.reloadList}
    />
  );

  return (
    <div className="flex h-[100svh] flex-col bg-abyss text-ink">
      <a
        href="#thread"
        className="sr-only z-50 rounded-md bg-champagne px-4 py-2 font-semibold text-abyss focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to conversation
      </a>
      <header className="flex items-center gap-3 border-b border-navy-2 px-4 py-3 sm:px-5">
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="rounded-md border border-navy-2 px-2.5 py-1.5 text-xs font-semibold text-ink lg:hidden"
          aria-label="Show conversations"
          aria-expanded={drawerOpen}
        >
          ☰
        </button>
        <Link href={paths.landing} className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-ink" aria-label={`${brand.name} home`}>
          <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-champagne shadow-[0_0_14px_2px_rgb(230_201_143/0.55)]" />
          <span className="hidden sm:inline">{brand.name}</span>
        </Link>
        <h1 className="min-w-0 flex-1 truncate text-sm font-medium text-ink-muted">
          {isNew ? "New conversation" : (conversation?.title ?? "Conversation")}
        </h1>
        <BackendStatus refreshKey={healthKey} />
      </header>

      {state.setup && (
        <SetupBanner
          error={state.setup}
          onRetry={() => {
            setHealthKey((k) => k + 1);
            ws.reloadList();
            if (!isNew) ws.refreshThread(selectedKey);
          }}
        />
      )}

      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-72 shrink-0 border-r border-navy-2 bg-navy/40 lg:block">{sidebar}</aside>

        {drawerOpen && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <button
              type="button"
              aria-label="Close conversations"
              className="absolute inset-0 h-full w-full bg-abyss/70"
              onClick={() => setDrawerOpen(false)}
            />
            <aside
              role="dialog"
              aria-modal="true"
              aria-label="Conversations"
              className="absolute inset-y-0 left-0 w-[min(20rem,85vw)] border-r border-navy-2 bg-navy shadow-2xl"
            >
              {sidebar}
            </aside>
          </div>
        )}

        <main id="thread" ref={mainRef} tabIndex={-1} className="flex min-w-0 flex-1 flex-col outline-none">
          <ThreadView
            conversation={conversation}
            thread={thread}
            turn={turn}
            awaiting={awaiting}
            notice={state.notices[selectedKey]}
            isNew={isNew}
            invalidId={invalidId}
            sendAgainDisabled={ws.sendBlockedReason !== null}
            onSendAgain={(id) => void ws.sendAgain(id)}
            onRefresh={() => ws.refreshThread(selectedKey)}
            onDismissNotice={() => ws.dismissNotice(selectedKey)}
          />
          {showComposer && (
            <Composer
              value={state.drafts[selectedKey] ?? ""}
              onChange={(text) => ws.setDraft(selectedKey, text)}
              onSend={(text) => void ws.send(text)}
              blockedReason={ws.sendBlockedReason}
              warning={
                awaiting?.phase === "expired" && awaiting.userStored
                  ? "The previous message has no recorded reply. If you send a new one, the unanswered message stays in the history the model sees."
                  : undefined
              }
            />
          )}
        </main>
      </div>

      <div aria-live="polite" className="sr-only" data-testid="announcer">
        {ws.announcement}
      </div>

      {deleteTarget && (
        <DeleteDialog
          conversation={deleteTarget}
          onConfirm={() => ws.deleteConversation(deleteTarget.id)}
          onClose={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}
