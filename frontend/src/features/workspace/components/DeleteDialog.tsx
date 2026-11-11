import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import type { ApiError } from "../../../api/client";
import type { Conversation } from "../../../api/types";
import { ErrorNotice } from "./Notices";

/** Accessible confirmation modal: focus moves in, Esc cancels, focus returns. */
export function DeleteDialog({
  conversation,
  onConfirm,
  onClose,
}: {
  conversation: Conversation;
  onConfirm: () => Promise<ApiError | null>;
  onClose: () => void;
}) {
  const titleId = useId();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    return () => opener?.focus?.();
  }, []);

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape" && !busy) {
      e.stopPropagation();
      onClose();
    }
    if (e.key === "Tab") {
      // Keep focus inside the dialog.
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled])");
      if (!focusable?.length) return;
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  const confirm = async () => {
    setBusy(true);
    const err = await onConfirm();
    setBusy(false);
    if (err) setError(err);
    else onClose();
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-abyss/75 px-4 backdrop-blur-sm" onKeyDown={onKeyDown}>
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-md rounded-2xl border border-navy-2 bg-navy p-6 shadow-2xl"
      >
        <h2 id={titleId} className="font-display text-lg font-semibold text-ink">
          Delete this conversation?
        </h2>
        <p className="mt-2 text-sm break-words text-ink-muted">
          “{conversation.title}” and all of its messages and tool records will be deleted from the database. This cannot be undone.
        </p>
        {error && (
          <div className="mt-4">
            <ErrorNotice error={error} />
          </div>
        )}
        <div className="mt-6 flex justify-end gap-2">
          <button
            ref={cancelRef}
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-full border border-navy-2 px-4 py-2 text-sm font-semibold text-ink hover:border-ink-muted"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={confirm}
            disabled={busy}
            className="rounded-full border border-danger/60 px-4 py-2 text-sm font-semibold text-danger hover:bg-danger/10 disabled:opacity-60"
          >
            {busy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
