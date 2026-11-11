import { useId, useRef, type FormEvent, type KeyboardEvent } from "react";
import { codePoints, validateMessage } from "../../../api/conversations";
import { LIMITS } from "../../../api/types";

/**
 * Message input. Enter sends, Shift+Enter inserts a newline. Sending is
 * disabled (with the reason shown) while another turn is active (D6).
 */
export function Composer({
  value,
  onChange,
  onSend,
  blockedReason,
  warning,
}: {
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string) => void;
  blockedReason: string | null;
  warning?: string;
}) {
  const id = useId();
  const ref = useRef<HTMLTextAreaElement>(null);
  const length = codePoints(value.trim());
  const invalid = value.trim() ? validateMessage(value) : null;
  const canSend = !blockedReason && !validateMessage(value);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (canSend) onSend(value);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={submit} className="border-t border-navy-2 bg-abyss/90 px-4 pt-3 pb-4 backdrop-blur sm:px-6" aria-label="Send a message">
      <div className="mx-auto max-w-3xl">
        {warning && <p className="mb-2 text-xs text-champagne">{warning}</p>}
        <label htmlFor={id} className="sr-only">
          Message
        </label>
        <div className="flex items-end gap-2 rounded-2xl border border-navy-2 bg-navy/80 p-2 focus-within:border-electric/60">
          <textarea
            id={id}
            ref={ref}
            value={value}
            rows={Math.min(8, Math.max(1, value.split("\n").length))}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Message Nexus…"
            aria-describedby={`${id}-hint`}
            aria-invalid={invalid ? true : undefined}
            className="max-h-64 min-h-[2.75rem] flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-muted/70"
          />
          <button
            type="submit"
            disabled={!canSend}
            className="shrink-0 rounded-xl bg-champagne px-4 py-2.5 text-sm font-semibold text-abyss transition-colors hover:bg-[#EFD7A6] disabled:cursor-not-allowed disabled:bg-navy-2 disabled:text-ink-muted"
          >
            Send
          </button>
        </div>
        <p id={`${id}-hint`} className="mt-1.5 flex flex-wrap justify-between gap-2 text-xs text-ink-muted">
          <span>{blockedReason ?? invalid ?? "Enter to send · Shift+Enter for a new line"}</span>
          {length > LIMITS.messageChars * 0.8 && (
            <span className={length > LIMITS.messageChars ? "text-danger" : ""}>
              {length.toLocaleString("en")} / {LIMITS.messageChars.toLocaleString("en")}
            </span>
          )}
        </p>
      </div>
    </form>
  );
}
