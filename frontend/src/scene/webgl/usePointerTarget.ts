import { useEffect, useRef, type RefObject } from "react";

export interface PointerState {
  /** Normalised device coordinates of the pointer over the canvas (-1..1). */
  ndcX: number;
  ndcY: number;
  /** True while a pointer is over the canvas area. */
  active: boolean;
}

/**
 * Tracks the pointer in a ref (never React state: no re-renders). Listens on
 * the window because the canvas sits BEHIND the page content with
 * pointer-events: none, so the page stays fully clickable.
 */
export function usePointerTarget(
  enabled: boolean,
  canvas: HTMLElement,
): RefObject<PointerState> {
  const state = useRef<PointerState>({ ndcX: 0, ndcY: 0, active: false });

  useEffect(() => {
    const s = state.current;
    s.active = false;
    if (!enabled) return;

    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const inside =
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom &&
        rect.width > 0 &&
        rect.height > 0;
      s.active = inside;
      if (!inside) return;
      s.ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      s.ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
    };
    const release = () => {
      s.active = false;
    };
    const onUp = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") release(); // touch / pen lift
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerdown", onMove, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    window.addEventListener("pointercancel", release);
    window.addEventListener("blur", release);
    document.documentElement.addEventListener("pointerleave", release);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", release);
      window.removeEventListener("blur", release);
      document.documentElement.removeEventListener("pointerleave", release);
      s.active = false;
    };
  }, [enabled, canvas]);

  return state;
}
