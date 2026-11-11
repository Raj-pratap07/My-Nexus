import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => cleanup());

// Node-environment test files (dev/*.test.ts) have no DOM to shim.
const hasDom = typeof window !== "undefined";

// jsdom has no WebGL. Report "no context" quietly (jsdom would otherwise log
// "Not implemented: HTMLCanvasElement.prototype.getContext").
if (hasDom) {
  Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    writable: true,
    value: vi.fn(() => null),
  });
}

// jsdom lacks matchMedia; default to "no preference". Tests override it.
if (hasDom && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: (query: string): MediaQueryList =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
        dispatchEvent: () => false,
      }) as MediaQueryList,
  });
}
