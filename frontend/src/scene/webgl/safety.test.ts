import { describe, expect, it, vi } from "vitest";
import { watchContextLoss, watchShaderErrors, type RendererDebug } from "./safety";

describe("watchContextLoss", () => {
  it("reports a lost context once and prevents the default", () => {
    const canvas = document.createElement("canvas");
    const onLost = vi.fn();
    const stop = watchContextLoss(canvas, onLost);
    const event = new Event("webglcontextlost", { cancelable: true });
    canvas.dispatchEvent(event);
    canvas.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    expect(onLost).toHaveBeenCalledOnce();
    expect(event.defaultPrevented).toBe(true);
    stop();
  });

  it("stops listening after cleanup", () => {
    const canvas = document.createElement("canvas");
    const onLost = vi.fn();
    watchContextLoss(canvas, onLost)();
    canvas.dispatchEvent(new Event("webglcontextlost", { cancelable: true }));
    expect(onLost).not.toHaveBeenCalled();
  });
});

describe("watchShaderErrors", () => {
  it("turns three's shader-error hook into a single failure callback", () => {
    const renderer: RendererDebug = { debug: { checkShaderErrors: false, onShaderError: null } };
    const onError = vi.fn();
    const stop = watchShaderErrors(renderer, onError);
    expect(renderer.debug.checkShaderErrors).toBe(true);
    renderer.debug.onShaderError?.();
    renderer.debug.onShaderError?.();
    expect(onError).toHaveBeenCalledOnce();
    stop();
    expect(renderer.debug.onShaderError).toBeNull();
  });
});
