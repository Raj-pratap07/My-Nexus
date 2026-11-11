/**
 * WebGL failure detection that does not depend on React error boundaries:
 * context loss and shader-compile errors happen outside React's render phase,
 * so they are reported through callbacks and the scene swaps to its fallback.
 */

export type Cleanup = () => void;

/**
 * Calls `onLost` once when the canvas loses its WebGL context (GPU reset,
 * driver crash, too many contexts, or WEBGL_lose_context). The static
 * fallback is shown from then on; this milestone does not try to restore.
 */
export function watchContextLoss(canvas: HTMLCanvasElement, onLost: () => void): Cleanup {
  let fired = false;
  const handler = (event: Event) => {
    event.preventDefault();
    if (fired) return;
    fired = true;
    onLost();
  };
  canvas.addEventListener("webglcontextlost", handler, false);
  return () => canvas.removeEventListener("webglcontextlost", handler, false);
}

/** The part of three's WebGLRenderer used here (keeps this module testable). */
export interface RendererDebug {
  debug: {
    checkShaderErrors: boolean;
    onShaderError: ((...args: never[]) => void) | null;
  };
}

/**
 * three normally only logs shader compile/link errors. This makes them
 * observable: `onError` is called once and the scene falls back.
 */
export function watchShaderErrors(renderer: RendererDebug, onError: () => void): Cleanup {
  let fired = false;
  renderer.debug.checkShaderErrors = true;
  renderer.debug.onShaderError = () => {
    if (fired) return;
    fired = true;
    onError();
  };
  return () => {
    renderer.debug.onShaderError = null;
  };
}

/** Best-effort GPU description for benchmark reports (may be masked by the browser). */
export function describeRenderer(gl: WebGLRenderingContext | WebGL2RenderingContext): {
  webgl2: boolean;
  renderer: string;
  vendor: string;
} {
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = ext
    ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL))
    : String(gl.getParameter(gl.RENDERER));
  const vendor = ext
    ? String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL))
    : String(gl.getParameter(gl.VENDOR));
  const webgl2 =
    typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
  return { webgl2, renderer, vendor };
}
