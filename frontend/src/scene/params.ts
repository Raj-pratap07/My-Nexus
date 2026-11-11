/**
 * Scene diagnostics read from the URL (for manual QA and benchmarking):
 *   ?scene=off   force the static fallback
 *   ?scene=fail  compile a deliberately broken shader (tests the shader-error path)
 *   ?scene=lose  lose the WebGL context 3 s after start (tests context loss)
 *   ?bench       run the rendering benchmark and print its result on the page
 *   ?particles=N override the particle count (clamped to a safe range)
 */
export type SceneMode = "auto" | "off" | "fail" | "lose" | "bench";

export interface SceneParams {
  mode: SceneMode;
  particles: number | null;
}

export const MIN_PARTICLES = 1_000;
export const MAX_PARTICLES = 60_000;

export function clampParticles(value: number): number {
  return Math.min(MAX_PARTICLES, Math.max(MIN_PARTICLES, Math.round(value)));
}

export function readSceneParams(search: string): SceneParams {
  const q = new URLSearchParams(search);
  const scene = q.get("scene");
  const mode: SceneMode = q.has("bench")
    ? "bench"
    : scene === "off" || scene === "fail" || scene === "lose"
      ? scene
      : "auto";
  const raw = q.get("particles");
  const n = raw === null ? Number.NaN : Number(raw);
  return { mode, particles: Number.isFinite(n) && n > 0 ? clampParticles(n) : null };
}
