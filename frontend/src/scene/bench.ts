/**
 * Rendering benchmark (?bench). Pure statistics + report types; the runner
 * lives in the lazy scene chunk. Frame times are measured between real
 * requestAnimationFrame callbacks, so results are capped by the display
 * refresh rate (vsync): ~16.7 ms at 60 Hz is the floor, not a measurement
 * limit of the scene.
 */

export interface FrameStats {
  frames: number;
  avgFps: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
}

export interface BenchLevelResult extends FrameStats {
  particles: number;
  dpr: number;
}

export interface BenchReport {
  levels: BenchLevelResult[];
  gpu: { webgl2: boolean; renderer: string; vendor: string } | null;
  viewport: { width: number; height: number; devicePixelRatio: number };
  userAgent: string;
  warmupMs: number;
  measureMs: number;
}

/** Ladder indices measured by ?bench (20k @ dpr<=2, 12k @ dpr<=1.5, 6k @ dpr 1). */
export const BENCH_LEVELS = [0, 2, 4] as const;
export const BENCH_WARMUP_MS = 1000;
export const BENCH_MEASURE_MS = 4000;

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx] ?? 0;
}

const round = (n: number, d = 2) => Math.round(n * 10 ** d) / 10 ** d;

export function frameStats(deltasMs: readonly number[]): FrameStats {
  if (deltasMs.length === 0) {
    return { frames: 0, avgFps: 0, p50Ms: 0, p95Ms: 0, p99Ms: 0, maxMs: 0 };
  }
  const sorted = [...deltasMs].sort((a, b) => a - b);
  const total = deltasMs.reduce((s, d) => s + d, 0);
  return {
    frames: deltasMs.length,
    avgFps: round(1000 / (total / deltasMs.length), 1),
    p50Ms: round(percentile(sorted, 50)),
    p95Ms: round(percentile(sorted, 95)),
    p99Ms: round(percentile(sorted, 99)),
    maxMs: round(sorted[sorted.length - 1] ?? 0),
  };
}
