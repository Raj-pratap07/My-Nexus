import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import { BENCH_MEASURE_MS, BENCH_WARMUP_MS, frameStats, type FrameStats } from "../bench";
import { FrameBudgetMonitor } from "../quality";

/**
 * Calls `onSlow` when frames stay over budget. Remount (key) it after each
 * quality change so the warm-up and window start fresh.
 */
export function QualityMonitor({ onSlow }: { onSlow: () => void }) {
  const monitor = useMemo(() => new FrameBudgetMonitor(), []);
  useFrame((_, delta) => {
    if (monitor.sample(delta * 1000, performance.now())) onSlow();
  });
  return null;
}

/**
 * Measures real time between frames for one benchmark level: a warm-up,
 * then BENCH_MEASURE_MS of samples. Remount (key) per level.
 */
export function BenchRunner({ onDone }: { onDone: (stats: FrameStats) => void }) {
  const run = useRef({ start: -1, last: -1, deltas: [] as number[], done: false });
  useFrame(() => {
    const now = performance.now();
    const r = run.current;
    if (r.done) return;
    if (r.start < 0) {
      r.start = now;
      r.last = now;
      return;
    }
    const delta = now - r.last;
    r.last = now;
    const elapsed = now - r.start;
    if (elapsed < BENCH_WARMUP_MS) return;
    r.deltas.push(delta);
    if (elapsed >= BENCH_WARMUP_MS + BENCH_MEASURE_MS) {
      r.done = true;
      onDone(frameStats(r.deltas));
    }
  });
  return null;
}
