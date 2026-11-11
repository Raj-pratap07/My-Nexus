/**
 * Rendering quality: a fixed ladder of (particle count, max pixel ratio)
 * levels, an initial choice from device signals, and a frame-budget monitor
 * that steps DOWN the ladder (never up, so it cannot oscillate).
 * Pure and three-free: part of the main chunk and unit-tested.
 */

export interface QualityLevel {
  readonly particles: number;
  readonly maxDpr: number;
}

/** Ordered best -> cheapest. Each step lowers pixel ratio or particle count. */
export const QUALITY_LADDER: readonly QualityLevel[] = [
  { particles: 20_000, maxDpr: 2 },
  { particles: 20_000, maxDpr: 1.5 },
  { particles: 12_000, maxDpr: 1.5 },
  { particles: 12_000, maxDpr: 1 },
  { particles: 6_000, maxDpr: 1 },
];

export const LOWEST_LEVEL = QUALITY_LADDER.length - 1;

export interface DeviceSignals {
  hardwareConcurrency?: number | undefined;
  /** GiB, from navigator.deviceMemory (Chromium only). */
  deviceMemory?: number | undefined;
  saveData?: boolean | undefined;
}

/**
 * Everyone starts at the 20,000-particle top level; only explicit
 * low-capability signals start lower. The monitor adapts from there.
 */
export function initialLevel(s: DeviceSignals): number {
  if (s.saveData) return LOWEST_LEVEL;
  const lowMemory = s.deviceMemory !== undefined && s.deviceMemory <= 2;
  const fewCores = s.hardwareConcurrency !== undefined && s.hardwareConcurrency <= 2;
  if (lowMemory || fewCores) return 2;
  return 0;
}

export function readDeviceSignals(nav: Navigator): DeviceSignals {
  const n = nav as Navigator & {
    deviceMemory?: number;
    connection?: { saveData?: boolean };
  };
  return {
    hardwareConcurrency: n.hardwareConcurrency,
    deviceMemory: n.deviceMemory,
    saveData: n.connection?.saveData,
  };
}

export function effectiveDpr(level: QualityLevel, devicePixelRatio: number): number {
  return Math.max(1, Math.min(level.maxDpr, devicePixelRatio || 1));
}

export interface FrameBudgetOptions {
  /** Frames slower than this on average trigger a step down (24 ms ~ 42 fps). */
  budgetMs?: number;
  /** Frames averaged per decision. */
  windowFrames?: number;
  /** Ignore everything for this long after start (shader compile, warm-up). */
  warmupMs?: number;
  /** After a step down, wait this long before judging again. */
  cooldownMs?: number;
  /** Deltas above this are pauses (hidden tab, debugger), not slow frames. */
  maxDeltaMs?: number;
}

/**
 * Feed it every frame's delta; `sample` returns true when the average over a
 * full window exceeds the budget. It never recommends stepping up.
 */
export class FrameBudgetMonitor {
  private readonly budgetMs: number;
  private readonly windowFrames: number;
  private readonly warmupMs: number;
  private readonly cooldownMs: number;
  private readonly maxDeltaMs: number;
  private startedAt: number | null = null;
  private quietUntil = 0;
  private sum = 0;
  private count = 0;

  constructor(o: FrameBudgetOptions = {}) {
    this.budgetMs = o.budgetMs ?? 24;
    this.windowFrames = o.windowFrames ?? 90;
    this.warmupMs = o.warmupMs ?? 1500;
    this.cooldownMs = o.cooldownMs ?? 3000;
    this.maxDeltaMs = o.maxDeltaMs ?? 250;
  }

  sample(deltaMs: number, nowMs: number): boolean {
    if (this.startedAt === null) this.startedAt = nowMs;
    if (nowMs - this.startedAt < this.warmupMs || nowMs < this.quietUntil) return false;
    if (deltaMs <= 0 || deltaMs > this.maxDeltaMs) return false;
    this.sum += deltaMs;
    this.count += 1;
    if (this.count < this.windowFrames) return false;
    const slow = this.sum / this.count > this.budgetMs;
    this.sum = 0;
    this.count = 0;
    if (slow) this.quietUntil = nowMs + this.cooldownMs;
    return slow;
  }
}
