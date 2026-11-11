import { describe, expect, it } from "vitest";
import { BENCH_LEVELS, frameStats } from "./bench";
import { sphereLayout } from "./layout";
import { MAX_PARTICLES, MIN_PARTICLES, readSceneParams } from "./params";
import {
  FrameBudgetMonitor,
  LOWEST_LEVEL,
  QUALITY_LADDER,
  effectiveDpr,
  initialLevel,
  readDeviceSignals,
} from "./quality";

describe("quality ladder", () => {
  it("starts at 20,000 particles and every step lowers dpr or particle count", () => {
    expect(QUALITY_LADDER[0]).toEqual({ particles: 20_000, maxDpr: 2 });
    for (let i = 1; i < QUALITY_LADDER.length; i++) {
      const prev = QUALITY_LADDER[i - 1]!;
      const cur = QUALITY_LADDER[i]!;
      expect(cur.particles <= prev.particles && cur.maxDpr <= prev.maxDpr).toBe(true);
      expect(cur.particles < prev.particles || cur.maxDpr < prev.maxDpr).toBe(true);
    }
    // Both kinds of reduction exist in the ladder.
    expect(new Set(QUALITY_LADDER.map((l) => l.particles)).size).toBeGreaterThan(1);
    expect(new Set(QUALITY_LADDER.map((l) => l.maxDpr)).size).toBeGreaterThan(1);
  });

  it("starts at the top level unless the device clearly signals low capability", () => {
    expect(initialLevel({})).toBe(0);
    expect(initialLevel({ hardwareConcurrency: 8, deviceMemory: 8 })).toBe(0);
    expect(initialLevel({ hardwareConcurrency: 4, deviceMemory: 4 })).toBe(0);
    expect(initialLevel({ deviceMemory: 2 })).toBe(2);
    expect(initialLevel({ hardwareConcurrency: 2 })).toBe(2);
    expect(initialLevel({ saveData: true, hardwareConcurrency: 16 })).toBe(LOWEST_LEVEL);
  });

  it("reads device signals defensively", () => {
    expect(readDeviceSignals({ hardwareConcurrency: 4 } as Navigator)).toEqual({
      hardwareConcurrency: 4,
      deviceMemory: undefined,
      saveData: undefined,
    });
  });

  it("caps pixel ratio per level and never goes below 1", () => {
    const top = QUALITY_LADDER[0]!;
    expect(effectiveDpr(top, 3)).toBe(2);
    expect(effectiveDpr(top, 1.25)).toBe(1.25);
    expect(effectiveDpr(QUALITY_LADDER[LOWEST_LEVEL]!, 2)).toBe(1);
    expect(effectiveDpr(top, 0)).toBe(1);
  });
});

describe("FrameBudgetMonitor", () => {
  function feed(m: FrameBudgetMonitor, deltaMs: number, frames: number, startMs: number) {
    let now = startMs;
    let fired = 0;
    for (let i = 0; i < frames; i++) {
      now += deltaMs;
      if (m.sample(deltaMs, now)) fired++;
    }
    return { fired, now };
  }

  it("never fires for frames within budget", () => {
    expect(feed(new FrameBudgetMonitor(), 16.7, 2000, 0).fired).toBe(0);
  });

  it("fires for sustained slow frames, after warm-up, then waits out a cooldown", () => {
    const m = new FrameBudgetMonitor({ warmupMs: 1000, windowFrames: 30, cooldownMs: 3000 });
    // First sample at t=40 ms, so warm-up lasts until t=1040 ms; nothing fires before.
    const warm = feed(m, 40, 20, 0); // t = 40 .. 800
    expect(warm.fired).toBe(0);
    // t = 840 .. 3200: the 30-frame window fills at t = 2200 -> exactly one step.
    const slow = feed(m, 40, 60, warm.now);
    expect(slow.fired).toBe(1);
    // Cooldown runs to t = 5200; the frames after it do not fill a new window.
    expect(feed(m, 40, 60, slow.now).fired).toBe(0);
  });

  it("ignores pauses (huge deltas) instead of treating them as slow frames", () => {
    const m = new FrameBudgetMonitor({ warmupMs: 0, windowFrames: 10 });
    expect(feed(m, 5000, 50, 0).fired).toBe(0);
  });
});

describe("scene params", () => {
  it("parses diagnostic modes", () => {
    expect(readSceneParams("").mode).toBe("auto");
    expect(readSceneParams("?scene=off").mode).toBe("off");
    expect(readSceneParams("?scene=fail").mode).toBe("fail");
    expect(readSceneParams("?scene=lose").mode).toBe("lose");
    expect(readSceneParams("?bench").mode).toBe("bench");
    expect(readSceneParams("?scene=nonsense").mode).toBe("auto");
  });

  it("clamps the particle override to a safe range and ignores junk", () => {
    expect(readSceneParams("?particles=500").particles).toBe(MIN_PARTICLES);
    expect(readSceneParams("?particles=999999").particles).toBe(MAX_PARTICLES);
    expect(readSceneParams("?particles=15000").particles).toBe(15_000);
    expect(readSceneParams("?particles=abc").particles).toBeNull();
    expect(readSceneParams("?particles=-5").particles).toBeNull();
  });
});

describe("benchmark statistics", () => {
  it("computes fps and percentiles", () => {
    const s = frameStats([...Array<number>(95).fill(16), ...Array<number>(5).fill(40)]);
    expect(s.frames).toBe(100);
    expect(s.p50Ms).toBe(16);
    expect(s.p95Ms).toBe(16);
    expect(s.p99Ms).toBe(40);
    expect(s.maxMs).toBe(40);
    expect(s.avgFps).toBeCloseTo(1000 / 17.2, 1);
  });

  it("handles an empty run", () => {
    expect(frameStats([]).frames).toBe(0);
  });

  it("benchmarks the top, middle and bottom of the ladder", () => {
    expect(BENCH_LEVELS.map((i) => QUALITY_LADDER[i]!.particles)).toEqual([20_000, 12_000, 6_000]);
  });
});

describe("sphere layout", () => {
  it("puts the sphere right of the text in landscape and above it in portrait", () => {
    expect(sphereLayout(1440, 900).position[0]).toBeGreaterThan(1.5);
    expect(sphereLayout(1000, 800).position[0]).toBeGreaterThan(0); // small landscape
    for (const [w, h] of [[390, 844], [820, 1180]] as const) {
      const portrait = sphereLayout(w, h);
      expect(portrait.position[0]).toBe(0);
      expect(portrait.position[1]).toBeGreaterThan(0);
      expect(portrait.scale).toBeLessThan(1);
    }
    // Portrait tablets get a larger sphere than phones.
    expect(sphereLayout(820, 1180).scale).toBeGreaterThan(sphereLayout(390, 844).scale);
  });

  it("keeps every portrait sphere inside the frame width", () => {
    const halfH = Math.tan((45 / 2) * (Math.PI / 180)) * 6;
    for (const [w, h] of [[390, 844], [820, 1180]] as const) {
      const { scale } = sphereLayout(w, h);
      const shell = 2.2 * scale; // the dense shell; the sparse halo may bleed
      expect(shell).toBeLessThan(halfH * (w / h) * 1.05);
    }
  });

  it("keeps the whole desktop sphere inside the frame (camera z=6, fov 45)", () => {
    const halfH = Math.tan((45 / 2) * (Math.PI / 180)) * 6;
    const halfW = halfH * (1440 / 900);
    const { position, scale } = sphereLayout(1440, 900);
    const outer = 2.2 * scale * 1.22; // radius x halo
    expect(position[0] + outer).toBeLessThan(halfW);
    expect(outer).toBeLessThan(halfH);
  });

  it("shrinks point size with the sphere so small layouts are not over-dense", () => {
    const sizes = [sphereLayout(1440, 900), sphereLayout(1000, 800), sphereLayout(390, 844)];
    expect(sizes.map((s) => s.pointScale)).toEqual([...sizes.map((s) => s.pointScale)].sort((a, b) => b - a));
  });
});
