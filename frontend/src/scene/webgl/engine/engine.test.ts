import { AdditiveBlending } from "three";
import { describe, expect, it, vi } from "vitest";
import { GOLD_TONE_THRESHOLD, buildParticleAttributes } from "./attributes";
import { BLUEGOLD_CONFIG, BLUEGOLD_LOOK, type ParticleConfig } from "./config";
import { fibonacciSphere } from "./distribution";
import { ATTRIBUTE_NAMES, createParticleGeometry } from "./geometry";
import { createParticleSystem, maxWorldRadius } from "./index";
import { createRng } from "./random";

const cfg = (count: number, over: Partial<ParticleConfig> = {}): ParticleConfig => ({
  ...BLUEGOLD_CONFIG,
  count,
  ...over,
});

function nearestNeighbourDistances(p: Float32Array): number[] {
  const n = p.length / 3;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = p[i * 3]! - p[j * 3]!;
      const dy = p[i * 3 + 1]! - p[j * 3 + 1]!;
      const dz = p[i * 3 + 2]! - p[j * 3 + 2]!;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    out.push(Math.sqrt(best));
  }
  return out;
}

const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
const cv = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2))) / m;
};

describe("random", () => {
  it("is deterministic per seed and stays in [0, 1)", () => {
    const a = createRng(42);
    const b = createRng(42);
    const c = createRng(43);
    const seqA = Array.from({ length: 1000 }, a);
    expect(Array.from({ length: 1000 }, b)).toEqual(seqA);
    expect(Array.from({ length: 1000 }, c)).not.toEqual(seqA);
    expect(seqA.every((x) => x >= 0 && x < 1)).toBe(true);
    expect(Math.abs(mean(seqA) - 0.5)).toBeLessThan(0.03);
  });
});

describe("fibonacciSphere", () => {
  it("returns exactly count points, all on the unit sphere", () => {
    for (const n of [1, 2, 7, 1000]) {
      const p = fibonacciSphere(n);
      expect(p.length).toBe(n * 3);
      for (let i = 0; i < n; i++) {
        expect(Math.hypot(p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!)).toBeCloseTo(1, 6);
      }
    }
  });

  it("keeps jittered points on the unit sphere", () => {
    const p = fibonacciSphere(2000, 0.35, createRng(1));
    for (let i = 0; i < 2000; i++) {
      expect(Math.hypot(p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!)).toBeCloseTo(1, 5);
    }
  });

  it("balances hemispheres and octants", () => {
    const n = 4000;
    const p = fibonacciSphere(n, 0.35, createRng(7));
    const upper = Array.from({ length: n }, (_, i) => p[i * 3 + 1]!).filter((y) => y > 0).length;
    expect(Math.abs(upper - n / 2) / n).toBeLessThan(0.01);
    const octants = new Array<number>(8).fill(0);
    for (let i = 0; i < n; i++) {
      const k = (p[i * 3]! > 0 ? 1 : 0) + (p[i * 3 + 1]! > 0 ? 2 : 0) + (p[i * 3 + 2]! > 0 ? 4 : 0);
      octants[k] = (octants[k] ?? 0) + 1;
    }
    for (const count of octants) expect(Math.abs(count - n / 8) / (n / 8)).toBeLessThan(0.05);
  });

  it("spaces points evenly (no clumps, no gaps)", () => {
    const n = 2000;
    const expected = Math.sqrt((4 * Math.PI) / n); // mean spacing on a unit sphere
    for (const [jitter, maxCv] of [
      [0, 0.12],
      [0.35, 0.3],
    ] as const) {
      const nn = nearestNeighbourDistances(fibonacciSphere(n, jitter, createRng(3)));
      expect(cv(nn)).toBeLessThan(maxCv);
      expect(Math.min(...nn)).toBeGreaterThan(expected * 0.3);
      expect(mean(nn)).toBeGreaterThan(expected * 0.75);
      expect(mean(nn)).toBeLessThan(expected * 1.15);
    }
  });
});

describe("buildParticleAttributes", () => {
  const attrs = buildParticleAttributes(cfg(20_000));

  it("allocates one entry per particle for every attribute", () => {
    expect(attrs.count).toBe(20_000);
    expect(attrs.position.length).toBe(60_000);
    expect(attrs.seed.length).toBe(80_000);
    expect(attrs.tone.length).toBe(20_000);
    expect(attrs.size.length).toBe(20_000);
  });

  it("places shell points on the unit sphere and halo points just outside it", () => {
    let halo = 0;
    for (let i = 0; i < attrs.count; i++) {
      const r = Math.hypot(attrs.position[i * 3]!, attrs.position[i * 3 + 1]!, attrs.position[i * 3 + 2]!);
      expect(r).toBeGreaterThanOrEqual(1 - 1e-5);
      expect(r).toBeLessThanOrEqual(1 + BLUEGOLD_CONFIG.haloDepth + 1e-5);
      if (r > 1 + 1e-4) halo++;
    }
    expect(Math.abs(halo / attrs.count - BLUEGOLD_CONFIG.haloFraction)).toBeLessThan(0.01);
  });

  it("colours about goldFraction of particles champagne gold", () => {
    const gold = Array.from(attrs.tone).filter((t) => t > GOLD_TONE_THRESHOLD).length;
    expect(Math.abs(gold / attrs.count - BLUEGOLD_CONFIG.goldFraction)).toBeLessThan(0.01);
    expect(Array.from(attrs.tone).every((t) => (t >= 0 && t <= 0.08) || (t >= 0.86 && t <= 1))).toBe(true);
  });

  it("keeps sizes and seeds in range", () => {
    expect(Math.min(...attrs.size)).toBeGreaterThanOrEqual(BLUEGOLD_CONFIG.sizeMin);
    expect(Math.max(...attrs.size)).toBeLessThanOrEqual(BLUEGOLD_CONFIG.sizeMax);
    expect(Array.from(attrs.seed).every((s) => s >= 0 && s < 1)).toBe(true);
  });

  it("is deterministic for a seed", () => {
    const a = buildParticleAttributes(cfg(500));
    const b = buildParticleAttributes(cfg(500));
    const c = buildParticleAttributes(cfg(500, { seed: 1 }));
    expect(b.position).toEqual(a.position);
    expect(b.tone).toEqual(a.tone);
    expect(c.position).not.toEqual(a.position);
  });
});

describe("geometry and material", () => {
  it("uploads static buffers with the shader's attribute names and a safe bounding sphere", () => {
    const geometry = createParticleGeometry(buildParticleAttributes(cfg(100)), 5);
    expect(Object.keys(geometry.attributes).sort()).toEqual([...ATTRIBUTE_NAMES].sort());
    expect(geometry.getAttribute("position").itemSize).toBe(3);
    expect(geometry.getAttribute("aSeed").itemSize).toBe(4);
    expect(geometry.getAttribute("aTone").itemSize).toBe(1);
    expect(geometry.getAttribute("aSize").itemSize).toBe(1);
    expect(geometry.boundingSphere?.radius).toBe(5);
  });

  it("bounding radius covers radius, halo, turbulence and pointer push", () => {
    const r = maxWorldRadius(cfg(1), BLUEGOLD_LOOK);
    expect(r).toBeGreaterThan(BLUEGOLD_LOOK.radius * (1 + BLUEGOLD_CONFIG.haloDepth) + BLUEGOLD_LOOK.turbulence);
  });

  it("builds an additive, transparent, depth-write-free material driven by uniforms", () => {
    const system = createParticleSystem({
      config: cfg(100),
      look: BLUEGOLD_LOOK,
      colors: { blue: "#3D8BFF", gold: "#E6C98F" },
      pixelRatio: 2,
      motion: 1,
    });
    const m = system.material;
    expect(m.blending).toBe(AdditiveBlending);
    expect(m.transparent).toBe(true);
    expect(m.depthWrite).toBe(false);
    expect(m.uniforms).toBe(system.uniforms);
    expect(system.uniforms.uPixelRatio.value).toBe(2);
    // Colours are stored in linear RGB (sRGB #3D8BFF -> linear blue channel 1, red < 0.06).
    expect(system.uniforms.uColorBlue.value.b).toBeCloseTo(1, 5);
    expect(system.uniforms.uColorBlue.value.r).toBeLessThan(0.06);
    expect(m.fragmentShader).not.toContain("undefinedSymbol");
  });

  it("dispose() frees both GPU resources", () => {
    const system = createParticleSystem({
      config: cfg(10),
      look: BLUEGOLD_LOOK,
      colors: { blue: "#3D8BFF", gold: "#E6C98F" },
      pixelRatio: 1,
      motion: 0,
    });
    const g = vi.spyOn(system.geometry, "dispose");
    const m = vi.spyOn(system.material, "dispose");
    system.dispose();
    expect(g).toHaveBeenCalledOnce();
    expect(m).toHaveBeenCalledOnce();
  });

  it("can build a deliberately broken shader for the ?scene=fail diagnostic", () => {
    const system = createParticleSystem({
      config: cfg(10),
      look: BLUEGOLD_LOOK,
      colors: { blue: "#3D8BFF", gold: "#E6C98F" },
      pixelRatio: 1,
      motion: 1,
      broken: true,
    });
    expect(system.material.fragmentShader).toContain("undefinedSymbol");
  });
});
