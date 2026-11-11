import type { ParticleConfig } from "./config";
import { fibonacciSphere } from "./distribution";
import { createRng } from "./random";

/** Typed arrays uploaded once to GPU buffers. Positions are in unit-sphere space. */
export interface ParticleAttributes {
  count: number;
  /** vec3: unit shell points, or halo points at radius (1, 1 + haloDepth]. */
  position: Float32Array;
  /** vec4 of independent [0,1) values: noise phase, breathing, twinkle. */
  seed: Float32Array;
  /** float: 0..0.08 for electric blue, 0.86..1 for champagne gold. */
  tone: Float32Array;
  /** float: per-particle size multiplier in [sizeMin, sizeMax]. */
  size: Float32Array;
}

export const GOLD_TONE_THRESHOLD = 0.5;

export function buildParticleAttributes(config: ParticleConfig): ParticleAttributes {
  const { count } = config;
  const rng = createRng(config.seed);
  const position = fibonacciSphere(count, config.jitter, rng);
  const seed = new Float32Array(count * 4);
  const tone = new Float32Array(count);
  const size = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    if (rng() < config.haloFraction) {
      const r = 1 + config.haloDepth * (0.15 + 0.85 * rng());
      position[i * 3] = (position[i * 3] ?? 0) * r;
      position[i * 3 + 1] = (position[i * 3 + 1] ?? 0) * r;
      position[i * 3 + 2] = (position[i * 3 + 2] ?? 0) * r;
    }
    for (let k = 0; k < 4; k++) seed[i * 4 + k] = rng();
    const gold = rng() < config.goldFraction;
    tone[i] = gold ? 0.86 + 0.14 * rng() : 0.08 * rng();
    // Bias toward small particles (a few larger "stars").
    const s = rng();
    size[i] = config.sizeMin + (config.sizeMax - config.sizeMin) * s * s;
  }
  return { count, position, seed, tone, size };
}
