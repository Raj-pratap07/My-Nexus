/** Geometry-level configuration of a particle sphere (changing it rebuilds buffers). */
export interface ParticleConfig {
  /** Number of particles (one vertex each). */
  count: number;
  /** Seed for every random choice: same seed, same sphere. */
  seed: number;
  /** Random tangential offset of shell points, as a fraction of point spacing. */
  jitter: number;
  /** Fraction of particles placed in a thin outer halo instead of on the shell. */
  haloFraction: number;
  /** Halo thickness: halo radii lie in (1, 1 + haloDepth]. */
  haloDepth: number;
  /** Fraction of champagne-gold particles (the rest are electric blue). */
  goldFraction: number;
  /** Per-particle size multiplier range. */
  sizeMin: number;
  sizeMax: number;
}

/** Visual (uniform-level) parameters: changing them never rebuilds buffers. */
export interface ParticleLook {
  radius: number;
  turbulence: number;
  pointSize: number;
  opacity: number;
  pointerRadius: number;
}

export const BLUEGOLD_CONFIG: Omit<ParticleConfig, "count"> = {
  seed: 20_260_511,
  jitter: 0.35,
  haloFraction: 0.08,
  haloDepth: 0.22,
  goldFraction: 0.22,
  sizeMin: 0.55,
  sizeMax: 1.6,
};

export const BLUEGOLD_LOOK: ParticleLook = {
  radius: 2.2,
  turbulence: 0.11,
  pointSize: 3.6,
  opacity: 0.78,
  pointerRadius: 1.15,
};
