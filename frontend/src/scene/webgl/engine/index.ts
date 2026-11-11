/**
 * BLUEGOLD particle engine: plain three.js, no React. Builds one GPU particle
 * system (static buffers + a ShaderMaterial driven only by uniforms).
 * Reusable outside the landing page.
 */
import type { BufferGeometry, ShaderMaterial } from "three";
import { buildParticleAttributes } from "./attributes";
import type { ParticleConfig, ParticleLook } from "./config";
import { createParticleGeometry } from "./geometry";
import { createParticleMaterial } from "./material";
import { createUniforms, type ParticleUniforms } from "./uniforms";

export { BLUEGOLD_CONFIG, BLUEGOLD_LOOK } from "./config";
export type { ParticleConfig, ParticleLook } from "./config";
export type { ParticleUniforms } from "./uniforms";

export interface ParticleSystemOptions {
  config: ParticleConfig;
  look: ParticleLook;
  colors: { blue: string; gold: string };
  pixelRatio: number;
  motion: number;
  /** Diagnostics only (?scene=fail). */
  broken?: boolean;
}

export interface ParticleSystem {
  geometry: BufferGeometry;
  material: ShaderMaterial;
  uniforms: ParticleUniforms;
  dispose(): void;
}

/** Largest world-space distance a particle can reach (for the bounding sphere). */
export function maxWorldRadius(config: ParticleConfig, look: ParticleLook): number {
  const breathing = 1.02;
  const pointerPush = 0.7;
  return look.radius * (1 + config.haloDepth) * breathing + look.turbulence * 1.4 + pointerPush;
}

export function createParticleSystem(o: ParticleSystemOptions): ParticleSystem {
  const geometry = createParticleGeometry(
    buildParticleAttributes(o.config),
    maxWorldRadius(o.config, o.look),
  );
  const uniforms = createUniforms({
    look: o.look,
    blue: o.colors.blue,
    gold: o.colors.gold,
    pixelRatio: o.pixelRatio,
    motion: o.motion,
  });
  const material = createParticleMaterial({ uniforms, broken: o.broken ?? false });
  return {
    geometry,
    material,
    uniforms,
    dispose() {
      geometry.dispose();
      material.dispose();
    },
  };
}
