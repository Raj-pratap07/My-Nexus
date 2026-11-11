import { BufferAttribute, BufferGeometry, Sphere, StaticDrawUsage, Vector3 } from "three";
import type { ParticleAttributes } from "./attributes";

/** Shader attribute names (must match particle.vert.glsl; a test checks this). */
export const ATTRIBUTE_NAMES = ["position", "aSeed", "aTone", "aSize"] as const;

/**
 * One BufferGeometry for the whole sphere. Every buffer is static: it is
 * uploaded to the GPU once and never rewritten from JavaScript.
 *
 * The vertex shader scales and displaces positions, so the bounding sphere
 * is set explicitly to cover the largest possible displacement; three would
 * otherwise compute it from the unit-space positions and cull wrongly.
 */
export function createParticleGeometry(
  attrs: ParticleAttributes,
  maxWorldRadius: number,
): BufferGeometry {
  const geometry = new BufferGeometry();
  const add = (name: string, array: Float32Array, size: number) => {
    const attribute = new BufferAttribute(array, size);
    attribute.setUsage(StaticDrawUsage);
    geometry.setAttribute(name, attribute);
  };
  add("position", attrs.position, 3);
  add("aSeed", attrs.seed, 4);
  add("aTone", attrs.tone, 1);
  add("aSize", attrs.size, 1);
  geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), maxWorldRadius);
  return geometry;
}
