import { AdditiveBlending, ShaderMaterial } from "three";
import noise from "./shaders/noise.glsl?raw";
import fragmentShader from "./shaders/particle.frag.glsl?raw";
import vertexBody from "./shaders/particle.vert.glsl?raw";
import type { ParticleUniforms } from "./uniforms";

/** The vertex shader as compiled: noise functions + particle stage. */
export const VERTEX_SHADER = `${noise}\n${vertexBody}`;
export const FRAGMENT_SHADER = fragmentShader;

/** A fragment shader that cannot compile; used only by ?scene=fail. */
export const BROKEN_FRAGMENT_SHADER = "void main() { gl_FragColor = vec4(undefinedSymbol); }";

export interface MaterialOptions {
  uniforms: ParticleUniforms;
  /** Diagnostics only: compile a broken fragment shader. */
  broken?: boolean;
}

/**
 * Additive, transparent, no depth writes: particles glow and overlap without
 * sorting, in a single draw call.
 */
export function createParticleMaterial({ uniforms, broken = false }: MaterialOptions) {
  return new ShaderMaterial({
    uniforms: uniforms as unknown as ShaderMaterial["uniforms"],
    vertexShader: VERTEX_SHADER,
    fragmentShader: broken ? BROKEN_FRAGMENT_SHADER : FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: AdditiveBlending,
    toneMapped: false,
  });
}
