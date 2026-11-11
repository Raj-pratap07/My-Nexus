// BLUEGOLD particle sphere: vertex stage. ALL animation happens here; the CPU
// only updates the uniforms below once per frame. snoise() is prepended from
// noise.glsl by material.ts.

uniform float uTime;            // seconds, paused while the scene is paused
uniform float uMotion;          // 1 = full motion, 0 = static (reduced motion)
uniform float uRadius;          // sphere radius in world units
uniform float uTurbulence;      // displacement amplitude in world units
uniform vec3 uPointer;          // pointer target in world space (plane z = 0)
uniform float uPointerStrength; // 0..1, eased in/out on the CPU
uniform float uPointerRadius;   // pointer influence radius (world units)
uniform float uSize;            // base point size in CSS pixels at distance 6
uniform float uPixelRatio;      // renderer pixel ratio

attribute vec4 aSeed;  // independent [0,1) values per particle
attribute float aTone; // 0 = electric blue, 1 = champagne gold
attribute float aSize; // per-particle size multiplier

varying float vTone;
varying float vAlpha;
varying float vGlow;

const float TAU = 6.28318530718;

void main() {
  float t = uTime * uMotion;
  vec3 dir = normalize(position);

  // Slow breathing of the radius, phase-shifted per particle.
  float breathe = 1.0 + 0.015 * sin(t * 0.5 + aSeed.x * TAU) * uMotion;
  vec3 base = position * uRadius * breathe;

  // Turbulence: simplex noise mostly along the normal, a little tangential.
  float n = snoise(base * 0.55 + vec3(0.0, t * 0.12, aSeed.y * 10.0));
  vec3 tangent = normalize(cross(dir, vec3(0.0, 1.0, 0.0)) + vec3(1e-4));
  vec3 displaced = base + (dir + tangent * 0.35) * n * uTurbulence * uMotion;

  vec4 world = modelMatrix * vec4(displaced, 1.0);
  // Facing: +1 toward the camera, -1 away (independent of the sphere's position).
  vec3 worldNormal = normalize(mat3(modelMatrix) * dir);

  // Pointer: push front-facing particles away from the pointer in screen space.
  vec2 toPointer = world.xy - uPointer.xy;
  float influence = (1.0 - smoothstep(0.0, uPointerRadius, length(toPointer)))
                  * uPointerStrength
                  * smoothstep(-0.2, 0.6, worldNormal.z);
  world.xy += normalize(toPointer + vec2(1e-4)) * 0.45 * influence;
  world.xyz += worldNormal * 0.25 * influence;

  vec4 mvPosition = viewMatrix * world;
  gl_Position = projectionMatrix * mvPosition;

  float twinkle = 0.75 + 0.25 * sin(t * (1.5 + aSeed.z * 2.0) + aSeed.w * TAU);
  float facing = smoothstep(-1.0, 0.8, worldNormal.z);
  // Rim light: particles where the surface turns away from the camera glow
  // brighter, which gives the sphere a defined 3D silhouette.
  float rim = pow(1.0 - abs(worldNormal.z), 2.0);
  vAlpha = mix(0.22, 1.0, facing) * (0.8 + 0.9 * rim) * mix(1.0, twinkle, uMotion);
  vTone = aTone;
  vGlow = influence;

  // Gold particles are drawn slightly larger so the champagne tone carries.
  float size = uSize * aSize * uPixelRatio * (6.0 / -mvPosition.z)
             * mix(1.0, 1.35, aTone) * (1.0 + influence * 0.6);
  gl_PointSize = clamp(size, 1.0, 64.0 * uPixelRatio);
}
