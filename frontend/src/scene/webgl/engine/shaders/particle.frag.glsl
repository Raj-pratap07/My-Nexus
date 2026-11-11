// BLUEGOLD particle sphere: fragment stage. A soft round particle with a
// bright core and a faint halo; with additive blending this gives the glow
// without any post-processing pass.

uniform float uOpacity;
uniform vec3 uColorBlue; // linear RGB
uniform vec3 uColorGold; // linear RGB

varying float vTone;
varying float vAlpha;
varying float vGlow;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  if (d > 0.5) discard;

  float core = exp(-d * d * 16.0);
  float halo = (1.0 - smoothstep(0.0, 0.5, d)) * 0.45;
  float alpha = (core + halo) * vAlpha * uOpacity;
  if (alpha < 0.003) discard;

  vec3 color = mix(uColorBlue, uColorGold, vTone);
  color += vGlow * 0.35 * mix(uColorBlue, vec3(1.0), 0.5);

  gl_FragColor = vec4(color, alpha);
  #include <colorspace_fragment>
}
