import type { Rng } from "./random";

export const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Fibonacci (golden-angle) sphere: `count` unit vectors spread evenly with no
 * clustering at the poles, in O(n). Point i sits at height
 * y = 1 - 2(i + 0.5)/n on a spiral turned by the golden angle each step.
 *
 * `jitter` (0..1, a fraction of the mean point spacing) nudges each point
 * along the surface with the given RNG so the spiral pattern is not visible;
 * points stay exactly on the unit sphere.
 */
export function fibonacciSphere(count: number, jitter = 0, rng?: Rng): Float32Array {
  const out = new Float32Array(count * 3);
  // Mean angular spacing between neighbours on a unit sphere of n points.
  const spacing = Math.sqrt((4 * Math.PI) / Math.max(count, 1));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = GOLDEN_ANGLE * i;
    let x = Math.cos(theta) * r;
    let z = Math.sin(theta) * r;
    let yy = y;
    if (jitter > 0 && rng) {
      // Offset in a random tangent direction, then re-project to the sphere.
      const a = rng() * Math.PI * 2;
      const d = jitter * spacing * Math.sqrt(rng());
      // Tangent basis at (x, yy, z).
      let tx = -z;
      let ty = 0;
      let tz = x;
      const tl = Math.hypot(tx, ty, tz);
      if (tl < 1e-6) {
        tx = 1;
        ty = 0;
        tz = 0;
      } else {
        tx /= tl;
        tz /= tl;
      }
      const bx = yy * tz - z * ty;
      const by = z * tx - x * tz;
      const bz = x * ty - yy * tx;
      x += (Math.cos(a) * tx + Math.sin(a) * bx) * d;
      yy += (Math.cos(a) * ty + Math.sin(a) * by) * d;
      z += (Math.cos(a) * tz + Math.sin(a) * bz) * d;
      const l = Math.hypot(x, yy, z);
      x /= l;
      yy /= l;
      z /= l;
    }
    out[i * 3] = x;
    out[i * 3 + 1] = yy;
    out[i * 3 + 2] = z;
  }
  return out;
}
