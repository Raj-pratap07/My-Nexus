/**
 * Where the sphere sits for a given canvas size (world units; camera at z = 6,
 * fov 45 => the visible half-height is ~2.49). Pure, so it is unit-tested.
 *   portrait (any width):        stacked above the text (text at the bottom);
 *                                larger on tablets than on phones
 *   wide landscape (>= 1024 px): right of the hero text
 *   other landscape:             right of centre, smaller
 * The CSS mirrors this with Tailwind's md:landscape: variant.
 */
export interface SphereLayout {
  position: [number, number, number];
  scale: number;
  /**
   * Point-size multiplier. Particle sizes are in pixels, so a smaller sphere
   * would otherwise look denser and brighter than a large one.
   */
  pointScale: number;
}

export function sphereLayout(width: number, height: number): SphereLayout {
  const aspect = height > 0 ? width / height : 1;
  if (aspect < 1) {
    return width >= 600
      ? { position: [0, 0.95, 0], scale: 0.56, pointScale: 0.8 }
      : { position: [0, 1.05, 0], scale: 0.46, pointScale: 0.68 };
  }
  if (width >= 1024 && aspect >= 1.15) return { position: [1.75, 0, 0], scale: 0.78, pointScale: 1 };
  return { position: [1.3, 0, 0], scale: 0.66, pointScale: 0.9 };
}
