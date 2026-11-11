import { describe, expect, it } from "vitest";
import css from "../styles/index.css?raw";
import { contrastRatio, relativeLuminance } from "./contrast";
import { colors, particleColors, textPairs } from "./tokens";

describe("design tokens", () => {
  it("Tailwind @theme colours mirror tokens.ts exactly", () => {
    const themed = new Map<string, string>();
    for (const m of css.matchAll(/--color-([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) {
      themed.set(m[1]!, m[2]!.toUpperCase());
    }
    const expected = new Map(Object.entries(colors).map(([k, v]) => [k, v.toUpperCase()]));
    expect(themed).toEqual(expected);
  });

  it("uses Space Grotesk for display and body text", () => {
    expect(css).toMatch(/--font-display:\s*"Space Grotesk Variable"/);
    expect(css).toMatch(/--font-sans:\s*"Space Grotesk Variable"/);
  });

  it("particles use the electric-blue and champagne-gold tokens", () => {
    expect(particleColors).toEqual({ blue: colors.electric, gold: colors.champagne });
  });

  it("contrast maths matches known WCAG values", () => {
    expect(contrastRatio("#000000", "#FFFFFF")).toBeCloseTo(21, 5);
    expect(contrastRatio("#777777", "#FFFFFF")).toBeCloseTo(4.48, 2);
    expect(relativeLuminance("#FFFFFF")).toBeCloseTo(1, 6);
  });

  it.each(textPairs)("text %s on %s meets WCAG AA (4.5:1)", (fg, bg) => {
    expect(contrastRatio(colors[fg], colors[bg])).toBeGreaterThanOrEqual(4.5);
  });
});
