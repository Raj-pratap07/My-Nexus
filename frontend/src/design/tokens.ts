/**
 * BLUEGOLD design tokens: the single source of truth.
 *
 * styles/index.css mirrors `colors` into Tailwind's @theme (a test keeps the
 * two identical), and the particle engine reads `particleColors`.
 * No three.js import here: this module is part of the main chunk.
 */

export const colors = {
  /** Page background: near-black with a navy cast. */
  abyss: "#05070D",
  /** Primary surface. */
  navy: "#0A1428",
  /** Raised surface / borders. */
  "navy-2": "#10203F",
  /** Electric blue accent. */
  electric: "#3D8BFF",
  /** Lighter electric blue for glows and accent text. */
  "electric-glow": "#7FB2FF",
  /** Champagne gold accent. */
  champagne: "#E6C98F",
  /** Deeper gold for hover / borders. */
  "champagne-deep": "#C9A961",
  /** Primary text. */
  ink: "#E8ECF4",
  /** Secondary text. */
  "ink-muted": "#9AA6BD",
  /** Failures and destructive actions (text and borders on dark surfaces). */
  danger: "#FF9A8F",
} as const;

export type ColorToken = keyof typeof colors;

/** Particle colours (sRGB hex; the engine converts them to linear RGB). */
export const particleColors = {
  blue: colors.electric,
  gold: colors.champagne,
} as const;

/**
 * Text/background pairs used by the UI. Each must meet WCAG AA for normal
 * text (4.5:1); tests enforce it.
 */
export const textPairs: ReadonlyArray<readonly [fg: ColorToken, bg: ColorToken]> = [
  ["ink", "abyss"],
  ["ink", "navy"],
  ["ink", "navy-2"],
  ["ink-muted", "abyss"],
  ["ink-muted", "navy"],
  ["ink-muted", "navy-2"],
  ["champagne", "abyss"],
  ["champagne", "navy"],
  ["electric-glow", "abyss"],
  ["electric-glow", "navy"],
  ["abyss", "champagne"], // primary button label
  ["danger", "abyss"],
  ["danger", "navy"],
  ["danger", "navy-2"],
  ["electric-glow", "navy-2"],
  ["champagne", "navy-2"],
];

export const motion = {
  /** UI transitions (ms). */
  fast: 150,
  base: 240,
  slow: 600,
  ease: "cubic-bezier(0.22, 1, 0.36, 1)",
} as const;
