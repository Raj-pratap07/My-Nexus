import type { BenchReport } from "./bench";
import type { SceneMode } from "./params";

/** Why the 3D scene gave way to the static fallback. */
export type SceneFailure = "unsupported" | "disabled" | "context-lost" | "shader-error" | "crashed";

export interface ActiveQuality {
  particles: number;
  dpr: number;
}

/** Props of the lazily loaded WebGL scene (the chunk boundary's contract). */
export interface SceneRootProps {
  reducedMotion: boolean;
  /** False while the tab is hidden or the scene is off screen: no rendering at all. */
  active: boolean;
  mode: SceneMode;
  particlesOverride: number | null;
  initialLevel: number;
  onFailure: (reason: Extract<SceneFailure, "context-lost" | "shader-error">) => void;
  onQualityChange?: (quality: ActiveQuality) => void;
  onBenchReport?: (report: BenchReport) => void;
}
