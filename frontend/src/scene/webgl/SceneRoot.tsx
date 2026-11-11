/**
 * The WebGL scene: the lazily loaded chunk boundary. Everything that imports
 * three.js or React Three Fiber is reached only through this module.
 */
import { Canvas, type GLProps } from "@react-three/fiber";
import { useCallback, useEffect, useRef, useState } from "react";
import { WebGLRenderer } from "three";
import {
  BENCH_LEVELS,
  BENCH_MEASURE_MS,
  BENCH_WARMUP_MS,
  type BenchLevelResult,
  type FrameStats,
} from "../bench";
import { LOWEST_LEVEL, QUALITY_LADDER, effectiveDpr } from "../quality";
import type { SceneRootProps } from "../types";
import { BenchRunner, QualityMonitor } from "./monitors";
import { ParticleSphere } from "./ParticleSphere";
import { describeRenderer, watchContextLoss, watchShaderErrors, type Cleanup } from "./safety";

const LOSE_CONTEXT_AFTER_MS = 3000;

/** What R3F passes to a renderer factory, taken from its own GLProps type. */
type RendererDefaults = Parameters<Extract<GLProps, (defaults: never) => unknown>>[0];

export default function SceneRoot(props: SceneRootProps) {
  const { mode, reducedMotion, active, particlesOverride, onFailure, onQualityChange, onBenchReport } =
    props;
  const bench = mode === "bench";

  const [level, setLevel] = useState<number>(() => (bench ? BENCH_LEVELS[0] : props.initialLevel));
  const [benchStep, setBenchStep] = useState(0);
  const benchResults = useRef<BenchLevelResult[]>([]);
  const gpu = useRef<ReturnType<typeof describeRenderer> | null>(null);
  const cleanups = useRef<Cleanup[]>([]);

  // Callbacks used inside the renderer factory must not go stale.
  const failRef = useRef(onFailure);
  failRef.current = onFailure;

  const quality = QUALITY_LADDER[level] ?? QUALITY_LADDER[LOWEST_LEVEL]!;
  // An explicit ?particles= override applies only at the top level; once the
  // monitor steps down, the ladder decides.
  const particles = !bench && particlesOverride !== null && level === props.initialLevel
    ? particlesOverride
    : quality.particles;
  const dpr = effectiveDpr(quality, typeof window === "undefined" ? 1 : window.devicePixelRatio);

  useEffect(() => onQualityChange?.({ particles, dpr }), [particles, dpr, onQualityChange]);
  useEffect(() => () => cleanups.current.splice(0).forEach((c) => c()), []);

  // Created by us (not R3F) so the failure hooks exist before the first compile.
  const createRenderer = useCallback((defaults: RendererDefaults) => {
    const renderer = new WebGLRenderer({
      ...(defaults as ConstructorParameters<typeof WebGLRenderer>[0]),
      antialias: false,
      alpha: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    cleanups.current.push(
      watchContextLoss(renderer.domElement, () => failRef.current("context-lost")),
      watchShaderErrors(renderer, () => failRef.current("shader-error")),
    );
    gpu.current = describeRenderer(renderer.getContext());
    if (mode === "lose") {
      const id = window.setTimeout(
        () => renderer.getContext().getExtension("WEBGL_lose_context")?.loseContext(),
        LOSE_CONTEXT_AFTER_MS,
      );
      cleanups.current.push(() => window.clearTimeout(id));
    }
    return renderer;
    // The factory runs once per Canvas; mode never changes for a page load.
  }, [mode]);

  const stepDown = useCallback(() => setLevel((l) => Math.min(l + 1, LOWEST_LEVEL)), []);

  const onBenchLevel = useCallback(
    (stats: FrameStats) => {
      benchResults.current.push({ ...stats, particles, dpr });
      const next = benchStep + 1;
      const nextLevel = BENCH_LEVELS[next];
      if (nextLevel !== undefined) {
        setBenchStep(next);
        setLevel(nextLevel);
        return;
      }
      onBenchReport?.({
        levels: benchResults.current,
        gpu: gpu.current,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
          devicePixelRatio: window.devicePixelRatio,
        },
        userAgent: navigator.userAgent,
        warmupMs: BENCH_WARMUP_MS,
        measureMs: BENCH_MEASURE_MS,
      });
    },
    [benchStep, particles, dpr, onBenchReport],
  );

  const frameloop = bench ? "always" : !active ? "never" : reducedMotion ? "demand" : "always";

  return (
    <Canvas
      gl={createRenderer}
      dpr={dpr}
      flat
      frameloop={frameloop}
      camera={{ position: [0, 0, 6], fov: 45, near: 0.1, far: 50 }}
      style={{ position: "absolute", inset: 0 }}
    >
      <ParticleSphere particles={particles} reducedMotion={reducedMotion && !bench} broken={mode === "fail"} />
      {!bench && !reducedMotion && level < LOWEST_LEVEL && (
        <QualityMonitor key={`q${level}`} onSlow={stepDown} />
      )}
      {bench && <BenchRunner key={`b${benchStep}`} onDone={onBenchLevel} />}
    </Canvas>
  );
}
