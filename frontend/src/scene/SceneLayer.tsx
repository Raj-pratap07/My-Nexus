import { Suspense, lazy, useCallback, useMemo, useRef, useState } from "react";
import type { BenchReport } from "./bench";
import { usePageActivity } from "./env/usePageActivity";
import { usePrefersReducedMotion } from "./env/usePrefersReducedMotion";
import { useWebGLSupport } from "./env/useWebGLSupport";
import { readSceneParams } from "./params";
import { initialLevel, readDeviceSignals } from "./quality";
import { SceneBoundary } from "./SceneBoundary";
import { SceneFallback } from "./SceneFallback";
import type { ActiveQuality, SceneFailure } from "./types";

/**
 * The only door to the WebGL scene. three.js and React Three Fiber are
 * fetched as a separate chunk AFTER the page content has rendered; if that
 * chunk, the renderer, a shader or the GPU context fails, this layer shows
 * the static fallback and the page is unaffected.
 */
export const loadSceneRoot = () => import("./webgl/SceneRoot");
const LazySceneRoot = lazy(loadSceneRoot);

export interface SceneLayerProps {
  className?: string;
}

export function SceneLayer({ className = "" }: SceneLayerProps) {
  const params = useMemo(
    () => readSceneParams(typeof window === "undefined" ? "" : window.location.search),
    [],
  );
  const supported = useWebGLSupport();
  const reducedMotion = usePrefersReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const active = usePageActivity(ref);
  const level = useMemo(
    () => (typeof navigator === "undefined" ? 0 : initialLevel(readDeviceSignals(navigator))),
    [],
  );

  const [failure, setFailure] = useState<SceneFailure | null>(() =>
    !supported ? "unsupported" : params.mode === "off" ? "disabled" : null,
  );
  const [quality, setQuality] = useState<ActiveQuality | null>(null);
  const [report, setReport] = useState<BenchReport | null>(null);

  const onFailure = useCallback((reason: SceneFailure) => {
    console.warn(`3D scene disabled (${reason}); showing the static fallback.`);
    setFailure((current) => current ?? reason);
  }, []);
  const onCrash = useCallback(() => setFailure((current) => current ?? "crashed"), []);

  return (
    <>
      <div
        ref={ref}
        aria-hidden="true"
        className={`pointer-events-none ${className}`}
        data-testid="scene-layer"
        data-scene={failure ?? "webgl"}
        data-particles={quality?.particles ?? ""}
        data-dpr={quality?.dpr ?? ""}
      >
        {failure ? (
          <SceneFallback />
        ) : (
          <SceneBoundary fallback={<SceneFallback />} onError={onCrash}>
            <Suspense fallback={null}>
              <LazySceneRoot
                mode={params.mode}
                particlesOverride={params.particles}
                initialLevel={level}
                reducedMotion={reducedMotion}
                active={active}
                onFailure={onFailure}
                onQualityChange={setQuality}
                onBenchReport={setReport}
              />
            </Suspense>
          </SceneBoundary>
        )}
      </div>
      {report && (
        <output
          id="bench-result"
          data-done="true"
          className="fixed bottom-3 left-3 z-50 max-h-[45vh] max-w-[min(92vw,640px)] overflow-auto rounded-lg border border-navy-2 bg-abyss/95 p-3 font-mono text-xs text-ink"
        >
          {JSON.stringify(report, null, 2)}
        </output>
      )}
    </>
  );
}
