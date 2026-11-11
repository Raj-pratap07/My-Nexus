import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SceneRootProps } from "./types";

/** Make WebGL "available" to detectWebGL (jsdom has none). */
function fakeWebGL(available: boolean) {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() =>
    available ? { getExtension: () => ({ loseContext() {} }) } : null) as never);
}

type ModuleFactory = Parameters<typeof vi.doMock<typeof import("./webgl/SceneRoot")>>[1];

async function loadLayer(sceneModule: NonNullable<ModuleFactory>) {
  vi.resetModules();
  vi.doMock("./webgl/SceneRoot", sceneModule);
  return (await import("./SceneLayer")).SceneLayer;
}

let received: SceneRootProps | null = null;
function RecordingScene(props: SceneRootProps) {
  received = props;
  return <div data-testid="webgl-scene" />;
}

beforeEach(() => {
  received = null;
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  vi.doUnmock("./webgl/SceneRoot");
  vi.restoreAllMocks();
});

describe("SceneLayer", () => {
  it("shows the static fallback and never loads the WebGL chunk without WebGL", async () => {
    fakeWebGL(false);
    const loaded = vi.fn();
    const SceneLayer = await loadLayer(() => {
      loaded();
      return { default: RecordingScene };
    });
    render(<SceneLayer />);
    expect(screen.getByTestId("scene-fallback")).toBeInTheDocument();
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", "unsupported");
    expect(loaded).not.toHaveBeenCalled();
  });

  it("loads the WebGL scene lazily when WebGL is available", async () => {
    fakeWebGL(true);
    const SceneLayer = await loadLayer(() => ({ default: RecordingScene }));
    render(<SceneLayer />);
    expect(await screen.findByTestId("webgl-scene")).toBeInTheDocument();
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", "webgl");
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("aria-hidden", "true");
    expect(received).toMatchObject({ mode: "auto", reducedMotion: false, active: true, initialLevel: 0 });
  });

  it("falls back when the scene chunk fails to load", async () => {
    fakeWebGL(true);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    const SceneLayer = await loadLayer(() => {
      throw new Error("Failed to fetch dynamically imported module");
    });
    render(<SceneLayer />);
    expect(await screen.findByTestId("scene-fallback")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", "crashed"),
    );
  });

  it("falls back when the scene throws while rendering", async () => {
    fakeWebGL(true);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    const SceneLayer = await loadLayer(() => ({
      default: () => {
        throw new Error("renderer could not be created");
      },
    }));
    render(<SceneLayer />);
    expect(await screen.findByTestId("scene-fallback")).toBeInTheDocument();
  });

  it.each(["context-lost", "shader-error"] as const)(
    "replaces the canvas with the fallback on %s",
    async (reason) => {
      fakeWebGL(true);
      vi.spyOn(console, "warn").mockImplementation(() => {});
      function FailingScene(props: SceneRootProps) {
        useEffect(() => props.onFailure(reason), [props]);
        return <div data-testid="webgl-scene" />;
      }
      const SceneLayer = await loadLayer(() => ({ default: FailingScene }));
      render(<SceneLayer />);
      await waitFor(() =>
        expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", reason),
      );
      expect(screen.getByTestId("scene-fallback")).toBeInTheDocument();
      expect(screen.queryByTestId("webgl-scene")).not.toBeInTheDocument();
    },
  );

  it("honours ?scene=off and forwards diagnostics and overrides", async () => {
    fakeWebGL(true);
    window.history.replaceState(null, "", "/?scene=off");
    let SceneLayer = await loadLayer(() => ({ default: RecordingScene }));
    const { unmount } = render(<SceneLayer />);
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-scene", "disabled");
    unmount();

    window.history.replaceState(null, "", "/?scene=lose&particles=9000");
    SceneLayer = await loadLayer(() => ({ default: RecordingScene }));
    render(<SceneLayer />);
    await screen.findByTestId("webgl-scene");
    expect(received).toMatchObject({ mode: "lose", particlesOverride: 9000 });
  });

  it("passes reduced motion and page visibility through to the scene", async () => {
    fakeWebGL(true);
    vi.spyOn(window, "matchMedia").mockImplementation(
      (q: string) =>
        ({
          matches: q.includes("reduce"),
          media: q,
          addEventListener: () => {},
          removeEventListener: () => {},
        }) as unknown as MediaQueryList,
    );
    const SceneLayer = await loadLayer(() => ({ default: RecordingScene }));
    render(<SceneLayer />);
    await screen.findByTestId("webgl-scene");
    expect(received?.reducedMotion).toBe(true);

    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(received?.active).toBe(false);
    visibility.mockReturnValue("visible");
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(received?.active).toBe(true);
  });

  it("exposes the active quality on the layer for QA", async () => {
    fakeWebGL(true);
    function ReportingScene(props: SceneRootProps) {
      useEffect(() => props.onQualityChange?.({ particles: 20_000, dpr: 2 }), [props]);
      return <div data-testid="webgl-scene" />;
    }
    const SceneLayer = await loadLayer(() => ({ default: ReportingScene }));
    render(<SceneLayer />);
    await waitFor(() =>
      expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-particles", "20000"),
    );
    expect(screen.getByTestId("scene-layer")).toHaveAttribute("data-dpr", "2");
  });
});
