/**
 * Structural guarantees that keep the page usable without WebGL and keep the
 * render loop on the GPU. Checked on the source text.
 */
import { describe, expect, it } from "vitest";

const sources = import.meta.glob<string>("/src/**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
});
const productionFiles = Object.entries(sources).filter(
  ([path]) => !/\.test\.tsx?$/.test(path) && !path.startsWith("/src/test/"),
);

describe("chunk boundary", () => {
  it("only src/scene/webgl/ imports three.js or React Three Fiber", () => {
    const offenders = productionFiles
      .filter(([path]) => !path.startsWith("/src/scene/webgl/"))
      .filter(([, src]) => /from\s+["'](three|@react-three\/fiber)["']/.test(src))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });

  it("nothing outside webgl/ imports webgl/ statically (only SceneLayer, lazily)", () => {
    const offenders = productionFiles
      .filter(([path]) => !path.startsWith("/src/scene/webgl/"))
      .filter(([, src]) => /^\s*import[^(]*from\s+["'][^"']*\/webgl\//m.test(src))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
    expect(sources["/src/scene/SceneLayer.tsx"]).toMatch(/import\(\s*["']\.\/webgl\/SceneRoot["']\s*\)/);
  });
});

describe("GPU-driven animation", () => {
  const sphere = sources["/src/scene/webgl/ParticleSphere.tsx"]!;

  it("renders the particles as a single THREE.Points object", () => {
    expect(sphere.match(/<points\b/g)).toHaveLength(1);
  });

  it("never rewrites particle buffers or React state per frame", () => {
    const frame = sphere.slice(sphere.indexOf("useFrame("));
    for (const forbidden of ["needsUpdate", "setAttribute", ".array[", "setState", "useState"]) {
      expect(frame, forbidden).not.toContain(forbidden);
    }
    expect(frame).toMatch(/uniforms/);
  });

  it("uses no post-processing passes", () => {
    const all = productionFiles.map(([, src]) => src).join("\n");
    for (const pass of ["EffectComposer", "UnrealBloomPass", "postprocessing"]) {
      expect(all).not.toContain(pass);
    }
  });
});

/** Static (non-dynamic) relative imports of a source file, resolved to /src paths. */
function staticImports(path: string): string[] {
  const src = sources[path] ?? "";
  const dir = path.slice(0, path.lastIndexOf("/"));
  const out: string[] = [];
  for (const m of src.matchAll(/^\s*(?:import|export)\s[^;]*?from\s+["'](\.[^"']+)["']/gm)) {
    const stack: string[] = [];
    for (const part of `${dir}/${m[1]}`.split("/")) {
      if (part === "..") stack.pop();
      else if (part !== ".") stack.push(part);
    }
    const base = stack.join("/");
    const hit = [base, `${base}.ts`, `${base}.tsx`, `${base}/index.ts`].find((c) => c in sources);
    if (hit) out.push(hit);
  }
  return out;
}

function reachable(entry: string): Set<string> {
  const seen = new Set<string>([entry]);
  const queue = [entry];
  while (queue.length) {
    for (const next of staticImports(queue.shift()!)) {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push(next);
      }
    }
  }
  return seen;
}

describe("workspace / 3D separation", () => {
  it("the workspace chunk cannot reach src/scene or three.js through static imports", () => {
    const graph = reachable("/src/features/workspace/WorkspacePage.tsx");
    expect(graph.size).toBeGreaterThan(10);
    expect([...graph].filter((p) => p.startsWith("/src/scene/"))).toEqual([]);
    for (const p of graph) expect(sources[p], p).not.toMatch(/from\s+["'](three|@react-three\/fiber)["']/);
  });

  it("the API layer is independent of UI, scene and workspace code", () => {
    const graph = reachable("/src/api/conversations.ts");
    expect([...graph].every((p) => p.startsWith("/src/api/"))).toBe(true);
  });

  it("the workspace is only loaded lazily", () => {
    const offenders = productionFiles
      .filter(([path]) => !path.startsWith("/src/features/workspace/"))
      .filter(([, src]) => /^\s*import[^(]*from\s+["'][^"']*features\/workspace/m.test(src))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
    expect(sources["/src/app/lazyWorkspace.ts"]).toMatch(
      /import\(\s*["']\.\.\/features\/workspace\/WorkspacePage["']\s*\)/,
    );
  });
});

describe("credential isolation in browser code", () => {
  it("never sets an Authorization header, reads process.env/import.meta.env or names the token variable", () => {
    for (const [path, src] of productionFiles) {
      expect(src, path).not.toMatch(/["']authorization["']|authorization\s*:/i);
      expect(src, path).not.toMatch(/process\.env/);
      expect(src, path).not.toMatch(/NEXUS_DEV_TOKEN/);
      expect(src, path).not.toMatch(/import\.meta\.env/);
    }
  });

  it("never injects HTML", () => {
    for (const [path, src] of productionFiles) {
      expect(src, path).not.toMatch(/dangerouslySetInnerHTML|\.innerHTML\s*=|insertAdjacentHTML/);
    }
  });

  it("talks to the backend only through the same-origin /api proxy", () => {
    for (const [path, src] of productionFiles) {
      expect(src, path).not.toMatch(/https?:\/\/(127\.0\.0\.1|localhost):8000/);
      if (path !== "/src/api/client.ts") expect(src, path).not.toMatch(/\bfetch\(/);
    }
  });
});
