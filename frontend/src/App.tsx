import { Suspense, useEffect } from "react";
import { ChunkBoundary } from "./app/ChunkBoundary";
import { Link } from "./app/Link";
import { WorkspacePage } from "./app/lazyWorkspace";
import { parseRoute, paths, usePathname } from "./app/router";
import { LandingPage } from "./features/landing/LandingPage";

/**
 * Application shell. The landing page (with the 3D scene) and the workspace
 * are separate routes; the workspace is a lazy chunk with no three.js.
 */
export function App() {
  const route = parseRoute(usePathname());

  useEffect(() => {
    document.title = route.name === "workspace" ? "Workspace · My-Nexus" : "My-Nexus";
  }, [route.name]);

  if (route.name === "landing") return <LandingPage />;
  if (route.name === "workspace") {
    return (
      <ChunkBoundary>
        <Suspense fallback={<WorkspaceLoading />}>
          <WorkspacePage conversationId={route.conversationId} invalidId={route.invalidId ?? false} />
        </Suspense>
      </ChunkBoundary>
    );
  }
  return (
    <main className="grid min-h-[100svh] place-items-center px-5 text-center">
      <div>
        <h1 className="font-display text-3xl font-semibold text-ink">Page not found</h1>
        <p className="mt-3 text-ink-muted">There is nothing at this address.</p>
        <div className="mt-6 flex justify-center gap-3">
          <Link href={paths.landing} className="rounded-full border border-electric/50 px-5 py-2.5 text-sm font-semibold text-electric-glow">
            Home
          </Link>
          <Link href={paths.workspace} className="rounded-full bg-champagne px-5 py-2.5 text-sm font-semibold text-abyss">
            Enter Nexus
          </Link>
        </div>
      </div>
    </main>
  );
}

function WorkspaceLoading() {
  return (
    <div role="status" className="grid min-h-[100svh] place-items-center text-sm text-ink-muted">
      Loading the workspace…
    </div>
  );
}
