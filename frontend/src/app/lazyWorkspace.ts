/**
 * The workspace is its own lazily loaded chunk. It never imports src/scene,
 * so entering it does not download or start three.js.
 */
import { lazy } from "react";

export const loadWorkspace = () => import("../features/workspace/WorkspacePage");

let preloaded = false;
/** Warm the chunk (on hover/focus of "Enter Nexus"); load errors surface later. */
export function preloadWorkspace(): void {
  if (preloaded) return;
  preloaded = true;
  loadWorkspace().catch(() => {
    preloaded = false;
  });
}

export const WorkspacePage = lazy(loadWorkspace);
