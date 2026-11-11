import type { MouseEvent, ReactNode } from "react";
import { preloadWorkspace } from "../../app/lazyWorkspace";
import { isPlainClick, navigate, paths } from "../../app/router";
import { motion } from "../../design/tokens";

const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * A real link to /nexus. A plain click fades the landing page out briefly
 * (instantly with reduced motion), then navigates client-side; leaving the
 * landing page unmounts the 3D scene, which disposes its renderer.
 */
export function EnterNexusLink({
  className,
  children,
  onLeave,
}: {
  className?: string;
  children: ReactNode;
  onLeave?: () => void;
}) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    if (!isPlainClick(e)) return;
    e.preventDefault();
    preloadWorkspace();
    if (prefersReducedMotion() || !onLeave) {
      navigate(paths.workspace);
      return;
    }
    onLeave();
    window.setTimeout(() => navigate(paths.workspace), motion.base);
  };
  return (
    <a
      href={paths.workspace}
      className={className}
      onClick={handle}
      onPointerEnter={preloadWorkspace}
      onFocus={preloadWorkspace}
    >
      {children}
    </a>
  );
}
