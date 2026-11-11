import { useState } from "react";
import { CapabilityGrid } from "./CapabilityGrid";
import { Hero } from "./Hero";
import { SiteFooter } from "./SiteFooter";
import { SiteHeader } from "./SiteHeader";

export function LandingPage() {
  // Set by "Enter Nexus": fades the page out just before navigating away.
  const [leaving, setLeaving] = useState(false);
  return (
    <div
      data-leaving={leaving || undefined}
      className="min-h-[100svh] bg-transparent text-ink transition-opacity duration-[240ms] ease-[var(--ease-premium)] data-[leaving]:opacity-0"
    >
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-champagne px-4 py-2 font-semibold text-abyss focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to content
      </a>
      <SiteHeader />
      <main id="main" tabIndex={-1} className="outline-none">
        <Hero onLeave={() => setLeaving(true)} />
        <CapabilityGrid />
      </main>
      <SiteFooter />
    </div>
  );
}
