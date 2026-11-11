import { SceneLayer } from "../../scene/SceneLayer";
import { hero } from "./content";
import { EnterNexusLink } from "./EnterNexusLink";

/**
 * The hero hosts the 3D scene in its own box, so scrolling the hero off
 * screen pauses rendering. The text is ordinary HTML above the canvas and
 * works with or without WebGL. A soft scrim keeps it readable over particles.
 */
export function Hero({ onLeave }: { onLeave?: () => void }) {
  return (
    <section
      id="top"
      aria-labelledby="hero-title"
      className="relative isolate -mt-[76px] flex min-h-[100svh] items-end overflow-hidden pt-[76px] md:landscape:items-center"
    >
      {/* Soft light behind the sphere: a CSS gradient, so no extra WebGL pass. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-20 bg-[radial-gradient(38%_42%_at_50%_24%,rgb(61_139_255/0.16),transparent_70%)] md:landscape:bg-[radial-gradient(30%_46%_at_70%_50%,rgb(61_139_255/0.16),rgb(230_201_143/0.03)_45%,transparent_72%)]"
      />
      <SceneLayer className="absolute inset-0 -z-10" />
      {/* Readability scrim: dark behind the text only, transparent over the sphere. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-[5] bg-[linear-gradient(to_top,rgb(5_7_13/0.96)_0%,rgb(5_7_13/0.7)_38%,transparent_70%)] md:landscape:bg-[linear-gradient(to_right,rgb(5_7_13/0.92)_0%,rgb(5_7_13/0.62)_38%,transparent_62%)]"
      />
      <div className="mx-auto w-full max-w-7xl px-5 pb-16 sm:px-8 md:landscape:pb-0">
        <div className="max-w-xl">
          <p className="mb-5 inline-flex items-center gap-2 rounded-full border border-navy-2 bg-navy/70 px-3.5 py-1.5 text-xs font-medium tracking-wide text-champagne backdrop-blur-sm">
            <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-electric-glow" />
            {hero.eyebrow}
          </p>
          <h1
            id="hero-title"
            className="font-display text-[2.35rem] leading-[1.05] font-semibold tracking-[-0.035em] text-ink sm:text-5xl md:text-6xl"
          >
            {hero.title}
          </h1>
          <p className="mt-6 max-w-lg text-base leading-relaxed text-ink-muted sm:text-lg">
            {hero.body}
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <EnterNexusLink
              onLeave={onLeave}
              className="inline-flex items-center rounded-full bg-champagne px-6 py-3 text-sm font-semibold text-abyss shadow-[0_0_32px_-6px_rgb(230_201_143/0.6)] transition-[transform,background-color] duration-200 ease-[var(--ease-premium)] hover:-translate-y-0.5 hover:bg-[#EFD7A6]"
            >
              {hero.primaryCta.label}
            </EnterNexusLink>
            <a
              href={hero.secondaryCta.href}
              className="inline-flex items-center rounded-full border border-electric/50 px-6 py-3 text-sm font-semibold text-electric-glow transition-colors duration-200 hover:border-electric-glow hover:bg-electric/10"
            >
              {hero.secondaryCta.label}
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
