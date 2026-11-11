import { brand } from "./content";

export function SiteHeader() {
  return (
    // In portrait layouts the sphere sits near the top of the hero; a
    // translucent backdrop keeps the navigation readable over it.
    <header className="relative z-20 bg-abyss/60 backdrop-blur-md md:landscape:bg-transparent md:landscape:backdrop-blur-none">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
        <a
          href="#top"
          className="group flex items-center gap-2.5 text-[15px] font-semibold tracking-tight text-ink"
          aria-label={`${brand.name} home`}
        >
          <span
            aria-hidden="true"
            className="h-2.5 w-2.5 rounded-full bg-champagne shadow-[0_0_14px_2px_rgb(230_201_143/0.55)]"
          />
          {brand.name}
        </a>
        <nav aria-label="Primary">
          <ul className="flex items-center gap-1 text-sm text-ink-muted sm:gap-2">
            <li>
              <a className="rounded-md px-3 py-2 transition-colors hover:text-ink" href="#available">
                Available
              </a>
            </li>
            <li>
              <a className="rounded-md px-3 py-2 transition-colors hover:text-ink" href="#planned">
                Planned
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}
