import { brand, footer } from "./content";

export function SiteFooter() {
  return (
    <footer className="border-t border-navy-2">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 px-5 py-10 text-sm text-ink-muted sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>
          <span className="font-semibold text-ink">{brand.name}</span> · {brand.tagline}
        </p>
        <p>{footer.note}</p>
      </div>
    </footer>
  );
}
