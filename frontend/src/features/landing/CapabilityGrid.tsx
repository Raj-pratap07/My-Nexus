import { available, planned, type Capability } from "./content";

function Card({ item, status }: { item: Capability; status: "Available now" | "Planned" }) {
  const isPlanned = status === "Planned";
  return (
    <li className="group relative rounded-2xl border border-navy-2 bg-navy/60 p-6 transition-colors duration-200 hover:border-electric/40">
      <span
        className={`mb-4 inline-block rounded-full px-2.5 py-1 text-[11px] font-semibold tracking-wide uppercase ${
          isPlanned
            ? "border border-champagne/40 text-champagne"
            : "bg-electric/15 text-electric-glow"
        }`}
      >
        {status}
      </span>
      <h3 className="text-lg font-semibold tracking-tight text-ink">{item.title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-ink-muted">{item.body}</p>
    </li>
  );
}

export function CapabilityGrid() {
  return (
    <div className="mx-auto max-w-7xl px-5 py-24 sm:px-8">
      <section id="available" aria-labelledby="available-title" className="scroll-mt-8">
        <h2 id="available-title" className="font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
          What works today
        </h2>
        <p className="mt-3 max-w-2xl text-ink-muted">
          The backend below is implemented and tested. This web page does not connect to it yet.
        </p>
        <ul className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {available.map((item) => (
            <Card key={item.title} item={item} status="Available now" />
          ))}
        </ul>
      </section>

      <section id="planned" aria-labelledby="planned-title" className="mt-24 scroll-mt-8">
        <h2 id="planned-title" className="font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
          Planned
        </h2>
        <p className="mt-3 max-w-2xl text-ink-muted">Not implemented yet.</p>
        <ul className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {planned.map((item) => (
            <Card key={item.title} item={item} status="Planned" />
          ))}
        </ul>
      </section>
    </div>
  );
}
