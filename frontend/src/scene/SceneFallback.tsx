import { colors } from "../design/tokens";

/**
 * Static stand-in for the particle sphere when WebGL is unavailable, lost or
 * failed. Plain SVG (no three.js): a deterministic Fibonacci dot pattern
 * projected orthographically, in the same palette and position as the 3D
 * sphere. Decorative only.
 */
const DOTS = (() => {
  const n = 420;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const out: { x: number; y: number; r: number; o: number; gold: boolean }[] = [];
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const rad = Math.sqrt(1 - y * y);
    const t = golden * i;
    const x = Math.cos(t) * rad;
    const z = Math.sin(t) * rad;
    // Tilt toward the viewer slightly and keep the front hemisphere brighter.
    const front = (z + 1) / 2;
    out.push({
      x: 50 + x * 40,
      y: 50 - y * 40,
      r: 0.35 + front * 0.55,
      o: 0.18 + front * 0.62,
      gold: i % 9 === 0 || i % 13 === 0,
    });
  }
  return out;
})();

export function SceneFallback() {
  return (
    <div className="absolute inset-0 overflow-hidden" data-testid="scene-fallback">
      <svg
        viewBox="0 0 100 100"
        className="absolute right-1/2 top-[8%] h-[46%] max-h-[540px] translate-x-1/2 md:landscape:right-[6%] md:landscape:top-1/2 md:landscape:h-[72%] md:landscape:max-h-[680px] md:landscape:translate-x-0 md:landscape:-translate-y-1/2"
        aria-hidden="true"
        focusable="false"
      >
        <defs>
          <radialGradient id="nexus-fallback-glow" cx="45%" cy="40%" r="55%">
            <stop offset="0" stopColor={colors.electric} stopOpacity="0.28" />
            <stop offset="0.6" stopColor={colors["navy-2"]} stopOpacity="0.18" />
            <stop offset="1" stopColor={colors.abyss} stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="50" cy="50" r="48" fill="url(#nexus-fallback-glow)" />
        {DOTS.map((d, i) => (
          <circle
            key={i}
            cx={d.x}
            cy={d.y}
            r={d.r}
            fill={d.gold ? colors.champagne : colors["electric-glow"]}
            opacity={d.o}
          />
        ))}
      </svg>
    </div>
  );
}
