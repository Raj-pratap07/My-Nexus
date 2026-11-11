/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { nexusApiProxy } from "./dev/nexusApiProxy.ts";

export default defineConfig({
  // nexusApiProxy serves /api on the dev and preview servers only. It adds
  // the backend credential server-side; see dev/nexusApiProxy.ts and
  // README.md. It refuses to start unless both servers bind to loopback.
  plugins: [react(), tailwindcss(), nexusApiProxy()],
  build: {
    target: "es2022",
    sourcemap: false,
    // The lazily loaded scene chunk is ~930 kB minified (~249 kB gzip): React
    // Three Fiber imports three's whole namespace to build its JSX elements,
    // which defeats tree-shaking. It loads after the page content, so it
    // never blocks it. The limit sits just above the measured size so that
    // unexpected growth still warns.
    chunkSizeWarningLimit: 1000,
  },
  server: { host: "127.0.0.1", strictPort: true },
  preview: { host: "127.0.0.1", strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    // Keep CSS processing on: with `css: false` Vitest also blanks `?raw`
    // imports of CSS, which the design-token parity test reads.
    css: true,
    restoreMocks: true,
  },
});
