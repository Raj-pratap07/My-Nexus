#!/usr/bin/env node
/**
 * Starts Vite (dev or preview) with NEXUS_DEV_TOKEN supplied at runtime,
 * without putting the token in shell history or any file.
 *
 *   npm run dev:secure        # vite
 *   npm run preview:secure    # vite preview
 *
 * In a terminal it asks for the token with input hidden. When stdin is not a
 * terminal (for example piped from a password manager) it reads one line.
 * The token is passed only in the child process environment; it is never
 * printed. If NEXUS_DEV_TOKEN is already set, it is used without prompting.
 */
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
// vite's "exports" hides bin/vite.js; resolve it from its package.json "bin".
const vitePkgPath = require.resolve("vite/package.json");
const vitePkg = JSON.parse(readFileSync(vitePkgPath, "utf8"));
const viteBin = join(dirname(vitePkgPath), typeof vitePkg.bin === "string" ? vitePkg.bin : vitePkg.bin.vite);
const args = process.argv.slice(2);

function readHidden(prompt) {
  return new Promise((resolve, reject) => {
    const stdin = process.stdin;
    if (!stdin.isTTY) {
      let data = "";
      stdin.setEncoding("utf8");
      stdin.on("data", (chunk) => (data += chunk));
      stdin.on("end", () => resolve(data.split(/\r?\n/)[0] ?? ""));
      stdin.on("error", reject);
      return;
    }
    process.stderr.write(prompt);
    let value = "";
    stdin.setRawMode(true);
    stdin.setEncoding("utf8");
    stdin.resume();
    const onData = (chunk) => {
      for (const ch of chunk) {
        if (ch === "\r" || ch === "\n") {
          stdin.setRawMode(false);
          stdin.pause();
          stdin.off("data", onData);
          process.stderr.write("\n");
          resolve(value);
          return;
        }
        if (ch === "\u0003") {
          // Ctrl+C
          stdin.setRawMode(false);
          process.stderr.write("\n");
          process.exit(130);
        }
        if (ch === "\u007f" || ch === "\b") value = value.slice(0, -1);
        else if (ch >= " ") value += ch;
      }
    };
    stdin.on("data", onData);
  });
}

const token =
  process.env.NEXUS_DEV_TOKEN ||
  (await readHidden("Backend NEXUS_DEV_TOKEN (input hidden): ")).trim();

if (!token) {
  process.stderr.write("No token entered; not starting.\n");
  process.exit(1);
}

const child = spawn(process.execPath, [viteBin, ...args], {
  stdio: "inherit",
  env: { ...process.env, NEXUS_DEV_TOKEN: token },
});
const forward = (signal) => child.kill(signal);
process.on("SIGINT", forward);
process.on("SIGTERM", forward);
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
