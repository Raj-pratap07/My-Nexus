#!/usr/bin/env node
/**
 * Proves the backend credential cannot reach the client bundle.
 *
 * Builds the app with NEXUS_DEV_TOKEN (and NEXUS_BACKEND_URL) set to random
 * sentinel values, then scans every emitted file. The check fails if a
 * sentinel, the variable names, or an Authorization header literal appears.
 * The sentinel is random per run and is never a real credential.
 */
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const outDir = join(root, "dist");
const sentinel = `SENTINEL${randomBytes(24).toString("hex")}`;
const require = createRequire(import.meta.url);
// vite's "exports" hides bin/vite.js; resolve it from its package.json "bin".
const vitePkgPath = require.resolve("vite/package.json");
const vitePkg = JSON.parse(readFileSync(vitePkgPath, "utf8"));
const viteBin = join(dirname(vitePkgPath), typeof vitePkg.bin === "string" ? vitePkg.bin : vitePkg.bin.vite);

const build = spawnSync(process.execPath, [viteBin, "build"], {
  cwd: root,
  stdio: ["ignore", "pipe", "pipe"],
  env: {
    ...process.env,
    NEXUS_DEV_TOKEN: sentinel,
    NEXUS_BACKEND_URL: `http://127.0.0.1:8000/${sentinel}`,
  },
});
if (build.status !== 0) {
  process.stderr.write(build.stdout.toString() + build.stderr.toString());
  process.stderr.write("check:bundle: build failed.\n");
  process.exit(1);
}
if (build.stdout.toString().includes(sentinel) || build.stderr.toString().includes(sentinel)) {
  process.stderr.write("check:bundle: FAIL - the sentinel token appeared in build output.\n");
  process.exit(1);
}

function* files(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) yield* files(path);
    else yield path;
  }
}

const forbidden = [
  [sentinel, "sentinel token value"],
  ["NEXUS_DEV_TOKEN", "token variable name"],
  ["NEXUS_BACKEND_URL", "backend URL variable name"],
];
const problems = [];
let scanned = 0;
for (const path of files(outDir)) {
  scanned++;
  const text = readFileSync(path).toString("latin1");
  for (const [needle, label] of forbidden) {
    if (text.includes(needle)) problems.push(`${relative(root, path)}: contains ${label}`);
  }
  if (/\.(js|html)$/.test(path) && /["']authorization["']\s*[:,]/i.test(text)) {
    problems.push(`${relative(root, path)}: sets an Authorization header`);
  }
}

if (problems.length) {
  process.stderr.write(`check:bundle: FAIL\n  ${problems.join("\n  ")}\n`);
  process.exit(1);
}
process.stdout.write(
  `check:bundle: OK - scanned ${scanned} files in dist/; no credential, credential variable or Authorization header found.\n`,
);
