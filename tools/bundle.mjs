#!/usr/bin/env node
/**
 * Single-file browser bundles (esbuild) — the CDN/`<script type="module">` path the workspace
 * packages could not offer: `@ares/core` imports the bare specifier `meshoptimizer`, which needs an
 * import map or a bundler to resolve. These outputs inline it (the WASM ships base64 inside the
 * meshoptimizer module, so nothing else is fetched).
 *
 *   packages/core/dist/bundle/ares-core.esm.js       ESM, `import { AresPlayer } from "…/ares-core.esm.js"`
 *   packages/core/dist/bundle/ares-core.iife.js      classic script, `window.ARES.AresPlayer`
 *   packages/core/dist/bundle/ares-decode-worker.js  the geometry decode worker, self-contained;
 *                                                     pass its URL as AresPlayerOptions.workerUrl
 *   packages/three/dist/bundle/ares-three.esm.js     ESM with `three` left external (peer)
 * Each also gets a .min.js. Run `npm run build` first (bundles from dist/, not src/).
 */
import { build } from "esbuild";
import { mkdir, stat } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const core = join(ROOT, "packages/core/dist");
const three = join(ROOT, "packages/three/dist");

async function ensureBuilt(p) {
  try { await stat(p); } catch { throw new Error(`${p} missing — run \`npm run build\` first`); }
}

const banner = { js: "/* ARES Volumetric — MIT. Bundles @ares/core + meshoptimizer (MIT). */" };

async function one(entry, outfile, opts) {
  for (const minify of [false, true]) {
    const out = minify ? outfile.replace(/\.js$/, ".min.js") : outfile;
    await build({ entryPoints: [entry], outfile: out, bundle: true, minify, sourcemap: true, target: "es2022", banner, logLevel: "warning", ...opts });
    console.log(`[bundle] ${out}`);
  }
}

await ensureBuilt(join(core, "index.js"));
await mkdir(join(core, "bundle"), { recursive: true });
await one(join(core, "index.js"), join(core, "bundle/ares-core.esm.js"), { format: "esm" });
// IIFE has no import.meta: the worker's default URL becomes unresolvable there, which the client
// turns into a rejected `ready` (main-thread decode). Hosts pass AresPlayerOptions.workerUrl.
await one(join(core, "index.js"), join(core, "bundle/ares-core.iife.js"), { format: "iife", globalName: "ARES", define: { "import.meta.url": "undefined" } });
await one(join(core, "decode-worker.js"), join(core, "bundle/ares-decode-worker.js"), { format: "esm" });
try {
  await ensureBuilt(join(three, "index.js"));
  await mkdir(join(three, "bundle"), { recursive: true });
  await one(join(three, "index.js"), join(three, "bundle/ares-three.esm.js"), { format: "esm", external: ["three"] });
} catch (e) {
  console.warn(`[bundle] skipping @ares/three: ${e.message}`);
}
