#!/usr/bin/env node
// Orchestrate a full coherent bake and land the result in the ARES app library (apps/demo/).
// Stages: (1) coherent-clip register+bake -> baked frames dir, (2) ares encode -> apps/demo/<name>.ares.
// Forwards the pipeline's honest [PROGRESS] lines and emits [STAGE]/[DONE] markers the app tails.
//
//   node run-coherent-bake.mjs <src-frames-dir> <out-name> [--tex-size 2048] [--work <dir>] [coherent flags...]
import { spawn } from "node:child_process";
import { mkdir, rm, stat, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..", "..");            // ares/
const args = process.argv.slice(2);
const srcDir = args[0];
const outName = (args[1] || "coherent-out").replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
const flag = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const TEX = flag("--tex-size", "2048");
const workDir = flag("--work", join(ROOT, ".coherent-bake-work"));
const outAres = join(ROOT, "apps", "demo", `${outName}.ares`);
if (!srcDir) { console.error("usage: run-coherent-bake.mjs <src-frames-dir> <out-name> [--tex-size 2048]"); process.exit(1); }

// pass-through coherent flags (anything after a `--` in argv, or the known set)
const passThrough = [];
for (const k of ["--gop", "--err-cut", "--stretch-cut", "--min-run", "--rounds", "--max-rounds", "--eps-mm", "--smooth-iters", "--rings", "--fallback-mm", "--workers", "--max-frames", "--python", "--arap-lambda", "--arap-gate", "--arap-outer", "--flow", "--flow-r", "--flow-min", "--flow-snap", "--flow-spread"]) {
  const v = flag(k, undefined); if (v !== undefined) passThrough.push(k, v);
}
if (args.includes("--gpu")) passThrough.push("--gpu");    // GPU Pass 2 (Warp) — ~100x, same 2048 quality
if (args.includes("--arap")) passThrough.push("--arap");  // ARAP registration — clean (no tear), keeps small

function runNode(scriptArgs, label) {
  return new Promise((resolve, reject) => {
    const p = spawn(process.execPath, scriptArgs, { cwd: ROOT });
    p.stdout.on("data", (d) => process.stdout.write(d));   // forwards [PROGRESS] lines verbatim
    p.stderr.on("data", (d) => process.stderr.write(d));
    p.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`${label} exited ${c}`))));
    p.on("error", reject);
  });
}

const t0 = Date.now();
try {
  await rm(workDir, { recursive: true, force: true });
  await mkdir(workDir, { recursive: true });

  console.log(`[STAGE] ${JSON.stringify({ stage: "register+bake", i: 1, n: 2, msg: "coherent register + bake (full-rez)" })}`);
  await runNode([join(ROOT, "tools/coherent/coherent-clip.mjs"), srcDir, workDir, ...passThrough], "coherent-clip");

  console.log(`[STAGE] ${JSON.stringify({ stage: "encode", i: 2, n: 2, msg: `encode -> ${outName}.ares @ ${TEX}` })}`);

  // Provenance enrichment (every setting bound to the clip). The encoder writes the
  // authoritative encode block into <out>.ares.meta.json; here we merge in the RESOLVED coherent
  // registration recipe (from coherent-clip's manifest) + run distribution + source, so the sidecar
  // captures the WHOLE pipeline — nothing about how this clip was made can be lost or mis-copied.
  const metaExtraArgs = [];
  try {
    const manPath = join(dirname(workDir), "coherent-manifest.json");
    const man = JSON.parse(await readFile(manPath, "utf8"));
    const metaExtra = {
      pipeline: "coherent",
      coherent: man.clip,   // gop, errCutMm, stretchCut, rounds, maxRounds, epsMm, smoothIters, minRun, bakeMode, gpu, …
      coherentRuns: (man.runs || []).map((r) => ({ template: r.templateGlobalFrame, start: r.startGlobalFrame, end: r.endGlobalFrame, len: r.length, cut: r.cutReason })),
      coherentCuts: man.cuts || [],
      source: { dir: srcDir },
    };
    const mePath = join(workDir, ".meta-extra.json");
    await writeFile(mePath, JSON.stringify(metaExtra));
    metaExtraArgs.push("--meta-extra-file", mePath);
  } catch (e) { console.log(`[run-coherent-bake] meta-extra skipped: ${e.message}`); }

  // --no-temporal is REQUIRED for coherent output: its multi-topology GOPs shred geometry through
  // the encoder's temporal (I+P) path. All-intra is the proven-safe path.
  await runNode([join(ROOT, "packages/encoder/dist/cli.js"), "encode", workDir, "-o", outAres, "--tex-size", TEX, "--fps", "30", "--no-temporal", ...metaExtraArgs], "encode");

  const sz = existsSync(outAres) ? (await stat(outAres)).size : 0;
  console.log(`[DONE] ${JSON.stringify({ ok: true, out: `/apps/demo/${outName}.ares`, name: `${outName}.ares`, sizeMB: +(sz / 1e6).toFixed(1), totalS: Math.round((Date.now() - t0) / 1000) })}`);
} catch (e) {
  console.log(`[DONE] ${JSON.stringify({ ok: false, error: String((e && e.message) || e) })}`);
  process.exit(1);
}
