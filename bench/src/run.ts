/** Phase 0 intra-representation bench (spec §13.4, §14 P0).
 *
 * Usage: node bench/dist/run.js [--frames 30] [--bits 14] [--reps 3]
 *                               [--clips talk,dance] [--skip-draco] [--skip-sweep]
 *
 * Measures, per clip × codec: bytes/frame, bytes/vertex, ratio vs raw, encode ms,
 * decode ms (median of reps), positional error (rel. bbox diagonal). Then sweeps
 * quantization 11→16 bits. Writes results/intra-latest.json + .csv and prints a table.
 * Real PLY sequences in bench/data/<clip>/*.ply join the corpus automatically.
 */
import { readdir, readFile, mkdir, writeFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { cpus, totalmem } from "node:os";
import { parsePly } from "@ares/encoder";
import { Clip, syntheticCorpus } from "./corpus.js";
import { makeCodecs, sweepCodecs, totalBytes, IntraCodec } from "./codecs.js";
import { positionError, aabb, vertexCount } from "./mesh.js";

const BENCH_DIR = dirname(dirname(fileURLToPath(import.meta.url))); // bench/
const DATA_DIR = join(BENCH_DIR, "data");
const OUT_DIR = join(BENCH_DIR, "results");

interface CodecStats {
  codec: string;
  bytesPerFrame: number;
  bytesPerVertex: number;
  ratioVsRaw: number;
  encodeMsMean: number;
  decodeMsMean: number;
  decodeMsMedian: number;
  errRms: number; // relative to bbox diagonal
  errMax: number;
  analyticError: boolean;
}

interface ClipResult {
  clip: string;
  cls: string;
  synthetic: boolean;
  frames: number;
  meanVerts: number;
  meanTris: number;
  codecs: CodecStats[];
}

interface SweepRow {
  clip: string; bits: number; codec: string;
  bytesPerFrame: number; bytesPerVertex: number; errRms: number; errMax: number;
}

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}
const has = (name: string) => process.argv.includes(name);
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
};
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

async function loadRealClips(maxFrames: number): Promise<Clip[]> {
  const clips: Clip[] = [];
  let dirs: string[] = [];
  try { dirs = await readdir(DATA_DIR); } catch { return clips; }
  for (const d of dirs) {
    let files: string[] = [];
    try { files = (await readdir(join(DATA_DIR, d))).filter((f) => f.toLowerCase().endsWith(".ply")).sort(); } catch { continue; }
    if (!files.length) continue;
    const frames = [];
    for (const f of files.slice(0, maxFrames)) {
      const m = parsePly(await readFile(join(DATA_DIR, d, f)));
      frames.push({ positions: m.positions, indices: m.indices });
    }
    clips.push({ name: d, description: `real capture (${files.length} PLY frames)`, cls: "A", frames, synthetic: false });
    console.log(`[bench] loaded real clip "${d}": ${frames.length} frames`);
  }
  return clips;
}

function benchCodec(codec: IntraCodec, clip: Clip, reps: number): CodecStats {
  const encodeMs: number[] = [];
  const decodeMs: number[] = [];
  let bytes = 0;
  let errRms = 0, errMax = 0;
  // warmup (JIT + wasm)
  {
    const e = codec.encode(clip.frames[0]!);
    codec.decode(e);
  }
  for (const frame of clip.frames) {
    const t0 = performance.now();
    const enc = codec.encode(frame);
    encodeMs.push(performance.now() - t0);
    bytes += totalBytes(enc);

    const times: number[] = [];
    let dec = null as ReturnType<IntraCodec["decode"]> | null;
    for (let r = 0; r < reps; r++) {
      const t1 = performance.now();
      dec = codec.decode(enc);
      times.push(performance.now() - t1);
    }
    decodeMs.push(median(times));

    if (!codec.analyticError && dec) {
      const ref = codec.reference ? codec.reference(frame, enc) : frame.positions;
      const { diag } = aabb(frame.positions);
      const e = positionError(ref, dec.positions, diag);
      errRms += e.rms; errMax = Math.max(errMax, e.max);
    }
  }
  const n = clip.frames.length;
  const verts = clip.frames.reduce((s, f) => s + vertexCount(f), 0) / n;
  if (codec.analyticError) {
    // uniform quantization bound at 14 bits over the AABB: step/2 max, step/sqrt(12) rms.
    // Draco quantizes per-axis over its own bbox; this matches qbin's analytic bound.
    const bits = Number(codec.name.replace(/\D/g, "")) || 14;
    const step = 1 / ((1 << bits) - 1);
    errMax = step / 2 * Math.sqrt(3);
    errRms = step / Math.sqrt(12);
  } else {
    errRms /= n;
  }
  return {
    codec: codec.name,
    bytesPerFrame: bytes / n,
    bytesPerVertex: bytes / n / verts,
    ratioVsRaw: 0, // filled by caller
    encodeMsMean: mean(encodeMs),
    decodeMsMean: mean(decodeMs),
    decodeMsMedian: median(decodeMs),
    errRms,
    errMax,
    analyticError: !!codec.analyticError,
  };
}

async function main() {
  const framesN = Number(arg("--frames") ?? process.env.ARES_BENCH_FRAMES ?? 30);
  const bits = Number(arg("--bits") ?? 14);
  const reps = Number(arg("--reps") ?? 3);
  const clipFilter = arg("--clips")?.split(",");
  const includeDraco = !has("--skip-draco");

  console.log(`[bench] ARES Phase 0 intra bench — frames=${framesN} bits=${bits} reps=${reps} draco=${includeDraco}`);
  console.log(`[bench] generating synthetic corpus (spec §13.2 taxonomy)...`);
  let clips = [...syntheticCorpus(framesN), ...(await loadRealClips(framesN))];
  if (clipFilter) clips = clips.filter((c) => clipFilter.includes(c.name));

  const codecs = await makeCodecs(bits, includeDraco);
  const results: ClipResult[] = [];

  for (const clip of clips) {
    const meanVerts = Math.round(clip.frames.reduce((s, f) => s + vertexCount(f), 0) / clip.frames.length);
    const meanTris = Math.round(clip.frames.reduce((s, f) => s + f.indices.length / 3, 0) / clip.frames.length);
    console.log(`[bench] clip "${clip.name}" (class ${clip.cls}, ~${meanVerts} verts, ${clip.frames.length} frames)`);
    const stats: CodecStats[] = [];
    for (const codec of codecs) {
      const s = benchCodec(codec, clip, reps);
      stats.push(s);
      console.log(
        `  ${s.codec.padEnd(12)} ${(s.bytesPerFrame / 1024).toFixed(1).padStart(8)} KB/f` +
        `  enc ${s.encodeMsMean.toFixed(2).padStart(7)} ms  dec ${s.decodeMsMedian.toFixed(2).padStart(6)} ms` +
        `  err(rms) ${s.errRms.toExponential(1)}${s.analyticError ? "*" : ""}`);
    }
    const rawBytes = stats.find((s) => s.codec === "raw-f32")!.bytesPerFrame;
    for (const s of stats) s.ratioVsRaw = s.bytesPerFrame / rawBytes;
    results.push({ clip: clip.name, cls: clip.cls, synthetic: clip.synthetic, frames: clip.frames.length, meanVerts, meanTris, codecs: stats });
  }

  // Quantization sweep (§13.4): bits 11→16, qbin + meshopt (+br), size × error.
  const sweep: SweepRow[] = [];
  if (!has("--skip-sweep")) {
    console.log(`[bench] quantization sweep 11→16 bits...`);
    for (let b = 11; b <= 16; b++) {
      const sweepSet = await sweepCodecs(Math.min(b, 16));
      for (const clip of clips) {
        for (const codec of sweepSet) {
          let bytes = 0, errRms = 0, errMax = 0;
          for (const frame of clip.frames) {
            const enc = codec.encode(frame);
            bytes += totalBytes(enc);
            const dec = codec.decode(enc);
            const ref = codec.reference ? codec.reference(frame, enc) : frame.positions;
            const { diag } = aabb(frame.positions);
            const e = positionError(ref, dec.positions, diag);
            errRms += e.rms; errMax = Math.max(errMax, e.max);
          }
          const n = clip.frames.length;
          const verts = clip.frames.reduce((s, f) => s + vertexCount(f), 0) / n;
          sweep.push({
            clip: clip.name, bits: b, codec: codec.name.replace(/\d+/, ""),
            bytesPerFrame: bytes / n, bytesPerVertex: bytes / n / verts,
            errRms: errRms / n, errMax,
          });
        }
      }
      console.log(`  ${b} bits done`);
    }
  }

  const out = {
    generated: new Date().toISOString(),
    env: {
      node: process.version,
      cpu: cpus()[0]?.model ?? "unknown",
      cores: cpus().length,
      memGB: Math.round(totalmem() / 2 ** 30),
      platform: process.platform,
      frames: framesN, bits, reps,
    },
    note: "Synthetic corpus stand-ins (spec §13.6): replace with real captures in bench/data/ before public claims. * = analytic quantization bound, not measured (Draco reorders vertices).",
    decoderFootprint: {
      "draco_decoder.wasm": 285948,
      "meshopt_decoder (wasm, embedded)": 0, // filled below
    } as Record<string, number>,
    results,
    sweep,
  };
  try {
    const { createRequire } = await import("node:module");
    const { statSync } = await import("node:fs");
    const req = createRequire(import.meta.url);
    const base = dirname(req.resolve("meshoptimizer"));
    // whole ESM decoder module incl. its custom-encoded embedded wasm — what a browser ships
    out.decoderFootprint["meshopt_decoder (wasm, embedded)"] = statSync(join(base, "meshopt_decoder.mjs")).size;
    out.decoderFootprint["draco_decoder.wasm"] =
      statSync(join(dirname(req.resolve("draco3d")), "draco_decoder.wasm")).size;
  } catch { out.decoderFootprint["meshopt_decoder (wasm, embedded)"] = -1; }

  await mkdir(OUT_DIR, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
  await writeFile(join(OUT_DIR, `intra-${stamp}.json`), JSON.stringify(out, null, 2));
  await writeFile(join(OUT_DIR, "intra-latest.json"), JSON.stringify(out, null, 2));

  const csv = ["clip,codec,bytesPerFrame,bytesPerVertex,ratioVsRaw,encodeMsMean,decodeMsMedian,errRms,errMax,analytic"];
  for (const r of results) for (const s of r.codecs) {
    csv.push([r.clip, s.codec, s.bytesPerFrame.toFixed(1), s.bytesPerVertex.toFixed(2),
      s.ratioVsRaw.toFixed(4), s.encodeMsMean.toFixed(3), s.decodeMsMedian.toFixed(3),
      s.errRms.toExponential(3), s.errMax.toExponential(3), String(s.analyticError)].join(","));
  }
  await writeFile(join(OUT_DIR, "intra-latest.csv"), csv.join("\n") + "\n");

  // Decision line (P0 exit criterion: intra default confirmed or revised)
  console.log("\n[bench] ——— P0 exit summary ———");
  for (const r of results) {
    const mo = r.codecs.find((c) => c.codec.startsWith("meshopt") && !c.codec.endsWith("br"));
    const mobr = r.codecs.find((c) => c.codec.startsWith("meshopt") && c.codec.endsWith("br"));
    const dr = r.codecs.find((c) => c.codec.startsWith("draco"));
    if (!mo) continue;
    const sizeLine = dr
      ? `meshopt+br ${(mobr!.bytesPerFrame / 1024).toFixed(0)}KB vs draco ${(dr.bytesPerFrame / 1024).toFixed(0)}KB/frame; ` +
        `decode ${mo.decodeMsMedian.toFixed(2)}ms vs ${dr.decodeMsMedian.toFixed(2)}ms`
      : `meshopt+br ${(mobr!.bytesPerFrame / 1024).toFixed(0)}KB/frame, decode ${mo.decodeMsMedian.toFixed(2)}ms`;
    console.log(`  ${r.clip.padEnd(8)} ${sizeLine}`);
  }
  console.log(`[bench] wrote ${join(OUT_DIR, "intra-latest.json")} (+ csv, timestamped copy)`);
  console.log(`[bench] report page: /bench/report/ (via tools/serve.mjs)`);
}

main().catch((e) => { console.error("[bench] error:", e); process.exit(1); });
