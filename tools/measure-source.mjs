#!/usr/bin/env node
/**
 * Backfill `source.{meshBytes,texBytes,totalBytes,fileCount}` into a clip's provenance sidecar.
 *
 * The encoder records these at encode time now, but clips baked before that only carry
 * `source.dir`. Re-encoding a 4910-frame capture for 15 minutes just to add three numbers is silly,
 * so this measures the source folder and patches the sidecar in place — provided the folder still
 * exists. It does NOT touch the .ares.
 *
 * It measures the same file set the encoder would have consumed (mesh + atlas, honouring the
 * recorded trim/max-frames window where the sidecar records one), so the numbers match what a fresh
 * encode would write. If it cannot reproduce the exact set it says so and writes nothing, rather
 * than recording a plausible-but-wrong total.
 *
 *   node tools/measure-source.mjs apps/demo/svf-export-full.ares
 *   node tools/measure-source.mjs --all apps/demo
 */
import { readdir, readFile, writeFile, stat } from "node:fs/promises";
import { join, extname, basename } from "node:path";

const MB = (b) => +(b / 1048576).toFixed(2);   // MiB — matches the app readout and `ares info`

async function sumBytes(names, dir) {
  let n = 0, missing = 0;
  for (const f of names) {
    try { n += (await stat(join(dir, f))).size; } catch { missing++; }
  }
  return { bytes: n, missing };
}

async function measure(aresPath) {
  const metaPath = aresPath + ".meta.json";
  let meta;
  try { meta = JSON.parse(await readFile(metaPath, "utf8")); }
  catch { return console.log(`skip ${basename(aresPath)} — no provenance sidecar`); }

  const src = meta.source;
  if (!src?.dir) return console.log(`skip ${basename(aresPath)} — sidecar records no source.dir`);
  if (src.totalBytes != null) return console.log(`skip ${basename(aresPath)} — already measured (${MB(src.totalBytes)} MB)`);

  let all;
  try { all = await readdir(src.dir); }
  catch { return console.log(`skip ${basename(aresPath)} — source folder is gone: ${src.dir}`); }

  // Detect the mesh type from the FOLDER, exactly as the encoder does (`isObj = objFiles.length > 0`).
  // Trusting `source.kind` was wrong: orchestrator-written sidecars (the coherent bakes) record only
  // `source.dir`, so `("" ).startsWith("obj")` was false and this hunted for .ply in a folder holding
  // 272 .obj — finding nothing.
  const objs = all.filter((f) => extname(f).toLowerCase() === ".obj").sort();
  const plys = all.filter((f) => extname(f).toLowerCase() === ".ply").sort();
  const isObj = objs.length > 0;
  const meshes = isObj ? objs : plys;
  const atlases = all.filter((f) => /atlas.*\.png$/i.test(f)).sort();

  // Reproduce the encoder's window. Bail loudly rather than guess: a total measured over the wrong
  // file set is worse than no total, because it looks authoritative.
  const from = meta.encode?.trim?.in ?? 0;
  const to = meta.encode?.trim?.out ?? meshes.length - 1;
  const usedMesh = meshes.slice(from, to + 1);
  const usedAtlas = atlases.slice(from, to + 1);

  // Refuse anything we can't reproduce EXACTLY. A wrong total is worse than none: it looks
  // authoritative and it's what the app quotes as the headline savings figure.
  // The empty case is the one that bit: an orchestrator's sidecar (coherent bakes) points source.dir
  // at a staging folder that holds no meshes, so this measured 0 files and cheerfully wrote
  // "source 0 MB → 0.0× smaller" into the provenance. A count guard alone missed it because those
  // sidecars carry no meshFrames to compare against.
  if (!usedMesh.length) {
    return console.log(`SKIP ${basename(aresPath)} — no ${isObj ? "OBJ" : "PLY"} frames in ${src.dir} ` +
      `(orchestrator sidecar pointing at a staging dir?). Not guessing.`);
  }
  if (src.meshFrames != null && usedMesh.length !== src.meshFrames) {
    return console.log(`SKIP ${basename(aresPath)} — cannot reproduce the encoded file set ` +
      `(found ${usedMesh.length} meshes in ${src.dir}, sidecar says ${src.meshFrames}). Not guessing.`);
  }

  const m = await sumBytes(usedMesh, src.dir);
  const t = await sumBytes(usedAtlas, src.dir);
  if (m.missing || t.missing) {
    return console.log(`SKIP ${basename(aresPath)} — ${m.missing + t.missing} source file(s) unreadable. Not guessing.`);
  }
  if (m.bytes + t.bytes === 0) {
    return console.log(`SKIP ${basename(aresPath)} — measured 0 bytes. Not recording that.`);
  }

  src.meshBytes = m.bytes;
  src.texBytes = t.bytes;
  src.totalBytes = m.bytes + t.bytes;
  src.fileCount = usedMesh.length + usedAtlas.length;
  src.measured = "backfilled by tools/measure-source.mjs (stat of the source folder)";

  const outBytes = meta.output?.sizeBytes ?? 0;
  await writeFile(metaPath, JSON.stringify(meta, null, 2) + "\n");
  console.log(`${basename(aresPath)}: source ${MB(src.totalBytes)} MB in ${src.fileCount} files ` +
    `(mesh ${MB(src.meshBytes)} + atlas ${MB(src.texBytes)})` +
    (outBytes ? ` → ${MB(outBytes)} MB .ares = ${(src.totalBytes / outBytes).toFixed(1)}× smaller` : ""));
}

const args = process.argv.slice(2);
if (args[0] === "--all") {
  const dir = args[1] ?? "apps/demo";
  for (const f of (await readdir(dir)).filter((f) => f.endsWith(".ares")).sort()) await measure(join(dir, f));
} else if (args.length) {
  for (const p of args) await measure(p);
} else {
  console.error("usage: node tools/measure-source.mjs <clip.ares>... | --all <dir>");
  process.exit(1);
}
