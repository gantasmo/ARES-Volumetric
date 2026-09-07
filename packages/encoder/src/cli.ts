#!/usr/bin/env node
/**
 * `ares` CLI (spec §5.2, §14 P5).
 *   ares synth  [-o out.ares] [--shape object|talk|splat] [--frames 60] [--fps 30] [--no-texture] [--sh-degree 0|1]
 *   ares encode <frames-dir> [-o out.ares] [--fps 30] [--max-frames N] [--gop 30]
 *               mesh input (OBJ/PLY + atlas PNGs):
 *               [--texture-codec vp9|av1] [--tex-size 1024] [--crf 32] [--no-texture]
 *               [--edits file.json] [--crop x0,y0,z0,x1,y1,z1] [--track] [--no-temporal]
 *               [--smooth N] [--smooth-temporal N] [--decimate ratio] [--repack-detect topology|image]
 *               splat input (SPZ / 3DGS PLY / .splat / glTF+KHR_gaussian_splatting / SOG, one file per frame):
 *               [--sh-degree 0..3] [--splat-min-alpha a] [--splat-box-alpha a] [--splat-order morton|none]
 *               [--quant-bits 8..16]
 *               both: [--trim-in N] [--trim-out N] [--up-axis x|y|z] [--center bottom|mass|none]
 *                     [--scale N] [--rotate x,y,z] [--translate x,y,z] [--meta-extra-file f.json]
 *               audio (both): [--audio file] [--audio-offset s] [--audio-bitrate kbps]
 *   ares export <file.ares> -o <out> [--frame N]      (.obj/.ply for meshes; .spz/.ply/.glb/.splat for splats)
 *   ares info   <file.ares>
 */
import { readdir, readFile, writeFile, stat } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { join, extname, basename } from "node:path";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { Demuxer, GeometryProfile, BlockType, decodeGeometryBlock, decodePFrameBlock, decodeSplatBlock, decodeSplatPBlock, dequantScale, meshoptReady, transformMatrix, type DecodedGeometry } from "@ares/core";
import { parsePly, parsePlyHeader, isSplatPlyHeader, parsePlySplat, writeSplatPly } from "./importers/ply.js";
import { parseObj } from "./importers/obj.js";
import { parseSpz, writeSpz } from "./importers/spz.js";
import { parseSplatFile, writeSplatFile } from "./importers/splat-file.js";
import { parseGltfSplat, writeGlbSplat } from "./importers/gltf-splat.js";
import { parseSog } from "./importers/sog.js";
import { filterSplatFrame, transformSplatFrame, decodedSplatToFrame, type SplatFrame } from "./splat-frame.js";
import { decodedMeshToFrame, writeObj, writeMeshPly } from "./export.js";
import { transcodeToOpus, type AudioTrackData } from "./audio-mux.js";
import { filterFrame, parseCropBox } from "./crop.js";
import { decimateFrame, simplifierReady } from "./decimate.js";
import { parseEditList, keepPredicateAt, prepareRangeAt, prepareRangeSdfAt, isRangeEnabled, type EditList, type EditRange } from "@ares/core";
import { resolveOffset, applyTransform, applyTransformNormals, transformAabb, type ModelTransform } from "@ares/core";
import type { Aabb } from "@ares/core";
import { collectCopyOps, applyGeoCopy, resolveSrcFragment, buildPieceFragment, type CopyOp } from "./frame-copy.js";
import {
  buildPatchedAtlasDir, pngSize, rasterizeUvFootprint, dilateMask,
  planRegionRelocation, measureTexelCollateral, identityPlan, type TexelPatchPlan,
} from "./texel-copy.js";
import { collectRecolorOps, partitionTrianglesByCentroid, hexOf, parseHexColor, type RecolorPatch } from "./recolor.js";
import { collectPaintOps, rasterizePaintWeights, dilatePaintWeights, type PaintPatch } from "./paint.js";
import { collectSculptOps, applySculptToFrame } from "./sculpt.js";
import { detectHoleLoopsForRange, appendCaps, findFreeTile, HOLE_TILE, type HoleTileFill } from "./hole-patch.js";
import { muxClip, muxClipWithStats, type MuxClip } from "./muxer.js";
import { synthClip, synthSplatClip } from "./synth.js";
import { proceduralAtlas } from "./png.js";
import { encodeTextureVideo, detectPattern, ffmpegAvailable, ffmpegPath, type TexCodec } from "./texture-video.js";
import type { EncodeMeshFrame } from "./geometry-encode.js";

function flag(a: string[], name: string): string | undefined {
  const i = a.indexOf(name);
  if (i < 0) return undefined;
  const v = a[i + 1];
  // A flag that takes a value followed by another flag (or nothing) is a usage error, not a value.
  if (v === undefined || (v.startsWith("--") && v.length > 2)) throw new Error(`${name}: expected a value`);
  return v;
}
const has = (a: string[], name: string) => a.includes(name);
/** Numeric flag with range validation; `def` when absent. */
function numFlag(a: string[], name: string, def: number, opts: { min?: number; max?: number; int?: boolean } = {}): number {
  const raw = flag(a, name);
  if (raw === undefined) return def;
  const v = Number(raw);
  if (!Number.isFinite(v) || (opts.int && !Number.isInteger(v))) throw new Error(`${name}: expected ${opts.int ? "an integer" : "a number"}, got ${JSON.stringify(raw)}`);
  if (opts.min !== undefined && v < opts.min) throw new Error(`${name}: ${v} is below the minimum ${opts.min}`);
  if (opts.max !== undefined && v > opts.max) throw new Error(`${name}: ${v} is above the maximum ${opts.max}`);
  return v;
}
const SPLAT_EXTS = new Set([".spz", ".splat", ".sog", ".glb", ".gltf"]);

/** Provenance sidecar: EVERY setting used to convert/create/import a clip is saved into a metadata
 *  file bound to the clip. Written next to every .ares the
 *  encoder produces, so the full recipe travels WITH the file and can be re-applied to another
 *  conversion — no more re-deriving lost settings. The encoder writes the authoritative encode-level
 *  block; orchestrators (coherent/4ds/serve) enrich it with their upstream settings by passing a
 *  JSON file via --meta-extra-file (merged in under whatever keys they set, e.g. pipeline/coherent). */
async function writeClipMeta(a: string[], outPath: string, base: Record<string, unknown>): Promise<void> {
  let extra: Record<string, unknown> = {};
  const mx = flag(a, "--meta-extra-file");
  if (mx) {
    try { extra = JSON.parse(await readFile(mx, "utf8")); }
    catch (e) { console.warn(`[ares] --meta-extra-file unreadable (${mx}): ${(e as Error).message}`); }
  }
  const meta = { schema: "ares-clip-meta/1", createdAt: new Date().toISOString(), ...base, ...extra };
  await writeFile(outPath + ".meta.json", JSON.stringify(meta, null, 2) + "\n");
  console.log(`[ares] wrote ${outPath}.meta.json — provenance (${Object.keys(meta).length} sections)`);
}

function atlasOf() {
  const a = proceduralAtlas(512);
  return { png: a.png, width: a.width, height: a.height };
}

/** Content fingerprint of a triangle index buffer (Task J fix 2: atlas-repack detection). */
function hashIndices(indices: Uint32Array): string {
  return createHash("sha1").update(Buffer.from(indices.buffer, indices.byteOffset, indices.byteLength)).digest("hex");
}

/** --repack-detect image: decode every atlas at 64² gray in ONE ffmpeg pass and mark frames whose
 *  consecutive mean-abs-diff crosses REPACK_MAD. A true repack measures ~37 luma; a stable layout
 *  ~4 (2026-07-16 stability probe) — the threshold sits in the empty middle. Needed because the
 *  topology heuristic wrongly flags stable-layout rebakes whose GEOMETRY changes per frame. */
const REPACK_MAD = 12;
async function detectRepackByImage(dir: string, pattern: string, startNumber: number, frameCount: number): Promise<Set<number>> {
  const { spawn } = await import("node:child_process");
  const args = ["-v", "error", "-start_number", String(startNumber), "-i", join(dir, pattern),
    "-frames:v", String(frameCount), "-vf", "scale=64:64", "-f", "rawvideo", "-pix_fmt", "gray", "-"];
  const buf: Buffer = await new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath(), args, { stdio: ["ignore", "pipe", "inherit"] });
    const chunks: Buffer[] = [];
    p.stdout.on("data", (c: Buffer) => chunks.push(c));
    p.on("error", reject);
    p.on("close", (code: number) => (code === 0 ? resolve(Buffer.concat(chunks)) : reject(new Error(`ffmpeg repack-detect exited ${code}`))));
  });
  const N = 64 * 64;
  if (buf.length < frameCount * N) throw new Error(`repack-detect: expected ${frameCount * N} bytes, got ${buf.length}`);
  const out = new Set<number>();
  for (let f = 1; f < frameCount; f++) {
    let sum = 0;
    const a = (f - 1) * N, b = f * N;
    for (let i = 0; i < N; i++) sum += Math.abs(buf[b + i]! - buf[a + i]!);
    if (sum / N > REPACK_MAD) out.add(f);
  }
  return out;
}

async function synth(a: string[]) {
  const out = flag(a, "-o") ?? "demo.ares";
  const shape = flag(a, "--shape") ?? "object";
  if (!["object", "talk", "splat"].includes(shape)) throw new Error(`--shape: expected object|talk|splat, got ${JSON.stringify(shape)}`);
  const frames = numFlag(a, "--frames", 60, { min: 1, int: true });
  const fps = numFlag(a, "--fps", 30, { min: 1 });
  if (shape === "splat") {
    const shDegree = numFlag(a, "--sh-degree", 0, { min: 0, max: 1, int: true }) as 0 | 1;
    const sclip = synthSplatClip(frames, fps, shDegree);
    const t0 = performance.now();
    const bytes = await muxClip({ fps, splatFrames: sclip.splatFrames, gopLength: numFlag(a, "--gop", 30, { min: 1, int: true }), splatTemporal: { mode: (flag(a, "--splat-temporal") ?? "auto") as "auto" | "index" | "nn" | "off" }, meta: { title: "ARES synth: splat", encoder: "ares-cli/0.2.0", generator: "synth" } });
    await writeFile(out, bytes);
    console.log(`[ares] synth splat: ${frames} frames @ ${fps}fps, ${sclip.splatFrames[0]!.count} splats/frame, SH degree ${shDegree}`);
    console.log(`[ares] wrote ${out} — ${(bytes.length / 1024).toFixed(1)} KB total, ${(bytes.length / frames / 1024).toFixed(1)} KB/frame, encoded in ${(performance.now() - t0).toFixed(0)}ms`);
    await info(["info", out]);
    return;
  }
  const clip = synthClip(shape as "object" | "talk", frames, fps);
  const texture = has(a, "--no-texture") ? undefined : atlasOf();
  const t0 = performance.now();
  const bytes = await muxClip({
    fps, frames: clip.frames, gopLength: 30, texture,
    meta: { title: `ARES synth: ${shape}`, encoder: "ares-cli/0.1.0", generator: "synth" },
  });
  await writeFile(out, bytes);
  const encMs = performance.now() - t0;
  const verts = clip.frames[0]!.positions.length / 3;
  console.log(`[ares] synth ${shape}: ${frames} frames @ ${fps}fps, ~${verts} verts/frame`);
  console.log(`[ares] wrote ${out} — ${(bytes.length / 1024).toFixed(1)} KB total, ${(bytes.length / frames / 1024).toFixed(1)} KB/frame, encoded in ${encMs.toFixed(0)}ms`);
  await info(["info", out]);
}

/** Find the single immediate subdirectory of `dir` that holds .obj/.ply frames. Returns null when
 *  zero or more than one qualify (ambiguous → leave it to the user). Lets the encoder accept a
 *  parent folder, mirroring the /enhance descent in tools/serve.mjs. */
async function findFramesSubdir(dir: string, names: string[]): Promise<string | null> {
  const hits: string[] = [];
  for (const name of names) {
    const p = join(dir, name);
    try {
      if (!(await stat(p)).isDirectory()) continue;
      const inner = await readdir(p);
      if (inner.some((f) => { const e = extname(f).toLowerCase(); return e === ".obj" || e === ".ply"; })) hits.push(p);
    } catch { /* ignore unreadable entries */ }
  }
  return hits.length === 1 ? hits[0]! : null;
}

/**
 * Shift an edit list from SOURCE frame numbering onto a trimmed window that starts at `from` and
 * holds `n` frames. Every frame index in the document is authored against the untrimmed clip (the
 * editor only ever sees the whole thing), so trimming without this would silently slide every range
 * by `from` frames — a delete authored on the head would land in the middle of the body.
 *
 * Ranges that fall wholly outside the window are dropped; partial ones are clamped. Keyframes are
 * shifted but NEVER dropped, even when they land outside [0, n): prepareRangeAt interpolates
 * between the keyframes bracketing a frame, so discarding an out-of-window one would change the
 * shape of the surviving in-window frames. A copy op whose srcFrame was trimmed away is dropped
 * outright with a warning — its source no longer exists, and silently copying from some other frame
 * would be a fabrication.
 */
function rebaseEditList(list: EditList, from: number, n: number): EditList {
  if (from === 0) return list;
  const ranges: EditRange[] = [];
  for (const r of list.ranges) {
    const s = r.startFrame - from, e = r.endFrame - from;
    if (e < 0 || s > n - 1) continue;                         // wholly outside the kept window
    const nr: EditRange = {
      ...r,
      startFrame: Math.max(0, s),
      endFrame: Math.min(n - 1, e),
      keyframes: r.keyframes.map((k) => ({ ...k, frame: k.frame - from })),
    };
    if (nr.copy) {
      const cp = nr.copy;
      const src = cp.srcFrame - from;
      const dst = cp.dstFrames.map((d: number) => d - from).filter((d: number) => d >= 0 && d < n);
      if (src < 0 || src >= n) {
        console.warn(`[ares] trim: dropping copy range ${r.id ?? `${r.startFrame}-${r.endFrame}`} — its srcFrame ${cp.srcFrame} is outside the trim`);
        continue;
      }
      if (!dst.length) {
        console.warn(`[ares] trim: dropping copy range ${r.id ?? `${r.startFrame}-${r.endFrame}`} — every dstFrame is outside the trim`);
        continue;
      }
      nr.copy = { ...cp, srcFrame: src, dstFrames: dst };
    }
    ranges.push(nr);
  }
  return { ...list, ranges };
}

/** --up-axis / --center / --scale / --rotate / --translate → ModelTransform (null when none given). */
function parseModelTransform(a: string[]): ModelTransform | null {
  const up = flag(a, "--up-axis"), ctr = flag(a, "--center"), rot = flag(a, "--rotate");
  const scl = flag(a, "--scale"), tr = flag(a, "--translate");
  if (up == null && ctr == null && rot == null && scl == null && tr == null) return null;
  if (up != null && !["x", "y", "z"].includes(up)) throw new Error(`--up-axis: expected x|y|z, got ${JSON.stringify(up)}`);
  if (ctr != null && !["bottom", "mass", "none"].includes(ctr)) throw new Error(`--center: expected bottom|mass|none, got ${JSON.stringify(ctr)}`);
  const triple = (s: string | undefined, what: string): [number, number, number] | undefined => {
    if (s == null) return undefined;
    const p = s.split(",").map((v) => Number(v.trim()));
    if (p.length !== 3 || p.some((v) => !Number.isFinite(v))) throw new Error(`${what}: expected three numbers "x,y,z", got ${JSON.stringify(s)}`);
    return p as [number, number, number];
  };
  let scale: number | undefined;
  if (scl != null) {
    scale = Number(scl);
    if (!Number.isFinite(scale) || scale === 0) throw new Error(`--scale: expected a non-zero number, got ${JSON.stringify(scl)}`);
  }
  return {
    upAxis: (up as ModelTransform["upAxis"]) ?? "y",
    center: (ctr as ModelTransform["center"]) ?? "bottom",
    rotate: triple(rot, "--rotate") ?? [0, 0, 0],
    translate: triple(tr, "--translate") ?? [0, 0, 0],
    scale: scale ?? 1,
  };
}

/** Natural sort so frame-000009 < frame-000010 AND frame9 < frame10 (lexicographic order misaligns unpadded names). */
function naturalSort(names: string[]): string[] {
  const key = (n: string) => n.split(/(\d+)/).map((t) => (/^\d+$/.test(t) ? t.padStart(12, "0") : t)).join("");
  return names.slice().sort((x, y) => (key(x) < key(y) ? -1 : key(x) > key(y) ? 1 : 0));
}

type SplatKind = "spz" | "splat" | "sog" | "gltf" | "ply";

/** --audio <file>: transcode to Opus (spec §11.5) trimmed to the clip's window; null when absent. */
async function audioFromFlags(a: string[], fps: number, from: number, frameCount: number): Promise<AudioTrackData | null> {
  const src = flag(a, "--audio");
  if (!src) return null;
  if (!(await ffmpegAvailable())) throw new Error("--audio needs ffmpeg (set FFMPEG or install it)");
  const offset = numFlag(a, "--audio-offset", 0);
  const bitrate = numFlag(a, "--audio-bitrate", 96, { min: 6, max: 510, int: true });
  const t0 = performance.now();
  // A trimmed clip's frame 0 is source frame `from`: the audio window follows it.
  const track = await transcodeToOpus(src, { bitrateKbps: bitrate, offsetSec: offset, startSec: from > 0 ? from / fps : undefined, durationSec: frameCount / fps + 0.25 });
  const bytes = track.packets.reduce((s, p) => s + p.data.byteLength, 0);
  console.log(`[ares] audio: ${basename(src)} → Opus ${bitrate} kb/s, ${track.channels === 1 ? "mono" : "stereo"}, ${track.packets.length} packets, ${(track.durationUs / 1e6).toFixed(2)}s, ${(bytes / 1024).toFixed(1)} KB in ${((performance.now() - t0) / 1000).toFixed(1)}s` +
    (offset ? ` (offset ${offset}s)` : ""));
  if (track.durationUs < (frameCount / fps) * 1e6 * 0.9) console.warn(`[ares] audio is shorter than the clip (${(track.durationUs / 1e6).toFixed(2)}s vs ${(frameCount / fps).toFixed(2)}s) — the tail plays silent`);
  return track;
}

async function encode(a: string[]) {
  let dir = a[1]!;
  const out = flag(a, "-o") ?? "out.ares";
  const fps = numFlag(a, "--fps", 30, { min: 0.001 });
  const maxFrames = flag(a, "--max-frames") ? numFlag(a, "--max-frames", Infinity, { min: 1, int: true }) : Infinity;
  const gop = numFlag(a, "--gop", 30, { min: 1, max: 65535, int: true });

  // Frame discovery. Ease-of-use: if the folder holds no meshes directly but a single subfolder does
  // (e.g. pointing at a parent like Foo/ that contains Foo_Volcap/mesh-*.obj), descend into it.
  const meshesIn = (names: string[]) => ({
    obj: naturalSort(names.filter((f) => extname(f).toLowerCase() === ".obj")),
    ply: naturalSort(names.filter((f) => extname(f).toLowerCase() === ".ply")),
    splat: naturalSort(names.filter((f) => SPLAT_EXTS.has(extname(f).toLowerCase()))),
  });
  let all = await readdir(dir);
  let { obj: objFiles, ply: plyFiles, splat: splatFiles } = meshesIn(all);
  if (!objFiles.length && !plyFiles.length && !splatFiles.length) {
    const subdir = await findFramesSubdir(dir, all);
    if (subdir) {
      console.log(`[ares] no meshes in ${dir}; using frames subfolder ${subdir}`);
      dir = subdir;
      all = await readdir(dir);
      ({ obj: objFiles, ply: plyFiles, splat: splatFiles } = meshesIn(all));
    }
  }
  // A single SOG directory (meta.json + webp images) is one splat frame.
  const sogDir = !objFiles.length && !plyFiles.length && !splatFiles.length && all.includes("meta.json");
  const isObj = objFiles.length > 0;
  // Splat detection: a PLY folder whose first file carries 3DGS attributes is a splat sequence.
  let splatKind: SplatKind | null = null;
  if (!isObj && plyFiles.length) {
    const head = new Uint8Array(await readFile(join(dir, plyFiles[0]!)));
    if (isSplatPlyHeader(parsePlyHeader(head))) splatKind = "ply";
  } else if (!isObj && !plyFiles.length && (splatFiles.length || sogDir)) {
    const ext = sogDir ? ".sog" : extname(splatFiles[0]!).toLowerCase();
    splatKind = ext === ".spz" ? "spz" : ext === ".splat" ? "splat" : ext === ".sog" ? "sog" : "gltf";
  }
  const discovered = (isObj ? objFiles : splatKind && splatKind !== "ply" ? (sogDir ? ["."] : splatFiles) : plyFiles).slice(0, maxFrames);
  if (!discovered.length) throw new Error(`no .obj/.ply mesh frames or .spz/.ply/.splat/.glb/.gltf/.sog splat frames found in ${dir}`);

  // Mesh-editor bake (docs/editor-v2-design.md §10). Parsed BEFORE the trim window is resolved
  // because the sidecar is where the editor persists its in/out points.
  let editList: EditList | null = null;
  const editsArg = flag(a, "--edits");
  if (editsArg) {
    editList = parseEditList(JSON.parse(await readFile(editsArg, "utf8")));
    const muted = editList.ranges.filter((r) => !isRangeEnabled(r)).length;
    if (muted) { console.log(`[ares] edits: ${muted} muted range(s) skipped`); editList.ranges = editList.ranges.filter(isRangeEnabled); }
  }

  // Clip trim (NLE in/out, inclusive SOURCE frame indices). Explicit flags beat the sidecar so a
  // scripted re-bake can override what the app authored. Applied HERE, before frames are read: the
  // trimmed-off frames are never imported, so a trim also costs less to encode instead of more.
  // Parse strictly and STOP on anything unreadable. A trim silently defaulting to 0 on a typo
  // ("--trim-in ten") would encode the wrong frames and look like a success — the same class of
  // failure as the Mathf.Max(64, texW) clamp that turned a bad field read into a garbage export.
  const trimSrc = editList?.trim;
  const num = (v: unknown, what: string): number => {
    const n = Number(v);
    if (!Number.isInteger(n)) throw new Error(`${what}: expected an integer frame index, got ${JSON.stringify(v)}`);
    return n;
  };
  const trimInRaw = flag(a, "--trim-in") ?? trimSrc?.in;
  const trimOutRaw = flag(a, "--trim-out") ?? trimSrc?.out;
  const trimIn = trimInRaw == null ? 0 : num(trimInRaw, "--trim-in");
  const trimOut = trimOutRaw == null ? discovered.length - 1 : num(trimOutRaw, "--trim-out");
  if (trimIn < 0 || trimIn >= discovered.length || trimOut < trimIn)
    throw new Error(`bad trim: in=${trimIn} out=${trimOut} against ${discovered.length} discovered frame(s) — need 0 <= in <= out < ${discovered.length}`);
  const from = trimIn, toExcl = Math.min(discovered.length, trimOut + 1);
  const files = discovered.slice(from, toExcl);
  if (!files.length) throw new Error(`trim ${trimIn}..${trimOut} selects no frames`);
  if (from !== 0 || toExcl !== discovered.length) {
    console.log(`[ares] trim: keeping source frames ${from}..${toExcl - 1} of ${discovered.length} ` +
      `(${files.length} frames, ${(files.length / fps).toFixed(2)}s) — dropping ${from} from the head, ${discovered.length - toExcl} from the tail`);
  }

  if (splatKind) {
    await encodeSplats(a, dir, files, splatKind, out, fps, gop, from, toExcl, discovered.length);
    return;
  }

  console.log(`[ares] ${files.length} ${isObj ? "OBJ" : "PLY"} frame(s) @ ${fps}fps, gop=${gop}`);
  const frames: EncodeMeshFrame[] = [];
  const t0 = performance.now();
  for (const f of files) {
    if (isObj) {
      const m = parseObj(await readFile(join(dir, f), "utf8"));
      frames.push({ positions: m.positions, uvs: m.uvs, indices: m.indices });
    } else {
      const m = parsePly(await readFile(join(dir, f)));
      frames.push({ positions: m.positions, uvs: m.uvs, normals: m.normals, indices: m.indices });
    }
  }
  const v0 = frames[0]!.positions.length / 3;
  console.log(`[ares] imported ${files.length} frames (~${v0} verts) in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // ---- Model transform: bake the up-axis / centring / scale into the geometry ------------------
  // Runs FIRST, before repack detection, edits, decimate and quantization — everything downstream
  // works in world space (crop boxes, brush points, SAM mask unprojection), so a transform applied
  // later would silently mean those were all authored against a different space.
  //
  // The clip-wide bounds are computed over EVERY frame and the resulting offset reused for all of
  // them. Centring each frame on its own bounds would re-centre the subject every frame — a walk
  // cycle would moonwalk in place instead of crossing the floor.
  const modelXf: ModelTransform | null = parseModelTransform(a);
  if (modelXf) {
    const bounds: Aabb = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (const fr of frames) {
      const p = fr.positions;
      for (let i = 0; i < p.length; i += 3) for (let k = 0; k < 3; k++) {
        const v = p[i + k]!;
        if (v < bounds.min[k]!) bounds.min[k] = v;
        if (v > bounds.max[k]!) bounds.max[k] = v;
      }
    }
    const offset = resolveOffset(bounds, modelXf);
    for (const fr of frames) {
      applyTransform(fr.positions, modelXf, offset);
      if (fr.normals) applyTransformNormals(fr.normals, modelXf);
    }
    const after = transformAabb(bounds, modelXf);
    const lo = after.min.map((v, i) => v + offset[i]!), hi = after.max.map((v, i) => v + offset[i]!);
    console.log(`[ares] transform: up=${modelXf.upAxis} center=${modelXf.center} scale=${modelXf.scale} ` +
      `rotate=[${modelXf.rotate}] translate=[${modelXf.translate}]`);
    console.log(`[ares]   bounds ${bounds.min.map((v) => v.toFixed(1))}..${bounds.max.map((v) => v.toFixed(1))} -> ` +
      `${lo.map((v) => v.toFixed(1))}..${hi.map((v) => v.toFixed(1))}  (height ${(hi[1]! - lo[1]!).toFixed(1)}, floor y=${lo[1]!.toFixed(3)})`);
  }

  // Atlas-repack detection (Task J fix 2): a repack frame is one whose index buffer's CONTENT
  // differs from the previous frame's (SHA1 hash compare) — computed here, from the freshly
  // imported, pre-edit frames, because a repack is a property of the SOURCE capture's topology,
  // not of any --edits/--decimate applied below (those can change per-frame indices for reasons
  // unrelated to the source atlas). Passed through to the texture encoder so its GOP segmentation
  // forces a real codec keyframe at every repack — the fix for inter-predicting straight across a
  // 73%-of-pixels atlas repack (the dominant warp/crack artifact on 4DViews captures).
  const repackFrames = new Set<number>();
  {
    let prevHash: string | null = null;
    for (let f = 0; f < frames.length; f++) {
      const h = hashIndices(frames[f]!.indices);
      if (prevHash !== null && h !== prevHash) repackFrames.add(f);
      prevHash = h;
    }
  }
  if (repackFrames.size) {
    console.log(`[ares] detected ${repackFrames.size} atlas-repack frame(s) (topology reset vs previous frame): ` +
      `${[...repackFrames].slice(0, 20).join(", ")}${repackFrames.size > 20 ? ", …" : ""} — texture GOPs will force a keyframe at each`);
  }

  // Time-ranged, keyframe-interpolated regions applied per frame via the shared centroid predicate.
  // The sidecar's frame indices are authored against the UNTRIMMED source, so they must be re-based
  // onto the kept window before anything below reads a frame number. --crop is the degenerate
  // one-keyframe keep-box edit list, so both flags flow through the same path (preview == bake) —
  // and it is built AFTER the re-base, in trimmed space, since frames.length is already trimmed.
  if (editList) editList = rebaseEditList(editList, from, files.length);
  const cropArg = flag(a, "--crop");
  if (cropArg) {
    const box = parseCropBox(cropArg);
    const cropRange = {
      mode: "keep" as const, startFrame: 0, endFrame: frames.length - 1,
      keyframes: [{ frame: 0, volumes: [{ type: "box" as const, min: box.min, max: box.max }] }],
    };
    editList = editList ? { ...editList, ranges: [...editList.ranges, cropRange] } : { aresEdits: 1, ranges: [cropRange] };
  }

  // Frame-to-frame copy (action:"copy"): "fix a bad frame from a
  // good one." Runs BEFORE delete/keep — a user can copy-replace a region and further delete/
  // crop in the same sidecar — and before decimate. keepPredicateAt (core) excludes action:
  // "copy" ranges from its own delete/keep sweeps, so the block below never re-processes (and
  // un-does) the pasted-in region. Geometry mutates here; the texel blit itself runs later,
  // next to the atlas-file read (it patches PNGs on disk) — but it executes the PLAN built
  // here, from frames exactly as imported: the texture block runs after delete/keep and
  // --decimate, and a footprint rasterized from already-decimated geometry would no longer
  // match what the geo half of the same op pasted.
  // The atlases must take the SAME window as the meshes, or every frame would wear the texture of a
  // frame `from` earlier.
  const atlasFiles = naturalSort(all.filter((f) => /atlas.*\.png$/i.test(f))).slice(0, maxFrames).slice(from, toExcl);
  const copyOps: CopyOp[] = editList ? collectCopyOps(editList.ranges, frames.length) : [];
  const recolorOps = editList ? collectRecolorOps(editList.ranges, frames.length) : [];
  const paintOps = editList ? collectPaintOps(editList.ranges, frames.length) : [];
  const copyPastes = new Map<CopyOp, Map<number, EncodeMeshFrame>>();
  const texelPlans = new Map<CopyOp, Map<number, TexelPatchPlan>>();
  const copyFragments = new Map<CopyOp, EncodeMeshFrame>();
  if (copyOps.length) {
    for (const op of copyOps) copyFragments.set(op, resolveSrcFragment(frames, op));
    // Atlas-space planning. This capture's atlas is REPACKED EVERY FRAME (measured: 57.8% of
    // the dst triangles sampling a same-coordinate patch at 30 frames' distance belong to other
    // body parts; 78.1% at ONE frame), so:
    //   "both"   → RELOCATE: translate the src footprint's charts (tiling the ones that don't
    //              fit whole) into space that is free in the dst frame's atlas after this op's
    //              own dst-region deletion; the pasted UVs move with their texels.
    //   "texels" → same-coordinate blit, but REFUSED above 5% collateral (dst triangles outside
    //              the region that sample the patched area).
    //   "geo"    → unchanged UVs; loud warning (only valid on temporally-coherent atlases).
    const planned = copyOps.some((o) => o.what !== "geo");
    const atlasPath = atlasFiles.length ? join(dir, atlasFiles[0]!) : null;
    if (planned && !atlasPath) {
      console.warn(`[ares] copy: no atlas PNGs found — texel copy skipped; pasting with unchanged UVs (they will sample whatever texture the file ends up with)`);
    }
    if (planned && atlasPath) {
      const { width: aw, height: ah } = pngSize(await readFile(atlasPath));
      const placedByDst = new Map<number, Uint8Array>(); // dst frame → charts already relocated there by earlier ops
      for (const op of copyOps) {
        const id = op.range.id ?? "?";
        if (op.what === "geo") continue;
        const inside = prepareRangeAt(op.range, op.srcFrame);
        if (op.what === "texels") {
          const foot = dilateMask(rasterizeUvFootprint(copyFragments.get(op)!, aw, ah), aw, ah, 2);
          const plans = new Map<number, TexelPatchPlan>();
          for (const d of op.dstFrames) {
            const { affected, collateral } = measureTexelCollateral(frames[d]!, foot, aw, ah, inside);
            const pct = affected ? (collateral / affected) * 100 : 0;
            if (pct > 5) {
              throw new Error(`copy ${id} what:"texels" (srcFrame ${op.srcFrame} → frame ${d}): ${collateral} of ${affected} dst triangles sampling the patch area (${pct.toFixed(1)}%) lie OUTSIDE the copy region — ` +
                `this capture's atlas is packed per frame, so a same-coordinate texel copy would overwrite other surfaces. Use what:"both" (relocating copy) instead.`);
            }
            if (collateral) console.warn(`[ares] copy ${id} what:"texels" → frame ${d}: ${collateral}/${affected} affected dst triangles (${pct.toFixed(1)}%) are outside the region — proceeding (≤5%)`);
            plans.set(d, identityPlan(foot, aw, ah));
          }
          texelPlans.set(op, plans);
        } else { // "both"
          const fragment = copyFragments.get(op)!;
          const plans = new Map<number, TexelPatchPlan>();
          const pastes = new Map<number, EncodeMeshFrame>();
          for (const d of op.dstFrames) {
            // Occupancy = the dst triangles that SURVIVE this op's own deletion: the geo half
            // removes the region from dst before pasting, so the region's old dst charts are
            // sampled by nothing after the bake — legitimate free space, and the roomiest kind
            // (it hosted the same body part the paste brings in). The raw inter-chart gutters
            // alone are NOT enough here: this region's largest chart is ~316k px with a 678×533
            // bbox, and no gutter gap that size exists in a 58-60%-occupied packed atlas.
            const kept = filterFrame(frames[d]!, (x, y, z) => !inside(x, y, z));
            const claimed = dilateMask(rasterizeUvFootprint(kept, aw, ah), aw, ah, 2);
            const prior = placedByDst.get(d);
            if (prior) for (let i = 0; i < claimed.length; i++) if (prior[i]) claimed[i] = 1;
            const { pieces, wholeCharts, tiledCharts } = planRegionRelocation(fragment, claimed, aw, ah);
            let mine = placedByDst.get(d);
            if (!mine) { mine = new Uint8Array(aw * ah); placedByDst.set(d, mine); }
            for (const p of pieces) {
              const dd = p.dy * aw + p.dx;
              for (let i = 0; i < p.pixels.length; i++) mine[p.pixels[i]! + dd] = 1;
            }
            plans.set(d, { width: aw, height: ah, components: pieces.filter((p) => p.pixels.length) });
            pastes.set(d, buildPieceFragment(fragment, pieces, aw, ah));
            console.log(`[ares] copy ${id} → frame ${d}: relocated region texels into free atlas space — ${wholeCharts} chart(s) whole, ${tiledCharts} tiled, ${pieces.length} piece(s)`);
          }
          texelPlans.set(op, plans);
          copyPastes.set(op, pastes);
        }
      }
    }
    for (const op of copyOps) {
      if (op.what === "texels") continue; // geometry untouched by a texels-only copy
      if (op.what === "geo") {
        console.warn(`[ares] copy ${op.range.id ?? "?"} what:"geo": pasted triangles keep their src UVs and sample the DST atlas — ` +
          `this capture's atlas is packed per frame, so expect wrong texels; use what:"both" unless the atlas is temporally coherent`);
      }
      const { stats, srcTris } = applyGeoCopy(frames, op, { fragment: copyFragments.get(op), pasteFragments: copyPastes.get(op) });
      if (srcTris === 0) {
        console.warn(`[ares] copy ${op.range.id ?? "?"}: srcFrame ${op.srcFrame}'s region matched 0 triangles — dst frame(s) only had the region deleted, nothing was pasted`);
      }
      for (const s of stats) {
        console.log(`[ares] copy ${op.range.id ?? "?"} (srcFrame ${op.srcFrame} → frame ${s.frame}): ${s.before} → ${s.after} tris/frame (pasted ${srcTris} src tris)`);
      }
    }
  }

  // Garment recolor, tier 1 (action:"recolor"): deterministic
  // masked retint/hue-shift of a region's texels, per frame — no geometry is touched. Runs
  // AFTER the copy geometry-apply loop above (so recoloring a region that a copy op just
  // pasted in samples the PASTED/relocated UVs, not the pre-paste ones — "recoloring a pasted
  // region" from the task spec) and BEFORE decimate (recolor plans against the ORIGINAL,
  // pre-decimate UVs, same reasoning as the copy footprint planning above). Unlike copy, each
  // frame resolves its OWN region via prepareRangeAt(r, f) (per-frame keyframe interpolation)
  // against its OWN mesh — no cross-frame relocation is needed or used (recolor.ts header).
  const recolorPatches = new Map<number, RecolorPatch[]>();
  if (recolorOps.length) {
    if (has(a, "--no-texture")) {
      console.warn(`[ares] recolor: --no-texture set — recolor only ever produces atlas-texel patches, skipping ${recolorOps.length} range(s)`);
    } else if (!atlasFiles.length) {
      console.warn(`[ares] recolor: no atlas PNGs found — skipping (recolor only ever produces atlas-texel patches)`);
    } else {
      const atlasPath = join(dir, atlasFiles[0]!);
      const { width: rw, height: rh } = pngSize(await readFile(atlasPath));
      for (const op of recolorOps) {
        const id = op.range.id ?? "?";
        let framesTouched = 0, totalSel = 0, totalDilated = 0, totalBled = 0;
        for (let f = op.startFrame; f <= op.endFrame; f++) {
          const inside = prepareRangeAt(op.range, f);
          const { sel, other } = partitionTrianglesByCentroid(frames[f]!, inside);
          if (!sel.length) continue; // region matches nothing at this frame — no-op, like a copy with 0 src tris
          const selFrag: EncodeMeshFrame = { positions: frames[f]!.positions, uvs: frames[f]!.uvs, indices: sel };
          const otherFrag: EncodeMeshFrame = { positions: frames[f]!.positions, uvs: frames[f]!.uvs, indices: other };
          const selFoot = rasterizeUvFootprint(selFrag, rw, rh);
          // Occupancy of the COMPLEMENT (every OTHER triangle's footprint): masks the dilation
          // ring so a 2px bleed cannot paint texels a neighboring UV chart actually samples —
          // the exact selected footprint always recolors regardless (those texels ARE the
          // region), only the dilation RING is dropped where it would land on foreign-chart
          // texels. Reuses the same occupancy rasterizer texel-copy.ts's relocation planner
          // uses (rasterizeUvFootprint) rather than a bespoke bleed test.
          const otherFoot = rasterizeUvFootprint(otherFrag, rw, rh);
          const dilated = dilateMask(selFoot, rw, rh, 2);
          const mask = new Uint8Array(rw * rh);
          let selPx = 0, dilPx = 0, bled = 0;
          for (let p = 0; p < mask.length; p++) {
            if (selFoot[p]) { mask[p] = 1; selPx++; continue; }
            if (!dilated[p]) continue;
            dilPx++;
            if (otherFoot[p]) { bled++; continue; } // would land on a neighboring chart's texels — excluded
            mask[p] = 1;
          }
          totalSel += selPx; totalDilated += dilPx; totalBled += bled; framesTouched++;
          const arr = recolorPatches.get(f) ?? [];
          arr.push({ mask, color: op.color, strength: op.strength, mode: op.mode, rangeId: id });
          recolorPatches.set(f, arr);
        }
        console.log(`[ares] recolor ${id} (frames ${op.startFrame}-${op.endFrame}, ${op.mode} ${hexOf(op.color)} strength ${op.strength}): ` +
          `${framesTouched} frame(s) touched, ${totalSel} selected texel(s), ${totalDilated} dilation-ring px total (${totalBled} would bleed onto a neighboring chart — masked out)`);
      }
    }
  }

  // Texture paint (action:"paint"): texel-granular soft brush over the range's world-anchored
  // region — sculpt+paint plan §B, build item 1. Same pipeline position as recolor (after copy,
  // before decimate), same per-frame own-mesh own-atlas resolution; differs only in per-texel
  // SDF-feathered weights (paint.ts). The complement footprint guards the 2 px edge dilation
  // exactly as recolor's dilation ring is guarded.
  const paintPatches = new Map<number, PaintPatch[]>();
  if (paintOps.length) {
    if (has(a, "--no-texture")) {
      console.warn(`[ares] paint: --no-texture set — paint only ever produces atlas-texel patches, skipping ${paintOps.length} range(s)`);
    } else if (!atlasFiles.length) {
      console.warn(`[ares] paint: no atlas PNGs found — skipping (paint only ever produces atlas-texel patches)`);
    } else {
      const atlasPath = join(dir, atlasFiles[0]!);
      const { width: rw, height: rh } = pngSize(await readFile(atlasPath));
      for (const op of paintOps) {
        const id = op.range.id ?? "?";
        let framesTouched = 0, totalPx = 0;
        for (let f = op.startFrame; f <= op.endFrame; f++) {
          const sdf = prepareRangeSdfAt(op.range, f);
          if (!sdf) continue; // keyframe-less range — same no-op law as delete preview
          const weights = rasterizePaintWeights(frames[f]!, sdf, rw, rh, op.feather);
          if (!weights) continue;
          // complement occupancy = footprint of triangles fully OUTSIDE the feathered region
          const { other } = partitionTrianglesByCentroid(frames[f]!, (x, y, z) => sdf(x, y, z) < op.feather);
          const otherFoot = rasterizeUvFootprint({ positions: frames[f]!.positions, uvs: frames[f]!.uvs, indices: other }, rw, rh);
          dilatePaintWeights(weights, rw, rh, 2, otherFoot);
          let px = 0;
          for (let p = 0; p < weights.length; p++) if (weights[p]! > 0) px++;
          totalPx += px; framesTouched++;
          const arr = paintPatches.get(f) ?? [];
          arr.push({ weights, brush: op.brush, color: op.color, strength: op.strength, rangeId: id });
          paintPatches.set(f, arr);
        }
        console.log(`[ares] paint ${id} (frames ${op.startFrame}-${op.endFrame}, ${op.brush}${op.color ? " " + hexOf(op.color) : ""} strength ${op.strength} feather ${op.feather.toFixed(0)}mm): ` +
          `${framesTouched} frame(s) touched, ${totalPx} weighted texel(s) total`);
      }
    }
  }

  // Sculpt (action:"sculpt"): world-anchored vertex displacement — after the texel ops (which
  // rasterize against the un-sculpted UV footprints they were authored on) and BEFORE delete/keep
  // and decimate, so a sculpted region that is also cropped is shaped first, then cut.
  const sculptOps = editList ? collectSculptOps(editList.ranges, frames.length) : [];
  for (const op of sculptOps) {
    const id = op.range.id ?? "?";
    let framesTouched = 0, verts = 0, feather = 0;
    for (let f = op.startFrame; f <= op.endFrame; f++) {
      const st = applySculptToFrame(frames[f]!, op, f);
      if (!st) continue;
      framesTouched++; verts += st.vertices; feather = st.feather;
    }
    console.log(`[ares] sculpt ${id} (${op.brush}, frames ${op.startFrame}-${op.endFrame}${op.brush === "move" ? `, offset [${op.offset}]` : `, amount ${op.amount}`}${op.brush === "smooth" ? `, ${op.iterations} it` : ""}): ` +
      `${framesTouched} frame(s) touched, ${verts} vertex moves total, feather ${feather.toFixed(1)}`);
  }

  if (editList) {
    let before = 0, after = 0;
    for (let f = 0; f < frames.length; f++) {
      const keep = keepPredicateAt(editList, f);
      before += frames[f]!.indices.length / 3;
      if (keep) frames[f] = filterFrame(frames[f]!, keep);
      after += frames[f]!.indices.length / 3;
    }
    console.log(`[ares] edits (${editList.ranges.length} range${editList.ranges.length === 1 ? "" : "s"}${editsArg ? ", " + editsArg : ""}): ` +
      `${Math.round(before / frames.length)} → ${Math.round(after / frames.length)} tris/frame (${((1 - after / before) * 100).toFixed(1)}% removed)`);
  }

  // Hole patching after deletion (patchHoles modifier on a plain delete range): runs AFTER
  // the delete/keep sweep above (loops are detected in the mesh the
  // deletion actually left behind) and BEFORE decimate (so decimate's LockBorder can pin the new
  // cap tile's seam vertices exactly like any other atlas-chart seam — see hole-patch.ts header).
  const holePatchRanges = editList
    ? editList.ranges.filter((r) => r.patchHoles && r.mode === "delete" && (r.action ?? "delete") === "delete")
    : [];
  const holeTileFills = new Map<number, HoleTileFill[]>(); // frame → tile fills, consumed by buildPatchedAtlasDir below
  if (editList) {
    for (const r of editList.ranges) {
      if (r.patchHoles && !holePatchRanges.includes(r)) {
        console.warn(`[ares] hole-patch ${r.id ?? "?"}: patchHoles is only meaningful on a plain delete range (mode "delete", no action or action "delete") — ignored`);
      }
    }
  }
  if (holePatchRanges.length) {
    let atlasW = 0, atlasH = 0;
    const haveAtlas = atlasFiles.length > 0;
    if (haveAtlas) { const s = pngSize(await readFile(join(dir, atlasFiles[0]!))); atlasW = s.width; atlasH = s.height; }
    if (!haveAtlas) console.warn(`[ares] hole-patch: no atlas PNGs found — cap geometry will be appended with UVs defaulting to (0,0)`);
    if (has(a, "--no-texture")) console.warn(`[ares] hole-patch: --no-texture set — cap geometry is appended but texel fill never runs (no texture-video track means nothing to paint)`);

    for (const r of holePatchRanges) {
      const id = r.id ?? "?";
      const explicitColor = r.patchHoles!.color ? parseHexColor(r.patchHoles!.color, id) : undefined;
      const startF = Math.max(0, r.startFrame), endF = Math.min(r.endFrame, frames.length - 1);
      let loopsFound = 0, loopsCapped = 0, loopsRejected = 0, nonManifoldTotal = 0, capTrisTotal = 0, tilesTotal = 0, tilesFailed = 0;
      for (let f = startF; f <= endF; f++) {
        const { accepted, rejectedRegion, nonManifoldVertices } = detectHoleLoopsForRange(frames[f]!, r, f, String(f));
        loopsFound += accepted.length + rejectedRegion;
        loopsRejected += rejectedRegion;
        nonManifoldTotal += nonManifoldVertices;
        if (!accepted.length) continue;

        let tileUvs: ([number, number] | null)[] = accepted.map(() => null);
        let placements: ({ x: number; y: number } | null)[] = accepted.map(() => null);
        if (haveAtlas && frames[f]!.uvs) {
          const occ = dilateMask(rasterizeUvFootprint(frames[f]!, atlasW, atlasH), atlasW, atlasH, 2);
          placements = accepted.map(() => {
            const p = findFreeTile(occ, atlasW, atlasH, HOLE_TILE);
            if (p) for (let dy = 0; dy < HOLE_TILE; dy++) for (let dx = 0; dx < HOLE_TILE; dx++) occ[(p.y + dy) * atlasW + (p.x + dx)] = 1;
            return p;
          });
          tileUvs = placements.map((p) => (p ? [(p.x + HOLE_TILE / 2) / atlasW, (p.y + HOLE_TILE / 2) / atlasH] : null));
          tilesFailed += placements.filter((p) => !p).length;
        }

        const { frame: patched, capTriangles, rimUvPerLoop } = appendCaps(frames[f]!, accepted, tileUvs);
        frames[f] = patched;
        loopsCapped += accepted.length;
        capTrisTotal += capTriangles;

        if (haveAtlas) {
          const jobs = holeTileFills.get(f) ?? [];
          placements.forEach((p, li) => {
            if (!p) return; // no free tile found for this loop — cap geometry stands, texel fill skipped for it
            const samplePx: [number, number][] = rimUvPerLoop[li]!.map(([u, v]) => [
              Math.min(atlasW - 1, Math.max(0, Math.floor(u * atlasW))),
              Math.min(atlasH - 1, Math.max(0, Math.floor(v * atlasH))),
            ]);
            jobs.push({ rect: { x: p.x, y: p.y, w: HOLE_TILE, h: HOLE_TILE }, color: explicitColor, samplePx, rangeId: id });
            tilesTotal++;
          });
          if (jobs.length) holeTileFills.set(f, jobs);
        }
      }
      console.log(`[ares] hole-patch ${id} (frames ${startF}-${endF}): ${loopsFound} loop(s) found, ${loopsRejected} rejected (outside the delete region — e.g. pre-existing capture boundaries), ` +
        `${loopsCapped} capped (${capTrisTotal} cap tri total), ${tilesTotal} tile(s) allocated` +
        (tilesFailed ? `, ${tilesFailed} tile alloc FAILED (no free space — cap stands, untextured)` : "") +
        (nonManifoldTotal ? `, ${nonManifoldTotal} non-manifold boundary vertex/vertices skipped` : ""));
    }
  }

  // Optional decimation (--decimate <ratio>, e.g. 0.6 keeps ~60% of triangles): meshopt
  // simplify with locked atlas-seam borders, so UVs map verbatim (decimate.ts). Runs after
  // edits (fewer triangles to simplify) and before smoothing/normals/quantization.
  const decimateArg = flag(a, "--decimate");
  if (decimateArg) {
    const ratio = Number(decimateArg);
    if (!(ratio > 0 && ratio < 1)) throw new Error(`--decimate expects a ratio in (0,1), got ${decimateArg}`);
    await simplifierReady();
    const td = performance.now();
    let before = 0, after = 0;
    for (let f = 0; f < frames.length; f++) {
      before += frames[f]!.indices.length / 3;
      frames[f] = decimateFrame(frames[f]!, ratio);
      after += frames[f]!.indices.length / 3;
    }
    console.log(`[ares] decimate ${ratio}: ${Math.round(before / frames.length)} → ${Math.round(after / frames.length)} tris/frame ` +
      `(${((1 - after / before) * 100).toFixed(1)}% removed, error-capped at 1% of extent) in ${((performance.now() - td) / 1000).toFixed(1)}s`);
  }

  // Texture-video track (spec §7.1): encode the per-frame atlas PNGs via ffmpeg → VP9/AV1.
  // (atlasFiles was computed up at the copy stage — the relocation planner needs it too.)
  let textureVideo: MuxClip["textureVideo"];
  const noTexture = has(a, "--no-texture");
  if (!noTexture && atlasFiles.length >= files.length) {
    if (!(await ffmpegAvailable())) {
      console.warn("[ares] ffmpeg not found — skipping texture (set FFMPEG or install ffmpeg). Geometry-only.");
    } else {
      const pat = detectPattern(atlasFiles);
      if (!pat) console.warn("[ares] could not detect atlas filename pattern — skipping texture.");
      else {
        const codecRaw = flag(a, "--texture-codec") ?? "vp9";
        if (codecRaw !== "vp9" && codecRaw !== "av1") throw new Error(`--texture-codec: expected vp9|av1, got ${JSON.stringify(codecRaw)}`);
        const codec = codecRaw as TexCodec;
        const size = numFlag(a, "--tex-size", 1024, { min: 16, max: 8192, int: true });
        if (size % 2) throw new Error(`--tex-size: ${size} is odd; 4:2:0 video needs even dimensions`);
        const crf = numFlag(a, "--crf", 32, { min: 0, max: 63, int: true });

        // Copy texels (frame-copy "texels"/"both", editor v3 §1): patch a scratch copy of the
        // atlas directory before ffmpeg reads it — encodeTextureVideo reads atlas PNGs straight
        // off disk, so the copy has to land there too. Source atlas dir is never modified.
        let texDir = dir;
        let cleanupTexDir: (() => Promise<void>) | undefined;
        const texelOps = copyOps.filter((o) => o.what !== "geo");
        if (texelOps.length || recolorPatches.size || holeTileFills.size || paintPatches.size) {
          const tp = performance.now();
          const patched = await buildPatchedAtlasDir(dir, atlasFiles, texelOps, texelPlans, recolorPatches, holeTileFills, paintPatches);
          texDir = patched.dir;
          cleanupTexDir = patched.cleanup;
          console.log(`[ares] atlas patch: ${patched.patchedFrames} atlas frame(s) patched (copy + recolor + paint + hole-patch, one decode each) in ${((performance.now() - tp) / 1000).toFixed(1)}s`);
        }

        // --repack-detect image: measure actual atlas content change instead of inferring a
        // repack from geometry topology. The topology heuristic is right for raw captures (a
        // topology reset there always re-packs the atlas), but WRONG for stable-layout rebakes
        // (v7-style: topology changes every frame while the atlas layout holds still) — it
        // would force a keyframe per frame and forfeit the entire inter-coding win. Image mode:
        // one ffmpeg pass decodes every atlas at 64² gray; consecutive mean-abs-diff over ~12
        // luma (measured: within-run ~4, at a true repack ~37) marks a repack.
        let texRepack = repackFrames;
        if (flag(a, "--repack-detect") === "image") {
          texRepack = await detectRepackByImage(texDir, pat.pattern, pat.startNumber, files.length);
          console.log(`[ares] image-based repack detection: ${texRepack.size} repack frame(s)` +
            (texRepack.size ? ` — ${[...texRepack].slice(0, 20).join(", ")}${texRepack.size > 20 ? ", …" : ""}` : ""));
        }
        console.log(`[ares] encoding texture video (${codec} ${size}²) from ${atlasFiles.length} atlas frames — this runs ffmpeg per GOP…`);
        const tt = performance.now();
        // .finally: the scratch atlas copy can be GBs — reclaim it even when ffmpeg throws.
        const tv = await encodeTextureVideo({ dir: texDir, pattern: pat.pattern, startNumber: pat.startNumber, frameCount: files.length, gopLength: gop, fps, codec, size, crf, repackFrames: texRepack })
          .finally(() => cleanupTexDir?.());
        const texBytes = tv.gops.reduce((s, g) => s + g.frames.reduce((ss, f) => ss + f.data.byteLength, 0), 0);
        textureVideo = { fourcc: tv.fourcc, width: tv.width, height: tv.height, gops: tv.gops };
        console.log(`[ares] texture video: ${tv.fourcc} ${tv.width}x${tv.height}, ${(texBytes / 1048576).toFixed(2)} MB (${(texBytes / files.length / 1024).toFixed(1)} KB/frame) in ${((performance.now() - tt) / 1000).toFixed(1)}s`);
      }
    }
  } else if (!noTexture && atlasFiles.length) {
    console.warn(`[ares] found ${atlasFiles.length} atlas PNGs but ${files.length} meshes — count mismatch, skipping texture.`);
  }

  const temporal = {
    track: has(a, "--track"),
    smoothSpatial: numFlag(a, "--smooth", 0, { min: 0, int: true }),
    smoothTemporal: numFlag(a, "--smooth-temporal", 0, { min: 0, int: true }),
    // --no-temporal: force all-intra geometry (proven-safe for multi-topology coherent bakes, whose
    // temporal I+P path shredded geometry across run/topology boundaries).
    forceIntra: has(a, "--no-temporal"),
  };
  const audio = await audioFromFlags(a, fps, from, frames.length);
  const clip: MuxClip = { fps, frames, gopLength: gop, textureVideo, temporal, audio: audio ?? undefined, meta: { title: dir, encoder: "ares-cli/0.2.0", source: isObj ? "obj" : "ply" } };
  const t1 = performance.now();
  const r = await muxClipWithStats(clip);
  await writeFile(out, r.bytes);
  const mode = r.temporalFrames && !r.intraFrames ? "temporal (I+P)" : r.temporalFrames ? "mixed I+P/intra" : "intra-only";
  console.log(`[ares] geometry: ${mode} — ${r.temporalFrames} temporal + ${r.intraFrames} intra frames` +
    (temporal.track ? `, mean track error ${(r.meanTrackError * 100).toFixed(2)}% of bbox` : ""));
  console.log(`[ares] wrote ${out} — ${(r.bytes.length / 1048576).toFixed(2)} MB, ${(r.bytes.length / files.length / 1024).toFixed(1)} KB/frame, muxed in ${((performance.now() - t1) / 1000).toFixed(1)}s`);

  // Bound-by-blood provenance. Records the FULL encode recipe (every flag that shaped this .ares) so
  // it can never be lost or mis-transposed onto another clip again.
  const usedTexture = !!textureVideo;

  // MEASURE the source we actually consumed, in bytes. Without this the app has nothing to compare
  // against but a hardcoded constant from whichever capture happened to be measured first — which is
  // a different clip's numbers presented as this one's. Measured over exactly `files`/`atlasFiles`
  // (post-trim, post-max-frames), so it's the bytes this .ares was really made from. Recorded here
  // rather than computed on demand because the source folder is routinely deleted once the .ares
  // exists — the provenance has to outlive it ("bound by blood").
  const sumBytes = async (names: string[], from: string): Promise<number> => {
    let n = 0;
    for (const f of names) { try { n += (await stat(join(from, f))).size; } catch { /* skip unreadable */ } }
    return n;
  };
  const srcMeshBytes = await sumBytes(files, dir);
  const srcTexBytes = await sumBytes(atlasFiles, dir);
  // MiB, matching every other size this project prints (the app's readout, `info` above). The ratio
  // is a pure byte ratio either way — it's the LABELS that have to agree, or a reader compares an SI
  // MB against a MiB and is quietly 4.9% out.
  const MB = (b: number) => +(b / 1048576).toFixed(2);
  console.log(`[ares] source measured: ${MB(srcMeshBytes + srcTexBytes)} MB in ${files.length + atlasFiles.length} files ` +
    `(mesh ${MB(srcMeshBytes)} + atlas ${MB(srcTexBytes)}) → ${MB(r.bytes.length)} MB .ares ` +
    `= ${((srcMeshBytes + srcTexBytes) / Math.max(1, r.bytes.length)).toFixed(1)}× smaller, 1 file`);

  await writeClipMeta(a, out, {
    output: { name: basename(out), sizeBytes: r.bytes.length, frames: files.length, fps, durationS: +(files.length / fps).toFixed(2) },
    source: {
      dir, kind: isObj ? "obj+atlas" : "ply+atlas", meshFrames: files.length,
      atlasFrames: atlasFiles.length, repackFrames: repackFrames.size,
      discoveredFrames: discovered.length,
      // The REAL bytes this clip was made from — what the app's savings comparison must read.
      meshBytes: srcMeshBytes, texBytes: srcTexBytes, totalBytes: srcMeshBytes + srcTexBytes,
      fileCount: files.length + atlasFiles.length,
      vertexCount: v0,
      measured: "encode-time stat of the exact files consumed",
    },
    encode: {
      gop,
      // Trim is provenance, not decoration: without it a 200-frame .ares cut from a 272-frame
      // capture has no record of WHICH 200, and its frame numbers no longer match the source's.
      trim: { in: from, out: toExcl - 1, sourceFrames: discovered.length, from: flag(a, "--trim-in") != null || flag(a, "--trim-out") != null ? "flags" : trimSrc ? "edits-sidecar" : "none" },
      texSize: usedTexture ? Number(flag(a, "--tex-size") ?? 1024) : null,
      textureCodec: usedTexture ? (flag(a, "--texture-codec") ?? "vp9") : null,
      crf: usedTexture ? Number(flag(a, "--crf") ?? 32) : null,
      quantBitsPos: 14, quantBitsUv: 16, // muxer defaults (UVs are 16-bit; the superblock field used to misreport 14)
      noTexture, forceIntra: temporal.forceIntra, track: temporal.track,
      smoothSpatial: temporal.smoothSpatial, smoothTemporal: temporal.smoothTemporal,
      decimate: decimateArg ? Number(decimateArg) : null,
      crop: cropArg ?? null,
      edits: editsArg ? basename(editsArg) : null,
      audio: audio ? { source: basename(flag(a, "--audio")!), bitrateKbps: numFlag(a, "--audio-bitrate", 96), offsetSec: numFlag(a, "--audio-offset", 0), packets: audio.packets.length } : null,
    },
    geometry: { mode, temporalFrames: r.temporalFrames, intraFrames: r.intraFrames },
    tooling: { encoder: "ares-cli/0.2.0", node: process.version, generatedBy: "ares encode" },
  });
  await info(["info", out]);
}

/**
 * Splat-profile encode (spec §6.8): one splat file per frame — SPZ (Scaniverse, Marble), 3DGS PLY
 * (every trainer, Polycam, Luma), .splat, glTF/GLB with KHR_gaussian_splatting, or SOG (SuperSplat).
 * Positions are taken in the file's own frame; --rotate/--up-axis/--center/--scale bake an
 * orientation exactly as for meshes (rotations and scales follow the transform).
 */
async function encodeSplats(a: string[], dir: string, files: string[], kind: SplatKind, out: string, fps: number, gop: number, from: number, toExcl: number, discoveredCount: number): Promise<void> {
  console.log(`[ares] ${files.length} ${kind.toUpperCase()} splat frame(s) @ ${fps}fps, gop=${gop}`);
  const t0 = performance.now();
  let frames: SplatFrame[] = [];
  for (const f of files) {
    const path = f === "." ? dir : join(dir, f);
    let fr: SplatFrame;
    if (kind === "sog") fr = await parseSog(path);
    else {
      const buf = new Uint8Array(await readFile(path));
      fr = kind === "spz" ? parseSpz(buf)
        : kind === "splat" ? parseSplatFile(buf)
        : kind === "gltf" ? parseGltfSplat(buf, { loadUri: (u) => new Uint8Array(readFileSync(join(dir, u))) })
        : parsePlySplat(buf);
    }
    frames.push(fr);
  }
  const n0 = frames[0]!.count;
  const maxDegree = frames.reduce((m, f) => Math.max(m, f.sh ? f.shDegree : 0), 0);
  console.log(`[ares] imported ${files.length} splat frame(s) (~${n0} splats, SH degree ${maxDegree}) in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const minAlpha = numFlag(a, "--splat-min-alpha", 0, { min: 0, max: 1 });
  if (minAlpha > 0) {
    let before = 0, after = 0;
    frames = frames.map((f) => { before += f.count; const g = filterSplatFrame(f, (i) => f.opacities[i]! >= minAlpha); after += g.count; return g; });
    console.log(`[ares] --splat-min-alpha ${minAlpha}: ${before} → ${after} splats (${((1 - after / Math.max(1, before)) * 100).toFixed(1)}% dropped)`);
  }

  const modelXf = parseModelTransform(a);
  if (modelXf) {
    const bounds: Aabb = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (const fr of frames) {
      const p = fr.positions;
      for (let i = 0; i < p.length; i += 3) for (let k = 0; k < 3; k++) {
        const v = p[i + k]!;
        if (v < bounds.min[k]!) bounds.min[k] = v;
        if (v > bounds.max[k]!) bounds.max[k] = v;
      }
    }
    const offset = resolveOffset(bounds, modelXf);
    const m = transformMatrix(modelXf, offset);
    for (const fr of frames) transformSplatFrame(fr, m);
    console.log(`[ares] transform: up=${modelXf.upAxis} center=${modelXf.center} scale=${modelXf.scale} rotate=[${modelXf.rotate}] translate=[${modelXf.translate}]`);
  }

  const shDegree = flag(a, "--sh-degree") !== undefined ? numFlag(a, "--sh-degree", 3, { min: 0, max: 3, int: true }) : undefined;
  const quantBits = numFlag(a, "--quant-bits", 14, { min: 8, max: 16, int: true });
  const orderRaw = flag(a, "--splat-order") ?? "morton";
  if (orderRaw !== "morton" && orderRaw !== "none") throw new Error(`--splat-order: expected morton|none, got ${JSON.stringify(orderRaw)}`);
  const boxAlpha = numFlag(a, "--splat-box-alpha", 0, { min: 0, max: 1 });
  const tmodeRaw = flag(a, "--splat-temporal") ?? "auto";
  if (!["auto", "index", "nn", "off"].includes(tmodeRaw)) throw new Error(`--splat-temporal: expected auto|index|nn|off, got ${JSON.stringify(tmodeRaw)}`);
  const splatTemporal = { mode: tmodeRaw as "auto" | "index" | "nn" | "off", matchDist: numFlag(a, "--splat-match", 0.01, { min: 0.0001, max: 0.5 }), minSurvive: numFlag(a, "--splat-min-survive", 0.5, { min: 0, max: 1 }) };

  const audio = await audioFromFlags(a, fps, from, frames.length);
  const t1 = performance.now();
  const r = await muxClipWithStats({
    fps, splatFrames: frames, gopLength: gop, quantBitsPos: quantBits, shDegree, splatMorton: orderRaw === "morton", splatBoxMinAlpha: boxAlpha,
    splatTemporal, audio: audio ?? undefined,
    meta: { title: dir, encoder: "ares-cli/0.2.0", source: kind, profile: "splat" },
  });
  await writeFile(out, r.bytes);
  console.log(`[ares] splat geometry: ${r.temporalFrames} P + ${r.intraFrames} I frame(s) (${tmodeRaw})`);
  console.log(`[ares] wrote ${out} — ${(r.bytes.length / 1048576).toFixed(2)} MB, ${(r.bytes.length / files.length / 1024).toFixed(1)} KB/frame, muxed in ${((performance.now() - t1) / 1000).toFixed(1)}s`);
  let srcBytes = 0;
  for (const f of files) { try { srcBytes += (await stat(f === "." ? dir : join(dir, f))).size; } catch { /* skip */ } }
  await writeClipMeta(a, out, {
    output: { name: basename(out), sizeBytes: r.bytes.length, frames: files.length, fps, durationS: +(files.length / fps).toFixed(2) },
    source: { dir, kind: `splat-${kind}`, splatFrames: files.length, discoveredFrames: discoveredCount, splatCount: n0, shDegree: maxDegree, totalBytes: srcBytes, fileCount: files.length },
    encode: {
      profile: "splat", gop, quantBitsPos: quantBits, shDegree: shDegree ?? maxDegree, splatOrder: orderRaw, splatMinAlpha: minAlpha, splatBoxAlpha: boxAlpha,
      splatTemporal: tmodeRaw, splatMatch: splatTemporal.matchDist, splatMinSurvive: splatTemporal.minSurvive, temporalFrames: r.temporalFrames, intraFrames: r.intraFrames,
      trim: { in: from, out: toExcl - 1, sourceFrames: discoveredCount },
      transform: modelXf,
    },
    tooling: { encoder: "ares-cli/0.2.0", node: process.version, generatedBy: "ares encode" },
  });
  await info(["info", out]);
}

/** Locate frame `idx` in a parsed file: its chunk, its block, and the keyframe block(s) leading to it. */
function locateFrame(file: ReturnType<typeof Demuxer.parse>, idx: number) {
  for (const gop of file.gopIndex) {
    if (idx < gop.frameStart || idx >= gop.frameStart + gop.frameCount) continue;
    const chunk = Demuxer.chunkAt(file, gop);
    const blocks = Demuxer.geometryBlocks(file, chunk);
    const local = idx - gop.frameStart;
    if (local >= blocks.length) break;
    let kf = local;
    while (kf > 0 && blocks[kf]!.type !== BlockType.GeometryI) kf--;
    return { chunk, blocks, local, kf };
  }
  throw new Error(`frame ${idx} is out of range (0..${file.header.frameCount - 1})`);
}

/** `ares export <in.ares> -o <out> [--frame N]` — one frame back to an interchange format. */
async function exportCmd(a: string[]) {
  const input = a[1]!;
  const out = flag(a, "-o");
  if (!out) throw new Error("export: -o <out.obj|ply|spz|glb|splat> is required");
  const idx = numFlag(a, "--frame", 0, { min: 0, int: true });
  const file = Demuxer.parse(new Uint8Array(await readFile(input)));
  await meshoptReady();
  const { chunk, blocks, local, kf } = locateFrame(file, idx);
  const ext = extname(out).toLowerCase();
  const bits = file.superblock.quantBitsPos;
  if (file.header.geometryProfile === GeometryProfile.SplatIPB) {
    let dec = decodeSplatBlock(blocks[kf]!.data);
    for (let f = kf + 1; f <= local; f++) dec = decodeSplatPBlock(blocks[f]!.data, dec);
    const frame = decodedSplatToFrame(dec, chunk.gopAabb, dequantScale(bits));
    const bytes = ext === ".spz" ? writeSpz(frame) : ext === ".ply" ? writeSplatPly(frame) : ext === ".glb" ? writeGlbSplat(frame) : ext === ".splat" ? writeSplatFile(frame) : null;
    if (!bytes) throw new Error(`export: splat frames export to .spz, .ply, .glb or .splat (got ${ext || "no extension"})`);
    await writeFile(out, bytes);
    console.log(`[ares] exported frame ${idx}: ${frame.count} splats, SH degree ${frame.shDegree} → ${out} (${(bytes.length / 1024).toFixed(1)} KB)`);
    return;
  }
  let g: DecodedGeometry = decodeGeometryBlock(blocks[kf]!.data);
  for (let f = kf + 1; f <= local; f++) {
    const p = decodePFrameBlock(blocks[f]!.data, g.positionsQ);
    g = { ...g, positionsQ: p.positionsQ, uvsQ: p.uvsQ ?? g.uvsQ, normalsQ: p.normalsQ ?? g.normalsQ };
  }
  const frame = decodedMeshToFrame(g, chunk.gopAabb, bits, file.superblock.normalEncoding);
  if (ext === ".obj") await writeFile(out, writeObj(frame, basename(input, ".ares")));
  else if (ext === ".ply") await writeFile(out, writeMeshPly(frame));
  else throw new Error(`export: mesh frames export to .obj or .ply (got ${ext || "no extension"})`);
  console.log(`[ares] exported frame ${idx}: ${g.vertexCount} verts, ${g.indexCount / 3} tris → ${out}` + (file.tracks.length > 1 ? " (texture not exported — the atlas is a video track)" : ""));
}

async function info(a: string[]) {
  const path = a[1]!;
  const file = Demuxer.parse(new Uint8Array(await readFile(path)));
  const h = file.header, s = file.superblock;
  const profile = h.geometryProfile === GeometryProfile.SplatIPB ? "splat" : h.geometryProfile === GeometryProfile.MeshIPB ? "mesh" : `profile ${h.geometryProfile}`;
  console.log(`[ares] ${path}: v${h.versionMajor}.${h.versionMinor}, ${h.frameCount} frames @ ${h.fps}fps, ${(Number(h.durationUs) / 1e6).toFixed(2)}s`);
  if (h.geometryProfile === GeometryProfile.SplatIPB) {
    await meshoptReady();
    let n = 0;
    try { const { blocks } = locateFrame(file, 0); n = decodeSplatBlock(blocks[0]!.data).count; } catch { /* leave 0 */ }
    console.log(`       geometry ${profile} · ${n} splats in frame 0 · SH degree ${s.shDegree} · quant ${s.quantBitsPos}b pos · gop=${s.gopLength}`);
  } else {
    console.log(`       geometry ${profile} intra=${h.intraCodec} · quant ${s.quantBitsPos}b pos / ${s.quantBitsUv}b uv · gop=${s.gopLength}`);
  }
  console.log(`       ${file.gopIndex.length} chunk(s), ${file.tracks.length} track(s): ${file.tracks.map((t) => t.codecFourcc).join(", ")}`);
  const vid = Demuxer.textureVideo(file);
  const tex = Demuxer.textureAtlas(file);
  console.log(`       texture: ${vid ? `${vid.fourcc} video ${vid.width}x${vid.height}` : tex ? `${tex.width}x${tex.height} still atlas, ${(tex.bytes.length / 1024).toFixed(1)} KB` : "none"}`);
  const at = Demuxer.audioTrack(file);
  if (at) {
    let packets = 0, bytes = 0, endUs = 0;
    for (const gop of file.gopIndex) for (const p of Demuxer.audioPackets(file, Demuxer.chunkAt(file, gop))) { packets++; bytes += p.data.byteLength; endUs = Math.max(endUs, p.ptsUs + p.durationUs); }
    console.log(`       audio: ${at.fourcc} 48 kHz ${at.channels === 1 ? "mono" : "stereo"}, ${packets} packets, ${(endUs / 1e6).toFixed(2)}s, ${(bytes / 1024).toFixed(1)} KB`);
  }
  console.log(`       AABB min [${s.aabb.min.map((v) => v.toFixed(2)).join(", ")}] max [${s.aabb.max.map((v) => v.toFixed(2)).join(", ")}]`);
}

const USAGE = `usage:
  ares synth  [-o out.ares] [--shape object|talk|splat] [--frames 60] [--fps 30] [--no-texture] [--sh-degree 0|1]
  ares encode <frames-dir> [-o out.ares] [--fps 30] [--max-frames N] [--gop 30]
              mesh input (OBJ/PLY + atlas-*.png):
              [--texture-codec vp9|av1] [--tex-size 1024] [--crf 32] [--no-texture]
              [--edits file.json] [--crop x0,y0,z0,x1,y1,z1] [--track] [--no-temporal]
              [--smooth N] [--smooth-temporal N] [--decimate ratio] [--repack-detect topology|image]
              splat input (one .spz / 3DGS .ply / .splat / .glb|.gltf (KHR_gaussian_splatting) / .sog per frame):
              [--sh-degree 0..3] [--splat-min-alpha a] [--splat-box-alpha a] [--splat-order morton|none] [--quant-bits 8..16]
              [--splat-temporal auto|index|nn|off] [--splat-match 0.01] [--splat-min-survive 0.5]   (P-frames: deltas + births/deaths)
              both:
              [--trim-in N] [--trim-out N]  (clip in/out, inclusive source frames)
              [--up-axis x|y|z] [--center bottom|mass|none] [--scale N] [--rotate x,y,z] [--translate x,y,z]
              [--meta-extra-file f.json]  (merged into the <out>.ares.meta.json provenance sidecar)
              [--audio file] [--audio-offset seconds] [--audio-bitrate kbps]   (any format ffmpeg reads → Opus 48 kHz)
  ares export <file.ares> -o <out> [--frame N]
              mesh → .obj | .ply      splat → .spz | .ply (3DGS) | .glb (KHR_gaussian_splatting) | .splat
  ares info   <file.ares>`;

async function main() {
  const a = process.argv.slice(2);
  const cmd = a[0];
  try {
    if (cmd === "synth") await synth(a);
    else if (cmd === "encode" && a[1]) await encode(a);
    else if (cmd === "export" && a[1]) await exportCmd(a);
    else if (cmd === "info" && a[1]) await info(a);
    else {
      console.error(USAGE);
      process.exit(cmd === "--help" || cmd === "-h" || cmd === "help" ? 0 : 1);
    }
  } catch (e) {
    console.error("[ares] error:", (e as Error).message);
    process.exit(1);
  }
}

main();
