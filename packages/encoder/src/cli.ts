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
 *   ares depth  <video> --depth <run-dir> [-o out.ares]   2D video + an ares-depth/1 run -> a 2.5D relief clip
 *               [--fov 55] [--near 0.5] [--far 6] [--grid 256] [--edge 0.08] [--sheets] [--stabilize 0.7]
 *               [--gop 30] [--tex-size 1024] [--texture-codec vp9|av1] [--crf 30] [--no-texture]
 *               [--no-audio] [--audio file] [--audio-offset s] [--audio-bitrate kbps] [--smooth-temporal N]
 *               [--up-axis x|y|z] [--center bottom|mass|none] [--scale N] [--rotate x,y,z] [--translate x,y,z]
 *               [--meta-extra-file f.json]
 *   ares export <file.ares> -o <out> [--frame N]      (.obj/.ply for meshes; .spz/.ply/.glb/.splat for splats)
 *   ares info   <file.ares>
 *   ares verify-edits <file.edits.json>               (mask2d/keyframe pre-flight, exit 1 on any issue)
 */
import { readdir, readFile, writeFile, stat, mkdtemp, rm, open } from "node:fs/promises";
import { readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, extname, basename } from "node:path";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { Demuxer, GeometryProfile, BlockType, decodeGeometryBlock, decodePFrameBlock, decodeSplatBlock, decodeSplatPBlock, dequantScale, meshoptReady, transformMatrix, type DecodedGeometry, type GopEntry } from "@ares/core";
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
import { parseEditList, validateMasks, keepPredicateAt, prepareRangeAt, prepareRangeSdfAt, isRangeEnabled, type EditList, type EditRange } from "@ares/core";
import { resolveOffset, applyTransform, applyTransformNormals, transformAabb, type ModelTransform } from "@ares/core";
import { HEADER_SIZE, type Aabb } from "@ares/core";
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
import { encodeTextureVideo, detectPattern, ffmpegAvailable, ffmpegPath, type TexCodec, type TextureGop } from "./texture-video.js";
import { openDepthRun, type DepthRunMeta } from "./depth-io.js";
import { stabilizeDepthStream, RgbMotionGate } from "./depth-stabilize.js";
import { buildDepthGrid, resampleMap, depthFrameToMesh, gridDepths, fillLayerToMesh, mergeMeshes, reliefDiscard, DepthHistogram, cutEdgeCount } from "./depth-mesh.js";
import { resizeRgbArea, resampleMask, guidedResample, buildFillLayer, composeLayeredAtlas, snapRamps, FILL_MIN_JUMP } from "./depth-layers.js";
import { fileStore, memoryStore, type FrameStore } from "./depth-store.js";
import { openRawFrames, encodeRawTextureGop, detectLetterbox, type CropRect } from "./video-frames.js";
import { MeshClipWriter } from "./muxer.js";
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
 * Ranges that fall wholly outside the window are dropped; partial ones are clamped. A USER
 * keyframe is shifted but NEVER dropped, even when it lands outside [0, n): prepareRangeAt
 * interpolates between the keyframes bracketing a frame, so discarding an out-of-window anchor
 * would change the shape of the surviving in-window frames. That reasoning inverts for a
 * `derived` keyframe — a propagation writes one per frame, so every surviving frame already
 * carries its own and nothing in-window interpolates through an out-of-window one. Those ARE
 * dropped, because at ~272 keyframes per range the alternative is carrying the whole trimmed-off
 * tail's decoded bitmaps through the bake. A copy op whose srcFrame was trimmed away is dropped
 * outright with a warning — its source no longer exists, and silently copying from some other frame
 * would be a fabrication.
 *
 * There is deliberately NO `from === 0` fast path: a trim-out-only bake (the common one — the
 * editor's out point moves far more often than its in point) still has to clamp endFrame to the
 * shorter window and still has to drop the derived keyframes past it. `trimmed` says whether the
 * window is actually narrower than the discovered clip, which is what gates the copy-payload
 * rebase below — see the comment there.
 */
function rebaseEditList(list: EditList, from: number, n: number, trimmed: boolean): EditList {
  const ranges: EditRange[] = [];
  let droppedDerived = 0;
  for (const r of list.ranges) {
    const s = r.startFrame - from, e = r.endFrame - from;
    // A copy range is exempt: its bake frames are copy.srcFrame/dstFrames, and frame-copy.ts
    // evaluates its region with prepareRangeAt(range, srcFrame) — the span never enters the
    // computation, so it is not the gate for this action. The srcFrame/dstFrames checks below are.
    if ((r.action ?? "delete") !== "copy" && (e < 0 || s > n - 1)) {
      console.log(`[ares] trim: dropping range ${r.id ?? `${r.startFrame}-${r.endFrame}`} — its span is wholly outside the ${n}-frame window`);
      continue;
    }
    const keyframes = r.keyframes
      .map((k) => ({ ...k, frame: k.frame - from }))
      .filter((k) => {
        if (k.derived !== true || (k.frame >= 0 && k.frame <= n - 1)) return true;
        droppedDerived++;
        return false;
      });
    // Every keyframe was derived and every one fell outside: the range now matches nothing. Drop
    // it rather than keep an empty-keyframe range, which for mode:"keep" would put every point
    // "outside all keep regions" and delete the frame whole (the trap parseEditList documents).
    if (!keyframes.length) {
      console.warn(`[ares] trim: dropping range ${r.id ?? `${r.startFrame}-${r.endFrame}`} — every keyframe it had is derived and outside the trim`);
      continue;
    }
    const nr: EditRange = {
      ...r,
      startFrame: Math.max(0, s),
      endFrame: Math.min(n - 1, e),
      keyframes,
    };
    // Only rebase the copy payload when the window is actually narrower than the clip. Untrimmed,
    // `src < 0 || src >= n` and the dstFrames filter below are the SAME bounds collectCopyOps
    // (frame-copy.ts) checks — but it throws where these warn-and-drop, and a partly out-of-range
    // dstFrames list drops silently. A typo'd frame index in a hand-authored sidecar has to keep
    // failing loudly (frame-copy.ts's header states that contract, and recolor.ts cites it as the
    // reason recolor clamps and copy does not).
    if (nr.copy && trimmed) {
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
      if (dst.length !== cp.dstFrames.length)
        console.warn(`[ares] trim: copy range ${r.id ?? `${r.startFrame}-${r.endFrame}`} pastes to ${dst.length} of ${cp.dstFrames.length} dstFrame(s) — the rest are outside the trim`);
      nr.copy = { ...cp, srcFrame: src, dstFrames: dst };
    }
    ranges.push(nr);
  }
  if (droppedDerived)
    console.log(`[ares] trim: dropped ${droppedDerived} derived keyframe(s) outside the ${n}-frame window (user keyframes are kept wherever they land)`);
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
  if (!(await ffmpegAvailable())) throw new Error("--audio needs ffmpeg: none found (FFMPEG, FFMPEG_PATH, C:/FFmpeg/bin, PATH)");
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
    // parseEditList throws on a malformed DOCUMENT; validateMasks reports a malformed VOLUME, which
    // bakes as a silent no-op instead (prepareVolume returns null and the range matches nothing).
    // Warn rather than abort: an edit list with one bad keyframe out of hundreds should still bake.
    // `ares verify-edits` is the same check with an exit code, for a scripted pre-flight.
    for (const issue of validateMasks(editList)) console.warn(`[ares] edits: ${issue}`);
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
  if (editList) editList = rebaseEditList(editList, from, files.length, from !== 0 || toExcl !== discovered.length);
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
      console.warn("[ares] ffmpeg not found (FFMPEG, FFMPEG_PATH, C:/FFmpeg/bin, PATH): texture skipped, geometry only");
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

/* ================== 2D video -> 2.5D relief volumetric (`ares depth`) ================== */

/** ffprobe, found the same way ffmpeg is: env override, then the sibling of whatever ffmpeg we resolved. */
function ffprobePath(): string {
  const fromEnv = process.env.FFPROBE || process.env.FFPROBE_PATH;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const ff = ffmpegPath();
  const sibling = ff.replace(/ffmpeg(\.exe)?$/i, (m) => (m.toLowerCase().endsWith(".exe") ? "ffprobe.exe" : "ffprobe"));
  if (sibling !== ff && existsSync(sibling)) return sibling;
  return "ffprobe";
}

/** Machine-readable progress for the dev server's bar. The exact shape is a contract — do not reword. */
const progress = (stage: string, i: number, n: number) => console.log(`[ares] progress ${stage} ${i}/${n}`);

/**
 * The FRAME SAMPLING CONTRACT, as the depth engines apply it. Every later decode of the same video
 * has to reproduce it verbatim or frame i of the texture is not frame i of the depth: `fps=F` is
 * present only when the run sampled at a fixed rate, and it comes BEFORE the scale. A letterbox
 * crop (encoder side only; the engines never crop) sits between the two: it selects no frames.
 */
const samplingFilter = (meta: DepthRunMeta, scale: string, crop: CropRect | null = null): string =>
  (meta.sampling?.fps != null ? `fps=${meta.sampling.fps},` : "") + (crop ? `crop=${crop.w}:${crop.h}:${crop.x}:${crop.y}:exact=1,` : "") + scale;

/** ffmpeg args that decode the run's own sampled frame set, cropped, to raw rgb24 at `w x h`. */
function sampledRgbArgs(video: string, meta: DepthRunMeta, frameCount: number, w: number, h: number, flags: string, crop: CropRect | null): string[] {
  return [
    "-hide_banner", "-loglevel", "error", "-i", video,
    "-vf", samplingFilter(meta, `scale=${w}:${h}:flags=${flags}`, crop),
    "-frames:v", String(frameCount),
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
  ];
}

/**
 * The crop of the run's maps that matches a source-pixel crop, and the source crop re-derived from
 * it so the texture decode and the maps cut the picture at the same place: map edges are rounded
 * inward to whole map pixels, the source edges then inward to whole source pixels, which leaves
 * the two within one source pixel of each other.
 */
function mapCrop(crop: CropRect, runW: number, runH: number, srcW: number, srcH: number): { map: { x: number; y: number; w: number; h: number }; source: CropRect } {
  const mx0 = Math.ceil((crop.x * runW) / srcW), mx1 = Math.floor(((crop.x + crop.w) * runW) / srcW);
  const my0 = Math.ceil((crop.y * runH) / srcH), my1 = Math.floor(((crop.y + crop.h) * runH) / srcH);
  const sx0 = Math.ceil((mx0 * srcW) / runW), sx1 = Math.floor((mx1 * srcW) / runW);
  const sy0 = Math.ceil((my0 * srcH) / runH), sy1 = Math.floor((my1 * srcH) / runH);
  return { map: { x: mx0, y: my0, w: mx1 - mx0, h: my1 - my0 }, source: { w: sx1 - sx0, h: sy1 - sy0, x: sx0, y: sy0 } };
}

/**
 * The RGB motion gate for every frame of the run, streamed: one decode at the (cropped) maps' own
 * W x H, two frames resident, the gate written to `store`. Returns false when the video yielded
 * fewer frames than the run holds, in which case the stabilizer derives its gate from the depth
 * change instead.
 */
async function rgbMotionToStore(video: string, meta: DepthRunMeta, frameCount: number, w: number, h: number, store: FrameStore<Uint8Array>, crop: CropRect | null): Promise<boolean> {
  const reader = openRawFrames(sampledRgbArgs(video, meta, frameCount, w, h, "area", crop), w * h * 3);
  const gate = new RgbMotionGate(w, h);
  const out = new Uint8Array(w * h);
  let t = 0;
  try {
    for (;;) {
      const batch = await reader.read(16);
      if (!batch.length) break;
      for (const f of batch) {
        if (t >= frameCount) break;
        gate.next(f, 0, out);
        store.write(t++, out);
      }
      progress("rgb", Math.min(t, frameCount), frameCount);
    }
  } finally { reader.close(); }
  progress("rgb", frameCount, frameCount);
  if (t < frameCount) {
    console.warn(`[ares] depth: the video yielded ${t} sampled frame(s), the run has ${frameCount} — the motion gate falls back to the depth change itself`);
    return false;
  }
  return true;
}

/** Does the source video carry an audio stream at all? (Default-on audio needs to know.) */
async function videoHasAudio(video: string): Promise<boolean> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  try {
    const { stdout } = await promisify(execFile)(ffprobePath(), ["-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video]);
    return /audio/i.test(String(stdout));
  } catch { return false; }
}

/**
 * `ares depth <video> --depth <run-dir>` — a flat 2D video plus a monocular depth run become a
 * 2.5D volumetric clip: a per-frame pinhole-unprojected relief mesh wearing the video as its
 * texture. The depth run comes from one of the engines (the Python GPU service or the browser
 * worker) and is consumed through the ares-depth/1 directory contract (depth-io.ts).
 *
 * Every stage STREAMS. A run is frames x H x W floats and a clip is frames x ~1.6 MB of mesh; a
 * four-minute video is gigabytes of each, so nothing here holds a clip: the run is read a frame at
 * a time, the stabilizer's passes go through scratch frame stores, and the second half of the
 * pipeline moves one GOP at a time from the stabilized store and the texture decoder, through
 * meshing, to the chunk writer.
 *
 *   pass A  decode RGB at the map size -> motion gate store
 *   pass B  stabilize (align, smooth, normalise) -> stabilized store
 *   pass C  per GOP: decode texture frames; per frame: image-guided resample onto the grid,
 *           unproject, cut silhouettes (or keep the sheet), decimate, fill layer; encode the GOP's
 *           texture and geometry; append the chunk
 *   finish  centre the clip (a translation of the chunk boxes), write header + index, copy chunks
 *
 * The UVs ARE the grid coordinates, so the video frame is the atlas — no packing step exists.
 */
async function depth(a: string[]) {
  const video = a[1]!;
  const runDir = flag(a, "--depth");
  if (!runDir) throw new Error("depth: --depth <run-dir> is required (a directory holding depth.json + depth.f32)");
  const out = flag(a, "-o") ?? "out.ares";

  const run = await openDepthRun(runDir);
  const scratch = await mkdtemp(join(tmpdir(), "ares-depthwork-"));
  const stores: { close(): void }[] = [];
  let writer: MeshClipWriter | null = null;
  try {
    const meta = run.meta;
    const runW = run.width, runH = run.height, runP = runW * runH;
    let frames = run.frames;
    const fps = meta.fps;
    const metric = meta.kind === "metric-depth";
    console.log(`[ares] depth run ${runDir}: ${meta.engine}/${meta.modelKey} (${meta.model}), ${meta.kind}, ${runW}x${runH}, ${frames} frame(s) @ ${fps}fps` +
      `${meta.temporal === "model" ? ", temporally consistent model" : ""}${run.hasMask ? `, subject mask ${JSON.stringify(meta.mask?.prompt ?? "")}` : ""}`);

    const fov = numFlag(a, "--fov", 55, { min: 1, max: 179 });
    // A relative run's depth range is a choice, and its ratio is what a viewer sees: the relief is
    // exact from the capture camera whatever the range, and from anywhere else content at depth z
    // is z times larger than it looks in the picture. At 0.5..6 m (12:1) the characters of a shot
    // sat at doll size in front of a background twelve times their scale; 2..6 m keeps it at 3:1.
    // A metric run is already in metres: 0.5..20 m clamps only what the model got wrong.
    const near = numFlag(a, "--near", metric ? 0.5 : 2, { min: 0.001 });
    const far = numFlag(a, "--far", metric ? 20 : 6, { min: 0.002 });
    if (far <= near) throw new Error(`--far (${far}) must be greater than --near (${near})`);
    const gridW = numFlag(a, "--grid", 256, { min: 2, max: 4096, int: true });
    const edge = numFlag(a, "--edge", 0.08, { min: 0, max: 100 });
    const sheets = has(a, "--sheets");
    const strength = numFlag(a, "--stabilize", 0.7, { min: 0, max: 1 });
    const median = numFlag(a, "--median", 5, { min: 1, max: 5, int: true });
    if (median % 2 === 0) throw new Error(`--median: expected 1, 3 or 5 frames, got ${median}`);
    const gop = numFlag(a, "--gop", 30, { min: 1, max: 65535, int: true });
    const smoothTemporal = numFlag(a, "--smooth-temporal", 0, { min: 0, int: true });
    const noTexture = has(a, "--no-texture");
    const noAudio = has(a, "--no-audio");
    const decimate = flag(a, "--decimate") != null ? numFlag(a, "--decimate", 1, { min: 0.01, max: 1 }) : 1;
    if (decimate < 1 && sheets) throw new Error("--decimate re-triangulates every frame and --sheets needs one topology for the whole clip; use one or the other");
    const guided = !has(a, "--no-guided");
    const snap = has(a, "--snap-ramps");
    const guideSigma = numFlag(a, "--guide-sigma", 14, { min: 0.5, max: 255 });
    const inpaint = has(a, "--inpaint");
    const band = numFlag(a, "--inpaint-band", Math.max(4, Math.round(gridW * 0.16)), { min: 1, max: 4096, int: true });

    const texSize = numFlag(a, "--tex-size", 1024, { min: 16, max: 8192, int: true });
    if (texSize % 4) throw new Error(`--tex-size: ${texSize} is not a multiple of 4; 4:2:0 video and the fill plate need it`);
    const codecRaw = flag(a, "--texture-codec") ?? "vp9";
    if (codecRaw !== "vp9" && codecRaw !== "av1") throw new Error(`--texture-codec: expected vp9|av1, got ${JSON.stringify(codecRaw)}`);
    const crf = numFlag(a, "--crf", 30, { min: 0, max: 63, int: true });

    const haveFfmpeg = await ffmpegAvailable();
    if (!haveFfmpeg) console.warn("[ares] ffmpeg is not available to the encoder: geometry only, no texture, no audio, and the stabilizer gates on the depth change instead of the image.");

    // ---- letterbox: the bars are not part of the picture, so they become neither geometry nor
    // texture, and they stay out of the normalisation range (a bar reads as infinitely far) ---------
    const cropArg = flag(a, "--crop") ?? "auto";
    const srcFullW = meta.sourceWidth ?? runW, srcFullH = meta.sourceHeight ?? runH;
    let crop: CropRect | null = null;
    let cropFrom = "none";
    if (cropArg === "auto") {
      if (!haveFfmpeg) console.log("[ares] letterbox: not detected (ffmpeg is not available)");
      else if (meta.sourceWidth == null || meta.sourceHeight == null) console.log("[ares] letterbox: not detected (the run does not record the source size)");
      else {
        try {
          const d = await detectLetterbox(video, srcFullW, srcFullH);
          if (d) { crop = d.crop; cropFrom = `detected over ${d.frames} frame(s)`; }
          else console.log("[ares] letterbox: none");
        } catch (e) { console.warn(`[ares] letterbox: detection failed (${(e as Error).message}); the full frame is used`); }
      }
    } else if (cropArg !== "none") {
      const m = /^(\d+):(\d+):(\d+):(\d+)$/.exec(cropArg);
      if (!m) throw new Error(`--crop: expected auto, none or W:H:X:Y in source pixels, got ${JSON.stringify(cropArg)}`);
      crop = { w: +m[1]!, h: +m[2]!, x: +m[3]!, y: +m[4]! };
      if (crop.w < 2 || crop.h < 2 || crop.x + crop.w > srcFullW || crop.y + crop.h > srcFullH) throw new Error(`--crop ${cropArg} does not fit inside the ${srcFullW}x${srcFullH} source`);
      cropFrom = "--crop";
    }
    let mapBox = { x: 0, y: 0, w: runW, h: runH };
    if (crop) {
      const c = mapCrop(crop, runW, runH, srcFullW, srcFullH);
      if (c.map.w < 2 || c.map.h < 2) throw new Error(`the crop ${crop.w}x${crop.h}+${crop.x}+${crop.y} leaves under 2x2 of the ${runW}x${runH} maps`);
      mapBox = c.map; crop = c.source;
      console.log(`[ares] letterbox: picture ${crop.w}x${crop.h}+${crop.x}+${crop.y} of ${srcFullW}x${srcFullH} (${cropFrom}), maps ${mapBox.w}x${mapBox.h}+${mapBox.x}+${mapBox.y} of ${runW}x${runH}`);
    }
    // From here on W x H is the picture: the cropped maps, and the source size is the crop's.
    const W = mapBox.w, H = mapBox.h, P = W * H;
    const srcW = crop ? crop.w : srcFullW, srcH = crop ? crop.h : srcFullH;
    const gridH = Math.max(2, Math.round(gridW * srcH / srcW));
    const cropped = mapBox.w !== runW || mapBox.h !== runH;
    /** Copy the picture's rows of one full run frame (`from`, runW x runH) into `to` (W x H). */
    const cutMap = <T extends Float32Array | Uint8Array>(from: T, to: T): T => {
      if (!cropped) { to.set(from.subarray(0, P)); return to; }
      for (let y = 0; y < H; y++) { const s = (mapBox.y + y) * runW + mapBox.x; to.set(from.subarray(s, s + W), y * W); }
      return to;
    };

    // Scratch stores: memory for a short clip, files past 256 MB so clip length is bounded by disk.
    const bigClip = frames * P * 4 > (1 << 28);
    const makeF32 = (name: string): FrameStore<Float32Array> => {
      const s = bigClip ? fileStore<Float32Array>(join(scratch, `${name}.f32`), 4, frames, P) : memoryStore((n) => new Float32Array(n), frames, P);
      stores.push(s); return s;
    };
    const makeU8 = (name: string): FrameStore<Uint8Array> => {
      const s = bigClip ? fileStore<Uint8Array>(join(scratch, `${name}.u8`), 1, frames, P) : memoryStore((n) => new Uint8Array(n), frames, P);
      stores.push(s); return s;
    };
    if (bigClip) console.log(`[ares] long clip: working set on disk in ${scratch} (${((frames * P * 9) / 1073741824).toFixed(1)} GB)`);

    // ---- pass A: RGB motion gate over the SAME sampled frames the depth engine consumed ----------
    let motion: FrameStore<Uint8Array> | null = null;
    if (haveFfmpeg) {
      const m = makeU8("motion");
      // Optional by design: a video ffmpeg cannot open still yields a clip from the depth alone.
      try { if (await rgbMotionToStore(video, meta, frames, W, H, m, crop)) motion = m; }
      catch (e) { console.warn(`[ares] depth: could not decode ${basename(video)} for the motion gate (${(e as Error).message}) — falling back to the depth change`); }
      if (!motion) m.close();
    } else progress("rgb", 0, frames);

    // ---- pass B: stabilize ------------------------------------------------------------------------
    const ts = performance.now();
    const raw = new Float32Array(runP);
    const stab = stabilizeDepthStream({
      width: W, height: H, frames, motion, makeF32, makeU8,
      readRaw: (t, o) => { run.readFrames(t, 1, raw); cutMap(raw, o); },
    }, {
      kind: meta.kind, strength, median,
      align: meta.temporal !== "model",
      grow: !has(a, "--no-grow"),
      onProgress: (i, n) => { if (i === n || i % 30 === 0) progress("stabilize", Math.floor((i * frames) / Math.max(1, n)), frames); },
    });
    const st = stab.stats;
    console.log(`[ares] stabilize (strength ${strength}, gate ${st.rgbGated ? "rgb" : "depth-derived"}${strength > 0 && median > 1 ? `, ${median}-frame median` : ""}${st.aligned ? "" : ", alignment off: the model is temporally consistent"}): scale ${st.scaleMin.toFixed(3)}..${st.scaleMax.toFixed(3)} (mean ${st.scaleMean.toFixed(3)}), ` +
      `${st.clamped} clamped, ${st.degenerate} degenerate, gate grown on ${st.grownFrames} frame(s), still-pixel alpha ${st.alphaStill.toFixed(3)}, mean motion ${st.motionMean.toFixed(1)}/255`);
    console.log(`[ares]   ${metric ? `depth percentiles ${stab.lo.toFixed(2)}..${stab.hi.toFixed(2)} m (kept in metres)` : `disparity percentiles ${stab.lo.toFixed(4)}..${stab.hi.toFixed(4)} -> [0,1], 1 = nearest`} in ${((performance.now() - ts) / 1000).toFixed(1)}s`);
    if (st.clamped > 0.1 * Math.max(1, st.aligned)) {
      console.warn(`[ares] stabilize: ${st.clamped} of ${st.aligned} frame fits hit the [0.5, 2] scale clamp; depth may drift across the clip. A temporally consistent engine (video-small, video-large) does not need the fit.`);
    }
    if (motion) motion.close();

    // ---- grid, atlas layout -----------------------------------------------------------------------
    const aspect = srcW / srcH;
    const texW = texSize, texH = texSize, plateH = inpaint ? texSize / 2 : 0, atlasH = texH + plateH;
    const vScale = texH / atlasH;
    const grid = buildDepthGrid(gridW, gridH, aspect, fov, { vScale, vOffset: 0 });
    const plateGrid = inpaint ? buildDepthGrid(gridW, gridH, aspect, fov, { vScale: plateH / atlasH, vOffset: vScale }) : null;
    console.log(`[ares] grid ${gridW}x${gridH} (aspect ${aspect.toFixed(3)} from ${srcW}x${srcH}), fov ${fov}deg vertical, z ${near}..${far}${metric ? " m" : ""}, ` +
      `${sheets ? "sheets (full grid, persistent topology, silhouettes discarded at draw time)" : `silhouette cull at edge ${edge}`}` +
      `${decimate < 1 ? `, decimate ${decimate}` : ""}${inpaint ? `, fill layer (band ${band} cells, plate ${texW}x${plateH})` : ""}`);

    // ---- model transform: identical law to `encode`, but defaulting to "stand it on the floor" ----
    // Rotation and scale are applied per frame; the centring offset needs the whole clip's bounds,
    // so it is applied at finish() as a translation of the chunk boxes.
    const modelXf: ModelTransform = parseModelTransform(a) ?? { upAxis: "y", center: "bottom", rotate: [0, 0, 0], translate: [0, 0, 0], scale: 1 };
    const bounds: Aabb = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };

    // ---- audio: default ON, taken from the video's own stream -------------------------------------
    let audio: AudioTrackData | null = null;
    let audioSource: string | null = null;
    if (!noAudio) {
      if (flag(a, "--audio")) { audio = await audioFromFlags(a, fps, 0, frames); audioSource = flag(a, "--audio")!; }
      else if (haveFfmpeg && (await videoHasAudio(video))) {
        audio = await audioFromFlags([...a, "--audio", video], fps, 0, frames);
        audioSource = video;
      } else if (haveFfmpeg) console.log(`[ares] audio: ${basename(video)} has no audio stream — none muxed`);
    }

    // ---- pass C: GOP by GOP -------------------------------------------------------------------------
    const wantTexture = !noTexture && haveFfmpeg;
    const needFrames = haveFfmpeg && (wantTexture || guided || inpaint || snap);
    if (!noTexture && !haveFfmpeg) console.warn("[ares] no texture: ffmpeg is not available to the encoder. Geometry-only.");
    if (inpaint && !wantTexture) console.warn("[ares] --inpaint builds the fill geometry, but its colours live in the texture and this clip has none.");
    const texReader = needFrames ? openRawFrames(sampledRgbArgs(video, meta, frames, texW, texH, "lanczos", crop), texW * texH * 3) : null;
    if (wantTexture) console.log(`[ares] texture video: ${codecRaw} ${texW}x${atlasH}, crf ${crf}, one closed GOP per ${gop} frames`);
    if (decimate < 1) await simplifierReady();
    writer = await MeshClipWriter.open({
      out, fps, gopLength: gop, audio: audio ?? undefined,
      textureVideo: wantTexture ? { fourcc: codecRaw === "av1" ? "AV01" : "VP09", width: texW, height: atlasH } : undefined,
      temporal: { forceIntra: !sheets, smoothTemporal },
    });

    const tm = performance.now();
    const fullTris = (gridW - 1) * (gridH - 1) * 2;
    const surfaceDepth = new DepthHistogram(near, far);
    // The silhouette cut carries its per-edge decisions from frame to frame (depth-mesh.ts
    // hysteresis): an edge near the threshold keeps its state instead of blinking.
    const cutState = sheets ? null : new Uint8Array(cutEdgeCount(gridW, gridH));
    let cutPrimed = false;
    let triSum = 0, fillTriSum = 0, texBytes = 0, done = 0, snapSum = 0;
    const map = new Float32Array(P), maskMap = run.hasMask ? new Uint8Array(P) : null, maskFull = run.hasMask ? new Uint8Array(runP) : null;
    const guideMap = new Uint8Array(P * 3), guideGrid = new Uint8Array(gridW * gridH * 3);
    const inflight: { meshes: EncodeMeshFrame[]; tex: Promise<TextureGop> | null }[] = [];
    let framesFailed = false;
    const drain = async (keep: number) => {
      while (inflight.length > keep) {
        const g = inflight.shift()!;
        const t = g.tex ? await g.tex : null;
        if (t) texBytes += t.frames.reduce((s, f) => s + f.data.byteLength, 0);
        writer!.writeGop(g.meshes, t ? t.frames : null);
        done += g.meshes.length;
        progress("texture", done, frames);
        progress("mux", done, frames);
      }
    };
    try {
      for (let t0 = 0; t0 < frames; t0 += gop) {
        let n = Math.min(gop, frames - t0);
        let texFrames: Uint8Array[] | null = null;
        if (texReader && !framesFailed) {
          try { texFrames = await texReader.read(n); }
          catch (e) {
            // Only the texture genuinely needs the video decoded: without one, a video ffmpeg cannot
            // open still yields a clip from the depth alone, resampled without the image guide.
            if (wantTexture || t0 > 0) throw e;
            console.warn(`[ares] depth: could not decode ${basename(video)} for the image guide (${(e as Error).message}); resampling unguided`);
            framesFailed = true; texFrames = null;
          }
        }
        if (texFrames) {
          if (texFrames.length < n) {
            // Never let the two halves drift: frame i of the geometry must be frame i of the texture.
            const have = t0 + texFrames.length;
            if (have < 1) throw new Error(`the video decode produced no frames from ${video}`);
            console.warn(`[ares] video frames (${have}) != depth frames (${frames}) — truncating the clip to ${have}`);
            n = texFrames.length; frames = have;
            if (!n) break;
          }
        }
        const meshes: EncodeMeshFrame[] = [];
        const atlasFrames: Uint8Array[] = [];
        for (let k = 0; k < n; k++) {
          const t = t0 + k;
          stab.out.read(t, map);
          const frame = texFrames ? texFrames[k]! : null;
          let gm: Float32Array;
          if (frame && (guided || inpaint || snap)) resizeRgbArea(frame, texW, texH, gridW, gridH, guideGrid);
          if (frame && guided) {
            resizeRgbArea(frame, texW, texH, W, H, guideMap);
            gm = guidedResample(map, W, H, guideMap, guideGrid, gridW, gridH, { sigmaRange: guideSigma });
          } else gm = resampleMap(map, W, H, gridW, gridH);
          let gmask: Uint8Array | null = null;
          if (maskMap && maskFull) { run.readMask(t, 1, maskFull); gmask = resampleMask(cutMap(maskFull, maskMap), W, H, gridW, gridH); }
          if (snap) snapSum += snapRamps(gm, gridDepths(gm, gridW * gridH, { kind: meta.kind, near, far }), gridW, gridH, frame ? guideGrid : null, FILL_MIN_JUMP);
          let m = depthFrameToMesh(gm, grid, { kind: meta.kind, near, far, edge, sheets, mask: gmask, hysteresis: cutState ? { state: cutState, primed: cutPrimed } : undefined });
          cutPrimed = true;
          surfaceDepth.add(m.positions);   // before decimation, which thins the vertices unevenly
          if (decimate < 1 && m.indices.length > 3) m = decimateFrame(m, decimate);
          triSum += m.indices.length / 3;
          if (inpaint && frame && plateGrid) {
            const z = gridDepths(gm, gridW * gridH, { kind: meta.kind, near, far });
            const fill = buildFillLayer(z, gridW, gridH, guideGrid, { edge, band, mask: gmask });
            const fm = fillLayerToMesh(fill, z, plateGrid, { edge, sheets });
            fillTriSum += fm.indices.length / 3;
            if (fm.indices.length) m = mergeMeshes(m, fm);
            if (wantTexture) atlasFrames.push(composeLayeredAtlas(frame, texW, texH, plateH, fill, gridW, gridH));
          } else if (wantTexture && frame) atlasFrames.push(frame);
          const p = m.positions;
          for (let i = 0; i < p.length; i += 3) for (let c = 0; c < 3; c++) {
            const v = p[i + c]!;
            if (v < bounds.min[c]!) bounds.min[c] = v;
            if (v > bounds.max[c]!) bounds.max[c] = v;
          }
          // Sheets share one uv/index buffer between frames; positions are always per frame.
          applyTransform(m.positions, modelXf, [0, 0, 0]);
          meshes.push(m);
        }
        progress("mesh", t0 + n, frames);
        const tex = wantTexture
          ? encodeRawTextureGop({ frames: atlasFrames, width: texW, height: atlasH, fps, gopLength: gop, codec: codecRaw as TexCodec, crf, workDir: scratch, tag: t0 })
          : null;
        // An unobserved rejection would take the process down before drain() reaches it.
        tex?.catch(() => { /* surfaced by the await in drain */ });
        inflight.push({ meshes, tex });
        await drain(3);
      }
      await drain(0);
    } finally { texReader?.close(); }
    progress("mesh", frames, frames);
    console.log(`[ares] meshed ${frames} frame(s): ${Math.round(triSum / frames)} tris/frame of ${fullTris} full-grid ` +
      `(${(100 - (triSum / frames / fullTris) * 100).toFixed(1)}% ${decimate < 1 ? "culled + decimated" : "culled"})` +
      `${inpaint ? `, fill layer ${Math.round(fillTriSum / frames)} tris/frame` : ""}${guided && needFrames ? ", image-guided resample" : ""}` +
      `${snap ? `, ${(snapSum / frames).toFixed(0)} ramp vertices snapped/frame` : ""} in ${((performance.now() - tm) / 1000).toFixed(1)}s`);
    if (wantTexture) console.log(`[ares] texture video: ${(texBytes / 1048576).toFixed(2)} MB (${(texBytes / frames / 1024).toFixed(1)} KB/frame)`);

    // ---- finish: centre, relief metadata, layout ----------------------------------------------------
    const offset = resolveOffset(bounds, modelXf);
    {
      const after = transformAabb(bounds, modelXf);
      const lo = after.min.map((v, i) => v + offset[i]!), hi = after.max.map((v, i) => v + offset[i]!);
      console.log(`[ares] transform: up=${modelXf.upAxis} center=${modelXf.center} scale=${modelXf.scale} rotate=[${modelXf.rotate}] translate=[${modelXf.translate}]`);
      console.log(`[ares]   bounds ${bounds.min.map((v) => v.toFixed(2))}..${bounds.max.map((v) => v.toFixed(2))} -> ${lo.map((v) => v.toFixed(2))}..${hi.map((v) => v.toFixed(2))}`);
    }
    const clipMeta: Record<string, string> = { title: basename(video), encoder: "ares-cli/0.2.0", source: "depth" };
    const discard = reliefDiscard(gridH, fov, edge, far);
    const unit = Math.abs(modelXf.scale ?? 1);
    const framing = surfaceDepth.framing();
    // The capture camera sits at the origin looking down -Z; carry both through the model transform.
    // Every relief records it: the player opens a relief at that camera, orbits it about `pivot`, and
    // sizes its sway from the depth span (`near`, `far`: the nearest and farthest 5 % of the surface).
    const fwd = applyTransform(new Float32Array([0, 0, -1]), modelXf, [0, 0, 0]);
    const fl = Math.hypot(fwd[0]!, fwd[1]!, fwd[2]!) || 1;
    clipMeta["relief.camera"] = offset.map((v) => +v.toFixed(6)).join(",");
    clipMeta["relief.forward"] = [fwd[0]! / fl, fwd[1]! / fl, fwd[2]! / fl].map((v) => +v.toFixed(6)).join(",");
    clipMeta["relief.fov"] = String(fov);
    clipMeta["relief.aspect"] = String(+(srcW / srcH).toFixed(6));
    if (framing) {
      clipMeta["relief.pivot"] = String(+(framing.pivot * unit).toFixed(6));
      clipMeta["relief.near"] = String(+(framing.near * unit).toFixed(6));
      clipMeta["relief.far"] = String(+(framing.far * unit).toFixed(6));
    }
    // A culled clip has no stretched triangles and no parked vertices; only sheets need the discard.
    if (sheets) {
      clipMeta["relief.slope"] = String(+discard.slope.toFixed(6));
      clipMeta["relief.depthMax"] = String(+(discard.depthMax * unit).toFixed(6));
    }
    console.log(`[ares] relief: capture camera at [${clipMeta["relief.camera"]}] looking [${clipMeta["relief.forward"]}], fov ${fov}deg, ` +
      `${framing ? `surface depth ${framing.near.toFixed(3)}..${framing.far.toFixed(3)}${metric ? " m" : ""} (5th..95th percentile), pivot ${framing.pivot.toFixed(3)}` : "no surface"}` +
      `${sheets ? `, draw-time discard slope ${discard.slope.toFixed(4)}` : ""}`);
    const t1 = performance.now();
    const r = writer.finish({ meta: clipMeta, translate: offset });
    writer = null;
    const mode = r.temporalFrames && !r.intraFrames ? "temporal (I+P)" : r.temporalFrames ? "mixed I+P/intra" : "intra-only";
    console.log(`[ares] geometry: ${mode} — ${r.temporalFrames} temporal + ${r.intraFrames} intra frames in ${r.chunks} chunk(s)`);
    console.log(`[ares] wrote ${out} — ${(r.sizeBytes / 1048576).toFixed(2)} MB, ${(r.sizeBytes / frames / 1024).toFixed(1)} KB/frame, laid out in ${((performance.now() - t1) / 1000).toFixed(1)}s`);

    let srcBytes = 0;
    try { srcBytes = (await stat(video)).size; } catch { /* the video may be gone by now */ }
    await writeClipMeta(a, out, {
      output: { name: basename(out), sizeBytes: r.sizeBytes, frames, fps, durationS: +(frames / fps).toFixed(2) },
      source: { video, kind: "2d-video+depth-run", depthRun: runDir, videoBytes: srcBytes, fileCount: 1, sourceWidth: meta.sourceWidth, sourceHeight: meta.sourceHeight, sourceFps: meta.sourceFps, sourceFrames: meta.sourceFrames },
      depth: {
        engine: meta.engine, model: meta.model, modelKey: meta.modelKey, kind: meta.kind, temporal: meta.temporal ?? "none",
        mapWidth: runW, mapHeight: runH, frames, sampling: meta.sampling, device: meta.device, dtype: meta.dtype,
        crop: crop ? { source: [crop.w, crop.h, crop.x, crop.y], map: [mapBox.w, mapBox.h, mapBox.x, mapBox.y], from: cropFrom } : null,
        fov, near, far, grid: [gridW, gridH], edge, sheets, stabilize: strength, median: strength > 0 ? median : 1,
        guided: guided && needFrames ? { sigmaRange: guideSigma } : null,
        snapRamps: snap ? { jump: FILL_MIN_JUMP, perFrame: +(snapSum / Math.max(1, frames)).toFixed(1) } : null,
        decimate: decimate < 1 ? decimate : null,
        inpaint: inpaint ? { band, plate: [texW, plateH], trisPerFrame: Math.round(fillTriSum / Math.max(1, frames)) } : null,
        mask: run.hasMask ? meta.mask : null,
        discard: sheets ? discard : null,
        framing: framing ? { pivot: +framing.pivot.toFixed(4), near: +framing.near.toFixed(4), far: +framing.far.toFixed(4) } : null,
        normalization: { lo: stab.lo, hi: stab.hi, units: metric ? "metres" : "disparity -> [0,1], 1 = nearest" },
        gate: st.rgbGated ? "rgb" : "depth-derived",
        scale: { min: st.scaleMin, max: st.scaleMax, mean: st.scaleMean, clamped: st.clamped, degenerate: st.degenerate, grownFrames: st.grownFrames },
      },
      encode: {
        gop, texSize: wantTexture ? texSize : null, atlas: wantTexture ? [texW, atlasH] : null, textureCodec: wantTexture ? codecRaw : null, crf: wantTexture ? crf : null,
        noTexture, forceIntra: !sheets, smoothTemporal,
        quantBitsPos: 14, quantBitsUv: 16,
        transform: modelXf,
        audio: audio ? { source: basename(audioSource ?? video), bitrateKbps: numFlag(a, "--audio-bitrate", 96), offsetSec: numFlag(a, "--audio-offset", 0), packets: audio.packets.length } : null,
      },
      geometry: { mode, temporalFrames: r.temporalFrames, intraFrames: r.intraFrames, chunks: r.chunks, trisPerFrame: Math.round(triSum / Math.max(1, frames)), fullGridTris: fullTris },
      tooling: { encoder: "ares-cli/0.2.0", node: process.version, generatedBy: "ares depth" },
    });
    await info(["info", out]);
  } finally {
    writer?.abort();
    for (const s of stores) s.close();
    run.close();
    await rm(scratch, { recursive: true, force: true });
  }
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
  // Never read whole: a clip runs to gigabytes, and `ares depth` ends every encode with this. The
  // head (everything before the first chunk) is read once; the audio totals and a splat clip's
  // first frame come from chunks read one at a time.
  const fh = await open(path, "r");
  try {
    const hdr = new Uint8Array(HEADER_SIZE);
    await fh.read(hdr, 0, HEADER_SIZE, 0);
    const first = Number(Demuxer.parseHeader(hdr).firstChunkOffset);
    const head = new Uint8Array(first);
    await fh.read(head, 0, first, 0);
    const file = Demuxer.parse(head);
    /** One chunk as a file of its own: its bytes at offset 1 (chunkAt rejects a chunk at 0). */
    const chunkFile = async (gop: GopEntry) => {
      const buf = new Uint8Array(1 + gop.byteLength);
      await fh.read(buf, 1, gop.byteLength, Number(gop.byteOffset));
      const f = { ...file, buf, gopIndex: [{ ...gop, byteOffset: 1n }] };
      return { f, chunk: Demuxer.chunkAt(f, f.gopIndex[0]!) };
    };
    const h = file.header, s = file.superblock;
    const profile = h.geometryProfile === GeometryProfile.SplatIPB ? "splat" : h.geometryProfile === GeometryProfile.MeshIPB ? "mesh" : `profile ${h.geometryProfile}`;
    console.log(`[ares] ${path}: v${h.versionMajor}.${h.versionMinor}, ${h.frameCount} frames @ ${h.fps}fps, ${(Number(h.durationUs) / 1e6).toFixed(2)}s`);
    if (h.geometryProfile === GeometryProfile.SplatIPB) {
      await meshoptReady();
      let n = 0;
      try {
        const { f, chunk } = await chunkFile(file.gopIndex[0]!);
        n = decodeSplatBlock(Demuxer.geometryBlocks(f, chunk)[0]!.data).count;
      } catch { /* leave 0 */ }
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
      for (const gop of file.gopIndex) {
        const { f, chunk } = await chunkFile(gop);
        for (const p of Demuxer.audioPackets(f, chunk)) { packets++; bytes += p.data.byteLength; endUs = Math.max(endUs, p.ptsUs + p.durationUs); }
      }
      console.log(`       audio: ${at.fourcc} 48 kHz ${at.channels === 1 ? "mono" : "stereo"}, ${packets} packets, ${(endUs / 1e6).toFixed(2)}s, ${(bytes / 1024).toFixed(1)} KB`);
    }
    console.log(`       AABB min [${s.aabb.min.map((v) => v.toFixed(2)).join(", ")}] max [${s.aabb.max.map((v) => v.toFixed(2)).join(", ")}]`);
  } finally { await fh.close(); }
}

/**
 * `ares verify-edits <file.edits.json>` — a sidecar pre-flight that needs no frames directory.
 * parseEditList throws on a malformed document; this additionally runs validateMasks, which is
 * the only check that catches the failure mode a bake CANNOT report: an unrecognised mask2d
 * matches nothing, so the encode succeeds and the edit silently did not happen. Exits 1 on any
 * issue so a script can gate a bake on it.
 */
async function verifyEdits(a: string[]) {
  const path = a[1]!;
  const list = parseEditList(JSON.parse(await readFile(path, "utf8")));
  const trim = list.trim ? `, trim ${list.trim.in}..${list.trim.out}` : "";
  console.log(`[ares] ${path}: ${list.ranges.length} range(s)${list.frameCount ? `, ${list.frameCount} source frames` : ""}${trim}`);
  const issues = validateMasks(list);
  list.ranges.forEach((r, i) => {
    const derived = r.keyframes.filter((k) => k.derived === true).length;
    const shapes = new Set<string>();
    for (const kf of r.keyframes) for (const v of kf.volumes ?? []) {
      shapes.add(v.type !== "mask2d" ? v.type
        : v.kind === "bitmap" && v.mask ? `mask2d bitmap ${v.mask.width}x${v.mask.height}`
        : `mask2d ${v.kind ?? "no kind"}`);
    }
    // Re-run per range for the count only: the printed issues come from the whole-document pass
    // above, which is the one that knows each range's real index for an id-less range.
    const bad = validateMasks({ ...list, ranges: [r] }).length;
    // `muted` is the one field of the same kind as mode/action that changes whether the bake reads
    // this range at all (encode filters on isRangeEnabled), so a report that gates a bake has to
    // say it — otherwise a muted range's issues read as a reason the bake will fail when the bake
    // will not even look at it.
    console.log(`       ${r.id ?? `#${i}`} · ${r.mode}${r.action && r.action !== "delete" ? ` ${r.action}` : ""}${isRangeEnabled(r) ? "" : " · muted"} · frames ${r.startFrame}..${r.endFrame} · ` +
      `${r.keyframes.length} kf (${r.keyframes.length - derived} user, ${derived} derived) · ${[...shapes].join(", ") || "no volumes"} · ${bad ? `${bad} issue(s)` : "ok"}`);
  });
  for (const issue of issues) console.log(`[ares] edits: ${issue}`);
  // exitCode, not exit(): the summary above is several console.log calls and a piped stdout on
  // Windows flushes asynchronously, so exiting here would truncate the very report being gated on.
  if (issues.length) process.exitCode = 1;
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
  ares depth  <video> --depth <run-dir> [-o out.ares]   (2D video + monocular depth run → 2.5D relief clip)
              unprojection: [--fov 55] [--near 2, or 0.5 for a metric run] [--far 6, or 20] [--grid 256]
                            [--edge 0.08] [--sheets]            (--sheets: full grid, persistent topology, silhouettes cut at draw time)
                            [--decimate ratio]                  (triangles kept per frame after the cut; excludes --sheets)
                            [--crop auto|none|W:H:X:Y]          (letterbox: auto detects the bars; W:H:X:Y in source pixels)
              surface:      [--no-guided] [--guide-sigma 14]    (image-guided resampling of depth onto the grid)
                            [--snap-ramps]                      (vertices partway down a silhouette ramp take one side's depth)
                            [--inpaint] [--inpaint-band cells]  (fill layer behind silhouettes, plate under the frame in the atlas)
              depth:        [--stabilize 0.7] [--no-grow]       (0 = raw per-frame depth, 1 = strongest temporal smoothing)
                            [--median 5]                        (temporal median frames after smoothing: 1 off, 3, 5)
              texture:      [--tex-size 1024] [--texture-codec vp9|av1] [--crf 30] [--no-texture]
              audio:        on by default from the video's own stream; [--no-audio] [--audio file] [--audio-offset s] [--audio-bitrate kbps]
              also:         [--gop 30] [--smooth-temporal N] [--up-axis x|y|z] [--center bottom|mass|none]
                            [--scale N] [--rotate x,y,z] [--translate x,y,z] [--meta-extra-file f.json]
  ares export <file.ares> -o <out> [--frame N]
              mesh → .obj | .ply      splat → .spz | .ply (3DGS) | .glb (KHR_gaussian_splatting) | .splat
  ares info   <file.ares>
  ares verify-edits <file.edits.json>   (per-range keyframe/mask2d report; exit 1 on any issue)`;

async function main() {
  const a = process.argv.slice(2);
  const cmd = a[0];
  try {
    if (cmd === "synth") await synth(a);
    else if (cmd === "encode" && a[1]) await encode(a);
    else if (cmd === "depth" && a[1]) await depth(a);
    else if (cmd === "export" && a[1]) await exportCmd(a);
    else if (cmd === "info" && a[1]) await info(a);
    else if (cmd === "verify-edits" && a[1]) await verifyEdits(a);
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
