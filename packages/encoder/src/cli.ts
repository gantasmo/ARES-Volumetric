#!/usr/bin/env node
/**
 * `ares` CLI (spec §5.2, §14 P5).
 *   ares synth  [-o out.ares] [--shape object|talk] [--frames 60] [--fps 30] [--no-texture]
 *   ares encode <frames-dir> [-o out.ares] [--fps 30] [--max-frames N] [--gop 30]
 *               [--texture-codec vp9|av1] [--tex-size 1024] [--crf 32] [--no-texture]
 *               [--edits file.json] [--crop x0,y0,z0,x1,y1,z1] [--track]
 *               [--trim-in N] [--trim-out N]              (clip in/out, inclusive source frames)
 *               [--up-axis x|y|z] [--center bottom|mass|none] [--scale N]
 *               [--rotate x,y,z] [--translate x,y,z]      (bake orientation into the model)
 *               [--smooth N] [--smooth-temporal N]        (OBJ/PLY + atlas → .ares)
 *   ares info   <file.ares>
 */
import { readdir, readFile, writeFile, stat } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import { Demuxer } from "@ares/core";
import { parsePly } from "./importers/ply.js";
import { parseObj } from "./importers/obj.js";
import { filterFrame, parseCropBox } from "./crop.js";
import { decimateFrame, simplifierReady } from "./decimate.js";
import { parseEditList, keepPredicateAt, prepareRangeAt, type EditList, type EditRange } from "@ares/core";
import { resolveOffset, applyTransform, applyTransformNormals, transformAabb, type ModelTransform } from "@ares/core";
import type { Aabb } from "@ares/core";
import { collectCopyOps, applyGeoCopy, resolveSrcFragment, buildPieceFragment, type CopyOp } from "./frame-copy.js";
import {
  buildPatchedAtlasDir, pngSize, rasterizeUvFootprint, dilateMask,
  planRegionRelocation, measureTexelCollateral, identityPlan, type TexelPatchPlan,
} from "./texel-copy.js";
import { collectRecolorOps, partitionTrianglesByCentroid, hexOf, parseHexColor, type RecolorPatch } from "./recolor.js";
import { detectHoleLoopsForRange, appendCaps, findFreeTile, HOLE_TILE, type HoleTileFill } from "./hole-patch.js";
import { muxClip, muxClipWithStats, type MuxClip } from "./muxer.js";
import { synthClip } from "./synth.js";
import { proceduralAtlas } from "./png.js";
import { encodeTextureVideo, detectPattern, ffmpegAvailable, type TexCodec } from "./texture-video.js";
import type { EncodeMeshFrame } from "./geometry-encode.js";

function flag(a: string[], name: string): string | undefined {
  const i = a.indexOf(name);
  return i >= 0 ? a[i + 1] : undefined;
}
const has = (a: string[], name: string) => a.includes(name);

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

async function synth(a: string[]) {
  const out = flag(a, "-o") ?? "demo.ares";
  const shape = (flag(a, "--shape") ?? "object") as "object" | "talk";
  const frames = Number(flag(a, "--frames") ?? 60);
  const fps = Number(flag(a, "--fps") ?? 30);
  const clip = synthClip(shape, frames, fps);
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
  await info([out]);
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

async function encode(a: string[]) {
  let dir = a[1]!;
  const out = flag(a, "-o") ?? "out.ares";
  const fps = Number(flag(a, "--fps") ?? 30);
  const maxFrames = flag(a, "--max-frames") ? Number(flag(a, "--max-frames")) : Infinity;
  const gop = Number(flag(a, "--gop") ?? 30);

  // Frame discovery. Ease-of-use: if the folder holds no meshes directly but a single subfolder does
  // (e.g. pointing at a parent like Foo/ that contains Foo_Volcap/mesh-*.obj), descend into it.
  const meshesIn = (names: string[]) => ({
    obj: names.filter((f) => extname(f).toLowerCase() === ".obj").sort(),
    ply: names.filter((f) => extname(f).toLowerCase() === ".ply").sort(),
  });
  let all = await readdir(dir);
  let { obj: objFiles, ply: plyFiles } = meshesIn(all);
  if (!objFiles.length && !plyFiles.length) {
    const subdir = await findFramesSubdir(dir, all);
    if (subdir) {
      console.log(`[ares] no meshes in ${dir}; using frames subfolder ${subdir}`);
      dir = subdir;
      all = await readdir(dir);
      ({ obj: objFiles, ply: plyFiles } = meshesIn(all));
    }
  }
  const isObj = objFiles.length > 0;
  const discovered = (isObj ? objFiles : plyFiles).slice(0, maxFrames);
  if (!discovered.length) throw new Error(`no .obj or .ply frames found in ${dir}`);

  // Mesh-editor bake (docs/editor-v2-design.md §10). Parsed BEFORE the trim window is resolved
  // because the sidecar is where the editor persists its in/out points.
  let editList: EditList | null = null;
  const editsArg = flag(a, "--edits");
  if (editsArg) editList = parseEditList(JSON.parse(await readFile(editsArg, "utf8")));

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

  console.log(`[ares] ${files.length} ${isObj ? "OBJ" : "PLY"} frame(s) @ ${fps}fps, gop=${gop}`);
  const frames: EncodeMeshFrame[] = [];
  const t0 = performance.now();
  for (const f of files) {
    if (isObj) {
      const m = parseObj(await readFile(join(dir, f), "utf8"));
      frames.push({ positions: m.positions, uvs: m.uvs, indices: m.indices });
    } else {
      const m = parsePly(await readFile(join(dir, f)));
      frames.push({ positions: m.positions, indices: m.indices });
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
  const modelXf: ModelTransform | null = (() => {
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
  })();
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
  const atlasFiles = all.filter((f) => /atlas.*\.png$/i.test(f)).sort().slice(0, maxFrames).slice(from, toExcl);
  const copyOps: CopyOp[] = editList ? collectCopyOps(editList.ranges, frames.length) : [];
  const recolorOps = editList ? collectRecolorOps(editList.ranges, frames.length) : [];
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
        const codec = ((flag(a, "--texture-codec") ?? "vp9") as TexCodec);
        const size = Number(flag(a, "--tex-size") ?? 1024);
        const crf = Number(flag(a, "--crf") ?? 32);

        // Copy texels (frame-copy "texels"/"both", editor v3 §1): patch a scratch copy of the
        // atlas directory before ffmpeg reads it — encodeTextureVideo reads atlas PNGs straight
        // off disk, so the copy has to land there too. Source atlas dir is never modified.
        let texDir = dir;
        let cleanupTexDir: (() => Promise<void>) | undefined;
        const texelOps = copyOps.filter((o) => o.what !== "geo");
        if (texelOps.length || recolorPatches.size || holeTileFills.size) {
          const tp = performance.now();
          const patched = await buildPatchedAtlasDir(dir, atlasFiles, texelOps, texelPlans, recolorPatches, holeTileFills);
          texDir = patched.dir;
          cleanupTexDir = patched.cleanup;
          console.log(`[ares] atlas patch: ${patched.patchedFrames} atlas frame(s) patched (copy + recolor + hole-patch, one decode each) in ${((performance.now() - tp) / 1000).toFixed(1)}s`);
        }

        console.log(`[ares] encoding texture video (${codec} ${size}²) from ${atlasFiles.length} atlas frames — this runs ffmpeg per GOP…`);
        const tt = performance.now();
        // .finally: the scratch atlas copy can be GBs — reclaim it even when ffmpeg throws.
        const tv = await encodeTextureVideo({ dir: texDir, pattern: pat.pattern, startNumber: pat.startNumber, frameCount: files.length, gopLength: gop, fps, codec, size, crf, repackFrames })
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
    smoothSpatial: Number(flag(a, "--smooth") ?? 0),
    smoothTemporal: Number(flag(a, "--smooth-temporal") ?? 0),
    // --no-temporal: force all-intra geometry (proven-safe for multi-topology coherent bakes, whose
    // temporal I+P path shredded geometry across run/topology boundaries).
    forceIntra: has(a, "--no-temporal"),
  };
  const clip: MuxClip = { fps, frames, gopLength: gop, textureVideo, temporal, meta: { title: dir, encoder: "ares-cli/0.1.0", source: isObj ? "obj" : "ply" } };
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
      quantBitsPos: 14, quantBitsUv: 14, // muxer defaults (no CLI override today)
      noTexture, forceIntra: temporal.forceIntra, track: temporal.track,
      smoothSpatial: temporal.smoothSpatial, smoothTemporal: temporal.smoothTemporal,
      decimate: decimateArg ? Number(decimateArg) : null,
      crop: cropArg ?? null,
      edits: editsArg ? basename(editsArg) : null,
    },
    geometry: { mode, temporalFrames: r.temporalFrames, intraFrames: r.intraFrames },
    tooling: { encoder: "ares-cli/0.1.0", node: process.version, generatedBy: "ares encode" },
  });
  await info([out]);
}

async function info(a: string[]) {
  const path = a[a.length - 1]!;
  const file = Demuxer.parse(await readFile(path));
  const h = file.header, s = file.superblock;
  console.log(`[ares] ${path}: v${h.versionMajor}.${h.versionMinor}, ${h.frameCount} frames @ ${h.fps}fps, ${(Number(h.durationUs) / 1e6).toFixed(2)}s`);
  console.log(`       geometry profile=${h.geometryProfile} intra=${h.intraCodec} · quant ${s.quantBitsPos}b pos / ${s.quantBitsUv}b uv · gop=${s.gopLength}`);
  console.log(`       ${file.gopIndex.length} chunk(s), ${file.tracks.length} track(s): ${file.tracks.map((t) => t.codecFourcc).join(", ")}`);
  const vid = Demuxer.textureVideo(file);
  const tex = Demuxer.textureAtlas(file);
  console.log(`       texture: ${vid ? `${vid.fourcc} video ${vid.width}x${vid.height}` : tex ? `${tex.width}x${tex.height} still atlas, ${(tex.bytes.length / 1024).toFixed(1)} KB` : "none"}`);
  console.log(`       AABB min [${s.aabb.min.map((v) => v.toFixed(2)).join(", ")}] max [${s.aabb.max.map((v) => v.toFixed(2)).join(", ")}]`);
}

async function main() {
  const a = process.argv.slice(2);
  const cmd = a[0];
  try {
    if (cmd === "synth") await synth(a);
    else if (cmd === "encode" && a[1]) await encode(a);
    else if (cmd === "info" && a[1]) await info(a);
    else {
      console.error("usage:\n  ares synth [-o out.ares] [--shape object|talk] [--frames 60] [--fps 30] [--no-texture]\n  ares encode <frames-dir> [-o out.ares] [--fps 30] [--max-frames N] [--gop 30]\n              [--texture-codec vp9|av1] [--tex-size 1024] [--crf 32] [--no-texture]\n              [--edits file.json] [--crop x0,y0,z0,x1,y1,z1] [--track] [--no-temporal]\n              [--smooth N] [--smooth-temporal N] [--decimate ratio]\n              [--meta-extra-file f.json]  (merged into the <out>.ares.meta.json provenance sidecar)\n  ares info <file.ares>");
      process.exit(1);
    }
  } catch (e) {
    console.error("[ares] error:", (e as Error).message);
    process.exit(1);
  }
}

main();
