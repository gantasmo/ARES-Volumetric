/**
 * Frame-to-frame copy — texel half (geometry half in frame-copy.ts).
 *
 * The texture-video encoder (texture-video.ts encodeTextureVideo) reads atlas PNGs straight off
 * disk by filename pattern — it never sees in-memory pixels — so a texel copy has to land on
 * disk too. We never touch the source atlas directory (read-only capture data): unaffected
 * frames are hard-linked into a scratch copy (falls back to a byte copy across volumes, e.g.
 * source on D:, scratch under the OS tmpdir on C:), and only the dst frames that need a texel
 * patch get their atlas PNG decoded, patched, and re-written. A sidecar with no texel-copy ops
 * still pays for the directory materialization (ffmpeg's image2 demuxer needs every frame
 * present and sequential) but decodes/writes nothing extra.
 *
 * v1.1 semantics (the v1 same-coordinate blit was WRONG for this capture, caught in review:
 * the atlas is REPACKED EVERY FRAME, so identical atlas coordinates mean different surface in
 * different frames — measured on the Daniel capture: 57.8% of the dst triangles sampling a
 * same-coordinate patch at 30 frames' distance belong to other body parts, 78.1% even at ONE
 * frame's distance; there is no coherence horizon). what:"both" therefore RELOCATES the copied
 * texels into atlas space that is FREE in the dst frame after the copy's own dst-region
 * deletion (planRegionRelocation: whole charts where they fit, UV tiles where they don't), the
 * blit follows those translations, and the pasted geometry's UVs move identically
 * (frame-copy.ts buildPieceFragment), so pasted triangles sample exactly the relocated texels
 * and NOTHING the dst frame's surviving triangles sample is overwritten. what:"texels" keeps
 * the same-coordinate blit (only valid for a hypothetically temporally-coherent atlas) but is
 * gated by measureTexelCollateral — >5% collateral refuses the bake.
 */
import { execFile } from "node:child_process";
import { readFile, writeFile, link, copyFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { encodePNG } from "./png.js";
import { ffmpegPath } from "./texture-video.js";
import type { EncodeMeshFrame } from "./geometry-encode.js";
import type { CopyOp } from "./frame-copy.js";
import { applyRecolorToImage, type RecolorPatch } from "./recolor.js";
import { applyPaintToImage, type PaintPatch } from "./paint.js";
import { applyHoleFillToImage, type HoleTileFill } from "./hole-patch.js";

const run = promisify(execFile);

interface Rgba { data: Uint8Array; width: number; height: number; }

/** PNG's signature + IHDR are fixed-layout (spec): width/height are the first two big-endian
 *  uint32s of the IHDR chunk data, at fixed byte offsets 16/20. Reading just these two ints
 *  needs no chunk walk or zlib — cheaper than decoding just to learn the size. Exported so
 *  cli.ts can learn the atlas raster dimensions at copy-planning time (placement runs on mesh
 *  UVs long before any PNG is decoded). */
export function pngSize(buf: Uint8Array): { width: number; height: number } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return { width: dv.getUint32(16, false), height: dv.getUint32(20, false) };
}

/** Decode a PNG to tightly-packed RGBA via ffmpeg. There is no PNG decoder in this repo (only
 *  png.ts's RGBA *encoder*), and the atlas PNGs here are plain 8-bit RGB (color type 2, no
 *  interlace) — ffmpeg is already a hard dependency of texture-video encoding, so shelling out
 *  to it here reuses that dependency instead of writing a filter-reconstruction PNG decoder. */
async function decodeAtlas(pngPath: string, scratchDir: string): Promise<Rgba> {
  const { width, height } = pngSize(await readFile(pngPath));
  const rawPath = join(scratchDir, `dec-${Math.random().toString(36).slice(2)}.raw`);
  await run(ffmpegPath(), ["-y", "-hide_banner", "-loglevel", "error", "-i", pngPath, "-f", "rawvideo", "-pix_fmt", "rgba", rawPath], { maxBuffer: 1 << 28 });
  const data = new Uint8Array(await readFile(rawPath));
  await rm(rawPath, { force: true });
  return { data, width, height };
}

async function linkOrCopy(src: string, dst: string): Promise<void> {
  try { await link(src, dst); } catch { await copyFile(src, dst); }
}

/** Mark every pixel a segment passes through (integer DDA). Endpoints floored + clamped. */
function rasterizeLine(mask: Uint8Array, w: number, h: number, x0: number, y0: number, x1: number, y1: number): void {
  const ax = Math.min(w - 1, Math.max(0, Math.floor(x0))), ay = Math.min(h - 1, Math.max(0, Math.floor(y0)));
  const bx = Math.min(w - 1, Math.max(0, Math.floor(x1))), by = Math.min(h - 1, Math.max(0, Math.floor(y1)));
  const steps = Math.max(Math.abs(bx - ax), Math.abs(by - ay), 1);
  for (let i = 0; i <= steps; i++) {
    const x = Math.round(ax + ((bx - ax) * i) / steps);
    const y = Math.round(ay + ((by - ay) * i) / steps);
    mask[y * w + x] = 1;
  }
}

/** Fill `mask` (width*height, row-major) for one UV-space triangle in ATLAS PIXEL coordinates.
 *  Winding-agnostic (edge signs normalized by the signed area) since OBJ triangles here aren't
 *  guaranteed consistently wound after the fragment's index remap. The PERIMETER is always
 *  rasterized too (DDA edges): center-sampling alone can miss needle tips and slivers entirely,
 *  and the v1.1 relocation needs every VERTEX's pixel in its piece's mask — otherwise a pasted
 *  vertex's UV would point at a texel that was never copied. Edges also keep each triangle's
 *  coverage connected, so a triangle cannot straddle two footprint components. */
function rasterizeTriangle(mask: Uint8Array, w: number, h: number, ax: number, ay: number, bx: number, by: number, cx: number, cy: number): void {
  rasterizeLine(mask, w, h, ax, ay, bx, by);
  rasterizeLine(mask, w, h, bx, by, cx, cy);
  rasterizeLine(mask, w, h, cx, cy, ax, ay);
  const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if (area === 0) return; // degenerate (zero UV area) — perimeter only
  const sign = area > 0 ? 1 : -1;
  const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
  const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
  for (let y = minY; y <= maxY; y++) {
    const py = y + 0.5;
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5;
      const w0 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * sign;
      const w1 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * sign;
      const w2 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * sign;
      if (w0 >= 0 && w1 >= 0 && w2 >= 0) mask[y * w + x] = 1;
    }
  }
}

/** A mesh's UV footprint, rasterized into atlas-pixel space (px = u·w, py = v·h —
 *  importers/obj.ts already flips V to top-left origin at import, matching PNG row order
 *  directly). Called with a src fragment (the region to copy) AND with a whole dst frame
 *  (its atlas occupancy — every rasterized pixel is one some dst triangle samples). */
export function rasterizeUvFootprint(fragment: EncodeMeshFrame, width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  const { uvs, indices } = fragment;
  if (!uvs) return mask; // no UVs on this mesh — nothing to copy
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 2, b = indices[t + 1]! * 2, c = indices[t + 2]! * 2;
    rasterizeTriangle(mask, width, height,
      uvs[a]! * width, uvs[a + 1]! * height,
      uvs[b]! * width, uvs[b + 1]! * height,
      uvs[c]! * width, uvs[c + 1]! * height);
  }
  return mask;
}

/** ~2px dilation (task spec) to avoid seam bleed: mark every pixel within radius of a hit
 *  pixel. Circular offsets so the dilation is isotropic rather than a diamond/box. */
export function dilateMask(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const offsets: Array<[number, number]> = [];
  const r2 = radius * radius;
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) if (dx * dx + dy * dy <= r2) offsets.push([dx, dy]);
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (const [dx, dy] of offsets) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) out[ny * w + nx] = 1;
      }
    }
  }
  return out;
}

/* ------------------------- v1.1 relocation planning (pure atlas-space, no fs) ------------------------- */

interface Placeable { pixels: Uint32Array; minX: number; minY: number; maxX: number; maxY: number; }

interface AtlasComponents {
  /** component id per pixel of the footprint mask (−1 = not in the footprint) */
  labels: Int32Array;
  components: Placeable[];
}

/** 4-connected component labeling of the dilated src footprint. Each component is one UV chart
 *  island (or a cluster of islands nearer than twice the dilation radius — those merge and then
 *  share one translation, which stays consistent). */
function labelComponents(mask: Uint8Array, w: number, h: number): AtlasComponents {
  const labels = new Int32Array(w * h).fill(-1);
  const components: Placeable[] = [];
  const stack: number[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start] !== -1) continue;
    const id = components.length;
    const pixels: number[] = [];
    let minX = w, minY = h, maxX = 0, maxY = 0;
    labels[start] = id;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop()!;
      pixels.push(p);
      const x = p % w, y = (p - x) / w;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && mask[p - 1] && labels[p - 1] === -1) { labels[p - 1] = id; stack.push(p - 1); }
      if (x < w - 1 && mask[p + 1] && labels[p + 1] === -1) { labels[p + 1] = id; stack.push(p + 1); }
      if (y > 0 && mask[p - w] && labels[p - w] === -1) { labels[p - w] = id; stack.push(p - w); }
      if (y < h - 1 && mask[p + w] && labels[p + w] === -1) { labels[p + w] = id; stack.push(p + w); }
    }
    components.push({ pixels: Uint32Array.from(pixels), minX, minY, maxX, maxY });
  }
  return { labels, components };
}

/** One translation such that every translated pixel of `piece` is in-bounds and lands on a FREE
 *  pixel of `claimed`, or null. Coarse-to-fine (stride 8 grid, then stride 1) with a sparse
 *  early-out sample of the piece's pixels before the full check. */
function placePiece(piece: Placeable, claimed: Uint8Array, w: number, h: number): { dx: number; dy: number } | null {
  const sample: number[] = [];
  for (let i = 0; i < piece.pixels.length; i += 61) sample.push(piece.pixels[i]!);
  const fits = (dx: number, dy: number, pixels: ArrayLike<number>): boolean => {
    const d = dy * w + dx;
    for (let i = 0; i < pixels.length; i++) if (claimed[pixels[i]! + d]) return false;
    return true;
  };
  for (const stride of [8, 1]) {
    for (let dy = -piece.minY; dy <= h - 1 - piece.maxY; dy += stride) {
      for (let dx = -piece.minX; dx <= w - 1 - piece.maxX; dx += stride) {
        if (!fits(dx, dy, sample)) continue;
        if (fits(dx, dy, piece.pixels)) return { dx, dy };
      }
    }
  }
  return null;
}

function stampPiece(claimed: Uint8Array, pixels: Uint32Array, dx: number, dy: number, w: number): void {
  const d = dy * w + dx;
  for (let i = 0; i < pixels.length; i++) claimed[pixels[i]! + d] = 1;
}

/** Rasterize + dilate the footprint of a subset of the fragment's triangles. */
function rasterizeTris(fragment: EncodeMeshFrame, tris: number[], w: number, h: number, dilation: number): Placeable {
  const { uvs, indices } = fragment;
  const mask = new Uint8Array(w * h);
  for (const t of tris) {
    const a = indices[t]! * 2, b = indices[t + 1]! * 2, c = indices[t + 2]! * 2;
    rasterizeTriangle(mask, w, h,
      uvs![a]! * w, uvs![a + 1]! * h, uvs![b]! * w, uvs![b + 1]! * h, uvs![c]! * w, uvs![c + 1]! * h);
  }
  const dil = dilateMask(mask, w, h, dilation);
  const pixels: number[] = [];
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let p = 0; p < dil.length; p++) {
    if (!dil[p]) continue;
    pixels.push(p);
    const x = p % w, y = (p - x) / w;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { pixels: Uint32Array.from(pixels), minX, minY, maxX, maxY };
}

/** One relocated unit: these triangles' texels move by (dx,dy), and frame-copy.ts
 *  buildPieceFragment moves their UVs identically. Empty `pixels` (pure UV-sliver triangles
 *  that rasterize nothing) means nothing was copied and dx=dy=0 (UVs left alone). */
export interface RelocationPiece { tris: number[]; pixels: Uint32Array; dx: number; dy: number; }

/** Tile size (atlas px) for charts that cannot be placed whole. 128 keeps pieces small enough
 *  to tuck into a ~60%-occupied atlas's gaps while limiting seam count; each piece carries its
 *  own 2px dilation ring, so seams have chart-seam quality. */
const RELOC_TILE = 128;

/**
 * Plan the relocation of a src fragment's texels into FREE space of one dst frame's atlas.
 *
 * Shape strategy (all three simpler strategies measurably fail on this capture's test region):
 * rigid whole-footprint translation is impossible (the region's charts scatter across the
 * atlas — footprint bbox 2034×1658 of 2048²); per-chart translation fails for large charts
 * (the region's biggest chart is 316k px with a 678×533 bbox, and no gap that size exists in a
 * 58-60%-occupied packed atlas even after the dst region's own charts are freed). So: try each
 * chart whole (fewest seams), and split only the charts that don't fit into RELOC_TILE UV tiles
 * of triangles, each placed independently.
 *
 * `claimed` must be the dst frame's dilated occupancy of the triangles that SURVIVE the copy's
 * own dst-region deletion (plus any previously placed charts); it is MUTATED — every placed
 * piece is stamped in. Throws with measured numbers when even a tile cannot be placed.
 */
export function planRegionRelocation(
  fragment: EncodeMeshFrame, claimed: Uint8Array, w: number, h: number,
): { pieces: RelocationPiece[]; wholeCharts: number; tiledCharts: number } {
  const foot = dilateMask(rasterizeUvFootprint(fragment, w, h), w, h, 2);
  const ac = labelComponents(foot, w, h);

  // triangle → chart id, via the label under the UV centroid (inside the triangle, hence inside
  // the rasterized footprint; slivers that rasterized nothing probe a 3px neighborhood, then
  // fall into the identity piece — their texels were never copied, so their UVs must not move).
  const trisByChart = new Map<number, number[]>();
  const orphanTris: number[] = [];
  const { uvs, indices } = fragment;
  if (!uvs) return { pieces: [], wholeCharts: 0, tiledCharts: 0 };
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]! * 2, b = indices[t + 1]! * 2, c = indices[t + 2]! * 2;
    const px = Math.min(w - 1, Math.max(0, Math.floor(((uvs[a]! + uvs[b]! + uvs[c]!) / 3) * w)));
    const py = Math.min(h - 1, Math.max(0, Math.floor(((uvs[a + 1]! + uvs[b + 1]! + uvs[c + 1]!) / 3) * h)));
    let id = ac.labels[py * w + px]!;
    search:
    for (let r = 1; r <= 3 && id < 0; r++) {
      for (let oy = -r; oy <= r; oy++) {
        for (let ox = -r; ox <= r; ox++) {
          const nx = px + ox, ny = py + oy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const l = ac.labels[ny * w + nx]!;
          if (l >= 0) { id = l; break search; }
        }
      }
    }
    if (id < 0) { orphanTris.push(t); continue; }
    const arr = trisByChart.get(id) ?? [];
    arr.push(t);
    trisByChart.set(id, arr);
  }

  const pieces: RelocationPiece[] = [];
  let wholeCharts = 0, tiledCharts = 0;
  const chartIds = [...trisByChart.keys()].sort((x, y) => ac.components[y]!.pixels.length - ac.components[x]!.pixels.length);
  for (const id of chartIds) {
    const tris = trisByChart.get(id)!;
    const chart = ac.components[id]!;
    const whole = placePiece(chart, claimed, w, h);
    if (whole) {
      stampPiece(claimed, chart.pixels, whole.dx, whole.dy, w);
      pieces.push({ tris, pixels: chart.pixels, dx: whole.dx, dy: whole.dy });
      wholeCharts++;
      continue;
    }
    tiledCharts++;
    // Split this chart's triangles into UV tiles (centroid rule) and place each tile.
    const byTile = new Map<number, number[]>();
    for (const t of tris) {
      const a = indices[t]! * 2, b = indices[t + 1]! * 2, c = indices[t + 2]! * 2;
      const tx = Math.floor((((uvs[a]! + uvs[b]! + uvs[c]!) / 3) * w) / RELOC_TILE);
      const ty = Math.floor((((uvs[a + 1]! + uvs[b + 1]! + uvs[c + 1]!) / 3) * h) / RELOC_TILE);
      const key = ty * 1024 + tx;
      const arr = byTile.get(key) ?? [];
      arr.push(t);
      byTile.set(key, arr);
    }
    for (const tileTris of byTile.values()) {
      const piece = rasterizeTris(fragment, tileTris, w, h, 2);
      if (!piece.pixels.length) { pieces.push({ tris: tileTris, pixels: piece.pixels, dx: 0, dy: 0 }); continue; }
      const off = placePiece(piece, claimed, w, h);
      if (!off) {
        let free = 0;
        for (let i = 0; i < claimed.length; i++) if (!claimed[i]) free++;
        throw new Error(
          `copy texels: cannot place a ${piece.pixels.length}px tile (bbox ${piece.maxX - piece.minX + 1}×${piece.maxY - piece.minY + 1}) ` +
          `anywhere in the ${free}px of free dst atlas space — mask a smaller region`);
      }
      stampPiece(claimed, piece.pixels, off.dx, off.dy, w);
      pieces.push({ tris: tileTris, pixels: piece.pixels, dx: off.dx, dy: off.dy });
    }
  }
  if (orphanTris.length) pieces.push({ tris: orphanTris, pixels: new Uint32Array(0), dx: 0, dy: 0 });
  return { pieces, wholeCharts, tiledCharts };
}

/**
 * Gate for what:"texels" (same-coordinate blit): of the dst frame's triangles whose UV centroid
 * samples inside the (dilated) patch footprint, how many lie OUTSIDE the copy region in world
 * space? Those are collateral — their texels get overwritten with unrelated src content when
 * the atlas is packed per frame. cli.ts refuses above 5%.
 */
export function measureTexelCollateral(
  dst: EncodeMeshFrame, foot: Uint8Array, w: number, h: number,
  insideRegion: (x: number, y: number, z: number) => boolean,
): { affected: number; collateral: number } {
  const { positions, uvs, indices } = dst;
  if (!uvs) return { affected: 0, collateral: 0 };
  let affected = 0, collateral = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t]!, i1 = indices[t + 1]!, i2 = indices[t + 2]!;
    const u = (uvs[i0 * 2]! + uvs[i1 * 2]! + uvs[i2 * 2]!) / 3;
    const v = (uvs[i0 * 2 + 1]! + uvs[i1 * 2 + 1]! + uvs[i2 * 2 + 1]!) / 3;
    const px = Math.min(w - 1, Math.max(0, Math.floor(u * w)));
    const py = Math.min(h - 1, Math.max(0, Math.floor(v * h)));
    if (!foot[py * w + px]) continue;
    affected++;
    const cx = (positions[i0 * 3]! + positions[i1 * 3]! + positions[i2 * 3]!) / 3;
    const cy = (positions[i0 * 3 + 1]! + positions[i1 * 3 + 1]! + positions[i2 * 3 + 1]!) / 3;
    const cz = (positions[i0 * 3 + 2]! + positions[i1 * 3 + 2]! + positions[i2 * 3 + 2]!) / 3;
    if (!insideRegion(cx, cy, cz)) collateral++;
  }
  return { affected, collateral };
}

/** One dst frame's texel patch: blit src pixel (x,y) → dst pixel (x+dx, y+dy) per component.
 *  what:"texels" ops use a single identity component (dx=dy=0, the whole footprint). */
export interface TexelPatchComponent { pixels: Uint32Array; dx: number; dy: number; }
export interface TexelPatchPlan { width: number; height: number; components: TexelPatchComponent[]; }

/** The identity plan (what:"texels" same-coordinate blit, post-collateral-gate). */
export function identityPlan(foot: Uint8Array, w: number, h: number): TexelPatchPlan {
  const pixels: number[] = [];
  for (let p = 0; p < foot.length; p++) if (foot[p]) pixels.push(p);
  return { width: w, height: h, components: [{ pixels: Uint32Array.from(pixels), dx: 0, dy: 0 }] };
}

function blitComponent(dst: Uint8Array, src: Uint8Array, c: TexelPatchComponent, w: number): void {
  const d = (c.dy * w + c.dx) * 4;
  for (let i = 0; i < c.pixels.length; i++) {
    const o = c.pixels[i]! * 4, q = o + d;
    dst[q] = src[o]!; dst[q + 1] = src[o + 1]!; dst[q + 2] = src[o + 2]!; dst[q + 3] = src[o + 3]!;
  }
}

export interface PatchedAtlasDir { dir: string; patchedFrames: number; cleanup(): Promise<void>; }

/**
 * Build a scratch copy of the atlas directory with texel-copy dst frames, recolor frames, AND
 * hole-patch tiles patched in place — ONE scratch dir, ONE decode per touched frame even when
 * several of these touch it (copy blits apply first, recolor second, hole-tile fill last, each
 * over the previous step's result — so a pasted-in
 * region can itself be recolored, and a capped hole can average a just-recolored rim texel).
 * Only copy ops with what !== "geo" are applied here. `plans` must hold one TexelPatchPlan per
 * (op, dstFrame); `recolorPatches` and `holeFills` must hold the patch(es) queued for each frame
 * index — all computed by the CALLER at planning time against frames exactly as imported (and,
 * for recolor/hole-patch, AFTER any copy geometry mutation — see cli.ts), before delete/keep or
 * decimate touch them (this function runs late in cli.ts, right before texture-video encoding, by
 * which point `frames` may already be decimated; planning from already-decimated geometry would
 * rasterize a footprint that no longer matches what the geometry half of the same op pasted).
 * Returns the directory to hand to encodeTextureVideo in place of the original, plus a cleanup()
 * to remove the scratch copy once the texture video has been encoded from it.
 */
export async function buildPatchedAtlasDir(
  dir: string, atlasFiles: string[], ops: CopyOp[], plans: Map<CopyOp, Map<number, TexelPatchPlan>>,
  recolorPatches?: Map<number, RecolorPatch[]>, holeFills?: Map<number, HoleTileFill[]>,
  paintPatches?: Map<number, PaintPatch[]>,
): Promise<PatchedAtlasDir> {
  const texOps = ops.filter((o) => o.what !== "geo");
  const work = await mkdtemp(join(tmpdir(), "ares-texcopy-"));
  const cleanup = () => rm(work, { recursive: true, force: true });
  if (!texOps.length && !recolorPatches?.size && !holeFills?.size && !paintPatches?.size) {
    for (const name of atlasFiles) await linkOrCopy(join(dir, name), join(work, name));
    return { dir: work, patchedFrames: 0, cleanup };
  }

  const patchesByDst = new Map<number, CopyOp[]>();
  for (const op of texOps) for (const d of op.dstFrames) {
    const arr = patchesByDst.get(d) ?? [];
    arr.push(op);
    patchesByDst.set(d, arr);
  }
  const srcCache = new Map<number, Rgba>(); // srcFrame index → decoded rgba (ops may share a srcFrame)
  let patchedFrames = 0;
  for (let i = 0; i < atlasFiles.length; i++) {
    const name = atlasFiles[i]!;
    const dstPath = join(work, name);
    const patches = patchesByDst.get(i);
    const recolors = recolorPatches?.get(i);
    const paints = paintPatches?.get(i);
    const holes = holeFills?.get(i);
    if ((!patches || !patches.length) && (!recolors || !recolors.length) && (!paints || !paints.length) && (!holes || !holes.length)) {
      await linkOrCopy(join(dir, name), dstPath);
      continue;
    }

    const dstImg = await decodeAtlas(join(dir, name), work);
    if (patches) {
      for (const op of patches) {
        let srcImg = srcCache.get(op.srcFrame);
        if (!srcImg) {
          srcImg = await decodeAtlas(join(dir, atlasFiles[op.srcFrame]!), work);
          srcCache.set(op.srcFrame, srcImg);
        }
        if (srcImg.width !== dstImg.width || srcImg.height !== dstImg.height) {
          throw new Error(`copy texels: atlas size mismatch — frame ${op.srcFrame} is ${srcImg.width}x${srcImg.height}, frame ${i} is ${dstImg.width}x${dstImg.height}`);
        }
        const plan = plans.get(op)?.get(i);
        if (!plan) throw new Error(`copy texels: no patch plan for range ${op.range.id ?? "?"} frame ${i} (internal — caller plans before patching)`);
        if (plan.width !== dstImg.width || plan.height !== dstImg.height) {
          throw new Error(`copy texels: plan raster ${plan.width}x${plan.height} but frame ${i} atlas is ${dstImg.width}x${dstImg.height}`);
        }
        for (const c of plan.components) blitComponent(dstImg.data, srcImg.data, c, dstImg.width);
      }
    }
    if (recolors) for (const rp of recolors) applyRecolorToImage(dstImg.data, rp); // over the copy result, in queued order
    if (paints) for (const pp of paints) applyPaintToImage(dstImg.data, pp, dstImg.width, dstImg.height); // over copy + recolor
    if (holes) for (const hf of holes) applyHoleFillToImage(dstImg.data, dstImg.width, hf); // over copy + recolor + paint, last
    await writeFile(dstPath, encodePNG(dstImg.data, dstImg.width, dstImg.height));
    patchedFrames++;
  }
  return { dir: work, patchedFrames, cleanup };
}
