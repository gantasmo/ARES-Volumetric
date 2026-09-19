/**
 * Depth-run directory I/O — the handoff between a depth engine (the Python GPU service or the
 * browser worker) and `ares depth`.
 *
 * A run is two files in one directory:
 *   depth.json  the sidecar below (schema "ares-depth/1"): model identity, map size, frame count,
 *               the frame-sampling recipe, and the source video's own dimensions.
 *   depth.f32   N x H x W float32 little-endian, frame-major then row-major, row 0 = top of the
 *               image, no header. `kind` says what the numbers mean:
 *                 "relative-disparity" — the model's raw affine-invariant inverse depth. LARGER =
 *                   NEARER, unnormalized, and neither scale nor shift is consistent between frames
 *                   (that inconsistency is what depth-stabilize.ts removes).
 *                 "metric-depth" — metres, larger = farther.
 *
 * A run whose sidecar says `done: false` is a run still being written (or one that was killed
 * partway). It is READ ANYWAY, up to the number of whole frames actually on disk, because a
 * half-finished 300-frame inference is still a usable 180-frame clip and re-running the GPU pass to
 * recover it costs minutes. `frames` in the result is always what the .f32 really holds, never what
 * the sidecar hoped for.
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { openSync, closeSync, readSync, fstatSync, existsSync } from "node:fs";
import { join } from "node:path";

export type DepthKind = "relative-disparity" | "metric-depth";

export interface DepthRunMeta {
  schema: string;
  engine: string;              // "service" | "browser"
  model: string;               // full model id
  modelKey: string;            // "small" | "base" | "large" | "metric-indoor-small" | ...
  kind: DepthKind;
  width: number;
  height: number;
  frames: number;
  fps: number;                 // frame rate of the SAMPLED sequence (the clip's fps)
  sampling: { fps: number | null; maxFrames: number | null };
  video: string;               // absolute source path
  sourceFps: number | null;
  sourceWidth: number | null;
  sourceHeight: number | null;
  sourceFrames: number | null;
  sourceDurationS: number | null;
  msPerFrame: number | null;
  device: string;
  dtype: string;
  done: boolean;
  /** "model" when the engine is temporally consistent by construction (Video-Depth-Anything);
   *  the stabilizer then skips its per-frame scale/shift alignment. Absent or "none" otherwise. */
  temporal?: "model" | "none";
  /** Present when the engine ran a subject mask pre-pass: `<dir>/<file>` is frames x H x W uint8,
   *  0 outside the subject and 255 inside, in the same frame order and orientation as depth.f32. */
  mask?: { file: string; prompt?: string; engine?: string; coverage?: number };
}

export interface DepthRun {
  meta: DepthRunMeta;
  /** frames x height x width float32, frame-major then row-major. */
  maps: Float32Array;
  frames: number;
  width: number;
  height: number;
}

const SCHEMA = "ares-depth/1";

function posInt(v: unknown, what: string): number {
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) throw new Error(`depth.json: ${what} must be a positive integer, got ${JSON.stringify(v)}`);
  return n;
}

/**
 * A depth run opened for frame-at-a-time reading. A run is `frames x H x W x 4` bytes, which for a
 * feature-length clip is past anything one buffer can hold (5,627 frames at 518x294 is 3.4 GB), so
 * the encoder never reads a run whole: it pulls one frame, or one GOP of frames, at a time.
 */
export interface DepthRunReader {
  meta: DepthRunMeta;
  frames: number;
  width: number;
  height: number;
  /** True when the run carries a subject mask (`meta.mask`) and its file is on disk. */
  hasMask: boolean;
  /** Read `count` whole frames starting at frame `t` into `out` (length >= count * W * H). */
  readFrames(t: number, count: number, out: Float32Array): void;
  /** Read `count` mask frames (0 / 255) starting at frame `t`. Throws when the run has no mask. */
  readMask(t: number, count: number, out: Uint8Array): void;
  close(): void;
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** readSync until `len` bytes are in, or throw: a short read here is a truncated run. */
function readFully(fd: number, into: Uint8Array, len: number, position: number, what: string): void {
  let got = 0;
  while (got < len) {
    const n = readSync(fd, into, got, len - got, position + got);
    if (n <= 0) throw new Error(`${what}: unexpected end of file at byte ${position + got}`);
    got += n;
  }
}

/** Open `<dir>/depth.json` + `<dir>/depth.f32` for streaming. Same validation as readDepthRun. */
export async function openDepthRun(dir: string): Promise<DepthRunReader> {
  const metaPath = join(dir, "depth.json");
  let meta: DepthRunMeta;
  try {
    meta = JSON.parse(await readFile(metaPath, "utf8")) as DepthRunMeta;
  } catch (e) {
    throw new Error(`cannot read ${metaPath}: ${(e as Error).message}`);
  }
  if (meta?.schema !== SCHEMA) throw new Error(`${metaPath}: expected schema ${JSON.stringify(SCHEMA)}, got ${JSON.stringify(meta?.schema)}`);
  if (meta.kind !== "relative-disparity" && meta.kind !== "metric-depth")
    throw new Error(`${metaPath}: kind must be "relative-disparity" or "metric-depth", got ${JSON.stringify(meta.kind)}`);
  const width = posInt(meta.width, "width");
  const height = posInt(meta.height, "height");
  const declared = posInt(meta.frames, "frames");
  if (!(Number(meta.fps) > 0)) throw new Error(`${metaPath}: fps must be > 0, got ${JSON.stringify(meta.fps)}`);

  const mapsPath = join(dir, "depth.f32");
  let fd: number, byteLength: number;
  try {
    fd = openSync(mapsPath, "r");
    byteLength = fstatSync(fd).size;
  } catch (e) {
    throw new Error(`cannot read ${mapsPath}: ${(e as Error).message}`);
  }
  let maskFd = -1;
  try {
    const frameBytes = width * height * 4;
    const whole = Math.floor(byteLength / frameBytes);
    if (byteLength % frameBytes !== 0) {
      // A torn tail is only ever the last frame of an in-flight write; anything else means the
      // sidecar's W/H do not describe this file, which would silently reinterpret every row.
      if (meta.done !== false) throw new Error(`${mapsPath}: ${byteLength} bytes is not a whole number of ${width}x${height} float32 frames (${frameBytes} B each)`);
      console.warn(`[ares] depth run: ${mapsPath} ends mid-frame (${byteLength} B, ${frameBytes} B/frame) — dropping the partial tail frame`);
    }
    let frames = whole;
    if (frames < 1) throw new Error(`${mapsPath}: holds no complete ${width}x${height} frame (${byteLength} bytes)`);
    if (frames > declared) frames = declared;
    if (frames < declared) {
      if (meta.done === false) console.warn(`[ares] depth run is INCOMPLETE (done:false): ${frames} of ${declared} frames on disk — encoding the ${frames} that exist`);
      else throw new Error(`${mapsPath}: declares ${declared} frames but holds ${frames} (${byteLength} bytes, ${frameBytes} B/frame)`);
    } else if (meta.done === false) {
      console.warn(`[ares] depth run is marked done:false but all ${frames} frames are present — encoding them`);
    }

    const P = width * height;
    if (meta.mask?.file) {
      const maskPath = join(dir, meta.mask.file);
      if (!existsSync(maskPath)) throw new Error(`${metaPath}: names a subject mask ${JSON.stringify(meta.mask.file)} that is not in the run directory`);
      maskFd = openSync(maskPath, "r");
      const have = Math.floor(fstatSync(maskFd).size / P);
      if (have < frames) throw new Error(`${maskPath}: holds ${have} mask frame(s), the run has ${frames}`);
    }

    const swap = LITTLE_ENDIAN ? null : new Uint8Array(4);
    const theFd = fd, theMaskFd = maskFd;
    return {
      meta: { ...meta, frames, width, height }, frames, width, height, hasMask: maskFd >= 0,
      readFrames(t, count, out) {
        if (t < 0 || count < 0 || t + count > frames) throw new Error(`depth run: frames ${t}..${t + count} are outside 0..${frames}`);
        if (out.length < count * P) throw new Error(`depth run: output holds ${out.length} floats, ${count * P} needed`);
        const bytes = new Uint8Array(out.buffer, out.byteOffset, count * frameBytes);
        readFully(theFd, bytes, count * frameBytes, t * frameBytes, mapsPath);
        if (swap) for (let o = 0; o < bytes.length; o += 4) { swap.set(bytes.subarray(o, o + 4)); bytes[o] = swap[3]!; bytes[o + 1] = swap[2]!; bytes[o + 2] = swap[1]!; bytes[o + 3] = swap[0]!; }
      },
      readMask(t, count, out) {
        if (theMaskFd < 0) throw new Error("depth run: this run carries no subject mask");
        if (t < 0 || count < 0 || t + count > frames) throw new Error(`depth run: mask frames ${t}..${t + count} are outside 0..${frames}`);
        readFully(theMaskFd, out, count * P, t * P, "mask");
      },
      close() { try { closeSync(theFd); } catch { /* already closed */ } if (theMaskFd >= 0) try { closeSync(theMaskFd); } catch { /* already closed */ } },
    };
  } catch (e) {
    try { closeSync(fd); } catch { /* ignore */ }
    if (maskFd >= 0) try { closeSync(maskFd); } catch { /* ignore */ }
    throw e;
  }
}

/** Read `<dir>/depth.json` + `<dir>/depth.f32` whole. For short runs and tests; the CLI streams
 *  through openDepthRun instead, because a long run does not fit one buffer. */
export async function readDepthRun(dir: string): Promise<DepthRun> {
  const r = await openDepthRun(dir);
  try {
    const maps = new Float32Array(r.frames * r.width * r.height);
    // One GOP-sized read at a time: a single readSync is capped at 2 GiB.
    const P = r.width * r.height, step = Math.max(1, Math.floor((1 << 26) / (P * 4)));
    for (let t = 0; t < r.frames; t += step) {
      const n = Math.min(step, r.frames - t);
      r.readFrames(t, n, maps.subarray(t * P, (t + n) * P));
    }
    return { meta: r.meta, maps, frames: r.frames, width: r.width, height: r.height };
  } finally { r.close(); }
}

/** Write a run directory (the shape the engines emit). Used by tests and by tooling that re-packs runs. */
export async function writeDepthRun(dir: string, meta: DepthRunMeta, maps: Float32Array): Promise<void> {
  const width = posInt(meta.width, "width"), height = posInt(meta.height, "height"), frames = posInt(meta.frames, "frames");
  if (maps.length !== frames * width * height)
    throw new Error(`writeDepthRun: maps holds ${maps.length} floats, expected ${frames * width * height} (${frames}x${height}x${width})`);
  await mkdir(dir, { recursive: true });
  const bytes = new Uint8Array(maps.length * 4);
  const dv = new DataView(bytes.buffer);
  for (let i = 0, o = 0; i < maps.length; i++, o += 4) dv.setFloat32(o, maps[i]!, true);
  await writeFile(join(dir, "depth.f32"), bytes);
  await writeFile(join(dir, "depth.json"), JSON.stringify({ ...meta, schema: meta.schema ?? SCHEMA }, null, 2) + "\n");
}
