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
 *
 * VOLUMETRIC KEYS (2026-09-19, all optional and additive under the same schema; written by the
 * service's geometry and body phases, tools/sam-service/depth_volume.py). Every per-frame file has
 * depth.f32's frame order, map size and row-0-top orientation, and is preallocated at full size by
 * the engine, so a key is present only once its files are complete:
 *   intrinsics  normalized pinhole K (fx = focal_px / W, fy = focal_px / H, cx, cy as fractions),
 *               the clip median of MoGe-2's per-frame estimates; `file` holds those estimates,
 *               frames x 4 float32 LE (fx, fy, cx, cy), NaN = no estimate for that frame
 *   metric      frames x H x W float32 LE, metric depth z in metres, 0 = invalid
 *   normals     frames x H x W x 3 int8, round(n * 127), unit normals in OpenCV camera space
 *               (camera-facing normals have n.z < 0), 0,0,0 = invalid
 *   body        frames x V x 3 float32 LE (SAM 3D Body vertices in OpenCV camera metres, computed
 *               with the map intrinsics), `faces` F x 3 uint32 LE written once, `valid` one byte per
 *               frame (1 = fitted on this frame, 0 = copied from the nearest valid frame)
 * openDepthRun checks every named file's size against the frame count, the way it checks the mask.
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
   *  0 outside the subject and 255 inside, in the same frame order and orientation as depth.f32.
   *  `filled` frames repeat a neighbouring frame's mask; `<dir>/<detected>`, when named, is one byte
   *  per frame: 1 where the tracker found the subject, 0 where the mask was copied. `<dir>/<ids>`,
   *  when named, is frames x H x W uint8: the tracker's object id on each mask pixel, 0 elsewhere;
   *  `objects` lists each id's detected frames, first and last frame and mean coverage. */
  mask?: {
    file: string; prompt?: string; engine?: string; coverage?: number; filled?: number; detected?: string;
    ids?: string; objects?: { id: number; frames: number; first: number; last: number; coverage: number }[];
  };
  /** Normalized clip intrinsics (see the header). `file` holds the per-frame estimates. */
  intrinsics?: DepthRunIntrinsics;
  /** Metric depth from the unmasked frame: `<dir>/<file>` is frames x H x W float32 LE, metres, 0 = invalid. */
  metric?: { file: string; model?: string; units?: string; dtype?: string; msPerFrame?: number | null; validFraction?: number; medianInMask?: number | null; workers?: DepthRunWorker[] };
  /** Camera-space normals: `<dir>/<file>` is frames x H x W x 3 int8, round(n * 127). */
  normals?: { file: string; space?: string; meanZInMask?: number | null };
  /** A body mesh per frame (see the header). */
  body?: DepthRunBody;
}

export interface DepthRunIntrinsics {
  fx: number; fy: number; cx: number; cy: number;
  source?: string; fovY?: number; fovX?: number;
  /** frames x 4 float32 LE per-frame estimates, NaN where a frame has none. */
  file?: string;
  /** Frames the median was taken over. */
  frames?: number;
}

export interface DepthRunBody {
  file: string;
  faces: string;
  valid: string;
  vertices: number;
  faceCount: number;
  units?: string;
  space?: string;
  engine?: string;
  model?: string;
  fp16?: boolean;
  boxes?: string;
  inference?: string;
  validFrames?: number;
  backfilled?: number;
  msPerFrame?: number | null;
  conventionFrame?: number | null;
  conventionIoU?: number | null;
  iouMean?: number | null;
  iouMin?: number | null;
  workers?: DepthRunWorker[];
}

/** One volume worker process of a geometry or body phase (tools/sam-service/depth_volume.py): the
 *  GPU, the frame range it was given, what it finished, and a reason when it was lost and its
 *  remaining frames went to another GPU. */
export interface DepthRunWorker {
  gpu: number;
  start: number;
  end: number;
  frames: number;
  retry?: boolean;
  lost?: string | null;
  msPerFrame?: number | null;
  loadMs?: number | null;
  wallMs?: number | null;
  peakMiB?: number | null;
  reservedMiB?: number | null;
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
  /** True when `meta.metric` names a file holding every frame. */
  hasMetric: boolean;
  /** True when `meta.normals` names a file holding every frame. */
  hasNormals: boolean;
  /** True when `meta.body` names body, faces and valid files holding every frame. */
  hasBody: boolean;
  /** The clip intrinsics in pixels of the run's maps (fx*W, fy*H, cx*W, cy*H), or null. */
  mapIntrinsics: { fx: number; fy: number; cx: number; cy: number } | null;
  /** Read `count` metric depth frames (metres, 0 = invalid) starting at `t`. */
  readMetric(t: number, count: number, out: Float32Array): void;
  /** Read `count` normal frames (W x H x 3 int8) starting at `t`. */
  readNormals(t: number, count: number, out: Int8Array): void;
  /** Read `count` body frames (V x 3 float32, OpenCV camera metres) starting at `t`. */
  readBody(t: number, count: number, out: Float32Array): void;
  /** The body's F x 3 face indices, validated against V. */
  bodyFaces(): Uint32Array;
  /** One byte per frame: 1 = the body was fitted on that frame, 0 = copied from a neighbour. */
  bodyValid(): Uint8Array;
  /** frames x 4 normalized (fx, fy, cx, cy) per frame, NaN where a frame has none; null without a file. */
  perFrameIntrinsics(): Float32Array | null;
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

/** Reverse every 4-byte group in place (big-endian hosts only). */
function swap4(bytes: Uint8Array): void {
  for (let o = 0; o + 3 < bytes.length; o += 4) {
    const a = bytes[o]!, b = bytes[o + 1]!;
    bytes[o] = bytes[o + 3]!; bytes[o + 1] = bytes[o + 2]!; bytes[o + 2] = b; bytes[o + 3] = a;
  }
}

/** A finite number, > 0 when `positive`. */
function finiteNum(v: unknown, what: string, positive: boolean): number {
  const n = Number(v);
  if (!Number.isFinite(n) || (positive && !(n > 0))) throw new Error(`depth.json: ${what} must be a finite number${positive ? " > 0" : ""}, got ${JSON.stringify(v)}`);
  return n;
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
  const extraFds: number[] = [];
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

    // ---- volumetric keys: each named file must exist and hold every frame ----------------------
    /** Open `<dir>/<file>` named by `key`, checking it holds at least `need` bytes. */
    const openSized = (key: string, file: unknown, need: number, unit: string): { fd: number; path: string; size: number } => {
      if (typeof file !== "string" || !file) throw new Error(`${metaPath}: ${key}.file must be a file name, got ${JSON.stringify(file)}`);
      const path = join(dir, file);
      if (!existsSync(path)) throw new Error(`${metaPath}: names a ${key} file ${JSON.stringify(file)} that is not in the run directory`);
      const f = openSync(path, "r");
      extraFds.push(f);
      const size = fstatSync(f).size;
      if (size < need) throw new Error(`${path}: holds ${size} bytes, ${need} needed for ${unit}`);
      return { fd: f, path, size };
    };
    let mapIntrinsics: DepthRunReader["mapIntrinsics"] = null;
    let intrFile: { fd: number; path: string } | null = null;
    if (meta.intrinsics) {
      const k = meta.intrinsics;
      const fx = finiteNum(k.fx, "intrinsics.fx", true), fy = finiteNum(k.fy, "intrinsics.fy", true);
      const cx = finiteNum(k.cx, "intrinsics.cx", false), cy = finiteNum(k.cy, "intrinsics.cy", false);
      mapIntrinsics = { fx: fx * width, fy: fy * height, cx: cx * width, cy: cy * height };
      if (k.file) intrFile = openSized("intrinsics", k.file, frames * 16, `${frames} frame(s) x 4 float32`);
    }
    const metricF = meta.metric ? openSized("metric", meta.metric.file, frames * P * 4, `${frames} ${width}x${height} float32 frame(s)`) : null;
    const normalsF = meta.normals ? openSized("normals", meta.normals.file, frames * P * 3, `${frames} ${width}x${height}x3 int8 frame(s)`) : null;
    let bodyF: { fd: number; path: string } | null = null, facesF: { fd: number; path: string; size: number } | null = null, validF: { fd: number; path: string } | null = null;
    let V = 0, F = 0;
    if (meta.body) {
      V = posInt(meta.body.vertices, "body.vertices");
      F = posInt(meta.body.faceCount, "body.faceCount");
      bodyF = openSized("body", meta.body.file, frames * V * 12, `${frames} frame(s) x ${V} vertices x 3 float32`);
      facesF = openSized("body.faces", meta.body.faces, F * 12, `${F} faces x 3 uint32`);
      validF = openSized("body.valid", meta.body.valid, frames, `${frames} frame flag(s)`);
    }

    const swap = LITTLE_ENDIAN ? null : new Uint8Array(4);
    const theFd = fd, theMaskFd = maskFd;
    const range = (what: string, t: number, count: number): void => {
      if (!Number.isInteger(t) || !Number.isInteger(count) || t < 0 || count < 0 || t + count > frames)
        throw new Error(`depth run: ${what} frames ${t}..${t + count} are outside 0..${frames}`);
    };
    /** Positional float32 read of `count` records of `per` floats starting at record `t`. */
    const readF32 = (src: { fd: number; path: string }, what: string, per: number, t: number, count: number, out: Float32Array): void => {
      range(what, t, count);
      if (out.length < count * per) throw new Error(`depth run: ${what} output holds ${out.length} floats, ${count * per} needed`);
      const bytes = new Uint8Array(out.buffer, out.byteOffset, count * per * 4);
      readFully(src.fd, bytes, bytes.length, t * per * 4, src.path);
      if (!LITTLE_ENDIAN) swap4(bytes);
    };
    let facesCache: Uint32Array | null = null;
    return {
      meta: { ...meta, frames, width, height }, frames, width, height, hasMask: maskFd >= 0,
      hasMetric: !!metricF, hasNormals: !!normalsF, hasBody: !!bodyF, mapIntrinsics,
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
      readMetric(t, count, out) {
        if (!metricF) throw new Error("depth run: this run carries no metric depth");
        readF32(metricF, "metric", P, t, count, out);
      },
      readNormals(t, count, out) {
        if (!normalsF) throw new Error("depth run: this run carries no normals");
        range("normals", t, count);
        if (out.length < count * P * 3) throw new Error(`depth run: normals output holds ${out.length} values, ${count * P * 3} needed`);
        readFully(normalsF.fd, new Uint8Array(out.buffer, out.byteOffset, count * P * 3), count * P * 3, t * P * 3, normalsF.path);
      },
      readBody(t, count, out) {
        if (!bodyF) throw new Error("depth run: this run carries no body mesh");
        readF32(bodyF, "body", V * 3, t, count, out);
      },
      bodyFaces() {
        if (!facesF) throw new Error("depth run: this run carries no body mesh");
        if (!facesCache) {
          const f = new Uint32Array(F * 3);
          const bytes = new Uint8Array(f.buffer);
          readFully(facesF.fd, bytes, bytes.length, 0, facesF.path);
          if (!LITTLE_ENDIAN) swap4(bytes);
          for (let i = 0; i < f.length; i++) if (f[i]! >= V) throw new Error(`${facesF.path}: face index ${f[i]} at ${i} is out of range for ${V} vertices`);
          facesCache = f;
        }
        return facesCache;
      },
      bodyValid() {
        if (!validF) throw new Error("depth run: this run carries no body mesh");
        const v = new Uint8Array(frames);
        readFully(validF.fd, v, frames, 0, validF.path);
        return v;
      },
      perFrameIntrinsics() {
        if (!intrFile) return null;
        const k = new Float32Array(frames * 4);
        readF32(intrFile, "intrinsics", 4, 0, frames, k);
        return k;
      },
      close() {
        try { closeSync(theFd); } catch { /* already closed */ }
        if (theMaskFd >= 0) try { closeSync(theMaskFd); } catch { /* already closed */ }
        for (const f of extraFds) try { closeSync(f); } catch { /* already closed */ }
      },
    };
  } catch (e) {
    try { closeSync(fd); } catch { /* ignore */ }
    if (maskFd >= 0) try { closeSync(maskFd); } catch { /* ignore */ }
    for (const f of extraFds) try { closeSync(f); } catch { /* ignore */ }
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

/** The optional per-run arrays writeDepthRun can add beside depth.f32 (the volumetric keys and the
 *  mask). Each is written to the file its meta key names; a missing key is created with the default
 *  name (mask.u8, metric.f32, normals.i8, intrinsics.f32, body.f32, body-faces.u32, body-valid.u8). */
export interface DepthRunExtras {
  /** frames x H x W, 0 / 255. */
  mask?: Uint8Array;
  /** frames x H x W metres, 0 = invalid. */
  metric?: Float32Array;
  /** frames x H x W x 3, round(n * 127). */
  normals?: Int8Array;
  /** frames x 4 normalized (fx, fy, cx, cy); needs meta.intrinsics for the clip values. */
  intrinsics?: Float32Array;
  /** frames x V x 3, OpenCV camera metres; needs bodyFaces and bodyValid. */
  body?: Float32Array;
  /** F x 3. */
  bodyFaces?: Uint32Array;
  /** frames bytes, 1 = fitted. */
  bodyValid?: Uint8Array;
}

/** A float32 or uint32 array as little-endian bytes. */
function leBytes(a: Float32Array | Uint32Array): Uint8Array {
  const bytes = new Uint8Array(a.length * 4);
  const dv = new DataView(bytes.buffer);
  if (a instanceof Float32Array) for (let i = 0, o = 0; i < a.length; i++, o += 4) dv.setFloat32(o, a[i]!, true);
  else for (let i = 0, o = 0; i < a.length; i++, o += 4) dv.setUint32(o, a[i]!, true);
  return bytes;
}

/** Write a run directory (the shape the engines emit). Used by tests and by tooling that re-packs runs. */
export async function writeDepthRun(dir: string, meta: DepthRunMeta, maps: Float32Array, extra: DepthRunExtras = {}): Promise<void> {
  const width = posInt(meta.width, "width"), height = posInt(meta.height, "height"), frames = posInt(meta.frames, "frames");
  const P = width * height;
  if (maps.length !== frames * P)
    throw new Error(`writeDepthRun: maps holds ${maps.length} floats, expected ${frames * P} (${frames}x${height}x${width})`);
  const need = (name: string, a: { length: number } | undefined, n: number): void => {
    if (a && a.length !== n) throw new Error(`writeDepthRun: ${name} holds ${a.length} values, expected ${n}`);
  };
  need("mask", extra.mask, frames * P);
  need("metric", extra.metric, frames * P);
  need("normals", extra.normals, frames * P * 3);
  need("intrinsics", extra.intrinsics, frames * 4);
  need("bodyValid", extra.bodyValid, frames);
  const m: DepthRunMeta = { ...meta };
  if (extra.intrinsics && !m.intrinsics) throw new Error("writeDepthRun: extra.intrinsics needs meta.intrinsics (the clip values)");
  if (!!extra.body !== !!extra.bodyFaces || !!extra.body !== !!extra.bodyValid) throw new Error("writeDepthRun: body, bodyFaces and bodyValid go together");
  if (extra.body && extra.bodyFaces) {
    if (extra.body.length % (frames * 3) !== 0) throw new Error(`writeDepthRun: body holds ${extra.body.length} floats, not frames (${frames}) x V x 3`);
    if (extra.bodyFaces.length % 3 !== 0) throw new Error(`writeDepthRun: bodyFaces holds ${extra.bodyFaces.length} indices, not a multiple of 3`);
    const V = extra.body.length / (frames * 3), F = extra.bodyFaces.length / 3;
    m.body = { file: "body.f32", faces: "body-faces.u32", valid: "body-valid.u8", units: "m", space: "opencv-camera", ...(m.body ?? {}), vertices: V, faceCount: F };
  }
  if (extra.mask) m.mask = { file: "mask.u8", ...(m.mask ?? {}) };
  if (extra.metric) m.metric = { file: "metric.f32", units: "m", ...(m.metric ?? {}) };
  if (extra.normals) m.normals = { file: "normals.i8", space: "opencv-camera", ...(m.normals ?? {}) };
  if (extra.intrinsics) m.intrinsics = { ...m.intrinsics!, file: m.intrinsics!.file ?? "intrinsics.f32" };
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "depth.f32"), leBytes(maps));
  if (extra.mask) await writeFile(join(dir, m.mask!.file), extra.mask);
  if (extra.metric) await writeFile(join(dir, m.metric!.file), leBytes(extra.metric));
  if (extra.normals) await writeFile(join(dir, m.normals!.file), new Uint8Array(extra.normals.buffer, extra.normals.byteOffset, extra.normals.length));
  if (extra.intrinsics) await writeFile(join(dir, m.intrinsics!.file!), leBytes(extra.intrinsics));
  if (extra.body && extra.bodyFaces && extra.bodyValid) {
    await writeFile(join(dir, m.body!.file), leBytes(extra.body));
    await writeFile(join(dir, m.body!.faces), leBytes(extra.bodyFaces));
    await writeFile(join(dir, m.body!.valid), extra.bodyValid);
  }
  await writeFile(join(dir, "depth.json"), JSON.stringify({ ...m, schema: m.schema ?? SCHEMA }, null, 2) + "\n");
}
