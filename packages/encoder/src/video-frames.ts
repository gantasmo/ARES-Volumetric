/**
 * Raw video frames through ffmpeg pipes, one GOP at a time in both directions.
 *
 * `openRawFrames` reads a decode (`-f rawvideo -`) as a sequence of fixed-size frames with
 * backpressure: nothing is pulled from ffmpeg until the caller asks for the next batch, so a
 * feature-length source costs one batch of memory, not the clip.
 *
 * `encodeRawTextureGop` is the other direction: a batch of rgb24 frames in, one closed VP9/AV1 GOP
 * out, split into coded frames. It replaces the PNG round trip the mesh importers use (extract every
 * frame to disk, re-read per GOP), which for a long clip is tens of gigabytes of scratch files that
 * exist only to be decoded again.
 */
import { spawn } from "node:child_process";
import { once } from "node:events";
import { readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { parseIvf } from "./ivf.js";
import { ffmpegPath, vp9FrameIsKey, type TexCodec, type TextureGop } from "./texture-video.js";

export interface RawFrameReader {
  readonly frameBytes: number;
  /** Frames delivered so far. */
  readonly delivered: number;
  /** The next `n` frames, fewer at the end of the stream, none after it. */
  read(n: number): Promise<Uint8Array[]>;
  /** Stop the decoder. Safe after EOF and safe to call twice. */
  close(): void;
}

/** Spawn `ffmpeg <args>` (which must write rawvideo to stdout) and frame its output. */
export function openRawFrames(args: string[], frameBytes: number): RawFrameReader {
  const p = spawn(ffmpegPath(), args, { stdio: ["ignore", "pipe", "pipe"] });
  let stderr = "";
  p.stderr.on("data", (c: Buffer) => { stderr = (stderr + c.toString("utf8")).slice(-2000); });
  let exit: number | null = null, spawnError: Error | null = null;
  const closed = new Promise<void>((resolve) => {
    p.on("error", (e) => { spawnError = e; resolve(); });
    p.on("close", (code) => { exit = code; resolve(); });
  });
  const it = p.stdout[Symbol.asyncIterator]() as AsyncIterator<Buffer>;
  let cur = new Uint8Array(frameBytes), fill = 0, ended = false, killed = false, delivered = 0;
  let carry: Buffer | null = null;

  return {
    frameBytes,
    get delivered() { return delivered; },
    async read(n) {
      const out: Uint8Array[] = [];
      while (out.length < n && !ended) {
        let chunk: Buffer;
        if (carry) { chunk = carry; carry = null; }
        else {
          const r = await it.next();
          if (r.done) { ended = true; break; }
          chunk = r.value;
        }
        let o = 0;
        while (o < chunk.length) {
          const take = Math.min(frameBytes - fill, chunk.length - o);
          cur.set(chunk.subarray(o, o + take), fill);
          fill += take; o += take;
          if (fill === frameBytes) {
            out.push(cur);
            cur = new Uint8Array(frameBytes); fill = 0;
            if (out.length === n) { if (o < chunk.length) carry = chunk.subarray(o); break; }
          }
        }
      }
      if (ended && !killed) {
        await closed;
        if (spawnError) throw new Error(`ffmpeg could not start: ${(spawnError as Error).message}`);
        if (exit !== 0 && delivered + out.length === 0) throw new Error(`ffmpeg decode exited ${exit}: ${stderr.trim().split(/\r?\n/).pop() ?? ""}`);
      }
      delivered += out.length;
      return out;
    },
    close() {
      if (killed) return;
      killed = true; ended = true;
      try { p.stdout.destroy(); } catch { /* already gone */ }
      try { p.kill(); } catch { /* already gone */ }
    },
  };
}

/** A crop in source pixels, in ffmpeg's `crop=w:h:x:y` order. */
export interface CropRect { w: number; h: number; x: number; y: number; }

/** Brightest gray value (0..255) a row or column may reach in every analysed frame and still be a bar. */
export const BAR_BLACK = 24;

/**
 * The bars in a luminance profile. `rowMax[y]` and `colMax[x]` are the brightest value row y and
 * column x reached over every analysed frame; a bar is a run of rows or columns at an edge that
 * never rose above `black`. Returns the picture between the bars, in profile pixels (x1 and y1
 * exclusive), or null when there is no bar, or when under a quarter of either axis is picture (a
 * black clip is not a letterbox).
 */
export function barsFromProfile(rowMax: Uint8Array, colMax: Uint8Array, black = BAR_BLACK): { x0: number; y0: number; x1: number; y1: number } | null {
  const lead = (a: Uint8Array) => { let i = 0; while (i < a.length && a[i]! <= black) i++; return i; };
  const trail = (a: Uint8Array) => { let i = a.length; while (i > 0 && a[i - 1]! <= black) i--; return i; };
  const y0 = lead(rowMax), y1 = trail(rowMax), x0 = lead(colMax), x1 = trail(colMax);
  if (y1 - y0 < rowMax.length / 4 || x1 - x0 < colMax.length / 4) return null;
  if (y0 === 0 && x0 === 0 && y1 === rowMax.length && x1 === colMax.length) return null;
  return { x0, y0, x1, y1 };
}

/**
 * Letterbox and pillarbox detection: the rows and columns of the source that stay black in every
 * keyframe. Keyframes are decoded alone (`-skip_frame nokey`, under a second for a four-minute
 * 1080p film) at half resolution in gray; a source with fewer than 8 keyframes is sampled at 2 fps
 * over its first minute instead. Returns the picture as a source-pixel crop, or null.
 */
export async function detectLetterbox(video: string, srcW: number, srcH: number): Promise<{ crop: CropRect; frames: number } | null> {
  const w = Math.max(2, Math.round(srcW / 2)), h = Math.max(2, Math.round(srcH / 2)), P = w * h;
  const profile = async (pre: string[], vf: string, cap: string[]) => {
    const r = openRawFrames(["-hide_banner", "-loglevel", "fatal", ...pre, "-i", video, "-vf", `${vf}scale=${w}:${h}:flags=area`,
      "-fps_mode", "passthrough", ...cap, "-f", "rawvideo", "-pix_fmt", "gray", "-"], P);
    const rowMax = new Uint8Array(h), colMax = new Uint8Array(w);
    let n = 0;
    try {
      for (;;) {
        const batch = await r.read(8);
        if (!batch.length) break;
        for (const f of batch) {
          for (let y = 0, i = 0; y < h; y++) {
            let m = rowMax[y]!;
            for (let x = 0; x < w; x++, i++) { const v = f[i]!; if (v > m) m = v; if (v > colMax[x]!) colMax[x] = v; }
            rowMax[y] = m;
          }
          n++;
        }
      }
    } finally { r.close(); }
    return { rowMax, colMax, n };
  };
  let p = await profile(["-skip_frame", "nokey"], "", []);
  if (p.n < 8) p = await profile([], "fps=2,", ["-frames:v", "120"]);
  const b = p.n ? barsFromProfile(p.rowMax, p.colMax) : null;
  if (!b) return null;
  // Inward to whole source pixels: a picture edge that falls inside a profile pixel is dropped.
  const x0 = Math.ceil((b.x0 * srcW) / w), x1 = Math.floor((b.x1 * srcW) / w);
  const y0 = Math.ceil((b.y0 * srcH) / h), y1 = Math.floor((b.y1 * srcH) / h);
  return { crop: { w: x1 - x0, h: y1 - y0, x: x0, y: y0 }, frames: p.n };
}

export interface RawGopOptions {
  frames: Uint8Array[];   // rgb24, width * height * 3 each
  width: number;
  height: number;
  fps: number;
  gopLength: number;
  codec: TexCodec;
  crf: number;
  /** Scratch directory for the IVF. */
  workDir: string;
  /** Distinguishes concurrent encodes in `workDir`. */
  tag: string | number;
}

/** One batch of rgb24 frames -> one closed GOP of coded frames. Same codec settings as encodeTextureVideo. */
export async function encodeRawTextureGop(o: RawGopOptions): Promise<TextureGop> {
  const ivfPath = join(o.workDir, `gop-${o.tag}.ivf`);
  const args = [
    "-y", "-hide_banner", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${o.width}x${o.height}`, "-framerate", String(o.fps), "-i", "-",
    "-frames:v", String(o.frames.length),
    "-pix_fmt", "yuv420p",
    ...(o.codec === "av1"
      ? ["-c:v", "libsvtav1", "-crf", String(o.crf), "-preset", "8", "-g", String(o.gopLength)]
      : ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", String(o.crf), "-g", String(o.gopLength), "-deadline", "good", "-cpu-used", "4"]),
    "-f", "ivf", ivfPath,
  ];
  const p = spawn(ffmpegPath(), args, { stdio: ["pipe", "ignore", "pipe"] });
  let stderr = "";
  p.stderr.on("data", (c: Buffer) => { stderr = (stderr + c.toString("utf8")).slice(-2000); });
  const done = new Promise<number | null>((resolve, reject) => { p.on("error", reject); p.on("close", resolve); });
  // A dead encoder surfaces as EPIPE on its stdin; the exit code below is the real report.
  p.stdin.on("error", () => { /* reported through `done` */ });
  for (const f of o.frames) {
    if (!p.stdin.writable) break;
    try { if (!p.stdin.write(f)) await once(p.stdin, "drain"); } catch { break; }
  }
  p.stdin.end();
  const code = await done;
  if (code !== 0) throw new Error(`ffmpeg texture encode exited ${code}: ${stderr.trim().split(/\r?\n/).pop() ?? ""}`);
  const ivf = parseIvf(new Uint8Array(await readFile(ivfPath)));
  await rm(ivfPath, { force: true });
  if (ivf.frames.length !== o.frames.length) throw new Error(`texture encode produced ${ivf.frames.length} coded frames for ${o.frames.length} inputs`);
  return { frames: ivf.frames.map((f, i) => ({ data: f.data, isKey: i === 0 || (o.codec === "vp9" ? vp9FrameIsKey(f.data) : false) })) };
}
