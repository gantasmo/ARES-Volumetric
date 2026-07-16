/**
 * Texture-video encoder (spec §7.1): the atlas PNG sequence → one closed video GOP per
 * geometry GOP, via ffmpeg (VP9 or AV1), split into per-frame coded chunks the runtime
 * feeds to WebCodecs `VideoDecoder`. This is the "one video, not thousands of images"
 * path that makes hardware texture decode possible.
 *
 * We encode each GOP separately so every GOP starts with a keyframe and is independently
 * decodable (matches spec §11.6 "one closed video GOP" per chunk) and seek stays cheap.
 */
import { execFile } from "node:child_process";
import { readFile, rm, mkdtemp } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { parseIvf } from "./ivf.js";

const run = promisify(execFile);

export type TexCodec = "vp9" | "av1";

export interface TextureFrame { data: Uint8Array; isKey: boolean; }
export interface TextureGop { frames: TextureFrame[]; }
export interface TextureVideo {
  codec: TexCodec;
  fourcc: string;   // container FourCC: VP09 / AV01
  width: number;
  height: number;
  gops: TextureGop[];
}

export interface EncodeTextureOptions {
  /** directory holding the atlas frames */
  dir: string;
  /** printf pattern, e.g. "atlas-f%05d.png" */
  pattern: string;
  /** 1-based file index of the first frame */
  startNumber: number;
  frameCount: number;
  gopLength: number;
  fps: number;
  codec?: TexCodec;   // default vp9
  size?: number;      // square output edge, default 1024
  crf?: number;       // default 32
  /**
   * Absolute (0-based, clip-wide) frame indices where the SOURCE atlas was repacked (Task J fix
   * 2: 4DViews captures repack the UV atlas at irregular topology-reset frames — measured ~73% of
   * atlas pixels change at a repack, 17x a normal inter-frame delta — and our fixed 30-frame GOP
   * texture segmentation was inter-predicting straight across those repacks, the dominant
   * warp/crack artifact). Every one of these frames is forced to a REAL codec keyframe so no
   * repack is ever inter-predicted. Optional/additive: omit or pass an empty set and encoding is
   * unchanged (one keyframe per GOP, at the GOP start, exactly as before).
   */
  repackFrames?: ReadonlySet<number>;
}

/** Exported for texel-copy.ts (frame-copy op, texel half): reuses the same ffmpeg discovery
 *  rather than forking it — that module also shells out to ffmpeg (PNG decode; no PNG decoder
 *  lives in this repo, and ffmpeg is already a hard dependency of texture encoding). */
export function ffmpegPath(): string {
  const fromEnv = process.env.FFMPEG || process.env.FFMPEG_PATH;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  for (const c of ["C:\\FFmpeg\\bin\\ffmpeg.exe", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]) {
    if (existsSync(c)) return c;
  }
  return "ffmpeg"; // rely on PATH
}

export async function ffmpegAvailable(): Promise<boolean> {
  try { await run(ffmpegPath(), ["-hide_banner", "-version"]); return true; } catch { return false; }
}

/**
 * VP9 uncompressed-header keyframe test (spec 6.2, profile 0/1/2 — our encode is always 8-bit
 * 4:2:0, i.e. profile 0, so the 1-bit "reserved_zero" that only exists for profile 3 is handled
 * but never hit in practice). Reads bits MSB-first from byte 0: frame_marker(2) must be 0b10,
 * then profile_low_bit, profile_high_bit, [reserved_zero if profile==3], show_existing_frame,
 * frame_type (0 = KEY_FRAME). Used to VERIFY forced keyframes actually landed in the bitstream
 * (Task J fix 2 validation) rather than trusting the ffmpeg request blindly.
 */
export function vp9FrameIsKey(data: Uint8Array): boolean {
  if (data.length < 1) return false;
  const b0 = data[0]!;
  const bitAt = (idx: number) => (b0 >> (7 - idx)) & 1;
  const frameMarker = (bitAt(0) << 1) | bitAt(1);
  if (frameMarker !== 2) return false; // not a valid VP9 frame marker
  const profile = (bitAt(3) << 1) | bitAt(2);
  let idx = 4;
  if (profile === 3) idx += 1; // reserved_zero
  const showExisting = bitAt(idx); idx += 1;
  if (showExisting) return false; // repeat-frame packet, not a new coded frame
  const frameType = bitAt(idx);
  return frameType === 0;
}

export async function encodeTextureVideo(opts: EncodeTextureOptions): Promise<TextureVideo> {
  const codec = opts.codec ?? "vp9";
  const size = opts.size ?? 1024;
  const crf = opts.crf ?? 32;
  const ff = ffmpegPath();
  const work = await mkdtemp(join(tmpdir(), "ares-tex-"));
  const gops: TextureGop[] = [];
  let width = size, height = size;
  const fourcc = codec === "av1" ? "AV01" : "VP09";
  const repack = opts.repackFrames;

  try {
    for (let start = 0; start < opts.frameCount; start += opts.gopLength) {
      const n = Math.min(opts.gopLength, opts.frameCount - start);
      const ivfPath = join(work, `gop${start}.ivf`);

      // Force a codec keyframe at every repack frame WITHIN this GOP (fix 2). The container's
      // GOP index / texture block is hard 1:1 aligned to geometry GOPs (muxer.ts: one
      // `textureVideo.gops[chunkIdx]` per geometry chunk) — splitting into extra video segments
      // across a GOP boundary isn't representable without a format change, so we keep one ffmpeg
      // invocation per GOP (already a fresh keyframe at local offset 0) and add MID-GOP forced
      // keyframes via `-force_key_frames` instead. `n` in the expr is ffmpeg's own frame counter
      // for THIS invocation (0-based), matching our GOP-local offsets exactly.
      const localForced: number[] = [];
      if (repack) for (let f = start + 1; f < start + n; f++) if (repack.has(f)) localForced.push(f - start);

      const args = [
        "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", String(opts.fps),
        "-start_number", String(opts.startNumber + start),
        "-i", join(opts.dir, opts.pattern),
        "-frames:v", String(n),
        "-vf", `scale=${size}:${size}:flags=lanczos`,
        "-pix_fmt", "yuv420p",
        ...(codec === "av1"
          ? ["-c:v", "libsvtav1", "-crf", String(crf), "-preset", "8", "-g", String(opts.gopLength)]
          : ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", String(crf), "-g", String(opts.gopLength), "-deadline", "good", "-cpu-used", "4"]),
        ...(localForced.length ? ["-force_key_frames", "expr:" + localForced.map((k) => `eq(n,${k})`).join("+")] : []),
        "-f", "ivf", ivfPath,
      ];
      await run(ff, args, { maxBuffer: 1 << 28 });
      const ivf = parseIvf(new Uint8Array(await readFile(ivfPath)));
      width = ivf.width; height = ivf.height;
      const forcedSet = new Set(localForced);
      gops.push({
        frames: ivf.frames.map((f, i) => ({
          data: f.data,
          // Frame 0 of every GOP is always a fresh-encoder-session keyframe. For VP9, verify every
          // OTHER frame directly against the bitstream (authoritative — force_key_frames placement
          // can in principle shift); AV1 OBU parsing isn't implemented, so trust the request there.
          isKey: i === 0 || (codec === "vp9" ? vp9FrameIsKey(f.data) : forcedSet.has(i)),
        })),
      });
      await rm(ivfPath, { force: true });
    }
  } finally {
    await rm(work, { recursive: true, force: true });
  }

  return { codec, fourcc, width, height, gops };
}

/** Detect the printf pattern + start index from a sorted atlas filename list. */
export function detectPattern(files: string[]): { pattern: string; startNumber: number } | null {
  const first = files[0];
  if (!first) return null;
  const m = first.match(/^(.*?)(\d+)(\.[A-Za-z0-9]+)$/);
  if (!m) return null;
  const [, prefix, digits, ext] = m;
  return { pattern: `${prefix}%0${digits!.length}d${ext}`, startNumber: parseInt(digits!, 10) };
}
