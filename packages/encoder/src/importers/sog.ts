/**
 * PlayCanvas SOG importer (the open "Spatially Ordered Gaussians" format SuperSplat publishes;
 * developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog). A SOG is `meta.json` plus
 * lossless WebP property images, either as a directory or bundled in a plain ZIP (`.sog`).
 *
 *   means_l / means_u  RGB   16-bit position per axis: q = (u << 8) | l, n = lerp(mins, maxs, q/65535),
 *                              p = sign(n)·(exp(|n|) − 1)
 *   quats              RGBA  smallest-three: comp = (c/255 − 0.5)·2/√2 for R,G,B, A − 252 = omitted index
 *                              in (w, x, y, z) order, omitted = √(1 − Σ)
 *   scales             RGB   codebook[byte] → log scale → exp
 *   sh0                RGBA  rgb: codebook[byte] = sh0 → colour = 0.5 + C0·sh0; a: opacity/255
 *   shN_centroids      RGB   palette of higher-order coefficients (codebook indices), entries in rows
 *   shN_labels         RG    16-bit palette index: r | g << 8
 *
 * Node has no WebP decoder, so the images go through ffmpeg (already a hard dependency of the
 * texture path) as raw RGBA; the dimensions come from the WebP header (VP8X / VP8L / VP8).
 */
import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { inflateRawSync } from "node:zlib";
import { spawn } from "node:child_process";
import { sh0ToColor, shRestCoeffs } from "@ares/core";
import { emptySplatFrame, type SplatFrame } from "../splat-frame.js";
import { ffmpegPath } from "../texture-video.js";

interface SogMeta {
  version: number;
  count: number;
  antialias?: boolean;
  means: { mins: number[]; maxs: number[]; files: string[] };
  scales: { codebook: number[]; files: string[] };
  quats: { files: string[] };
  sh0: { codebook: number[]; files: string[] };
  shN?: { count: number; bands: number; codebook: number[]; files: string[] };
}

// ---- minimal ZIP reader (stored + deflate; no encryption, no zip64) ------------------------------
export function readZipEntries(buf: Uint8Array): Map<string, Uint8Array> {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("ZIP: end of central directory not found");
  const count = dv.getUint16(eocd + 10, true);
  let cd = dv.getUint32(eocd + 16, true);
  const out = new Map<string, Uint8Array>();
  const td = new TextDecoder();
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(cd, true) !== 0x02014b50) throw new Error("ZIP: bad central directory entry");
    const method = dv.getUint16(cd + 10, true);
    const csize = dv.getUint32(cd + 20, true), usize = dv.getUint32(cd + 24, true);
    const nlen = dv.getUint16(cd + 28, true), elen = dv.getUint16(cd + 30, true), clen = dv.getUint16(cd + 32, true);
    const lho = dv.getUint32(cd + 42, true);
    const name = td.decode(buf.subarray(cd + 46, cd + 46 + nlen));
    if (dv.getUint32(lho, true) !== 0x04034b50) throw new Error(`ZIP: bad local header for ${name}`);
    const lnlen = dv.getUint16(lho + 26, true), lelen = dv.getUint16(lho + 28, true);
    const start = lho + 30 + lnlen + lelen;
    const raw = buf.subarray(start, start + csize);
    let data: Uint8Array;
    if (method === 0) data = raw;
    else if (method === 8) data = new Uint8Array(inflateRawSync(raw));
    else throw new Error(`ZIP: compression method ${method} unsupported for ${name}`);
    if (data.length !== usize) throw new Error(`ZIP: ${name} inflated to ${data.length}, expected ${usize}`);
    out.set(name.replace(/^.*\//, ""), data); // flatten: SOG entries live at the root
    cd += 46 + nlen + elen + clen;
  }
  return out;
}

// ---- WebP dimensions from the container header ---------------------------------------------------
export function webpSize(b: Uint8Array): { width: number; height: number } {
  const tag = (o: number) => String.fromCharCode(b[o]!, b[o + 1]!, b[o + 2]!, b[o + 3]!);
  if (tag(0) !== "RIFF" || tag(8) !== "WEBP") throw new Error("WebP: bad RIFF header");
  const chunk = tag(12);
  if (chunk === "VP8X") {
    const w = 1 + (b[24]! | (b[25]! << 8) | (b[26]! << 16));
    const h = 1 + (b[27]! | (b[28]! << 8) | (b[29]! << 16));
    return { width: w, height: h };
  }
  if (chunk === "VP8L") {
    const bits = b[21]! | (b[22]! << 8) | (b[23]! << 16) | (b[24]! << 24);
    return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
  }
  if (chunk === "VP8 ") {
    return { width: (b[26]! | (b[27]! << 8)) & 0x3fff, height: (b[28]! | (b[29]! << 8)) & 0x3fff };
  }
  throw new Error(`WebP: unknown first chunk ${chunk}`);
}

async function decodeWebpRgba(bytes: Uint8Array): Promise<{ width: number; height: number; rgba: Uint8Array }> {
  const { width, height } = webpSize(bytes);
  const out: Buffer = await new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath(), ["-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"], { stdio: ["pipe", "pipe", "inherit"] });
    const chunks: Buffer[] = [];
    p.stdout.on("data", (c: Buffer) => chunks.push(c));
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve(Buffer.concat(chunks)) : reject(new Error(`ffmpeg webp decode exited ${code}`))));
    p.stdin.on("error", () => { /* ffmpeg may close early on error; the close handler reports it */ });
    p.stdin.end(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  });
  if (out.length !== width * height * 4) throw new Error(`WebP: decoded ${out.length} bytes for ${width}×${height}×4`);
  return { width, height, rgba: new Uint8Array(out.buffer, out.byteOffset, out.byteLength) };
}

/** Load a SOG from a bundled .sog (zip) path or a directory holding meta.json + images. */
export async function parseSog(path: string): Promise<SplatFrame> {
  let files: Map<string, Uint8Array>;
  if ((await stat(path)).isDirectory()) {
    files = new Map();
    for (const name of await readdir(path)) files.set(name, new Uint8Array(await readFile(join(path, name))));
  } else {
    const bytes = new Uint8Array(await readFile(path));
    if (bytes[0] === 0x50 && bytes[1] === 0x4b) files = readZipEntries(bytes);
    else if (path.toLowerCase().endsWith("meta.json")) {
      const dir = path.slice(0, -"meta.json".length) || ".";
      files = new Map();
      for (const name of await readdir(dir)) files.set(name, new Uint8Array(await readFile(join(dir, name))));
    } else throw new Error("SOG: expected a .sog zip, a directory, or a meta.json path");
  }
  return parseSogFiles(files);
}

export async function parseSogFiles(files: Map<string, Uint8Array>): Promise<SplatFrame> {
  const metaBytes = files.get("meta.json");
  if (!metaBytes) throw new Error("SOG: meta.json missing");
  const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as SogMeta;
  if (meta.version !== 2) throw new Error(`SOG: version ${meta.version} unsupported (expected 2)`);
  const n = meta.count;
  const img = async (name: string) => {
    const b = files.get(name);
    if (!b) throw new Error(`SOG: image ${name} missing`);
    return decodeWebpRgba(b);
  };
  const [meansL, meansU, quats, scales, sh0] = await Promise.all([
    img(meta.means.files[0]!), img(meta.means.files[1]!), img(meta.quats.files[0]!), img(meta.scales.files[0]!), img(meta.sh0.files[0]!),
  ]);
  for (const im of [meansL, meansU, quats, scales, sh0]) if (im.width * im.height < n) throw new Error("SOG: image too small for count");

  const degree = meta.shN ? Math.min(3, Math.max(0, meta.shN.bands)) : 0;
  const f = emptySplatFrame(n, degree);
  f.antialiased = !!meta.antialias;
  const unlog = (v: number) => Math.sign(v) * (Math.exp(Math.abs(v)) - 1);
  const mins = meta.means.mins, maxs = meta.means.maxs;
  const toComp = (c: number) => (c / 255 - 0.5) * 2 / Math.SQRT2;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    for (let a = 0; a < 3; a++) {
      const q = (meansU.rgba[o + a]! << 8) | meansL.rgba[o + a]!;
      f.positions[i * 3 + a] = unlog(mins[a]! + (maxs[a]! - mins[a]!) * (q / 65535));
    }
    const a = toComp(quats.rgba[o]!), b = toComp(quats.rgba[o + 1]!), c = toComp(quats.rgba[o + 2]!);
    const mode = quats.rgba[o + 3]! - 252;
    const d = Math.sqrt(Math.max(0, 1 - (a * a + b * b + c * c)));
    let w: number, x: number, y: number, z: number;
    switch (mode) {
      case 0: w = d; x = a; y = b; z = c; break;
      case 1: w = a; x = d; y = b; z = c; break;
      case 2: w = a; x = b; y = d; z = c; break;
      case 3: w = a; x = b; y = c; z = d; break;
      default: throw new Error(`SOG: quaternion mode byte ${quats.rgba[o + 3]} out of range (252–255)`);
    }
    f.rotations[i * 4] = x; f.rotations[i * 4 + 1] = y; f.rotations[i * 4 + 2] = z; f.rotations[i * 4 + 3] = w;
    f.scales[i * 3] = Math.exp(meta.scales.codebook[scales.rgba[o]!]!);
    f.scales[i * 3 + 1] = Math.exp(meta.scales.codebook[scales.rgba[o + 1]!]!);
    f.scales[i * 3 + 2] = Math.exp(meta.scales.codebook[scales.rgba[o + 2]!]!);
    f.colors[i * 3] = sh0ToColor(meta.sh0.codebook[sh0.rgba[o]!]!);
    f.colors[i * 3 + 1] = sh0ToColor(meta.sh0.codebook[sh0.rgba[o + 1]!]!);
    f.colors[i * 3 + 2] = sh0ToColor(meta.sh0.codebook[sh0.rgba[o + 2]!]!);
    f.opacities[i] = sh0.rgba[o + 3]! / 255;
  }
  if (degree > 0 && meta.shN && f.sh) {
    const [cent, labels] = await Promise.all([img(meta.shN.files[0]!), img(meta.shN.files[1]!)]);
    const coeffs = shRestCoeffs(degree);            // 3 / 8 / 15 per channel
    const perRow = Math.max(1, Math.floor(cent.width / coeffs));
    const k3 = coeffs * 3;
    const cb = meta.shN.codebook;
    for (let i = 0; i < n; i++) {
      const label = labels.rgba[i * 4]! | (labels.rgba[i * 4 + 1]! << 8);
      const row = Math.floor(label / perRow), colBase = (label % perRow) * coeffs;
      for (let c = 0; c < coeffs; c++) {
        const px = ((row * cent.width) + colBase + c) * 4;
        f.sh[i * k3 + c * 3] = cb[cent.rgba[px]!]!;
        f.sh[i * k3 + c * 3 + 1] = cb[cent.rgba[px + 1]!]!;
        f.sh[i * k3 + c * 3 + 2] = cb[cent.rgba[px + 2]!]!;
      }
    }
  }
  return f;
}
