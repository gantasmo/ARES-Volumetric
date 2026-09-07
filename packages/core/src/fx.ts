/**
 * Playback effects (VFX on volumes and splats) — non-destructive, evaluated at render time in both
 * renderers from one packed uniform block, authored as a keyframed track in the edit sidecar.
 *
 * Effects (every one defaults to off):
 *   clip      — a world-space plane: geometry / splats on its negative side are discarded
 *               (cross-sections, reveals). `clipNormal` need not be unit; `clipOffset` is the
 *               signed distance from the origin along it.
 *   dissolve  — 3D value-noise threshold: 0 = intact, 1 = gone; `dissolveScale` sets the grain
 *               (cells per world unit) and `dissolveEdge` colours the burning rim.
 *   tint      — colour multiply mixed in by `tintMix`.
 *   rim       — fresnel rim light (meshes: vertex normal; splats: a fixed facing term) of
 *               `rimColor` × `rim`.
 *   scanlines — hologram bands along world Y: strength × frequency (bands per world unit).
 *   wobble    — sinusoidal vertex/centre displacement: amplitude (world units) × frequency.
 *   splatJitter / splatScale / splatOpacity — per-splat position noise (world units), size and
 *               opacity multipliers (splat clips only).
 * `time` advances from the player's clock unless a host pins it.
 *
 * The packed layout (8 × vec4 = 128 bytes, std140-compatible) is shared by WGSL and GLSL:
 *   0: clipNormal.xyz, clipOffset        1: clipOn, dissolve, dissolveScale, scanlines
 *   2: tint.rgb, tintMix                 3: rimColor.rgb, rim
 *   4: dissolveEdge.rgb, wobbleAmp       5: wobbleFreq, time, scanFreq, splatJitter
 *   6: camPos.xyz, splatScale            7: splatOpacity, 0, 0, 0
 */

export interface FxParams {
  clipOn: boolean;
  clipNormal: [number, number, number];
  clipOffset: number;
  dissolve: number;
  dissolveScale: number;
  dissolveEdge: [number, number, number];
  tint: [number, number, number];
  tintMix: number;
  rim: number;
  rimColor: [number, number, number];
  scanlines: number;
  scanFreq: number;
  wobbleAmp: number;
  wobbleFreq: number;
  splatJitter: number;
  splatScale: number;
  splatOpacity: number;
  /** seconds; undefined = follow the player clock */
  time?: number;
}

export const FX_DEFAULTS: FxParams = {
  clipOn: false, clipNormal: [0, 1, 0], clipOffset: 0,
  dissolve: 0, dissolveScale: 4, dissolveEdge: [0.79, 0.6, 0.35],
  tint: [1, 1, 1], tintMix: 0,
  rim: 0, rimColor: [0.79, 0.6, 0.35],
  scanlines: 0, scanFreq: 40,
  wobbleAmp: 0, wobbleFreq: 2,
  splatJitter: 0, splatScale: 1, splatOpacity: 1,
};

export function isFxIdentity(p: FxParams): boolean {
  return !p.clipOn && p.dissolve <= 0 && p.tintMix <= 0 && p.rim <= 0 && p.scanlines <= 0 && p.wobbleAmp <= 0 &&
    p.splatJitter <= 0 && p.splatScale === 1 && p.splatOpacity === 1;
}

/** Fill the 32-float uniform block. `timeSec` is used when `p.time` is undefined. */
export function packFx(p: FxParams, timeSec: number, camPos: [number, number, number], out = new Float32Array(32)): Float32Array {
  out[0] = p.clipNormal[0]; out[1] = p.clipNormal[1]; out[2] = p.clipNormal[2]; out[3] = p.clipOffset;
  out[4] = p.clipOn ? 1 : 0; out[5] = p.dissolve; out[6] = p.dissolveScale; out[7] = p.scanlines;
  out[8] = p.tint[0]; out[9] = p.tint[1]; out[10] = p.tint[2]; out[11] = p.tintMix;
  out[12] = p.rimColor[0]; out[13] = p.rimColor[1]; out[14] = p.rimColor[2]; out[15] = p.rim;
  out[16] = p.dissolveEdge[0]; out[17] = p.dissolveEdge[1]; out[18] = p.dissolveEdge[2]; out[19] = p.wobbleAmp;
  out[20] = p.wobbleFreq; out[21] = p.time ?? timeSec; out[22] = p.scanFreq; out[23] = p.splatJitter;
  out[24] = camPos[0]; out[25] = camPos[1]; out[26] = camPos[2]; out[27] = p.splatScale;
  out[28] = p.splatOpacity; out[29] = 0; out[30] = 0; out[31] = 0;
  return out;
}

/** A keyframed effects track (sidecar `edits.fx`): params interpolate linearly between keyframes and hold outside. */
export interface FxKeyframe { frame: number; params: Partial<FxParams>; }
export interface FxTrack { keyframes: FxKeyframe[]; }

const NUMERIC: (keyof FxParams)[] = ["clipOffset", "dissolve", "dissolveScale", "tintMix", "rim", "scanlines", "scanFreq", "wobbleAmp", "wobbleFreq", "splatJitter", "splatScale", "splatOpacity"];
const COLORS: (keyof FxParams)[] = ["clipNormal", "dissolveEdge", "tint", "rimColor"];

/** Merge a partial over defaults (or a base). */
export function mergeFx(base: FxParams, over: Partial<FxParams> | null | undefined): FxParams {
  if (!over) return { ...base };
  const out: FxParams = { ...base, ...over } as FxParams;
  for (const k of COLORS) { const v = over[k]; if (Array.isArray(v) && v.length === 3) (out as unknown as Record<string, unknown>)[k] = [v[0], v[1], v[2]]; }
  return out;
}

/** Evaluate the track at `frame` on top of `base`. Null / empty track → base. */
export function evalFxTrack(track: FxTrack | null | undefined, frame: number, base: FxParams = FX_DEFAULTS): FxParams {
  const kfs = track?.keyframes;
  if (!kfs || !kfs.length) return { ...base };
  const sorted = kfs.slice().sort((a, b) => a.frame - b.frame);
  if (frame <= sorted[0]!.frame) return mergeFx(base, sorted[0]!.params);
  const last = sorted[sorted.length - 1]!;
  if (frame >= last.frame) return mergeFx(base, last.params);
  let a = sorted[0]!, b = sorted[1]!;
  for (let i = 0; i < sorted.length - 1; i++) if (frame >= sorted[i]!.frame && frame <= sorted[i + 1]!.frame) { a = sorted[i]!; b = sorted[i + 1]!; break; }
  const t = b.frame === a.frame ? 0 : (frame - a.frame) / (b.frame - a.frame);
  const pa = mergeFx(base, a.params), pb = mergeFx(base, b.params);
  const out: FxParams = { ...pa };
  for (const k of NUMERIC) (out as unknown as Record<string, number>)[k] = (pa[k] as number) * (1 - t) + (pb[k] as number) * t;
  for (const k of COLORS) { const va = pa[k] as [number, number, number], vb = pb[k] as [number, number, number]; (out as unknown as Record<string, unknown>)[k] = [va[0] * (1 - t) + vb[0] * t, va[1] * (1 - t) + vb[1] * t, va[2] * (1 - t) + vb[2] * t]; }
  // Booleans hold from the earlier keyframe until the later one is reached.
  out.clipOn = t < 1 ? pa.clipOn : pb.clipOn;
  return out;
}

/** Shared WGSL/GLSL-equivalent helpers live in the renderers; this is the CPU reference for tests. */
export function valueNoise3(x: number, y: number, z: number): number {
  const hash = (i: number, j: number, k: number) => { const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453; return s - Math.floor(s); };
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const sm = (v: number) => v * v * (3 - 2 * v);
  const ux = sm(fx), uy = sm(fy), uz = sm(fz);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  return lerp(
    lerp(lerp(hash(ix, iy, iz), hash(ix + 1, iy, iz), ux), lerp(hash(ix, iy + 1, iz), hash(ix + 1, iy + 1, iz), ux), uy),
    lerp(lerp(hash(ix, iy, iz + 1), hash(ix + 1, iy, iz + 1), ux), lerp(hash(ix, iy + 1, iz + 1), hash(ix + 1, iy + 1, iz + 1), ux), uy), uz);
}
