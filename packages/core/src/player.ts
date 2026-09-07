/**
 * AresPlayer — the public runtime (spec §10.9). Fetches one .ares (single request),
 * demuxes it, decodes geometry frames off the hot float path, and drives the
 * renderer from a playback clock decoupled from the render loop (§10.1).
 *
 * Renderer: WebGPU by default; WebGL2 fallback (§10.4) when navigator.gpu is absent
 * or forceGL2 is set. Texture: WebCodecs VP9/AV1 video track (§7.1) or a still PNG
 * atlas. Geometry decodes on the main thread, or on a worker with useWorker (§10.7).
 */
import { Demuxer, type AresFile } from "./demuxer.js";
import { decodeGeometryBlock, decodePFrameBlock, decodeSplatBlock, decodeSplatPBlock, meshoptReady } from "./geometry.js";
import { BlockType, GeometryProfile } from "./format.js";
import { dequantScale, type Aabb } from "./quant.js";
import { WebGPURenderer } from "./renderer.js";
import { WebGL2Renderer, type AresRenderer, type SplatParams } from "./renderer-gl2.js";
import { WorkerGeometryDecoder, type WorkerDecodeResult } from "./worker-decode.js";
import { keepPredicateAt, filterIndicesByPredicate, type EditList } from "./edits.js";
import { rasterizeIds, type IdBuffer } from "./raster.js";
import { orbitViewProj, orbitMatrices, orbitViewHeight, multiply, type OrbitState } from "./camera.js";
import { SplatSorter } from "./splat-sort.js";
import type { DecodedSplat } from "./splat.js";
import { FX_DEFAULTS, packFx, mergeFx, evalFxTrack, type FxParams, type FxTrack } from "./fx.js";
import { invert as invertMat } from "./camera.js";
import { gridLod, type GridLod } from "./overlay.js";
import { resolveOffset, transformAabb, transformMatrix, isIdentityTransform, type ModelTransform } from "./transform.js";
import { TextureVideo, type CodedTextureFrame } from "./texture-video.js";
import { AudioTrack, type AudioPacketRef } from "./audio.js";

export interface AresPlayerOptions {
  canvas: HTMLCanvasElement;
  src: string | Uint8Array;
  loop?: boolean;
  autoOrbit?: boolean;
  /** Decode geometry blocks on a worker thread (spec §10.7). Default false (main-thread decode). */
  useWorker?: boolean;
  /** Worker script URL for useWorker — needed with the single-file bundles (dist/bundle/ares-decode-worker.js). */
  workerUrl?: string | URL;
  /** Play the file's audio track when it has one (default true). Browsers need a user gesture first. */
  audio?: boolean;
  /** Initial volume 0..1 (default 1) and mute. */
  volume?: number;
  muted?: boolean;
  /** Force the WebGL2 fallback renderer even when WebGPU is available (testing, spec §10.4). */
  forceGL2?: boolean;
  onFrame?: (ptsSec: number, frameIndex: number) => void;
  onStats?: (stats: PlayerStats) => void;
  /** Fired once when a "once"-mode clip reaches its final frame and auto-pauses (demo flips Play). */
  onEnded?: () => void;
}

export interface PlayerStats {
  ttffMs: number;
  cpuMsPerFrame: number;   // rolling main-thread decode+upload time per presented frame
  decodeMsPerFrame: number;
  fps: number;
  frameIndex: number;
  frameCount: number;
  requestCount: number;    // network requests to first play (1 for single-file .ares)
  vertexCount: number;
  avgFrameKB: number;
  fileKB: number;
  textureLabel: string;    // "VP9 video 1024²" | "still atlas" | "none"
  audioLabel: string;      // "Opus 48 kHz stereo" | "none"
  geometryMode: string;    // "meshopt intra" | "meshopt I+P (temporal)" | "meshopt mixed (Nx intra + Mx P of F frames)"
}

interface FrameRef { block: Uint8Array; type: BlockType; keyframeIndex: number; gopBox: Aabb; ptsUs: number; }

export class AresPlayer {
  private raf = 0;
  private playing = false;
  private lastNow = 0;
  private clockUs = 0;
  private presented = -1;
  private indexCount = 0;
  private orbit: OrbitState = { azimuth: 0.6, elevation: 0.22, distance: 13, target: [0, 0, 0] };
  private baseDistance = 13;
  private dragging = false;
  private panning = false;
  private lastPtr = [0, 0];
  /** Idle auto-orbit; hosts can switch it off (e.g. when restoring a saved camera for A/B compare). */
  autoOrbit = true;
  private invLevels: number;

  // Ground grid. ON by default: the floor is a spatial REFERENCE, not an editing tool — it is what
  // tells you whether a capture stands on the ground, which way it faces, and how big it is.
  private gridOn = true;
  /** User-chosen cell size in world units; 0 = auto (the decade ladder in overlay.ts gridLod). */
  private gridStep = 0;
  private gridFloorY = 0;
  private gridLast: GridLod = { step0: 1, step1: 10, fineFade: 1 };
  /** Origin tripod arm length, derived from the clip's AABB — see GridParams.axisLen. */
  private gridAxisLen = 1;

  /** Live model transform + its resolved offset (see setModelTransform). null = identity. */
  private modelXf: ModelTransform | null = null;
  private modelOffset: [number, number, number] = [0, 0, 0];
  private modelMat: Float32Array | null = null;

  // Playback effects (fx.ts): live params, a keyframed track from the sidecar, and a host override
  // layer (audio-reactive modulation) — merged in that order at render time.
  private fx: FxParams = { ...FX_DEFAULTS };
  private fxTrack: FxTrack | null = null;
  private fxOverride: Partial<FxParams> | null = null;
  private fxData = new Float32Array(32);

  // Gaussian splat profile (spec §6.8): the file's geometry_profile selects this path at create().
  private readonly splat: boolean;
  private curSplat: DecodedSplat | null = null;
  private sorter = new SplatSorter();

  // Playback trim (clip in/out, inclusive frame indices; -1 out = to the last frame). Playback loops
  // inside it; an explicit seek OUTSIDE still holds, because you have to be able to look at the
  // frames you are about to cut in order to decide where to cut them.
  private trimIn = 0;
  private trimOut = -1;

  // stats
  private ttffMs = 0;
  private fpsEma = 0;
  private cpuEma = 0;
  private decEma = 0;

  private constructor(
    private readonly opts: AresPlayerOptions,
    private readonly file: AresFile,
    private readonly renderer: AresRenderer,
    private readonly frames: FrameRef[],
    private readonly fileBytes: number,
    private readonly textureVideo: TextureVideo | null,
    private readonly textureLabel: string,
    private readonly audio: AudioTrack | null,
  ) {
    this.invLevels = dequantScale(file.superblock.quantBitsPos);
    this.splat = file.header.geometryProfile === GeometryProfile.SplatIPB;
    this.autoOrbit = opts.autoOrbit !== false;
    this.frameToAabb(file.superblock.aabb);
    // Size the origin tripod to the clip. A capture is a person either way, but "a person" is ~1700
    // world units in a mm clip and ~1.7 in a metre one; the old fixed 300 was invisible in one and a
    // set of arms across the whole scene in the other.
    const bb = file.superblock.aabb;
    this.gridAxisLen = 0.12 * Math.max(1e-6, bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]);
    // Both renderers build their grid/tripod resources lazily on setGrid(), so the ON default has to
    // be pushed once here — a field initialised to `true` alone would leave them never compiled.
    // updateGridParams FIRST: setGrid reads gridParams.axisLen to build the tripod's vertices.
    this.updateGridParams();
    this.renderer.setGrid(this.gridOn);
    this.attachPointer();
  }

  /**
   * Orthographic projection on/off. Perspective is right for LOOKING; ortho is right for MEASURING
   * — it removes the wide-angle "fisheye" and makes an axis-aligned plane project to an exact
   * straight line, which is what the crop guides trade on. pickRaster() shares orbitViewProj, so
   * picking follows automatically and can't disagree with the pixels.
   */
  setOrtho(on: boolean): void {
    this.orbit.ortho = on;
    this.renderCurrent();
  }
  isOrtho(): boolean { return !!this.orbit.ortho; }

  /** Snapshot the orbit camera (for carrying a viewpoint across source switches / A-B compares). */
  getCamera(): { azimuth: number; elevation: number; distance: number; target: [number, number, number]; ortho?: boolean } {
    const o = this.orbit;
    // `ortho` rides along: hosts feed this straight back into orbitViewProj (the crop guides do),
    // and dropping it there would silently project through a different camera than the renderer.
    return { azimuth: o.azimuth, elevation: o.elevation, distance: o.distance, target: [o.target[0], o.target[1], o.target[2]], ortho: !!o.ortho };
  }

  /** Restore a camera snapshot. Does not touch autoOrbit — set `player.autoOrbit = false` to hold it. */
  setCamera(c: { azimuth: number; elevation: number; distance: number; target: [number, number, number]; ortho?: boolean }): void {
    this.orbit = { azimuth: c.azimuth, elevation: c.elevation, distance: c.distance, target: [c.target[0], c.target[1], c.target[2]], ortho: c.ortho ?? this.orbit.ortho };
  }

  /** The clip's world-space bounding box (superblock AABB) — the mesh editor's crop range. */
  getAabb(): { min: [number, number, number]; max: [number, number, number] } {
    const a = this.file.superblock.aabb;
    return { min: [a.min[0], a.min[1], a.min[2]], max: [a.max[0], a.max[1], a.max[2]] };
  }

  /**
   * Live model transform — the import-time orientation fix (up-axis / centre / scale / rotate).
   * PREVIEW ONLY: it folds into the mesh's viewProj, it does not touch the decoded geometry. The
   * bake applies the identical transform to the positions themselves (encoder --up-axis/--center/
   * --scale/--rotate/--translate) via the SAME core evaluator, so preview == bake.
   *
   * The grid and origin tripod deliberately do NOT move with it: they are the world reference you
   * are transforming the model INTO. That is the whole point of "stand it on the ground plane".
   */
  setModelTransform(t: ModelTransform | null): void {
    this.modelXf = t && !isIdentityTransform(t) ? t : null;
    if (!this.modelXf) { this.modelOffset = [0, 0, 0]; this.modelMat = null; this.renderer.setModelMatrix(null); }
    else {
      // Offset resolves against the CLIP-WIDE AABB, once — not per frame. Centring each frame on
      // its own bounds would re-centre the subject every frame and a walk would moonwalk in place.
      this.modelOffset = resolveOffset(this.file.superblock.aabb, this.modelXf);
      this.modelMat = transformMatrix(this.modelXf, this.modelOffset);
      this.renderer.setModelMatrix(this.modelMat);
    }
    this.sorter.invalidate();
    this.renderCurrent();
  }
  getModelTransform(): ModelTransform | null { return this.modelXf; }

  /** The clip's AABB as the viewer currently SEES it — i.e. with any model transform applied. This
   *  is what "frame the subject" and any world-space readout must measure against. */
  getTransformedAabb(): Aabb {
    const base = this.file.superblock.aabb;
    if (!this.modelXf) return { min: [base.min[0], base.min[1], base.min[2]], max: [base.max[0], base.max[1], base.max[2]] };
    const r = transformAabb(base, this.modelXf);
    const o = this.modelOffset;
    return { min: [r.min[0] + o[0], r.min[1] + o[1], r.min[2] + o[2]], max: [r.max[0] + o[0], r.max[1] + o[1], r.max[2] + o[2]] };
  }

  /** Frame the subject: point the camera at the (transformed) bounds and pull back to fit. Unity's F. */
  focus(): void {
    const b = this.getTransformedAabb();
    const c: [number, number, number] = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
    const radius = Math.max(1e-6, Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) / 2);
    this.orbit.target = c;
    // orbitViewHeight(d) is the world height on screen at distance d; solve it for "the bounding
    // sphere fits", with a little margin so the subject isn't jammed against the frame edge.
    this.orbit.distance = (radius * 2.3) / (2 * Math.tan((50 * Math.PI) / 180 / 2));
    this.baseDistance = this.orbit.distance;
    this.renderCurrent();
  }

  /** Live crop preview (world space); null disables. Rendering only — bake via the encoder. */
  setCrop(crop: { min: [number, number, number]; max: [number, number, number] } | null): void {
    this.renderer.setCrop(crop);
    this.renderCurrent();
  }

  /** Wireframe render mode (mesh editor). */
  setWireframe(on: boolean): void {
    this.renderer.setWireframe(on);
    this.renderCurrent();
  }

  /** Lit (lambert) vs unlit — unlit shows video-textured captures' baked lighting verbatim. */
  setLit(on: boolean): void {
    this.renderer.setLit(on);
    this.renderCurrent();
  }

  /** Viewport shading: textured (default) vs untextured clay — judge FORM without the atlas. */
  setTextured(on: boolean): void {
    this.renderer.setTextured(on);
    this.renderCurrent();
  }

  /** Analysis views: "shaded" (default), "normals", "uv" (checker), "depth", "points" (point-cloud view). */
  setShadeMode(mode: "shaded" | "normals" | "uv" | "depth" | "points"): void {
    const idx = { shaded: 0, normals: 1, uv: 2, depth: 3, points: 4 }[mode] ?? 0;
    this.renderer.setShadeMode(idx);
    this.renderCurrent();
  }
  /** Point size in pixels for the point-cloud view. */
  setPointSize(px: number): void {
    this.renderer.setPointSize(px);
    this.renderCurrent();
  }

  /** Idle auto-orbit speed in radians per second (default 0.2; a turntable recording sets 2π/duration). */
  orbitSpeed = 0.2;

  /**
   * The presented frame's geometry as floats (mesh profile): positions, UVs, normals and indices,
   * dequantized on the CPU — for in-browser export (OBJ) and measurement. Null for splat clips or
   * before the first frame is decoded.
   */
  exportFrame(): { positions: Float32Array; uvs?: Float32Array; normals?: Float32Array; indices: Uint32Array; frameIndex: number } | null {
    if (this.splat || !this.curPosQ || !this.curIndices || this.presented < 0) return null;
    const box = this.frames[this.presented]!.gopBox;
    const s = this.invLevels;
    const sx = (box.max[0] - box.min[0]) * s, sy = (box.max[1] - box.min[1]) * s, sz = (box.max[2] - box.min[2]) * s;
    const q = this.curPosQ;
    const n = q.length >> 2;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = box.min[0] + q[i * 4]! * sx;
      positions[i * 3 + 1] = box.min[1] + q[i * 4 + 1]! * sy;
      positions[i * 3 + 2] = box.min[2] + q[i * 4 + 2]! * sz;
    }
    let uvs: Float32Array | undefined;
    if (this.curUvsQ && this.curUvsQ.length >= n * 2) { uvs = new Float32Array(n * 2); for (let i = 0; i < n * 2; i++) uvs[i] = this.curUvsQ[i]! / 65535; }
    let normals: Float32Array | undefined;
    const nq = this.curNormalsQ;
    if (nq && nq.length >= n * 4) {
      normals = new Float32Array(n * 3);
      if (this.file.superblock.normalEncoding === 1) {
        const o = new Int16Array(nq.buffer, nq.byteOffset, n * 2);
        for (let i = 0; i < n; i++) {
          let ox = o[i * 2]! / 32767, oy = o[i * 2 + 1]! / 32767;
          let oz = 1 - Math.abs(ox) - Math.abs(oy);
          if (oz < 0) { const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1), ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1); ox = tx; oy = ty; }
          const l = Math.hypot(ox, oy, oz) || 1;
          normals[i * 3] = ox / l; normals[i * 3 + 1] = oy / l; normals[i * 3 + 2] = oz / l;
        }
      } else {
        for (let i = 0; i < n; i++) { normals[i * 3] = nq[i * 4]! / 127; normals[i * 3 + 1] = nq[i * 4 + 1]! / 127; normals[i * 3 + 2] = nq[i * 4 + 2]! / 127; }
      }
    }
    return { positions, uvs, normals, indices: this.curIndices.slice(), frameIndex: this.presented };
  }

  /** Infinite ground grid + origin tripod overlay. Default ON. */
  setGrid(on: boolean): void {
    this.gridOn = on;
    this.renderer.setGrid(on);
    this.renderCurrent();
  }
  isGrid(): boolean { return this.gridOn; }

  /**
   * Grid cell size in WORLD units, or 0 for auto (cells resize with the zoom so the grid stays
   * readable at every scale — see overlay.ts gridLod). Units are the clip's own: ARES millimetres
   * for the Microsoft/daniel captures, metres for 4DViews bakes (the unit-scale trap), so
   * the caller converts from whatever it shows the user.
   */
  setGridStep(worldStep: number): void {
    this.gridStep = Number.isFinite(worldStep) && worldStep > 0 ? worldStep : 0;
    this.renderCurrent();
  }
  getGridStep(): number { return this.gridStep; }
  /** The cell sizes actually drawn last frame — for a live "what am I looking at" readout. */
  getGridLod(): GridLod { return this.gridLast; }
  /** Ground-plane height in world units (the captures' feet sit at ~y = 0). */
  setGridFloor(y: number): void { this.gridFloorY = y; this.renderCurrent(); }

  /** Recompute the grid's LOD from the live camera. Called once per render — that per-frame
   *  recompute IS the feature: it is what lets one shader cover every zoom without popping. */
  private updateGridParams(): void {
    const viewHeight = orbitViewHeight(this.orbit.distance);
    const pxPerUnit = this.opts.canvas.height / Math.max(1e-9, viewHeight);
    const lod = gridLod(pxPerUnit, 12, this.gridStep);
    this.gridLast = lod;
    this.renderer.setGridParams({
      ...lod,
      floorY: this.gridFloorY,
      axisLen: this.gridAxisLen,
      centerX: this.orbit.target[0], centerZ: this.orbit.target[2],
      // Fade completes well inside orbitViewProj's far plane (20 × distance), so the grid dissolves
      // into the distance rather than being sliced off by a clip plane.
      fadeRadius: viewHeight * 8,
    });
  }

  /**
   * Rasterize the CURRENT frame's full topology into a triangle-id + depth buffer from the CURRENT
   * camera (editor picking, design §5.2 CPU flavor). Same viewProj math as the render, so pixel
   * (px,py) in the buffer corresponds to (px/w, py/h) of the canvas. Also returns the world
   * positions accessor pieces needed to resolve a hit id to a surface point.
   */
  pickRaster(w = 320, h = 320): { buf: IdBuffer; triCentroid: (triBase: number) => [number, number, number] } | null {
    if (this.splat || !this.curPosQ || !this.curIndices || this.presented < 0) return null;
    const ref = this.frames[this.presented]!;
    const aspect = this.opts.canvas.width / Math.max(1, this.opts.canvas.height);
    const vp = orbitViewProj(this.orbit, aspect);
    const posQ = this.curPosQ, idx = this.curIndices, box = ref.gopBox;
    const buf = rasterizeIds(posQ, idx, box, this.invLevels, vp, w, h);
    const sx = (box.max[0] - box.min[0]) * this.invLevels;
    const sy = (box.max[1] - box.min[1]) * this.invLevels;
    const sz = (box.max[2] - box.min[2]) * this.invLevels;
    const triCentroid = (triBase: number): [number, number, number] => {
      const a = idx[triBase]!, b = idx[triBase + 1]!, c = idx[triBase + 2]!;
      return [
        box.min[0] + ((posQ[a * 4]! + posQ[b * 4]! + posQ[c * 4]!) / 3) * sx,
        box.min[1] + ((posQ[a * 4 + 1]! + posQ[b * 4 + 1]! + posQ[c * 4 + 1]!) / 3) * sy,
        box.min[2] + ((posQ[a * 4 + 2]! + posQ[b * 4 + 2]! + posQ[c * 4 + 2]!) / 3) * sz,
      ];
    };
    return { buf, triCentroid };
  }

  /** The canvas aspect used for rendering/picking (mask2d volumes serialize it with the camera). */
  getViewAspect(): number {
    return this.opts.canvas.width / Math.max(1, this.opts.canvas.height);
  }

  /**
   * Render the current frame and capture it as a PNG data URL (segmentation input). Copies
   * through a 2d canvas — WebGPU/WebGL back buffers may be cleared after compositing, but
   * drawImage immediately after a render reads the just-submitted frame. maxDim caps the long
   * side; the mask maps back to full-viewport NDC regardless of capture scale.
   */
  captureFrame(maxDim = 1024): { dataUrl: string; width: number; height: number } {
    this.renderCurrent();
    const c = this.opts.canvas;
    const scale = Math.min(1, maxDim / Math.max(1, Math.max(c.width, c.height)));
    const w = Math.max(1, Math.round(c.width * scale));
    const h = Math.max(1, Math.round(c.height * scale));
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const ctx = off.getContext("2d")!;
    ctx.drawImage(c, 0, 0, w, h);
    return { dataUrl: off.toDataURL("image/png"), width: w, height: h };
  }

  /** Auto-frame the orbit camera on the clip's bounding box (any scale/origin). */
  private frameToAabb(aabb: { min: [number, number, number]; max: [number, number, number] }): void {
    const center: [number, number, number] = [
      (aabb.min[0] + aabb.max[0]) / 2, (aabb.min[1] + aabb.max[1]) / 2, (aabb.min[2] + aabb.max[2]) / 2,
    ];
    const diag = Math.hypot(aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]) || 1;
    this.baseDistance = diag * 0.95;   // 50° vertical FOV with headroom
    this.orbit = { azimuth: 0.4, elevation: 0.05, distance: this.baseDistance, target: center };
  }

  static async create(opts: AresPlayerOptions): Promise<AresPlayer> {
    const t0 = performance.now();
    const bytes = typeof opts.src === "string"
      ? new Uint8Array(await (await fetch(opts.src)).arrayBuffer())
      : opts.src;
    const file = Demuxer.parse(bytes);
    await meshoptReady();

    // Flatten every geometry frame to a byte-range ref (no decode yet); collect coded texture frames.
    const usPerFrame = 1e6 / file.header.fps;
    const frames: FrameRef[] = [];
    const vinfo = Demuxer.textureVideo(file);
    const texFrames: (CodedTextureFrame | null)[] = new Array(file.header.frameCount).fill(null);
    const atrack = Demuxer.audioTrack(file);
    const audioPackets: AudioPacketRef[] = [];
    for (const gop of file.gopIndex) {
      const chunk = Demuxer.chunkAt(file, gop);
      if (atrack) for (const p of Demuxer.audioPackets(file, chunk)) audioPackets.push(p);
      const blocks = Demuxer.geometryBlocks(file, chunk);
      let keyframeIndex = gop.frameStart;
      blocks.forEach((b, i) => {
        const gi = gop.frameStart + i;
        if (b.type === BlockType.GeometryI) keyframeIndex = gi; // I-frame is its own keyframe; P-frames inherit
        frames.push({ block: b.data, type: b.type, keyframeIndex, gopBox: chunk.gopAabb, ptsUs: gi * usPerFrame });
      });
      if (vinfo) for (const tf of Demuxer.textureFrames(file, chunk)) texFrames[tf.frameIndex] = { data: tf.data, isKey: tf.isKey, frameIndex: tf.frameIndex };
    }

    // WebGPU when available; WebGL2 fallback when it isn't (spec §10.4). forceGL2 is for testing.
    const renderer: AresRenderer = (opts.forceGL2 || !navigator.gpu)
      ? await WebGL2Renderer.create(opts.canvas)
      : await WebGPURenderer.create(opts.canvas);
    renderer.resize(opts.canvas.width, opts.canvas.height);
    renderer.setNormalEncoding(file.superblock.normalEncoding); // 0 = legacy i8×4, 1 = oct16

    // Texture: video track (spec §7.1) if present, else still atlas (§7.7).
    let textureVideo: TextureVideo | null = null;
    if (vinfo) {
      const tv = new TextureVideo(
        { fourcc: vinfo.fourcc, width: vinfo.width, height: vinfo.height },
        (i) => texFrames[i] ?? null,
        file.superblock.gopLength || 30,
        file.header.frameCount,
      );
      textureVideo = (await tv.configure()) ? tv : (console.warn("[ares] texture video unsupported:", tv.error), null);
    } else {
      const atlas = Demuxer.textureAtlas(file);
      if (atlas) {
        const bitmap = await createImageBitmap(new Blob([atlas.bytes as BlobPart], { type: "image/png" }));
        renderer.setTextureFromBitmap(bitmap);
        bitmap.close();
      }
    }

    const textureLabel = vinfo
      ? `${vinfo.fourcc === "AV01" ? "AV1" : "VP9"} video ${vinfo.width}²`
      : (Demuxer.textureAtlas(file) ? `still atlas ${file.superblock.texture.width}²` : "none");

    // Audio (spec §11.5 OPUS): WebCodecs decode → Web Audio; the audio clock then leads the video.
    let audio: AudioTrack | null = null;
    if (atrack && opts.audio !== false && audioPackets.length) {
      const at = new AudioTrack(atrack, audioPackets);
      if (at.available && await AudioTrack.isSupported(atrack)) {
        audio = at;
        if (opts.volume !== undefined) at.setVolume(opts.volume);
        if (opts.muted) at.setMuted(true);
      } else console.warn("[ares] audio track present but Opus decode is unsupported here — playing silent");
    }
    const player = new AresPlayer(opts, file, renderer, frames, bytes.byteLength, textureVideo, textureLabel, audio);

    // Worker-thread geometry decode (spec §10.7): opt-in; falls back to main-thread decode
    // if the worker can't boot (e.g. module workers without import-map inheritance).
    // Splat frames decode on the main thread in this build (one meshopt call per stream).
    if (opts.useWorker && !player.splat) {
      const dec = new WorkerGeometryDecoder(opts.workerUrl);
      try { await dec.ready; player.workerDec = dec; }
      catch (e) { console.warn("[ares] worker decode unavailable, using main-thread decode:", e); dec.dispose(); }
    }

    // Present frame 0 → TTFF.
    player.present(0);
    player.ttffMs = performance.now() - t0;
    player.renderCurrent();
    player.emitStats();
    opts.onFrame?.(0, 0);
    return player;
  }

  /**
   * Decode + upload the given frame if it isn't already resident. Returns decode+upload ms.
   * `prefetch` opts into worker prefetch of the NEXT frame — only the advancing paths
   * (loop/tick) set it; seek/edit-preview presents keep curPosQ resident for pickRaster.
   */
  private present(idx: number, prefetch = false): number {
    if (idx === this.presented) { this.pumpTexture(idx); return 0; }
    const ref = this.frames[idx];
    if (!ref) return 0;
    const t0 = performance.now();
    if (this.splat) {
      // Splat profile: I-frames are self-contained; P-frames (dynamic splat profile) apply position
      // deltas + births/deaths to the previous frame, re-rolled from the keyframe on any jump.
      const dec = this.decodeSplatFrame(idx);
      const decMsS = performance.now() - t0;
      this.curSplat = dec;
      this.curVerts = dec.count;
      this.renderer.uploadSplats(dec);
      this.presented = idx;
      this.decodedIdx = idx;
      this.sorter.invalidate();
      const totalS = performance.now() - t0;
      this.decEma = this.decEma ? this.decEma * 0.9 + decMsS * 0.1 : decMsS;
      this.cpuEma = this.cpuEma ? this.cpuEma * 0.9 + totalS * 0.1 : totalS;
      return totalS;
    }
    let positionsQ: Uint16Array;
    let decMs: number;
    if (this.workerDone && this.workerDone.idx === idx) {
      const done = this.workerDone;               // worker prefetched this frame (spec §10.7)
      this.workerDone = null;
      positionsQ = this.applyWorkerResult(idx, done.res);
      decMs = done.res.decodeMs;                  // decoded off-thread; report the worker's time
    } else {
      positionsQ = this.decodeGeomFrame(idx);
      decMs = performance.now() - t0;
    }
    this.renderer.writePositions(positionsQ);
    // Editor v2 live preview: filter this frame's triangles by the interpolated edit regions
    // (same centroid predicate as the bake — docs/editor-v2-design.md §6.4/§10).
    if (this.editPreview) this.applyEditPreview(idx, positionsQ);
    else if (this.previewFiltered && this.curIndices) {
      this.renderer.uploadTopology(this.curIndices);     // preview turned off → restore full topology
      this.indexCount = this.curIndices.length;
      this.previewFiltered = false;
    }
    this.presented = idx;
    if (prefetch) this.pumpWorker();              // start decoding the next frame while this renders

    this.pumpTexture(idx);
    const totalMs = performance.now() - t0;
    this.decEma = this.decEma ? this.decEma * 0.9 + decMs * 0.1 : decMs;
    this.cpuEma = this.cpuEma ? this.cpuEma * 0.9 + totalMs * 0.1 : totalMs;
    return totalMs;
  }

  /**
   * Reconstruct a frame's quantized positions, handling I-frames and P-frame deltas
   * (spec §11.6.2). Uploads topology/UVs once per GOP (on the I-frame); a P-frame accumulates
   * from the previous frame, and a non-sequential jump re-rolls from the covering keyframe.
   */
  private curPosQ: Uint16Array | null = null;
  private curUvsQ: Uint16Array | null = null;
  private curNormalsQ: Int8Array | null = null;
  private decodedIdx = -1;
  private uploadedGopKey = -1;
  private curVerts = 0;
  private curIndices: Uint32Array | null = null;
  private editPreview: EditList | null = null;
  private previewFiltered = false;

  /** Live edit-list preview (editor v2): re-filters every presented frame; null disables. */
  setEditPreview(list: EditList | null): void {
    this.editPreview = list;
    const cur = this.presented;
    if (cur >= 0) { this.presented = -1; this.present(cur); this.renderCurrent(); this.emitStats(); }
  }

  /** Filter the current topology by the interpolated regions at this frame (centroid rule). */
  private applyEditPreview(idx: number, posQ: Uint16Array): void {
    if (!this.curIndices || !this.editPreview) return;
    const keep = keepPredicateAt(this.editPreview, idx);
    let filtered = this.curIndices;
    if (keep) {
      const box = this.frames[idx]!.gopBox;
      const s = this.invLevels;
      const sx = (box.max[0] - box.min[0]) * s, sy = (box.max[1] - box.min[1]) * s, sz = (box.max[2] - box.min[2]) * s;
      filtered = filterIndicesByPredicate({
        x: (i) => box.min[0] + posQ[i * 4]! * sx,
        y: (i) => box.min[1] + posQ[i * 4 + 1]! * sy,
        z: (i) => box.min[2] + posQ[i * 4 + 2]! * sz,
      }, this.curIndices, keep);
    }
    this.renderer.uploadTopology(filtered);
    this.indexCount = filtered.length;
    this.previewFiltered = true;
  }

  private decodeGeomFrame(idx: number): Uint16Array {
    const ref = this.frames[idx]!;
    const kf = ref.keyframeIndex;
    // Fast path: the next P-frame in the GOP we're already decoding → apply one delta.
    if (ref.type === BlockType.GeometryPB && this.decodedIdx === idx - 1 && this.curPosQ && this.uploadedGopKey === kf) {
      const p = decodePFrameBlock(ref.block, this.curPosQ);
      this.curPosQ = p.positionsQ;
      if (p.uvsQ) { this.renderer.uploadUVs(p.uvsQ); this.curUvsQ = p.uvsQ; }       // per-frame UVs (re-atlased texture)
      if (p.normalsQ) { this.renderer.uploadNormals(p.normalsQ); this.curNormalsQ = p.normalsQ; }
      this.decodedIdx = idx;
      return this.curPosQ;
    }
    // (Re)start at the keyframe: decode the I-frame (uploads persistent topology + UVs), then
    // roll P-frame deltas forward to idx. For all-intra clips kf === idx, so this is one I-frame.
    const gi = decodeGeometryBlock(this.frames[kf]!.block);
    this.curIndices = gi.indices;                        // kept for edit-preview filtering
    this.renderer.uploadTopology(gi.indices);
    if (gi.uvsQ) this.renderer.uploadUVs(gi.uvsQ);
    if (gi.normalsQ) this.renderer.uploadNormals(gi.normalsQ);
    this.indexCount = gi.indexCount;
    this.curVerts = gi.vertexCount;
    this.uploadedGopKey = kf;
    let posQ = gi.positionsQ;
    let lastNormals: Int8Array | undefined = gi.normalsQ;
    let lastUvs: Uint16Array | undefined = gi.uvsQ;
    for (let f = kf + 1; f <= idx; f++) {
      const p = decodePFrameBlock(this.frames[f]!.block, posQ);
      posQ = p.positionsQ;
      if (p.normalsQ) lastNormals = p.normalsQ;
      if (p.uvsQ) lastUvs = p.uvsQ;
    }
    if (lastUvs) this.renderer.uploadUVs(lastUvs);       // present-frame UVs (may differ from the I-frame's)
    if (lastNormals) this.renderer.uploadNormals(lastNormals);
    this.curUvsQ = lastUvs ?? null;
    this.curNormalsQ = lastNormals ?? null;
    this.curPosQ = posQ;
    this.decodedIdx = idx;
    return posQ;
  }

  // --- worker decode (spec §10.7): keep the NEXT frame decoding off-thread while this one renders ---
  /** Set by create() when opts.useWorker and the worker booted; null → sync decode path. */
  workerDec: WorkerGeometryDecoder | null = null;
  private workerBusy = false;
  private workerDone: { idx: number; res: WorkerDecodeResult } | null = null;

  /** Kick a decode of the next frame on the worker if it's idle (sequential prefetch). */
  private pumpWorker(): void {
    if (!this.workerDec || this.workerBusy || this.splat) return;
    const n = this.frames.length;
    const next = this.loopMode === "loop" ? (this.presented + 1) % n : this.presented + 1;
    if (next >= n || next === this.presented) return;
    if (this.workerDone?.idx === next) return;             // already decoded, waiting to present
    this.workerDone = null;                                // stale prefetch for a frame we skipped past
    const ref = this.frames[next];
    if (!ref) return;
    let job: Promise<WorkerDecodeResult>;
    if (ref.type === BlockType.GeometryI) {
      job = this.workerDec.decodeI(ref.block);
    } else if (this.curPosQ && this.decodedIdx === next - 1 && this.uploadedGopKey === ref.keyframeIndex) {
      // Hand the current positions to the worker (transferred, zero-copy) as the P-frame base;
      // null them locally so the sync path can never touch a detached buffer mid-flight.
      const prev = this.curPosQ;
      this.curPosQ = null; this.decodedIdx = -1;
      job = this.workerDec.decodePB(ref.block, prev);
    } else {
      return; // mid-GOP jump — the sync path re-rolls from the keyframe on demand
    }
    this.workerBusy = true;
    job.then(
      (res) => { this.workerBusy = false; this.workerDone = { idx: next, res }; },
      (e) => { this.workerBusy = false; console.warn("[ares] worker decode failed, sync fallback:", e); },
    );
  }

  /** Upload a worker decode result (GOP topology on I-frames) and adopt it as the decode state. */
  private applyWorkerResult(idx: number, res: WorkerDecodeResult): Uint16Array {
    if (res.indices) {
      this.curIndices = res.indices;                     // kept for edit-preview filtering
      this.renderer.uploadTopology(res.indices);
      this.indexCount = res.indexCount ?? res.indices.length;
      this.curVerts = res.vertexCount ?? this.curVerts;
      this.uploadedGopKey = this.frames[idx]!.keyframeIndex;
    }
    if (res.uvsQ) { this.renderer.uploadUVs(res.uvsQ); this.curUvsQ = res.uvsQ; }
    if (res.normalsQ) { this.renderer.uploadNormals(res.normalsQ); this.curNormalsQ = res.normalsQ; }
    this.curPosQ = res.positionsQ;
    this.decodedIdx = idx;
    return res.positionsQ;
  }

  // --- texture sync: apply decoder output independently of geometry advancement ---
  // VideoDecoder emission is async, so a single-shot present (seek, scrub, pause, edit-preview
  // refresh, compare's per-rAF seek) races it: the exact frame often lands a few ms AFTER the
  // present call. Pumping on every loop/tick/seek — including when the geometry index is
  // unchanged — picks that frame up; without this the texture froze on whatever was last
  // applied until playback advanced (the reported desync on pause/scrub/edit/compare).
  private texAppliedIdx = -1;
  private texSettleFor = -1;
  private texSettleTries = 0;
  private texRetryPending = false;
  /** Poll the texture decoder for the freshest frame ≤ idx and upload it if it's new. */
  private pumpTexture(idx: number): void {
    if (!this.textureVideo || idx < 0) return;
    if (idx !== this.texSettleFor) { this.texSettleFor = idx; this.texSettleTries = 0; }
    // Retries widen the feed look-ahead (capped inside a GOP): some hardware decoders hold
    // frames until more input arrives.
    const vf = this.textureVideo.present(idx, 4 + Math.min(12, this.texSettleTries));
    if (vf) {
      const ts = Number(vf.timestamp);
      if (ts !== this.texAppliedIdx) { this.renderer.setTextureFromVideoFrame(vf); this.texAppliedIdx = ts; this.texApplied++; }
      if (ts === idx) { this.texSettleTries = 0; return; }   // exact frame on screen — settled
    }
    // No rAF loop running (e.g. compare-driven or startPaused players): retry briefly so a
    // paused/scrubbed frame still converges to its exact texture. The internal loop, when
    // active, pumps every frame anyway.
    if (!this.raf && !this.texRetryPending && this.texSettleTries < 30) {
      this.texRetryPending = true;
      this.texSettleTries++;
      requestAnimationFrame(() => {
        this.texRetryPending = false;
        if (this.presented === idx && !this.raf) { this.pumpTexture(idx); this.renderCurrent(); }
      });
    }
  }

  private renderCurrent(): void {
    const ref = this.frames[this.presented] ?? this.frames[0]!;
    const aspect = this.opts.canvas.width / Math.max(1, this.opts.canvas.height);
    if (this.gridOn) this.updateGridParams();
    if (this.splat) {
      const m = orbitMatrices(this.orbit, aspect);
      this.uploadFx(m.eye);
      const dec = this.curSplat;
      if (dec) {
        const mv = this.modelMat ? multiply(m.view, this.modelMat) : m.view;
        const order = this.sorter.update(dec.positionsQ, dec.count, ref.gopBox, this.invLevels, mv, this.presented);
        if (order) this.renderer.setSplatOrder(order);
      }
      this.renderer.renderSplats({ view: m.view, proj: m.proj, ortho: m.ortho, width: this.opts.canvas.width, height: this.opts.canvas.height },
        ref.gopBox, this.invLevels, dec?.count ?? 0);
      return;
    }
    const mm = orbitMatrices(this.orbit, aspect);
    const vp = mm.viewProj;
    this.uploadFx(mm.eye);
    // Depth view window: the orbit distance ± half the clip's diagonal (recomputed per frame so the
    // gradient always spans the subject, whatever the zoom).
    const bb = this.file.superblock.aabb;
    const diag = Math.hypot(bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]) || 1;
    this.renderer.setDepthRange(Math.max(1e-6, this.orbit.distance - diag * 0.5), this.orbit.distance + diag * 0.5);
    this.renderer.render(vp, ref.gopBox, this.invLevels, this.indexCount);
  }

  /** Splat frame reconstruction: sequential P-frames chain off `curSplat`; anything else re-rolls from the keyframe. */
  private decodeSplatFrame(idx: number): DecodedSplat {
    const ref = this.frames[idx]!;
    if (ref.type === BlockType.GeometryI) return decodeSplatBlock(ref.block);
    if (this.curSplat && this.decodedIdx === idx - 1 && this.frames[idx - 1]!.keyframeIndex === ref.keyframeIndex) return decodeSplatPBlock(ref.block, this.curSplat);
    let cur = decodeSplatBlock(this.frames[ref.keyframeIndex]!.block);
    for (let f = ref.keyframeIndex + 1; f <= idx; f++) cur = decodeSplatPBlock(this.frames[f]!.block, cur);
    return cur;
  }

  /** Live playback effects (merged over the current values); see fx.ts for the parameters. */
  setFx(p: Partial<FxParams>): void { this.fx = mergeFx(this.fx, p); this.renderCurrent(); }
  getFx(): FxParams { return { ...this.fx }; }
  resetFx(): void { this.fx = { ...FX_DEFAULTS }; this.renderCurrent(); }
  /** Keyframed effects track (sidecar `edits.fx`); null clears. Evaluated per presented frame on top of the live params. */
  setFxTrack(track: FxTrack | null): void { this.fxTrack = track && track.keyframes?.length ? track : null; this.renderCurrent(); }
  /** A transient layer on top of everything (audio-reactive modulation); null clears. Does not repaint by itself. */
  setFxOverride(p: Partial<FxParams> | null): void { this.fxOverride = p; }
  /** Effective params for the presented frame. */
  currentFx(): FxParams {
    const base = this.fxTrack ? evalFxTrack(this.fxTrack, Math.max(0, this.presented), this.fx) : this.fx;
    return this.fxOverride ? mergeFx(base, this.fxOverride) : base;
  }
  /** Audio loudness 0..1 for reactive effects (0 without an audio track or while paused). */
  getAudioLevel(): number { return this.audio?.level() ?? 0; }
  private uploadFx(eye: [number, number, number]): void {
    // Rim/dissolve run in MODEL space; bring the camera position across the model transform.
    let cam: [number, number, number] = eye;
    if (this.modelMat) {
      const inv = invertMat(this.modelMat);
      if (inv) cam = [inv[0]! * eye[0] + inv[4]! * eye[1] + inv[8]! * eye[2] + inv[12]!, inv[1]! * eye[0] + inv[5]! * eye[1] + inv[9]! * eye[2] + inv[13]!, inv[2]! * eye[0] + inv[6]! * eye[1] + inv[10]! * eye[2] + inv[14]!];
    }
    packFx(this.currentFx(), performance.now() / 1000, cam, this.fxData);
    this.renderer.setFx(this.fxData);
  }

  /** True when the file carries the Gaussian splat profile (spec §6.8) rather than meshes. */
  isSplat(): boolean { return this.splat; }
  /** Splat count / SH degree of the presented frame (null for mesh clips). */
  getSplatInfo(): { count: number; shDegree: number } | null {
    return this.splat && this.curSplat ? { count: this.curSplat.count, shDegree: this.curSplat.shDegree } : null;
  }
  /** Live splat display multipliers (size, opacity) — preview-only VFX hooks. */
  setSplatParams(p: Partial<SplatParams>): void {
    this.renderer.setSplatParams(p);
    this.renderCurrent();
  }

  play(): void {
    if (this.playing) return;
    const { in: lo, out: hi } = this.getTrim();
    const usPerFrame = 1e6 / this.file.header.fps;
    // A "once" clip parked on its out point restarts from the in point instead of no-op'ing.
    if (this.loopMode === "once" && this.presented >= hi) this.clockUs = lo * usPerFrame;
    // Pressing play from inside a trimmed-off head jumps to the in point: the trim says those frames
    // are gone, and the timeline already dims them — playing through them would contradict both.
    if (this.clockUs < lo * usPerFrame) this.clockUs = lo * usPerFrame;
    this.playing = true;
    this.lastNow = performance.now();
    this.audioAnchor();
    if (!this.raf) this.loop(this.lastNow);
  }
  pause(): void { this.playing = false; this.audio?.stop(); }

  // --- audio ↔ clock. The audio context is the master while it runs: clockUs = clock at anchor +
  // (audio media time − media time at anchor). Any non-sequential frame (loop wrap, seek, reverse)
  // re-anchors; reverse playback (ping-pong) is silent.
  private audioClockAt = 0;
  private audioMediaAt = 0;
  private audioAnchor(): void {
    if (!this.audio || !this.playing) return;
    const idx = this.frameIndexForClock();
    const mediaUs = this.frames[idx]?.ptsUs ?? 0;
    this.audioClockAt = this.clockUs;
    this.audioMediaAt = mediaUs;
    this.audio.start(mediaUs);
  }
  private audioTick(idx: number, prevIdx: number): void {
    if (!this.audio) return;
    if (!this.playing) { if (this.audio.isRunning) this.audio.stop(); return; }
    const forwardStep = idx === prevIdx || idx === prevIdx + 1;
    if (!forwardStep) {
      if (idx < prevIdx && this.loopMode === "pingpong" && idx !== this.getTrim().in) { this.audio.stop(); return; }   // reverse leg: silent
      this.audioAnchor();
      return;
    }
    if (!this.audio.isRunning) { this.audioAnchor(); return; }
    const m = this.audio.currentMediaUs();
    if (m !== null) this.clockUs = this.audioClockAt + (m - this.audioMediaAt);
    this.audio.pump(this.audioMediaAt + (this.clockUs - this.audioClockAt));
  }
  /** Volume 0..1 and mute for the audio track (no-ops without one). */
  setVolume(v: number): void { this.audio?.setVolume(v); }
  setMuted(m: boolean): void { this.audio?.setMuted(m); }
  hasAudio(): boolean { return !!this.audio; }
  isMuted(): boolean { return !!this.audio?.isMuted; }
  get isPlaying(): boolean { return this.playing; }

  /** Seek to seconds via the GOP index (spec §9.2). */
  seek(seconds: number): void {
    this.clockUs = Math.max(0, seconds * 1e6);
    const idx = this.frameIndexForClock();
    this.present(idx);
    this.renderCurrent();
    this.emitStats();
    this.opts.onFrame?.((this.frames[idx]?.ptsUs ?? 0) / 1e6, idx);
    if (this.playing) this.audioAnchor(); else this.audio?.stop();
  }

  setTier(_t: "auto" | number): void { /* single tier in P1 (spec §7.6 ladder is P3) */ }

  /** Playback behavior at the clip ends. "loop" = wrap to 0 (default, = the old opts.loop!==false);
   *  "pingpong" = reverse at each end (triangle wave); "once" = clamp on the last frame + auto-pause.
   *  The demo flips this live via the transport's loop-mode button; reverse play is served by the
   *  same seek/present re-roll used for scrubbing, so no decoder changes are needed. */
  loopMode: "loop" | "pingpong" | "once" = "loop";

  /**
   * Clip trim (NLE in/out points, inclusive frame indices). `outFrame` null/-1 = the last frame.
   * Playback loops/ping-pongs/stops inside [in, out] instead of over the whole clip; a bake applies
   * the same window (encoder `--trim-in`/`--trim-out`), so what you play is what you ship.
   *
   * Deliberately NOT enforced on seek: an explicit seek outside the trim still holds that frame.
   * Trimming is a judgement about where a clip should start and end, and you cannot make it without
   * looking at the frames on the far side of the cut.
   */
  setTrim(inFrame: number, outFrame: number | null): void {
    const total = this.frames.length;
    const lo = Math.max(0, Math.min(total - 1, Math.round(inFrame) || 0));
    const hi = outFrame == null || outFrame < 0 ? -1 : Math.max(lo, Math.min(total - 1, Math.round(outFrame)));
    this.trimIn = lo; this.trimOut = hi;
  }
  getTrim(): { in: number; out: number } {
    return { in: this.trimIn, out: this.trimOut < 0 ? Math.max(0, this.frames.length - 1) : this.trimOut };
  }

  private frameIndexForClock(): number {
    const usPerFrame = 1e6 / this.file.header.fps;
    // +1e-6 frames: k/fps seconds is not exactly representable — without the guard, seek(40/30)
    // computes 39.999999999999986 and floors to frame 39 (an off-by-one on ~half of all frames).
    const raw = Math.floor(this.clockUs / usPerFrame + 1e-6);
    const { in: lo, out: hi } = this.getTrim();
    const span = hi - lo + 1;
    // At or before the out point the clock maps straight through — that keeps a seek to ANY frame
    // exact (including inside a trimmed-off head, so you can still inspect what you're cutting).
    // Only running PAST the out point hands control to the loop mode, which wraps within the trim.
    if (raw <= hi) return Math.max(0, raw);
    if (this.loopMode === "pingpong" && span > 1) {
      const period = 2 * (span - 1);            // forward lo→hi, back hi→lo+1, repeat (ends not doubled)
      const p = (((raw - lo) % period) + period) % period;
      return lo + (p < span ? p : period - p);  // triangle wave over [lo, hi]
    }
    if (this.loopMode === "loop") return lo + ((((raw - lo) % span) + span) % span);
    return hi;                                   // once — clamp on the out point
  }

  /**
   * Host-driven advance + render (used by the Three/React wrappers), as an alternative
   * to the internal rAF loop. Advances the clock unconditionally; dtSec in seconds.
   */
  tick(dtSec: number): void {
    this.clockUs += dtSec * 1e6;
    if (this.autoOrbit && !this.dragging) this.orbit.azimuth += dtSec * this.orbitSpeed;
    const idx = this.frameIndexForClock();
    if (idx !== this.presented) {
      this.present(idx, true); this.opts.onFrame?.(this.frames[idx]!.ptsUs / 1e6, idx);
    } else this.pumpTexture(idx);
    this.renderCurrent();
    this.emitStats();
  }

  private loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, (now - this.lastNow) / 1000);
    this.lastNow = now;
    if (this.playing) this.clockUs += dt * 1e6;
    if (this.autoOrbit && !this.dragging) this.orbit.azimuth += dt * this.orbitSpeed;

    // Audio-led clock: when the track is running, its context time replaces the accumulator above.
    if (this.audio && this.playing && this.audio.isRunning) {
      const m = this.audio.currentMediaUs();
      if (m !== null) this.clockUs = this.audioClockAt + (m - this.audioMediaAt);
    }
    const prevIdx = this.presented;
    const idx = this.frameIndexForClock();
    const advanced = idx !== this.presented;
    if (advanced) {
      this.present(idx, true);
      this.opts.onFrame?.(this.frames[idx]!.ptsUs / 1e6, idx);
      if (this.loopMode === "once" && idx >= this.getTrim().out) {
        this.playing = false;               // reached the out point once → auto-pause (fires onEnded once)
        this.audio?.stop();
        this.opts.onEnded?.();
      }
    } else if (idx === this.presented) {
      this.pumpTexture(idx);   // paused/held frame: still collect async decoder output
    }
    this.audioTick(idx, prevIdx);
    this.renderCurrent();

    const fps = dt > 0 ? 1 / dt : 0;
    this.fpsEma = this.fpsEma ? this.fpsEma * 0.9 + fps * 0.1 : fps;
    this.emitStats();
  };

  resize(width: number, height: number): void {
    this.opts.canvas.width = width;
    this.opts.canvas.height = height;
    this.renderer.resize(width, height);
    this.renderCurrent();
  }

  /** Diagnostics for the texture path (frames applied, decoder error). */
  textureDebug(): { hasVideo: boolean; applied: number; error: string | null } {
    return { hasVideo: !!this.textureVideo, applied: this.texApplied, error: this.textureVideo?.error ?? null };
  }
  private texApplied = 0;

  getStats(): PlayerStats {
    const avgBytes = this.frames.reduce((s, f) => s + f.block.byteLength, 0) / Math.max(1, this.frames.length);
    // Per-GOP mode, not a file-wide `.some()` flip (Task J discrepancy fix): a GOP is "temporal"
    // when its topology was stable enough for P-frame deltas (1 I-frame + N P-frames); an "intra"
    // GOP re-encodes every one of its frames as its own I-frame. Every GOP contributes exactly one
    // I-frame when ALL gops are temporal, so iCount === gopIndex.length iff the whole file is
    // temporal; iCount > gopIndex.length means at least one GOP fell back to intra. The old
    // `.some(type === GeometryPB)` check mislabeled files that are almost entirely intra as
    // "temporal" whenever even a single small GOP (e.g. a short tail remainder) happened to pass
    // the encoder's topology-stability check — measured on bridgette-08d.ares: 451 of 455 frames
    // are intra (15 of 16 GOPs), but one 5-frame tail GOP's 4 P-frames flipped the whole label.
    let iCount = 0, pbCount = 0;
    for (const f of this.frames) { if (f.type === BlockType.GeometryPB) pbCount++; else iCount++; }
    const numGops = this.file.gopIndex.length;
    const geometryMode = this.splat
      ? (pbCount === 0 ? `splat intra (SH ${this.file.superblock.shDegree})` : `splat I+P (SH ${this.file.superblock.shDegree})`)
      : pbCount === 0 ? "meshopt intra"
      : iCount === numGops ? "meshopt I+P (temporal)"
      : `meshopt mixed (${iCount} intra + ${pbCount} P of ${this.frames.length} frames)`;
    return {
      ttffMs: this.ttffMs,
      cpuMsPerFrame: this.cpuEma,
      decodeMsPerFrame: this.decEma,
      fps: this.fpsEma,
      frameIndex: this.presented,
      frameCount: this.frames.length,
      requestCount: 1,
      vertexCount: this.curVerts,
      avgFrameKB: avgBytes / 1024,
      fileKB: this.fileBytes / 1024,
      textureLabel: this.textureLabel,
      audioLabel: this.audio ? `Opus 48 kHz ${this.audio ? (Demuxer.audioTrack(this.file)?.channels === 1 ? "mono" : "stereo") : ""}` : "none",
      geometryMode,
    };
  }
  private emitStats(): void { this.opts.onStats?.(this.getStats()); }

  private attachPointer(): void {
    const c = this.opts.canvas;
    // Left-drag (button 0) orbits; middle-drag (button 1, the scroll-wheel button) pans XY; wheel zooms.
    c.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      this.panning = e.button === 1 || e.pointerType === "mouse" && e.buttons === 4;
      if (this.panning) e.preventDefault();        // suppress middle-click autoscroll
      this.lastPtr = [e.clientX, e.clientY];
      // Forwarded events (editor nav pass-through) carry pointerIds the canvas can't capture.
      try { c.setPointerCapture(e.pointerId); } catch { /* synthetic/forwarded pointer */ }
    });
    const end = (e: PointerEvent) => { this.dragging = false; this.panning = false; try { c.releasePointerCapture(e.pointerId); } catch { /* ignore */ } };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPtr[0]!, dy = e.clientY - this.lastPtr[1]!;
      if (this.panning) {
        // Move the orbit target in the camera's screen plane so content follows the cursor.
        const a = this.orbit.azimuth, ce = Math.cos(this.orbit.elevation), se = Math.sin(this.orbit.elevation);
        const rx = Math.cos(a), rz = -Math.sin(a);                          // camera right (world)
        const ux = -se * Math.sin(a), uy = ce, uz = -se * Math.cos(a);      // camera up (world)
        const h = (c as HTMLCanvasElement).clientHeight || 1;
        const k = (2 * this.orbit.distance * Math.tan((50 * Math.PI) / 180 / 2)) / h; // world units / pixel
        this.orbit.target[0] += (-rx * dx + ux * dy) * k;
        this.orbit.target[1] += (uy * dy) * k;
        this.orbit.target[2] += (-rz * dx + uz * dy) * k;
      } else {
        this.orbit.azimuth -= dx * 0.01;
        this.orbit.elevation = Math.max(-1.4, Math.min(1.4, this.orbit.elevation + dy * 0.01));
      }
      this.lastPtr = [e.clientX, e.clientY];
      this.renderIfIdle();
    });
    // Middle button on some browsers still tries autoscroll on mousedown; cancel it there too.
    c.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.orbit.distance = Math.max(this.baseDistance * 0.15, Math.min(this.baseDistance * 5, this.orbit.distance * (1 + Math.sign(e.deltaY) * 0.08))); this.renderIfIdle(); }, { passive: false });
  }

  /**
   * Repaint after a camera change when the internal rAF loop ISN'T running (paused, startPaused,
   * or host-driven players). loop() repaints every tick, so during playback camera moves are drawn
   * for free — but a paused player has no loop, so drag/zoom mutated `orbit` and nothing repainted
   * until play() started, which then "jumped" the model to the accumulated orbit. (The axis gizmo
   * runs its own rAF off the same camera, so it moved while the model sat still — the tell.)
   * Coalesced to one frame so a drag burst costs one draw.
   */
  private idleRaf = 0;
  private renderIfIdle(): void {
    if (this.raf || this.idleRaf) return;
    this.idleRaf = requestAnimationFrame(() => {
      this.idleRaf = 0;
      if (this.raf) return;              // the real loop took over in the meantime
      this.renderCurrent();
    });
  }

  dispose(): void {
    if (this.raf) cancelAnimationFrame(this.raf);
    if (this.idleRaf) cancelAnimationFrame(this.idleRaf);
    this.raf = 0;
    this.idleRaf = 0;
    this.playing = false;
    this.audio?.dispose();
    this.textureVideo?.dispose();
    this.workerDec?.dispose();
    this.renderer.dispose();
  }
}
