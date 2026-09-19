/**
 * Image-guided depth resampling and the occlusion fill layer for `ares depth`.
 *
 * GUIDED RESAMPLING. A monocular depth map is soft exactly where a relief needs it sharp: the
 * model's silhouette is a ramp several pixels wide, and resampling that ramp onto the vertex grid
 * leaves vertices hanging between the subject and the wall behind it. The video frame knows where
 * the edge really is. Joint bilateral resampling (Kopf et al. 2007) weights each depth sample by
 * how closely its colour matches the colour at the vertex, so a vertex on the subject's side of an
 * image edge takes its depth from subject samples only. It applies whichever way the grid and the
 * map compare in size: growing (grid wider than the map) it is an edge-aware upsampler, shrinking
 * it is an edge-aware area average.
 *
 * FILL LAYER. A relief cut at its silhouettes has nothing behind them, so orbiting off the capture
 * axis opens a hole the shape of the subject. The fill layer is the second layer of a layered
 * depth image: behind each silhouette, the background surface is continued under the foreground
 * (depth and colour both extended inward from the far side of the cut), meshed, and textured from
 * a second region of the atlas. It is an extrapolation, soft by construction, and it is only ever
 * seen through a gap.
 */

/** Area-average an rgb24 image down (or point-sample up) to `dw x dh`. */
export function resizeRgbArea(src: Uint8Array, sw: number, sh: number, dw: number, dh: number, out?: Uint8Array): Uint8Array {
  const dst = out ?? new Uint8Array(dw * dh * 3);
  const rx = sw / dw, ry = sh / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = Math.min(sh - 1, Math.floor(y * ry)), y1 = Math.max(y0 + 1, Math.min(sh, Math.round((y + 1) * ry)));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.min(sw - 1, Math.floor(x * rx)), x1 = Math.max(x0 + 1, Math.min(sw, Math.round((x + 1) * rx)));
      let r = 0, g = 0, b = 0;
      for (let yy = y0; yy < y1; yy++) {
        let j = (yy * sw + x0) * 3;
        for (let xx = x0; xx < x1; xx++, j += 3) { r += src[j]!; g += src[j + 1]!; b += src[j + 2]!; }
      }
      const n = (y1 - y0) * (x1 - x0), o = (y * dw + x) * 3;
      dst[o] = (r / n + 0.5) | 0; dst[o + 1] = (g / n + 0.5) | 0; dst[o + 2] = (b / n + 0.5) | 0;
    }
  }
  return dst;
}

/** Nearest-sample a W x H byte mask onto the grid (a mask is a label; it is never averaged). */
export function resampleMask(mask: Uint8Array, W: number, H: number, gridW: number, gridH: number): Uint8Array {
  const out = new Uint8Array(gridW * gridH);
  for (let y = 0; y < gridH; y++) {
    const sy = Math.min(H - 1, Math.floor(((y + 0.5) / gridH) * H));
    for (let x = 0; x < gridW; x++) out[y * gridW + x] = mask[sy * W + Math.min(W - 1, Math.floor(((x + 0.5) / gridW) * W))]!;
  }
  return out;
}

export interface GuidedResampleOptions {
  /** Colour tolerance, RMS over RGB in 0..255 units. Default 14. */
  sigmaRange?: number;
}

/**
 * Joint bilateral resample of a `W x H` map onto the `gridW x gridH` vertex lattice.
 * `guideMap` is the frame at the map's size and `guideGrid` the same frame at the grid's size,
 * both rgb24.
 */
export function guidedResample(map: Float32Array, W: number, H: number, guideMap: Uint8Array, guideGrid: Uint8Array, gridW: number, gridH: number, opts: GuidedResampleOptions = {}): Float32Array {
  if (map.length < W * H) throw new Error(`guidedResample: map holds ${map.length} floats, expected ${W * H}`);
  if (guideMap.length < W * H * 3) throw new Error(`guidedResample: the map-size guide holds ${guideMap.length} bytes, expected ${W * H * 3}`);
  if (guideGrid.length < gridW * gridH * 3) throw new Error(`guidedResample: the grid-size guide holds ${guideGrid.length} bytes, expected ${gridW * gridH * 3}`);
  const sigmaR = opts.sigmaRange ?? 14;
  // Range weight by squared RGB distance, tabulated: 3 * 255^2 entries would be 195k, so index by
  // the distance quantised to 1/4 of the tolerance's variance step instead.
  const inv2s2 = 1 / (2 * sigmaR * sigmaR * 3);
  const lutN = 1024, lutMax = 18 * sigmaR * sigmaR * 3;       // exp(-9) beyond this: zero
  const lut = new Float32Array(lutN + 1);
  for (let i = 0; i <= lutN; i++) lut[i] = Math.exp(-((i / lutN) * lutMax) * inv2s2);
  const lutK = lutN / lutMax;

  const rx = W / gridW, ry = H / gridH;
  // The footprint follows the resampling ratio: one vertex stands for rx x ry map pixels when
  // shrinking, and needs its four bilinear neighbours plus a ring when growing.
  const radX = Math.max(2, Math.ceil(rx)), radY = Math.max(2, Math.ceil(ry));
  const sigX = Math.max(1, rx * 0.75), sigY = Math.max(1, ry * 0.75);
  const wx = new Float32Array(2 * radX + 1), wy = new Float32Array(2 * radY + 1);
  const out = new Float32Array(gridW * gridH);
  for (let gy = 0; gy < gridH; gy++) {
    const cy = (gy + 0.5) * ry - 0.5;
    const iy = Math.round(cy);
    for (let k = -radY; k <= radY; k++) { const d = iy + k - cy; wy[k + radY] = Math.exp(-(d * d) / (2 * sigY * sigY)); }
    for (let gx = 0; gx < gridW; gx++) {
      const cx = (gx + 0.5) * rx - 0.5;
      const ix = Math.round(cx);
      for (let k = -radX; k <= radX; k++) { const d = ix + k - cx; wx[k + radX] = Math.exp(-(d * d) / (2 * sigX * sigX)); }
      const go = (gy * gridW + gx) * 3;
      const r0 = guideGrid[go]!, g0 = guideGrid[go + 1]!, b0 = guideGrid[go + 2]!;
      let sum = 0, wsum = 0, plain = 0, pw = 0;
      for (let ky = -radY; ky <= radY; ky++) {
        const y = iy + ky;
        if (y < 0 || y >= H) continue;
        const wyv = wy[ky + radY]!;
        for (let kx = -radX; kx <= radX; kx++) {
          const x = ix + kx;
          if (x < 0 || x >= W) continue;
          const i = y * W + x, j = i * 3;
          const dr = guideMap[j]! - r0, dg = guideMap[j + 1]! - g0, db = guideMap[j + 2]! - b0;
          const d2 = dr * dr + dg * dg + db * db;
          const ws = wyv * wx[kx + radX]!;
          const w = ws * (d2 >= lutMax ? 0 : lut[(d2 * lutK) | 0]!);
          const v = map[i]!;
          sum += w * v; wsum += w;
          plain += ws * v; pw += ws;
        }
      }
      // No sample resembles the vertex (a one-vertex highlight, a codec artefact): fall back to the
      // unguided average rather than amplifying whichever sample happened to be least unlike it.
      out[gy * gridW + gx] = wsum > 1e-4 * pw ? sum / wsum : plain / pw;
    }
  }
  return out;
}

/**
 * Snap flying vertices to a side. A monocular model draws a silhouette as a ramp several pixels
 * wide, and resampling onto the grid leaves vertices partway down it: nearer than the wall behind
 * by more than `jump`, farther than the subject in front by more than `jump`. Such a vertex belongs
 * to neither surface, the silhouette cut removes every triangle it touches, and it anchors the
 * fill layer at a depth no surface has. Each one takes the value of whichever of its nearest and
 * farthest neighbours it resembles in colour (in disparity when there is no guide), so the
 * silhouette becomes a single step between two real surfaces.
 *
 * `gridMap` is the resampled map (disparity or metres, `kind` says which), modified in place;
 * `z` is the same map as depth. Two passes, each deciding from the values the pass started with,
 * close ramps two vertices wide. Returns the number of vertices snapped.
 */
export function snapRamps(gridMap: Float32Array, z: Float32Array, gridW: number, gridH: number, guide: Uint8Array | null, jump: number): number {
  const n = gridW * gridH;
  const src = new Float32Array(n), srcZ = new Float32Array(n), take = new Int32Array(n);
  let snapped = 0;
  for (let pass = 0; pass < 2; pass++) {
    src.set(gridMap.subarray(0, n)); srcZ.set(z.subarray(0, n)); take.fill(-1);
    for (let y = 0; y < gridH; y++) for (let x = 0; x < gridW; x++) {
      const i = y * gridW + x, zi = srcZ[i]!;
      let jn = -1, jf = -1;
      const look = (j: number) => { if (jn < 0 || srcZ[j]! < srcZ[jn]!) jn = j; if (jf < 0 || srcZ[j]! > srcZ[jf]!) jf = j; };
      if (x > 0) look(i - 1);
      if (x + 1 < gridW) look(i + 1);
      if (y > 0) look(i - gridW);
      if (y + 1 < gridH) look(i + gridW);
      if (jn < 0 || !(zi - srcZ[jn]! > jump * srcZ[jn]!) || !(srcZ[jf]! - zi > jump * zi)) continue;
      let toNear: boolean;
      if (guide) {
        const d = (j: number) => { const a = i * 3, b = j * 3; const r = guide[a]! - guide[b]!, g = guide[a + 1]! - guide[b + 1]!, bl = guide[a + 2]! - guide[b + 2]!; return r * r + g * g + bl * bl; };
        toNear = d(jn) <= d(jf);
      } else toNear = 1 / srcZ[jn]! - 1 / zi <= 1 / zi - 1 / srcZ[jf]!;
      take[i] = toNear ? jn : jf;
    }
    let changed = 0;
    for (let i = 0; i < n; i++) { const j = take[i]!; if (j >= 0) { gridMap[i] = src[j]!; z[i] = srcZ[j]!; changed++; } }
    snapped += changed;
    if (!changed) break;
  }
  return snapped;
}

export interface FillLayerOptions {
  /** Relative depth jump that marks a silhouette; the same value the relief is cut with. */
  edge: number;
  /** How far under the foreground the background is continued, in grid cells. */
  band: number;
  /**
   * Relative depth jump a cut needs before the background behind it is continued. Default
   * FILL_MIN_JUMP. Near the far plane the edge rule fires on model noise: at zero disparity a
   * change of 0.7 % of the normalised range is already an 8 % depth jump, so a sky is full of small
   * cuts. What a cut uncovers when the view moves scales with its jump, and a cut that small
   * uncovers nothing a fill is needed for; seeded, it floods the sky with a band of its own, and
   * the sky vertices beside a real silhouette then belong to it instead of anchoring the fill.
   */
  minJump?: number;
  /** Subject mask on the grid (0 = absent from the relief). Masked-out vertices take no part. */
  mask?: Uint8Array | null;
}

/** Default FillLayerOptions.minJump. */
export const FILL_MIN_JUMP = 0.25;

export interface FillLayer {
  /** Per grid vertex: distance in cells from the silhouette on the near side, -1 outside the band. */
  dist: Int16Array;
  /** Per grid vertex: fill depth where dist >= 0 (undefined elsewhere). */
  z: Float32Array;
  /** Per grid vertex: 1 where the vertex is the far end of a cut the band starts from. */
  anchor: Uint8Array;
  /** rgb24 on the grid: the fill colour inside the band, the frame's own colour elsewhere. */
  colour: Uint8Array;
  /** Vertices in the band. */
  count: number;
}

/**
 * Continue the background under the foreground along every silhouette.
 *
 * A cut edge has a near end (the occluder) and a far end (what it hides). The band is every vertex
 * reachable from a near end through uncut edges in at most `band` steps: the occluder's own
 * surface, walked inward from its outline. Depth and colour enter at the outline from the far ends
 * and are carried inward ring by ring, each vertex averaging the ring before it, which continues
 * the background's depth flat and its colour as a smooth blend of what borders the hole.
 */
export function buildFillLayer(z: Float32Array, gridW: number, gridH: number, guideGrid: Uint8Array, opts: FillLayerOptions): FillLayer {
  const n = gridW * gridH;
  const edge = opts.edge, mask = opts.mask ?? null;
  const dist = new Int16Array(n).fill(-1);
  const fz = new Float32Array(n);
  const anchor = new Uint8Array(n);
  const colour = guideGrid.slice(0, n * 3);
  const acc = new Float32Array(n * 4);       // seed accumulators: z, r, g, b
  const cnt = new Uint16Array(n);
  const present = (i: number) => !mask || mask[i]! >= 128;
  const cut = (a: number, b: number) => { const za = z[a]!, zb = z[b]!; return Math.abs(za - zb) > edge * (za < zb ? za : zb); };
  const minJump = Math.max(edge, opts.minJump ?? FILL_MIN_JUMP);

  const seed = (near: number, far: number) => {
    if (z[far]! - z[near]! <= minJump * z[near]!) return;
    anchor[far] = 1;
    acc[near * 4] = acc[near * 4]! + z[far]!;
    acc[near * 4 + 1] = acc[near * 4 + 1]! + guideGrid[far * 3]!;
    acc[near * 4 + 2] = acc[near * 4 + 2]! + guideGrid[far * 3 + 1]!;
    acc[near * 4 + 3] = acc[near * 4 + 3]! + guideGrid[far * 3 + 2]!;
    cnt[near] = cnt[near]! + 1;
  };
  for (let y = 0; y < gridH; y++) for (let x = 0; x < gridW; x++) {
    const i = y * gridW + x;
    if (!present(i)) continue;
    if (x + 1 < gridW && present(i + 1) && cut(i, i + 1)) { if (z[i]! < z[i + 1]!) seed(i, i + 1); else seed(i + 1, i); }
    if (y + 1 < gridH && present(i + gridW) && cut(i, i + gridW)) { if (z[i]! < z[i + gridW]!) seed(i, i + gridW); else seed(i + gridW, i); }
  }

  let ring: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!cnt[i]) continue;
    const c = cnt[i]!;
    dist[i] = 0;
    fz[i] = acc[i * 4]! / c;
    colour[i * 3] = acc[i * 4 + 1]! / c; colour[i * 3 + 1] = acc[i * 4 + 2]! / c; colour[i * 3 + 2] = acc[i * 4 + 3]! / c;
    ring.push(i);
  }
  let count = ring.length;
  const nb = [0, 0, 0, 0];
  for (let d = 1; d <= opts.band && ring.length; d++) {
    const next: number[] = [];
    for (const i of ring) {
      const x = i % gridW, y = (i - x) / gridW;
      let k = 0;
      if (x > 0) nb[k++] = i - 1;
      if (x + 1 < gridW) nb[k++] = i + 1;
      if (y > 0) nb[k++] = i - gridW;
      if (y + 1 < gridH) nb[k++] = i + gridW;
      for (let q = 0; q < k; q++) {
        const j = nb[q]!;
        if (dist[j] !== -1 || !present(j) || cut(i, j)) continue;
        dist[j] = d;
        next.push(j);
      }
    }
    // Values after membership, so a vertex averages every ring-(d-1) neighbour, not just the one
    // that happened to reach it first.
    for (const j of next) {
      const x = j % gridW, y = (j - x) / gridW;
      let k = 0;
      if (x > 0) nb[k++] = j - 1;
      if (x + 1 < gridW) nb[k++] = j + 1;
      if (y > 0) nb[k++] = j - gridW;
      if (y + 1 < gridH) nb[k++] = j + gridW;
      let sz = 0, sr = 0, sg = 0, sb = 0, c = 0;
      for (let q = 0; q < k; q++) {
        const i = nb[q]!;
        if (dist[i] !== d - 1 || cut(i, j)) continue;
        sz += fz[i]!; sr += colour[i * 3]!; sg += colour[i * 3 + 1]!; sb += colour[i * 3 + 2]!; c++;
      }
      fz[j] = sz / c;
      colour[j * 3] = sr / c; colour[j * 3 + 1] = sg / c; colour[j * 3 + 2] = sb / c;
    }
    count += next.length;
    ring = next;
  }
  relaxBand(dist, fz, colour, gridW, gridH, cut);
  // The fill is a layer BEHIND the surface it sits under, never through it.
  for (let i = 0; i < n; i++) if (dist[i]! >= 0) { const floor = z[i]! * (1 + edge); if (fz[i]! < floor) fz[i] = floor; }
  return { dist, z: fz, anchor, colour, count };
}

/** Gauss-Seidel sweeps relaxBand runs over the band. */
const RELAX_SWEEPS = 16;

/**
 * Smooth the band's depth and colour into one surface. The rings carry each outline vertex's
 * depth straight inward, so neighbouring rows seeded from different backgrounds keep their full
 * difference all the way in: measured on a film frame, one vertical pair in ten inside the band
 * differed by more than the edge rule, and the fill rendered as strips. Each sweep replaces every
 * interior band vertex (ring 1 and deeper) by the mean of its band neighbours across uncut edges;
 * the outline (ring 0) is held, so the fill still meets the background it continues.
 */
function relaxBand(dist: Int16Array, fz: Float32Array, colour: Uint8Array, gridW: number, gridH: number, cut: (a: number, b: number) => boolean): void {
  const inner: number[] = [];
  for (let i = 0; i < dist.length; i++) if (dist[i]! > 0) inner.push(i);
  if (!inner.length) return;
  const rgb = new Float32Array(dist.length * 3);
  for (const i of inner) { rgb[i * 3] = colour[i * 3]!; rgb[i * 3 + 1] = colour[i * 3 + 1]!; rgb[i * 3 + 2] = colour[i * 3 + 2]!; }
  for (let i = 0; i < dist.length; i++) if (dist[i] === 0) { rgb[i * 3] = colour[i * 3]!; rgb[i * 3 + 1] = colour[i * 3 + 1]!; rgb[i * 3 + 2] = colour[i * 3 + 2]!; }
  const nb = [0, 0, 0, 0];
  for (let s = 0; s < RELAX_SWEEPS; s++) {
    for (let q = 0; q < inner.length; q++) {
      const i = inner[s & 1 ? inner.length - 1 - q : q]!;
      const x = i % gridW, y = (i - x) / gridW;
      let k = 0;
      if (x > 0) nb[k++] = i - 1;
      if (x + 1 < gridW) nb[k++] = i + 1;
      if (y > 0) nb[k++] = i - gridW;
      if (y + 1 < gridH) nb[k++] = i + gridW;
      let sz = 0, sr = 0, sg = 0, sb = 0, c = 0;
      for (let m = 0; m < k; m++) {
        const j = nb[m]!;
        if (dist[j]! < 0 || cut(i, j)) continue;
        sz += fz[j]!; sr += rgb[j * 3]!; sg += rgb[j * 3 + 1]!; sb += rgb[j * 3 + 2]!; c++;
      }
      if (!c) continue;
      fz[i] = sz / c; rgb[i * 3] = sr / c; rgb[i * 3 + 1] = sg / c; rgb[i * 3 + 2] = sb / c;
    }
  }
  for (const i of inner) { colour[i * 3] = rgb[i * 3]! + 0.5; colour[i * 3 + 1] = rgb[i * 3 + 1]! + 0.5; colour[i * 3 + 2] = rgb[i * 3 + 2]! + 0.5; }
}

/**
 * The atlas of a two-layer relief: the video frame on top, the fill plate under it.
 * `frame` is rgb24 `texW x texH`; the plate is `texW x plateH`: the frame again, with the band
 * replaced by the fill colours (bilinear over the grid). Returns rgb24 `texW x (texH + plateH)`.
 */
export function composeLayeredAtlas(frame: Uint8Array, texW: number, texH: number, plateH: number, fill: FillLayer, gridW: number, gridH: number): Uint8Array {
  const out = new Uint8Array(texW * (texH + plateH) * 3);
  out.set(frame.subarray(0, texW * texH * 3));
  const plate = out.subarray(texW * texH * 3);
  resizeRgbArea(frame, texW, texH, texW, plateH, plate);
  if (!fill.count) return out;
  const { dist, colour } = fill;
  for (let y = 0; y < plateH; y++) {
    // Texel centre -> grid coordinate: vertex (gx, gy) sits at u = (gx + 0.5) / gridW.
    const fy = ((y + 0.5) / plateH) * gridH - 0.5;
    const y0 = Math.max(0, Math.min(gridH - 1, Math.floor(fy))), y1 = Math.min(gridH - 1, y0 + 1);
    const ty = Math.max(0, Math.min(1, fy - y0));
    for (let x = 0; x < texW; x++) {
      const fx = ((x + 0.5) / texW) * gridW - 0.5;
      const x0 = Math.max(0, Math.min(gridW - 1, Math.floor(fx))), x1 = Math.min(gridW - 1, x0 + 1);
      const a = y0 * gridW + x0, b = y0 * gridW + x1, c = y1 * gridW + x0, d = y1 * gridW + x1;
      if (dist[a]! < 0 && dist[b]! < 0 && dist[c]! < 0 && dist[d]! < 0) continue;
      const tx = Math.max(0, Math.min(1, fx - x0));
      const o = (y * texW + x) * 3;
      for (let ch = 0; ch < 3; ch++) {
        const top = colour[a * 3 + ch]! + (colour[b * 3 + ch]! - colour[a * 3 + ch]!) * tx;
        const bot = colour[c * 3 + ch]! + (colour[d * 3 + ch]! - colour[c * 3 + ch]!) * tx;
        plate[o + ch] = (top + (bot - top) * ty + 0.5) | 0;
      }
    }
  }
  return out;
}
