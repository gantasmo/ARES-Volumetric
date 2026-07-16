/**
 * Viewport overlay support — the origin axis tripod (real line geometry) and the LOD ladder for the
 * INFINITE ground grid (which is not geometry at all: both renderers raymarch it analytically in a
 * fragment shader, so it extends forever and fades into the distance instead of ending at a wall).
 *
 * Why analytic instead of a line buffer: a finite line list has to pick an extent and a cell size up
 * front. Whatever you pick is wrong at some zoom — either the grid visibly ENDS (a wall in the middle
 * of the floor) or, once the cells fall under a pixel, the lines alias into moiré. The shader has the
 * fragment's exact ground-plane hit point, so it can size cells to the CURRENT zoom, antialias each
 * line with screen-space derivatives, and fade out with distance. See `gridLod` for the ladder.
 *
 * Muted grey palette, no glow. World units are the clip's own (ARES millimetres for
 * the Microsoft/daniel captures, metres for 4DViews bakes — the unit-scale trap); the
 * grid is unit-agnostic by construction because every size it uses is derived from the live camera.
 */
export interface TripodOpts {
  floorY?: number;      // ground-plane height (world units)
  axisLen?: number;     // origin tripod arm length (world units)
}

/**
 * Origin axis tripod — three short colored arms at (0, floorY, 0), as an interleaved
 * Float32Array [x,y,z, r,g,b] per vertex (2 verts per line) + the vertex count. Same muted RGB as
 * the axis gizmo (gizmo.js), so "which way is X" reads identically in the corner widget and in the
 * world. This is the ONLY overlay geometry now — the grid itself is shader-side.
 */
export function buildOriginTripod(opts: TripodOpts = {}): { data: Float32Array; count: number } {
  const y = opts.floorY ?? 0;
  const axisLen = opts.axisLen ?? 300;
  const xCol: [number, number, number] = [0.878, 0.376, 0.373];
  const yCol: [number, number, number] = [0.494, 0.765, 0.416];
  const zCol: [number, number, number] = [0.373, 0.573, 0.878];

  const verts: number[] = [];
  const line = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, c: [number, number, number]) => {
    verts.push(x0, y0, z0, c[0], c[1], c[2], x1, y1, z1, c[0], c[1], c[2]);
  };
  line(0, y, 0, axisLen, y, 0, xCol);
  line(0, y, 0, 0, y + axisLen, 0, yCol);   // the Y arm rises from the floor, not from absolute 0
  line(0, y, 0, 0, y, axisLen, zCol);
  return { data: new Float32Array(verts), count: verts.length / 6 };
}

/** The two cell sizes the grid shader draws, and how visible the finer of the two is. */
export interface GridLod {
  /** Fine cell size, world units — drawn at `fineFade` opacity. */
  step0: number;
  /** Coarse cell size (10 × step0), world units — always drawn at full opacity. */
  step1: number;
  /** 0..1 opacity of the fine level. */
  fineFade: number;
}

/**
 * Pick the grid's cell sizes for the current zoom.
 *
 * AUTO (`explicitStep <= 0`) walks a decade ladder and CROSS-FADES, which is the only way an
 * infinite grid can stay readable at every zoom without popping. The trick is that decade cell sizes
 * are NESTED — every line of the 10× grid is also a line of the 1× grid — so drawing
 *
 *     alpha = max( grid(step1), grid(step0) * fineFade )
 *
 * is continuous across a level change: as you zoom out, `fineFade` runs 1 → 0 and the fine lines
 * dissolve, leaving exactly the step1 lines at full opacity; at the instant the level ticks over,
 * the new step0 IS the old step1 (drawn at fineFade = 1, i.e. still full) and the new step1's lines
 * were already being drawn at full as a subset of the old step1's. Nothing changes brightness at
 * the boundary, so the ladder is seamless in both directions.
 *
 * EXPLICIT step: the user asked for exactly that increment, so the fine level never fades. step1
 * stays at 10× as a readable coarse reference.
 *
 * @param pxPerUnit  device pixels per world unit at the orbit target
 * @param minPx      smallest cell, in pixels, the fine level is allowed to shrink to before the
 *                   ladder steps up a decade
 * @param explicitStep  a user-chosen cell size in world units; <= 0 means auto
 */
export function gridLod(pxPerUnit: number, minPx = 12, explicitStep = 0): GridLod {
  if (explicitStep > 0 && Number.isFinite(explicitStep)) {
    return { step0: explicitStep, step1: explicitStep * 10, fineFade: 1 };
  }
  if (!Number.isFinite(pxPerUnit) || pxPerUnit <= 0) return { step0: 1, step1: 10, fineFade: 1 };
  // t = log10(world units spanned by `minPx` pixels) — the decade at which a cell is exactly minPx.
  const t = Math.log10(minPx / pxPerUnit);
  const level = Math.floor(t);
  const frac = t - level;
  const step0 = Math.pow(10, level);
  return { step0, step1: step0 * 10, fineFade: 1 - frac };
}

/** Everything the grid shader needs for one frame. All sizes/positions are world units. */
export interface GridParams extends GridLod {
  /** Ground-plane height. */
  floorY: number;
  /** The grid fades out radially around this point (the orbit target) on the XZ plane. */
  centerX: number;
  centerZ: number;
  /** Distance from (centerX, centerZ) at which the grid has fully faded to nothing. */
  fadeRadius: number;
  /**
   * Origin tripod arm length. MUST be derived from the clip, never a constant: ARES world units
   * differ per capture (millimetres for the Microsoft/daniel clips, metres for 4DViews bakes),
   * so a fixed length is either invisible or, at metre scale, three coloured arms
   * shooting hundreds of units across the scene.
   */
  axisLen: number;
}
