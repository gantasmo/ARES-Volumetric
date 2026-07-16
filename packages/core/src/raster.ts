/**
 * CPU triangle-id rasterizer (editor v2 picking, docs/editor-v2-design.md §5.2 CPU flavor).
 *
 * Rasterizes the current frame's triangles into a small id+depth buffer from the current camera.
 * This is what gives the editor Blender's "solid mode selects only what's VISIBLE" semantics
 * (front-most triangle per pixel) and surface-hit positions for the brush — without a GPU ID pass
 * or readback. ~20k tris into 320² is a few ms, and it only runs on selection gestures, never per
 * rendered frame.
 */
import type { Aabb } from "./quant.js";

export interface IdBuffer {
  w: number;
  h: number;
  /** triangle index (i*3 into the index buffer) per pixel, -1 = background */
  ids: Int32Array;
  /** NDC z per pixel (smaller = closer), +Inf = background */
  depth: Float32Array;
}

/**
 * @param positionsQ quantized positions, stride 4 (u16 x,y,z,pad) — the player's live frame state
 * @param indices    full triangle list
 * @param box        the frame's GOP AABB (dequant range)
 * @param invLevels  dequant scale (1 / quant levels) — same value the shader uses
 * @param viewProj   column-major mat4 (camera.ts conventions)
 */
export function rasterizeIds(
  positionsQ: Uint16Array,
  indices: Uint32Array,
  box: Aabb,
  invLevels: number,
  viewProj: Float32Array,
  w: number,
  h: number,
): IdBuffer {
  const ids = new Int32Array(w * h).fill(-1);
  const depth = new Float32Array(w * h).fill(Infinity);
  const vcount = positionsQ.length / 4;

  // Project every vertex once: screen px/py + ndc z (w-divided). wOk marks in-front-of-camera.
  const px = new Float32Array(vcount), py = new Float32Array(vcount), pz = new Float32Array(vcount);
  const wOk = new Uint8Array(vcount);
  const m = viewProj;
  const sx = (box.max[0] - box.min[0]) * invLevels;
  const sy = (box.max[1] - box.min[1]) * invLevels;
  const sz = (box.max[2] - box.min[2]) * invLevels;
  for (let v = 0; v < vcount; v++) {
    const x = box.min[0] + positionsQ[v * 4]! * sx;
    const y = box.min[1] + positionsQ[v * 4 + 1]! * sy;
    const z = box.min[2] + positionsQ[v * 4 + 2]! * sz;
    const cw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
    if (cw <= 1e-6) { wOk[v] = 0; continue; }
    const inv = 1 / cw;
    const cx = (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * inv;
    const cy = (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * inv;
    const cz = (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * inv;
    px[v] = (cx * 0.5 + 0.5) * w;
    py[v] = (1 - (cy * 0.5 + 0.5)) * h;
    pz[v] = cz;
    wOk[v] = 1;
  }

  // Scanline each triangle's bbox with edge functions; nearest-z wins per pixel.
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
    if (!wOk[a] || !wOk[b] || !wOk[c]) continue;
    const ax = px[a]!, ay = py[a]!, bx = px[b]!, by = py[b]!, cx = px[c]!, cy = py[c]!;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (area === 0) continue;
    const inv = 1 / area;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
    if (x1 < x0 || y1 < y0) continue;
    const za = pz[a]!, zb = pz[b]!, zc = pz[c]!;
    for (let yy = y0; yy <= y1; yy++) {
      const rowBase = yy * w;
      const pyc = yy + 0.5;
      for (let xx = x0; xx <= x1; xx++) {
        const pxc = xx + 0.5;
        // barycentric via edge functions (sign-normalized by 1/area → orientation-agnostic)
        const w0 = ((bx - pxc) * (cy - pyc) - (by - pyc) * (cx - pxc)) * inv;
        const w1 = ((cx - pxc) * (ay - pyc) - (cy - pyc) * (ax - pxc)) * inv;
        const w2 = 1 - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
        const z = w0 * za + w1 * zb + w2 * zc;
        const o = rowBase + xx;
        if (z < depth[o]!) { depth[o] = z; ids[o] = t; }
      }
    }
  }
  return { w, h, ids, depth };
}
