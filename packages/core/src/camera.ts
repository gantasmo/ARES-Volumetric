/** Minimal column-major mat4 math + an orbit camera for the P1 renderer. No dependencies. */

export type Mat4 = Float32Array; // length 16, column-major (WebGPU/WGSL convention)

export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far * nf;          // WebGPU clip z ∈ [0,1]
  m[11] = -1;
  m[14] = far * near * nf;
  return m;
}

/**
 * Orthographic projection (no perspective divide — parallel lines stay parallel).
 * `height` is the world height of the view box; width follows from the aspect.
 * z maps to WebGPU clip [0,1] like `perspective` above: z_view=-near -> 0, z_view=-far -> 1.
 */
export function orthographic(height: number, aspect: number, near: number, far: number): Mat4 {
  const t = height / 2, r = t * aspect;
  const m = new Float32Array(16);
  m[0] = 1 / r;
  m[5] = 1 / t;
  m[10] = 1 / (near - far);
  m[14] = near / (near - far);
  m[15] = 1;
  return m;
}

export function lookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): Mat4 {
  const z0 = eye[0] - target[0], z1 = eye[1] - target[1], z2 = eye[2] - target[2];
  let zl = Math.hypot(z0, z1, z2) || 1;
  const zx = z0 / zl, zy = z1 / zl, zz = z2 / zl;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const m = new Float32Array(16);
  m[0] = xx; m[1] = yx; m[2] = zx; m[3] = 0;
  m[4] = xy; m[5] = yy; m[6] = zy; m[7] = 0;
  m[8] = xz; m[9] = yz; m[10] = zz; m[11] = 0;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}

/**
 * Full 4×4 inverse (column-major), or null when the matrix is singular. Cofactor expansion —
 * no assumption of affinity, because the one caller that needs it inverts a PROJECTION
 * (viewProj), whose bottom row is not [0,0,0,1] under perspective.
 *
 * Used by the infinite ground grid: it reconstructs a world-space ray per pixel by unprojecting
 * the fragment's NDC at z=0 and z=1, which only works with a true inverse of viewProj.
 */
export function invert(m: Mat4): Mat4 | null {
  const a00 = m[0]!, a01 = m[1]!, a02 = m[2]!, a03 = m[3]!;
  const a10 = m[4]!, a11 = m[5]!, a12 = m[6]!, a13 = m[7]!;
  const a20 = m[8]!, a21 = m[9]!, a22 = m[10]!, a23 = m[11]!;
  const a30 = m[12]!, a31 = m[13]!, a32 = m[14]!, a33 = m[15]!;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  const d = 1 / det;
  const o = new Float32Array(16);
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d;
  return o;
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4]!, b1 = b[c * 4 + 1]!, b2 = b[c * 4 + 2]!, b3 = b[c * 4 + 3]!;
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r]! * b0 + a[4 + r]! * b1 + a[8 + r]! * b2 + a[12 + r]! * b3;
    }
  }
  return o;
}

export interface OrbitState {
  azimuth: number; elevation: number; distance: number; target: [number, number, number];
  /** Orthographic instead of perspective. Measuring work (crop planes, alignment) needs it: under
   *  perspective an axis-aligned plane projects to a REGION, not a line, so a straight guide can
   *  only ever approximate the cut. In ortho, edge-on planes are exact lines. */
  ortho?: boolean;
}

const FOV_Y = (50 * Math.PI) / 180;
/** The world height the perspective camera sees at the orbit distance. Reused by the ortho box so
 *  toggling projection keeps the subject the same size instead of jumping. */
export const orbitViewHeight = (distance: number): number => 2 * distance * Math.tan(FOV_Y / 2);

/** viewProj for an orbit camera looking at `target` from spherical (azimuth, elevation, distance). */
export function orbitViewProj(o: OrbitState, aspect: number): Mat4 {
  const ce = Math.cos(o.elevation), se = Math.sin(o.elevation);
  const eye: [number, number, number] = [
    o.target[0] + o.distance * ce * Math.sin(o.azimuth),
    o.target[1] + o.distance * se,
    o.target[2] + o.distance * ce * Math.cos(o.azimuth),
  ];
  const view = lookAt(eye, o.target, [0, 1, 0]);
  // near/far scale with distance so the camera works at any model scale (unit or mm).
  const near = o.distance * 0.01, far = o.distance * 20;
  const proj = o.ortho
    ? orthographic(orbitViewHeight(o.distance), aspect, near, far)
    : perspective(FOV_Y, aspect, near, far);
  return multiply(proj, view);
}
