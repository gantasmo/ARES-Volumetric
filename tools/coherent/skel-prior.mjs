/**
 * Skeleton pose prior (SAM-3D-Body) for coherent registration — the motion-robust fix.
 *
 * WHY: ARAP-alone registration abandons fast-moving limbs. Its 40mm confidence gate downweights any
 * nearest-point match >40mm, so when a limb moves >40mm/frame the region stops following the source and
 * is carried RIGIDLY from the template pose — the surface diverges up to ~170mm (run4 f149) while edge
 * lengths stay ~1.0x (which is why the maxStretch metric called it "clean"; it was not).
 *
 * FIX: use the per-frame SAM-3D-Body skeleton (127 joints, each with a global 3x3 rotation) to LBS-warp
 * the template into the target pose FIRST, and feed that as the ARAP initialization. Nearest-point then
 * starts inside the gate on the correct limb → the gate stays open → ARAP only cleans up a small
 * non-rigid residual. The skeleton owns the big articulation; ARAP owns the detail.
 *
 * Pose format (ares/sam3d-results/.../poses/fNNNNN_pose.json), validated 2026-07-13:
 *   pred_joint_coords : (127,3) root-relative joint positions, METERS, joint0 = root pinned at origin.
 *   pred_global_rots  : (127,3,3) per-joint GLOBAL rotation (body-local frame; joint0 = identity).
 *   global_rot        : (3,) axis-angle of the whole body's global orientation (radians).
 * These are in SAM-3D camera-ish space (Y-down); we solve one similarity map to ARES world (Y-up, mm).
 *
 * No bone hierarchy needed: each joint carries its own global rotation, so the template->N transform of
 * joint j is the rigid motion (R_N(j) R_t(j)^T about the joint), composed with the body global_rot delta.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

/* --------------------------------- mat3 (row-major flat[9]) --------------------------------- */
const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
function matmul(a, b) { // a*b, row-major
  const o = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}
function transpose(a) { return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]; }
function matvec(a, v) {
  return [a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
          a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
}
function det3(a) {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}
// Rodrigues: axis-angle vector (magnitude = angle, radians) -> rotation matrix (row-major).
function rodrigues(a) {
  const th = Math.hypot(a[0], a[1], a[2]);
  if (th < 1e-9) return I3();
  const kx = a[0] / th, ky = a[1] / th, kz = a[2] / th;
  const s = Math.sin(th), c = Math.cos(th), C = 1 - c;
  return [
    c + kx * kx * C,      kx * ky * C - kz * s, kx * kz * C + ky * s,
    ky * kx * C + kz * s, c + ky * ky * C,      ky * kz * C - kx * s,
    kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C,
  ];
}

/* ------------------------------------- pose loading ------------------------------------- */
export async function loadPose(posesDir, frame) {
  const p = JSON.parse(await readFile(join(posesDir, `f${String(frame).padStart(5, "0")}_pose.json`), "utf8"));
  const J = p.pred_joint_coords.length;
  const jc = new Float64Array(J * 3);
  for (let j = 0; j < J; j++) { jc[j * 3] = p.pred_joint_coords[j][0]; jc[j * 3 + 1] = p.pred_joint_coords[j][1]; jc[j * 3 + 2] = p.pred_joint_coords[j][2]; }
  const R = new Array(J);
  for (let j = 0; j < J; j++) { const m = p.pred_global_rots[j]; R[j] = [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]; }
  return { J, jc, R, gr: p.global_rot.slice(0, 3) };
}

// Body-space joint position at a frame: apply the whole-body global_rot to the root-relative coords.
// (Returns SAM-3D world up to the unknown root translation, which the calibration/prealign absorbs.)
function bodyJoint(pose, j) { return matvec(rodrigues(pose.gr), [pose.jc[j * 3], pose.jc[j * 3 + 1], pose.jc[j * 3 + 2]]); }
// Body-space joint global orientation: global_rot composed with the joint's body-local rotation.
function bodyRot(pose, j) { return matmul(rodrigues(pose.gr), pose.R[j]); }

/* ---------------------------------- SAM-3D -> ARES calibration ---------------------------------- */
// Fit similarity (scale s, rotation Rmap, translation tau) placing the template-frame joints inside the
// ARES template mesh. Rmap is auto-selected from proper-rotation axis-flip candidates by the metric
// "median joint distance into the mesh AABB" (a good map puts every joint inside the body volume).
function bbox(pos) {
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) { const v = pos[i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
  return { mn, mx, ctr: [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2], ext: [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]] };
}
// The 24 proper-rotation axis-permutation/flip matrices (signed permutations with det +1).
function axisCandidates() {
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const out = [];
  for (const p of perms) for (let s = 0; s < 8; s++) {
    const sg = [(s & 1) ? -1 : 1, (s & 2) ? -1 : 1, (s & 4) ? -1 : 1];
    const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let r = 0; r < 3; r++) M[r * 3 + p[r]] = sg[r];
    if (Math.abs(det3(M) - 1) < 1e-6) out.push(M);
  }
  return out; // 24 proper rotations
}
export function calibrate(templatePose, aresPositions) {
  const jb = (() => { // body-space joint cloud (template)
    const arr = new Float64Array(templatePose.J * 3);
    for (let j = 0; j < templatePose.J; j++) { const q = bodyJoint(templatePose, j); arr[j * 3] = q[0]; arr[j * 3 + 1] = q[1]; arr[j * 3 + 2] = q[2]; }
    return arr;
  })();
  const meshBox = bbox(aresPositions), jointBox = bbox(jb);
  // scale from vertical (tallest) extent ratio — robust to which axis is "up" pre-rotation
  const jointHeight = Math.max(...jointBox.ext) || 1;
  const meshHeight = Math.max(...meshBox.ext) || 1;
  const s = meshHeight / jointHeight;
  const jctr = jointBox.ctr;
  // choose Rmap by joints-inside-mesh: after mapping, fraction of joints within the mesh AABB (+ margin)
  const cands = axisCandidates();
  let best = null;
  const margin = 0.06 * Math.max(...meshBox.ext);
  for (const Rmap of cands) {
    const tau = [meshBox.ctr[0], meshBox.ctr[1], meshBox.ctr[2]];
    let inside = 0;
    for (let j = 0; j < templatePose.J; j++) {
      const rel = [jb[j * 3] - jctr[0], jb[j * 3 + 1] - jctr[1], jb[j * 3 + 2] - jctr[2]];
      const w = matvec(Rmap, rel).map((x, k) => s * x + tau[k]);
      let ok = true;
      for (let k = 0; k < 3; k++) if (w[k] < meshBox.mn[k] - margin || w[k] > meshBox.mx[k] + margin) ok = false;
      if (ok) inside++;
    }
    // prefer maps that also put the head (min-Y-in-ARES should be feet, so joints should span up)
    if (!best || inside > best.inside) best = { Rmap, tau, inside };
  }
  return { s, Rmap: best.Rmap, tau: best.tau, jctr, insideFrac: best.inside / templatePose.J };
}
// Map a body-space joint position (relative to template joint-centroid) into ARES world.
function toAres(calib, q) {
  const rel = [q[0] - calib.jctr[0], q[1] - calib.jctr[1], q[2] - calib.jctr[2]];
  const w = matvec(calib.Rmap, rel);
  return [calib.s * w[0] + calib.tau[0], calib.s * w[1] + calib.tau[1], calib.s * w[2] + calib.tau[2]];
}

/* --------------------------------------- LBS weights --------------------------------------- */
// For each (compacted) template vertex, the K nearest joints (in ARES template space) + inverse-distance
// weights. K small so a torso vertex isn't dragged by hand joints. Joints beyond the K-th are ignored.
export function buildSkinWeights(restC, nc, calib, templatePose, K = 4) {
  const Qt = new Float64Array(templatePose.J * 3); // template joints in ARES
  for (let j = 0; j < templatePose.J; j++) { const a = toAres(calib, bodyJoint(templatePose, j)); Qt[j * 3] = a[0]; Qt[j * 3 + 1] = a[1]; Qt[j * 3 + 2] = a[2]; }
  const idx = new Int32Array(nc * K), wt = new Float64Array(nc * K);
  const cand = [];
  for (let i = 0; i < nc; i++) {
    cand.length = 0;
    for (let j = 0; j < templatePose.J; j++) {
      const dx = restC[i * 3] - Qt[j * 3], dy = restC[i * 3 + 1] - Qt[j * 3 + 1], dz = restC[i * 3 + 2] - Qt[j * 3 + 2];
      cand.push([dx * dx + dy * dy + dz * dz, j]);
    }
    cand.sort((a, b) => a[0] - b[0]);
    let wsum = 0;
    for (let k = 0; k < K; k++) { const d2 = cand[k][0]; const w = 1 / (d2 + 1e-6); idx[i * K + k] = cand[k][1]; wt[i * K + k] = w; wsum += w; }
    for (let k = 0; k < K; k++) wt[i * K + k] /= wsum;
  }
  return { Qt, idx, wt, K };
}

/* ------------------------------------- the warp (ARAP init) ------------------------------------- */
// LBS-warp the template (compacted rest positions restC) into frame N's pose. Returns compacted init.
// v_N = Σ_j w_j [ M(j) (v - Qt(j)) + QN(j) ],  M(j) = Rmap (G_N(j) G_t(j)^T) Rmap^T (ARES-space rotation).
export function skelWarpInit(restC, nc, calib, skin, templatePose, framePose, opts = {}) {
  const { J } = templatePose, { idx, wt, K, Qt } = skin;
  // Precompute per-joint ARES rotation M(j) and frame-N ARES joint position QN(j).
  const RmapT = transpose(calib.Rmap);
  const M = new Array(J), QN = new Float64Array(J * 3);
  const noRot = !!opts.noRot; // pure translational LBS (immune to per-joint rotation-convention errors)
  for (let j = 0; j < J; j++) {
    if (noRot) { M[j] = I3(); }
    else { const Gt = bodyRot(templatePose, j), Gn = bodyRot(framePose, j); M[j] = matmul(matmul(calib.Rmap, matmul(Gn, transpose(Gt))), RmapT); }
    const a = toAres(calib, bodyJoint(framePose, j)); QN[j * 3] = a[0]; QN[j * 3 + 1] = a[1]; QN[j * 3 + 2] = a[2];
  }
  const out = new Float64Array(nc * 3);
  for (let i = 0; i < nc; i++) {
    const vx = restC[i * 3], vy = restC[i * 3 + 1], vz = restC[i * 3 + 2];
    let ax = 0, ay = 0, az = 0;
    for (let k = 0; k < K; k++) {
      const j = idx[i * K + k], w = wt[i * K + k], Mj = M[j];
      const ox = vx - Qt[j * 3], oy = vy - Qt[j * 3 + 1], oz = vz - Qt[j * 3 + 2];
      const rx = Mj[0] * ox + Mj[1] * oy + Mj[2] * oz;
      const ry = Mj[3] * ox + Mj[4] * oy + Mj[5] * oz;
      const rz = Mj[6] * ox + Mj[7] * oy + Mj[8] * oz;
      ax += w * (rx + QN[j * 3]); ay += w * (ry + QN[j * 3 + 1]); az += w * (rz + QN[j * 3 + 2]);
    }
    out[i * 3] = ax; out[i * 3 + 1] = ay; out[i * 3 + 2] = az;
  }
  // Optional rigid pre-align: translate warped centroid onto the source centroid (removes global drift
  // from the dancing-in-place root translation we don't model). Requires opts.srcCentroid.
  if (opts.srcCentroid) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < nc; i++) { cx += out[i * 3]; cy += out[i * 3 + 1]; cz += out[i * 3 + 2]; }
    cx /= nc; cy /= nc; cz /= nc;
    const dx = opts.srcCentroid[0] - cx, dy = opts.srcCentroid[1] - cy, dz = opts.srcCentroid[2] - cz;
    for (let i = 0; i < nc; i++) { out[i * 3] += dx; out[i * 3 + 1] += dy; out[i * 3 + 2] += dz; }
  }
  return out;
}
