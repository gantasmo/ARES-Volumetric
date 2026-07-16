// The crop guides need screen->world along ONE axis. The projection of an axis-aligned line is a
// line, so ndc = (A + B*v) / (C + D*v) and it inverts analytically. Verify that round-trips exactly
// against the renderer's OWN orbitViewProj before building any UI on it.
import { orbitViewProj } from "../../../packages/core/dist/index.js";

const aabb = { min: [-157.65, -12.88, -49.24], max: [948.35, 1725.13, 428.01] };  // capture-scale mm
const centre = () => [(aabb.min[0] + aabb.max[0]) / 2, (aabb.min[1] + aabb.max[1]) / 2, (aabb.min[2] + aabb.max[2]) / 2];

const projPoint = (vp, p) => {
  const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  return [(vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12]) / w,
          (vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13]) / w];
};
const axisNdc = (vp, ax, comp, v) => { const c = centre(); c[ax] = v; return projPoint(vp, c)[comp]; };
const ndcToAxis = (vp, ax, comp, target) => {
  const c = centre();
  let A = vp[comp] * c[0] + vp[4 + comp] * c[1] + vp[8 + comp] * c[2] + vp[12 + comp];
  let C = vp[3] * c[0] + vp[7] * c[1] + vp[11] * c[2] + vp[15];
  const B = vp[4 * ax + comp], D = vp[4 * ax + 3];
  A -= B * c[ax]; C -= D * c[ax];
  const den = B - target * D;
  if (Math.abs(den) < 1e-9) return null;
  return (target * C - A) / den;
};

const target = centre();
const dist = Math.hypot(aabb.max[0] - aabb.min[0], aabb.max[1] - aabb.min[1], aabb.max[2] - aabb.min[2]) * 0.95;

let worst = 0, tested = 0, nulls = 0;
for (const az of [0, 0.4, 0.8, 1.57, 2.4, 3.14, 4.0, 5.5]) {
  for (const el of [-1.2, -0.5, 0, 0.22, 0.9, 1.3]) {
    for (const aspect of [1.0, 1.777, 0.6]) {
      const vp = orbitViewProj({ azimuth: az, elevation: el, distance: dist, target }, aspect);
      for (let ax = 0; ax < 3; ax++) {
        for (const comp of [0, 1]) {
          for (const t of [0, 0.25, 0.5, 0.75, 1]) {
            const v = aabb.min[ax] + t * (aabb.max[ax] - aabb.min[ax]);
            const ndc = axisNdc(vp, ax, comp, v);
            const back = ndcToAxis(vp, ax, comp, ndc);
            tested++;
            if (back === null) { nulls++; continue; }
            const err = Math.abs(back - v);
            if (err > worst) worst = err;
          }
        }
      }
    }
  }
}
console.log(`round-trips tested : ${tested}`);
console.log(`degenerate (null)  : ${nulls}`);
console.log(`worst error        : ${worst.toExponential(3)} mm   (model spans ~${(aabb.max[1] - aabb.min[1]).toFixed(0)} mm)`);
console.log(worst < 1e-6 ? "PASS — inverse is exact" : "FAIL — inverse is wrong");

// Axis->screen-direction classification: at azimuth 0 / elevation 0 we look down -Z, so X must run
// across the screen, Y up it, Z into it. If this is wrong the rulers label the wrong axes.
const cropAxes = (vp) => {
  const c = centre(), base = projPoint(vp, c), d = [];
  for (let ax = 0; ax < 3; ax++) {
    const p = c.slice(); p[ax] += (aabb.max[ax] - aabb.min[ax]) * 0.25 || 1;
    const q = projPoint(vp, p);
    d.push([Math.abs(q[0] - base[0]), Math.abs(q[1] - base[1])]);
  }
  const mag = d.map(([x, y]) => Math.hypot(x, y));
  let depth = 0; for (let i = 1; i < 3; i++) if (mag[i] < mag[depth]) depth = i;
  const rest = [0, 1, 2].filter((i) => i !== depth);
  const h = d[rest[0]][0] >= d[rest[1]][0] ? rest[0] : rest[1];
  const vv = rest[0] === h ? rest[1] : rest[0];
  return { h, v: vv, depth };
};
const N = ["X", "Y", "Z"];
console.log("\naxis mapping by view (h = across screen, v = up screen, depth = into screen):");
for (const [name, az, el] of [["front", 0, 0], ["right", Math.PI / 2, 0], ["back", Math.PI, 0], ["top", 0, 1.5], ["3/4", 0.6, 0.22]]) {
  const m = cropAxes(orbitViewProj({ azimuth: az, elevation: el, distance: dist, target }, 1.777));
  console.log(`  ${name.padEnd(6)} h=${N[m.h]} v=${N[m.v]} depth=${N[m.depth]}`);
}
