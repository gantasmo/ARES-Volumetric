import { test } from "node:test";
import assert from "node:assert/strict";
import {
  robustFit, fitTileField, smoothTileFields, evalTileField, normalGradients, gradientVote, poissonDetail, median3,
} from "../dist/depth-metric.js";

/** Deterministic uniform [0, 1). */
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

const near = (got, want, tol, what) => assert.ok(Math.abs(got - want) <= tol, `${what}: got ${got}, want ${want} +- ${tol}`);

/**
 * A surface seen through a pinhole: `zAt(u, v)` is the depth along the ray of map point (u, v).
 * Returns the depth at every pixel centre and the exact surface normals quantized to int8 (n * 127),
 * oriented toward the camera (OpenCV axes: x right, y down, z forward).
 */
function pinholeSurface(W, H, K, zAt) {
  const z = new Float32Array(W * H), normals = new Int8Array(W * H * 3);
  const h = 1e-3;
  const pt = (u, v) => { const d = zAt(u, v); return [d * (u - K.cx) / K.fx, d * (v - K.cy) / K.fy, d]; };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const u = x + 0.5, v = y + 0.5, i = y * W + x;
    const pu1 = pt(u + h, v), pu0 = pt(u - h, v), pv1 = pt(u, v + h), pv0 = pt(u, v - h);
    const Pu = [0, 1, 2].map((k) => (pu1[k] - pu0[k]) / (2 * h)), Pv = [0, 1, 2].map((k) => (pv1[k] - pv0[k]) / (2 * h));
    let n = [Pu[1] * Pv[2] - Pu[2] * Pv[1], Pu[2] * Pv[0] - Pu[0] * Pv[2], Pu[0] * Pv[1] - Pu[1] * Pv[0]];
    const l = Math.hypot(n[0], n[1], n[2]) * (n[2] > 0 ? -1 : 1);
    n = n.map((c) => c / l);
    for (let k = 0; k < 3; k++) normals[i * 3 + k] = Math.round(n[k] * 127);
    z[i] = zAt(u, v);
  }
  return { z, normals };
}

/**
 * Estimate and metric target related by a spatially varying affine field: z = a*e + b (linear) or
 * 1/z = a*e + b (disparity), a and b ramps across the frame, target noise +-2 mm.
 */
function affineScene(kind, W, H, seed = 3) {
  const P = W * H, rnd = lcg(seed);
  const est = new Float32Array(P), target = new Float32Array(P), truth = new Float32Array(P), valid = new Uint8Array(P).fill(1);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, u = x / W, v = y / H;
    const e = 0.5 + 0.2 * Math.sin((2 * Math.PI * x) / 23) * Math.sin((2 * Math.PI * y) / 19) + 0.1 * u;
    let z;
    if (kind === "linear") z = (-2 - 1.0 * u + 0.5 * v) * e + (4 + 0.6 * v - 0.3 * u);
    else z = 1 / ((0.5 + 0.1 * u - 0.05 * v) * e + (0.05 + 0.05 * v));
    est[i] = e; truth[i] = z; target[i] = z + (rnd() - 0.5) * 0.004;
  }
  return { est, target, truth, valid };
}

const median = (arr) => { const s = Float64Array.from(arr).sort(); return s[s.length >> 1]; };

/* ----------------------------------------- robustFit ----------------------------------------- */

test("robustFit: recovers a and b for both models with 20 % outliers", () => {
  const n = 2000;
  for (const kind of ["linear", "disparity"]) {
    const rnd = lcg(kind === "linear" ? 11 : 12);
    const d = new Float64Array(n), z = new Float64Array(n);
    const A = kind === "linear" ? -3 : 0.3, B = kind === "linear" ? 5 : 0.15;
    for (let i = 0; i < n; i++) {
      d[i] = 0.2 + 0.7 * rnd();
      const zt = kind === "linear" ? A * d[i] + B : 1 / (A * d[i] + B);
      z[i] = zt + (rnd() - 0.5) * 0.004;
      if (i % 5 === 0) z[i] = zt * (rnd() < 0.5 ? 0.6 + 0.25 * rnd() : 1.15 + 0.5 * rnd());   // 20 % gross outliers
    }
    const f = robustFit(d, z, n, kind);
    assert.ok(f, `${kind}: a fit`);
    assert.equal(f.kind, kind);
    near(f.a, A, Math.abs(A) * 0.01, `${kind} a`);
    near(f.b, B, Math.abs(B) * 0.01, `${kind} b`);
    assert.ok(f.medres < 0.002, `${kind}: inlier medres ${f.medres} m`);
    assert.ok(f.n >= 0.78 * n && f.n <= 0.82 * n, `${kind}: ${f.n} inliers of ${n}, the outliers are 400`);
  }
  // Non-finite samples are skipped, not fitted.
  const d = [0.1, 0.2, 0.3, NaN, 0.5, 0.6], z = [1.1, 1.2, 1.3, 1.4, Infinity, 1.6];
  const f = robustFit(d, z, 6, "linear");
  near(f.a, 1, 1e-9, "a over the finite samples");
  near(f.b, 1, 1e-9, "b over the finite samples");
  assert.equal(f.n, 4);
});

test("robustFit: null below 3 samples or without spread; bad arguments name the value", () => {
  assert.equal(robustFit([0.1, 0.2], [1, 2], 2, "linear"), null);
  assert.equal(robustFit([0.4, 0.4, 0.4, 0.4], [1, 2, 3, 4], 4, "linear"), null);
  assert.equal(robustFit([0.1, 0.2, 0.3, 0.4], [1, -2, 3, -4], 4, "disparity"), null, "z <= 0 has no disparity: 2 samples left");
  assert.throws(() => robustFit([1, 2, 3], [1, 2, 3], 4, "linear"), /n must be an integer in \[0, 3\], got 4/);
  assert.throws(() => robustFit([1, 2, 3], [1, 2, 3], 3, "inverse"), /kind must be .* got inverse/);
  assert.throws(() => robustFit([1, 2, 3], [1, 2, 3], 3, "linear", -1), /floor must be a finite number >= 0, got -1/);
});

/* ---------------------------------------- tile field ----------------------------------------- */

test("fitTileField: a spatially varying affine field is recovered; the tiled residual is under a quarter of the global one", () => {
  const W = 424, H = 240;
  for (const kind of ["linear", "disparity"]) {
    const s = affineScene(kind, W, H);
    const f = fitTileField(s.est, s.target, s.valid, W, H);
    assert.ok(f, `${kind}: a field`);
    assert.equal(f.kind, kind, `${kind}: the model competition picks the generating model`);
    assert.equal(f.tile, 36, "default tile = round(72 * W / 848)");
    assert.equal(f.tilesX, 12); assert.equal(f.tilesY, 7);
    assert.equal(f.samples, (W / 2) * (H / 2), "stride 2 on both axes");
    assert.ok(f.global.medres > 0.01, `${kind}: one affine cannot follow the field (global medres ${f.global.medres} m)`);
    assert.ok(f.medresTiled < 0.25 * f.global.medres, `${kind}: tiled ${f.medresTiled} m vs global ${f.global.medres} m`);
    // medresGlobal is over every sample, the set medresTiled is over; the inlier figure is below it.
    assert.ok(f.medresGlobal >= f.global.medres && f.medresTiled < 0.25 * f.medresGlobal, `${kind}: global ${f.medresGlobal} m over every sample`);
    const z = evalTileField(f, s.est, s.valid);
    const err = new Float64Array(W * H);
    for (let i = 0; i < W * H; i++) err[i] = Math.abs(z[i] - s.truth[i]);
    const m = median(err);
    assert.ok(Math.abs(m - f.medresTiled) < 0.004, `${kind}: evaluated field matches the fit residual (${m} vs ${f.medresTiled})`);
  }
});

test("fitTileField: too few samples is null; evalTileField zeroes masked, invalid and non-positive cells", () => {
  const W = 64, H = 48;
  const s = affineScene("linear", W, H);
  const sparse = new Uint8Array(W * H);
  for (let i = 0; i < 300; i++) sparse[i] = 1;          // 300 pixels, 75 on the stride-2 lattice
  assert.equal(fitTileField(s.est, s.target, sparse, W, H), null);
  assert.throws(() => fitTileField(s.est, s.target, s.valid, W, H, { tile: 0 }), /tile must be an integer >= 1, got 0/);
  assert.throws(() => fitTileField(s.est, s.target.subarray(0, 10), s.valid, W, H), /target holds 10 values, expected 3072/);
  const f = fitTileField(s.est, s.target, s.valid, W, H);
  assert.ok(f);
  const est = s.est.slice();
  est[5] = NaN;
  const mask = s.valid.slice();
  mask[7] = 0;
  const z = evalTileField(f, est, mask);
  assert.equal(z[5], 0); assert.equal(z[7], 0);
  assert.ok(z[9] > 0);
  const neg = { ...f, a: new Float32Array(f.a.length).fill(0), b: new Float32Array(f.b.length).fill(-1) };
  assert.ok(evalTileField(neg, s.est, null).every((v) => v === 0), "z <= 0 is no depth");
});

test("smoothTileFields: a one-frame spike in one tile's a is removed; null and other-model frames are filled", () => {
  const W = 424, H = 240;
  const s = affineScene("linear", W, H);
  const base = fitTileField(s.est, s.target, s.valid, W, H);
  const T = 11, j = 3 * base.tilesX + 5, k = 2 * base.tilesX + 8;
  const drift = (t) => 1 + 0.001 * t;                     // slow drift the smoothing must keep
  const frames = [];
  for (let t = 0; t < T; t++) {
    frames.push({ ...base, a: base.a.map((v) => v * drift(t)), b: base.b.slice(), w: base.w.slice(), global: { ...base.global } });
  }
  frames[5].a[j] *= 1.25;                                 // the spike
  frames[8] = null;                                       // a frame without a fit
  frames[2] = { ...frames[2], kind: "disparity", a: new Float32Array(base.a.length).fill(0.3), b: new Float32Array(base.a.length).fill(0.1),
    global: { ...base.global, kind: "disparity" } };      // a frame where the other model won
  const out = smoothTileFields(frames, 3);
  assert.equal(out.length, T);
  for (let t = 0; t < T; t++) {
    assert.ok(out[t], `frame ${t} present`);
    assert.equal(out[t].kind, "linear");
    for (const tile of [j, k]) {
      const want = base.a[tile] * drift(t);
      assert.ok(Math.abs(out[t].a[tile] - want) <= 0.002 * Math.abs(want), `frame ${t} tile ${tile}: a ${out[t].a[tile]} vs ${want}`);
      assert.ok(Math.abs(out[t].b[tile] - base.b[tile]) <= 1e-5 * Math.abs(base.b[tile]), `frame ${t} tile ${tile}: b`);
    }
  }
  assert.equal(out[8].samples, 0, "a filled frame carries no samples");
  assert.equal(out[8].global.n, 0);
  assert.equal(out[2].samples, 0, "the other-model frame is filled, not averaged");
  assert.equal(out[5].samples, base.samples);
  assert.notEqual(out[5].a, frames[5].a, "new arrays");
  assert.ok(Math.abs(frames[5].a[j] - base.a[j] * drift(5) * 1.25) < 1e-6, "input untouched");
  // radius 0: fill only, the spike stays.
  const raw = smoothTileFields(frames, 0);
  near(raw[5].a[j], base.a[j] * drift(5) * 1.25, 1e-5, "radius 0 keeps the spike");
  assert.deepEqual(smoothTileFields([null, null], 2), [null, null]);
  assert.throws(() => smoothTileFields(frames, 1.5), /radius must be an integer >= 0, got 1.5/);
  const other = fitTileField(s.est, s.target, s.valid, W, H, { tile: 40 });
  assert.throws(() => smoothTileFields([base, other], 2), /frame 1 has tile 40/);
});

/* --------------------------------------- normal detail --------------------------------------- */

const K160 = { fx: 150, fy: 150, cx: 80, cy: 60 };

test("normalGradients + poissonDetail: a 1 cm Gaussian dent on a 2 m plane is recovered from its normals with the right sign", () => {
  const W = 160, H = 120, sigma = 4, u0 = 80.5, v0 = 60.5;
  for (const amp of [0.01, -0.01]) {                      // a dent (away from the camera) and a bump
    const zAt = (u, v) => 2 + amp * Math.exp(-((u - u0) ** 2 + (v - v0) ** 2) / (2 * sigma * sigma));
    const { z: truth, normals } = pinholeSurface(W, H, K160, zAt);
    const zb = new Float32Array(W * H).fill(2);           // the base lacks the dent
    const g = normalGradients(normals, W, H, K160);
    assert.ok(g.ok.every((v) => v === 1), "every normal usable");
    const { r, iters, rms } = poissonDetail(zb, W, H, g);
    assert.ok(iters > 0 && iters <= 120, `iters ${iters}`);
    assert.ok(rms > 0);
    const c = (v0 - 0.5) * W + (u0 - 0.5);                // the pixel whose centre is the dent centre
    const got = zb[c] * Math.exp(r[c]) - 2;
    assert.ok(Math.sign(got) === Math.sign(amp), `amp ${amp}: recovered ${got} m at the centre`);
    assert.ok(Math.abs(got) >= 0.6 * Math.abs(amp) && Math.abs(got) <= 1.1 * Math.abs(amp), `amp ${amp}: recovered ${got} m of ${amp}`);
    // Shape: the recovered residual correlates with the true log residual over the dent.
    let sxy = 0, sxx = 0, syy = 0;
    for (let y = 40; y < 80; y++) for (let x = 60; x < 100; x++) {
      const i = y * W + x, t = Math.log(truth[i] / 2), q = r[i];
      sxy += t * q; sxx += t * t; syy += q * q;
    }
    assert.ok(sxy / Math.sqrt(sxx * syy) > 0.9, `correlation ${sxy / Math.sqrt(sxx * syy)}`);
  }
});

test("poissonDetail: a flat plane is left untouched, fronto-parallel exactly and tilted within quantization", () => {
  const W = 160, H = 120;
  const fronto = pinholeSurface(W, H, K160, () => 2);
  const r0 = poissonDetail(fronto.z, W, H, normalGradients(fronto.normals, W, H, K160));
  assert.ok(r0.r.every((v) => v === 0), "fronto-parallel: r = 0");
  assert.equal(r0.iters, 0);
  // Plane through (0, 0, 2) tilted 30 degrees about x: n = (0, -sin 30, -cos 30).
  const nY = -Math.sin(Math.PI / 6), nZ = -Math.cos(Math.PI / 6);
  const tilted = pinholeSurface(W, H, K160, (u, v) => (2 * nZ) / (nY * (v - K160.cy) / K160.fy + nZ));
  const { r } = poissonDetail(tilted.z, W, H, normalGradients(tilted.normals, W, H, K160));
  let mx = 0;
  for (const v of r) mx = Math.max(mx, Math.abs(v));
  assert.ok(mx < 5e-4, `tilted: max |r| ${mx} (1 mm at 2 m)`);
});

test("poissonDetail: rms is over the solved cells, not diluted by base cells without an edge", () => {
  const W = 160, H = 120, sigma = 4, u0 = 40.5, v0 = 60.5;
  const zAt = (u, v) => 2 + 0.01 * Math.exp(-((u - u0) ** 2 + (v - v0) ** 2) / (2 * sigma * sigma));
  const { normals } = pinholeSurface(W, H, K160, zAt);
  const zb = new Float32Array(W * H).fill(2);
  const g = normalGradients(normals, W, H, K160);
  for (let y = 0; y < H; y++) for (let x = W / 2; x < W; x++) g.ok[y * W + x] = 0;   // right half: base, no gradient
  const { r, rms } = poissonDetail(zb, W, H, g);
  let ss = 0, n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W / 2; x++) { ss += r[y * W + x] ** 2; n++; }
  for (let y = 0; y < H; y++) for (let x = W / 2; x < W; x++) assert.equal(r[y * W + x], 0, "no edge, no residual");
  near(rms, Math.sqrt(ss / n), 1e-9, "rms over the left half's cells");
});

test("gradientVote: positive for the right flipY, negative for the wrong one", () => {
  const W = 160, H = 120;
  const nY = -Math.sin(Math.PI / 6), nZ = -Math.cos(Math.PI / 6);
  const { z, normals } = pinholeSurface(W, H, K160, (u, v) => (2 * nZ) / (nY * (v - K160.cy) / K160.fy + nZ));
  const right = gradientVote(z, normalGradients(normals, W, H, K160, false), W, H);
  const wrong = gradientVote(z, normalGradients(normals, W, H, K160, true), W, H);
  assert.ok(right > 0, `right convention: ${right}`);
  assert.ok(wrong < 0, `flipped convention: ${wrong}`);
  // The global sign of the normals cancels: the same map negated votes the same way.
  const neg = normals.map((v) => -v);
  assert.ok(gradientVote(z, normalGradients(neg, W, H, K160, false), W, H) > 0, "negated normals, same vote");
});

test("poissonDetail: detail does not bleed across a 5 cm depth jump", () => {
  const W = 120, H = 80, jumpX = 60, u0 = 53.5, v0 = 40.5, sigma = 3;
  const K = { fx: 110, fy: 110, cx: 60, cy: 40 };
  // Left half at 2.00 m with a 1 cm dent next to the jump, right half a flat 2.05 m.
  const zAt = (u, v) => (u < jumpX ? 2 + 0.01 * Math.exp(-((u - u0) ** 2 + (v - v0) ** 2) / (2 * sigma * sigma)) : 2.05);
  const { normals } = pinholeSurface(W, H, K, zAt);
  const zb = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) zb[y * W + x] = x < jumpX ? 2 : 2.05;
  const g = normalGradients(normals, W, H, K);
  const { r } = poissonDetail(zb, W, H, g);
  let right = 0;
  for (let y = 0; y < H; y++) for (let x = jumpX; x < W; x++) right = Math.max(right, Math.abs(r[y * W + x]));
  assert.equal(right, 0, "the right half is its own system with no detail: r = 0");
  assert.ok(r[(v0 - 0.5) * W + (u0 - 0.5)] > 0.002, "the dent on the left is recovered");
  // Keeping the jump edges (edgeDrop 0.1 m) makes the solve fight the 5 cm step: detail bleeds.
  const kept = poissonDetail(zb, W, H, g, { edgeDrop: 0.1 });
  let bleed = 0;
  for (let y = 0; y < H; y++) bleed = Math.max(bleed, Math.abs(kept.r[y * W + jumpX]));
  assert.ok(bleed > 0.002, `with the jump edges kept the right half moves by ${bleed}`);
  assert.throws(() => poissonDetail(zb, W, H, g, { lambda: 0 }), /lambda must be a finite number > 0, got 0/);
  assert.throws(() => poissonDetail(zb, W, H, g, { iters: -1 }), /iters must be an integer >= 0, got -1/);
});

/* ------------------------------------------ median3 ------------------------------------------ */

test("median3: a one-frame spike is removed, a step is kept, 0 is missing", () => {
  const prev = Float32Array.from([1, 1, 1, 0, 1, -0.01, 3]);
  const cur = Float32Array.from([9, 2, 1, 5, 0, 0.02, 3]);
  const next = Float32Array.from([1, 2, 2, 6, 1, -0.012, 0]);
  const out = median3(prev, cur, next);
  assert.deepEqual([...out].map((v) => +v.toFixed(4)), [1, 2, 1, 5, 0, -0.01, 3]);
  // spike -> neighbour value; step up at t and at t+1 both kept; missing neighbour -> own value;
  // missing centre stays missing; signed residuals; missing next -> own value.
  assert.deepEqual([...median3(null, cur, next)], [...cur], "clip start: the frame passes through");
  assert.deepEqual([...median3(prev, cur, null)], [...cur], "clip end: the frame passes through");
  const inPlace = cur.slice();
  assert.equal(median3(prev, inPlace, next, inPlace), inPlace);
  assert.deepEqual([...inPlace], [...out], "out may alias cur");
  assert.throws(() => median3(prev.subarray(0, 3), cur, next), /prev holds 3 values, cur 7/);
});

/* ------------------------------------------- timing ------------------------------------------ */

test("fitTileField + poissonDetail on a 512x288 frame (timing reported, loose ceiling)", (t) => {
  const W = 512, H = 288, P = W * H;
  const K = { fx: 460, fy: 460, cx: 256, cy: 144 };
  const s = affineScene("disparity", W, H, 5);
  // Subject: an upright ellipse, a third of the frame.
  const mask = new Uint8Array(P);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const ex = (x + 0.5 - W / 2) / (W * 0.18), ey = (y + 0.5 - H / 2) / (H * 0.62);
    mask[y * W + x] = ex * ex + ey * ey < 1 ? 1 : 0;
  }
  const { normals } = pinholeSurface(W, H, K, (u, v) => 3 + 0.02 * Math.sin(u / 5) * Math.sin(v / 7));
  const g = normalGradients(normals, W, H, K);
  const runs = [];
  let f, d;
  for (let rep = 0; rep < 6; rep++) {
    const t0 = performance.now();
    f = fitTileField(s.est, s.target, mask, W, H);
    const zb = evalTileField(f, s.est, mask);
    d = poissonDetail(zb, W, H, g);
    runs.push(performance.now() - t0);
  }
  const warm = runs.slice(1).sort((a, b) => a - b);
  const ms = warm[warm.length >> 1];
  t.diagnostic(`512x288, subject ${(100 * mask.reduce((a, v) => a + v, 0) / P).toFixed(1)} %: fit + eval + solve ${ms.toFixed(1)} ms (median of ${warm.length} warm), ${d.iters} CG iterations`);
  assert.ok(f && d.iters > 0);
  assert.ok(ms < 600, `${ms} ms; budget 150 ms, ceiling 4x for a loaded machine`);
});
