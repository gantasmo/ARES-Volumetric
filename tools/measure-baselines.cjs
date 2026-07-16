/**
 * measure-baselines.cjs — produce the size-comparison numbers for a volcap capture using
 * ONLY real measurements. Nothing here is derived from how ARES compresses; each format is
 * measured (or, for 4DViews, a clearly-labelled native-bitrate estimate).
 *
 *   raw   : sum the source OBJ (geometry) and PNG (texture) folders on disk.
 *   draco : encode every OBJ frame with real draco3d (glTF-standard pos14/uv12/nrm10 edgebreaker)
 *           and sum the bytes; texture = the capture's own PNG atlases (a per-frame glTF/Draco
 *           sequence ships per-frame images — Draco compresses geometry only).
 *   ares  : walk the .ares container and sum payload bytes by block type (geometry vs VP9 texture).
 *   4dv   : 4DViews .4ds is proprietary and can't be produced here; estimated at a stated bitrate.
 *
 * Usage:  node tools/measure-baselines.cjs [captureDir] [ares file] [--sample N] [--mbps M]
 * Default captureDir = ../Daniel_Microsoft_Volcap/Daniel_Volcap, ares = apps/demo/daniel.ares.
 */
const fs = require("fs");
const path = require("path");
const draco3d = require("draco3d");

const ROOT = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const flag = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && args[i - 1].startsWith("--")));

const CAPTURE = positional[0] || path.resolve(ROOT, "..", "Daniel_Microsoft_Volcap", "Daniel_Volcap");
const ARES = positional[1] || path.resolve(ROOT, "apps", "demo", "daniel.ares");
const SAMPLE = Number(flag("--sample", 1));       // 1 = every frame
// 4DViews is tier-dependent: ~2 Mbps @ 720p streaming (anecdotal, unverified) up to ~16 Mbps
// standard (~120 MB / 60 s, published). Both are lower/near ARES's fidelity (1024² + full mesh), so
// not fidelity-matched. Default = 720p tier; pass --mbps 16 for standard.
const MBPS_4DV = Number(flag("--mbps", 2));
// A REAL .4ds measured by DATA RATE (no content read — size ÷ duration). Defaults are a measured
// DESKTOP_HR file (238,222,244 B / 15.2 s @ 29.99 fps = 125 Mbps). Pass --4ds-bytes / --4ds-dur to
// use your own. This is the HR tier whose fidelity is comparable-to-above ARES.
const FDS_BYTES = Number(flag("--4ds-bytes", 238222244));
const FDS_DUR = Number(flag("--4ds-dur", 15.2));
const FDS_LABEL = flag("--4ds-label", "DESKTOP_HR");
const FPS = 30;

const MB = (b) => b / 1048576;
const fmt = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : mb.toFixed(1) + " MB");

// ---- raw folder sizes -------------------------------------------------------
function folderBytes(dir, ext) {
  let bytes = 0, n = 0;
  for (const f of fs.readdirSync(dir)) {
    if (!f.toLowerCase().endsWith(ext)) continue;
    bytes += fs.statSync(path.join(dir, f)).size; n++;
  }
  return { bytes, n };
}

// ---- real Draco geometry ----------------------------------------------------
function parseObj(text) {
  const pos = [], uv = [], nrm = [], faces = [];
  for (const l of text.split("\n")) {
    const c0 = l.charCodeAt(0);
    if (c0 === 118) { // v
      const c1 = l.charCodeAt(1);
      const p = l.split(/\s+/);
      if (c1 === 32) pos.push(+p[1], +p[2], +p[3]);
      else if (c1 === 116) uv.push(+p[1], +p[2]);
      else if (c1 === 110) nrm.push(+p[1], +p[2], +p[3]);
    } else if (c0 === 102 && l.charCodeAt(1) === 32) {
      const p = l.split(/\s+/);
      faces.push((p[1].split("/")[0] | 0) - 1, (p[2].split("/")[0] | 0) - 1, (p[3].split("/")[0] | 0) - 1);
    }
  }
  return { pos, uv, nrm, faces };
}

async function dracoGeom(dir) {
  const enc = await draco3d.createEncoderModule({});
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".obj")).sort();
  const picked = files.filter((_, i) => i % SAMPLE === 0);
  let bytes = 0;
  for (const f of picked) {
    const { pos, uv, nrm, faces } = parseObj(fs.readFileSync(path.join(dir, f), "utf8"));
    const nPoints = pos.length / 3, nFaces = faces.length / 3;
    const mesh = new enc.Mesh(), mb = new enc.MeshBuilder();
    mb.AddFacesToMesh(mesh, nFaces, new Uint32Array(faces));
    mb.AddFloatAttributeToMesh(mesh, enc.POSITION, nPoints, 3, new Float32Array(pos));
    if (uv.length) mb.AddFloatAttributeToMesh(mesh, enc.TEX_COORD, nPoints, 2, new Float32Array(uv));
    if (nrm.length) mb.AddFloatAttributeToMesh(mesh, enc.NORMAL, nPoints, 3, new Float32Array(nrm));
    const encoder = new enc.Encoder();
    encoder.SetAttributeQuantization(enc.POSITION, 14);
    encoder.SetAttributeQuantization(enc.TEX_COORD, 12);
    encoder.SetAttributeQuantization(enc.NORMAL, 10);
    encoder.SetSpeedOptions(5, 5);
    encoder.SetEncodingMethod(enc.MESH_EDGEBREAKER_ENCODING);
    const out = new enc.DracoInt8Array();
    const len = encoder.EncodeMeshToDracoBuffer(mesh, out);
    if (len <= 0) throw new Error("Draco encode failed for " + f);
    bytes += len;
    enc.destroy(out); enc.destroy(encoder); enc.destroy(mb); enc.destroy(mesh);
  }
  return { bytes: bytes * (files.length / picked.length), frames: files.length, measured: picked.length };
}

// ---- ARES container split ---------------------------------------------------
function aresSplit(file) {
  const buf = fs.readFileSync(file);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const gopIndexOffset = Number(dv.getBigUint64(36, true));
  const frameCount = dv.getUint32(16, true);
  let p = gopIndexOffset;
  const nChunks = dv.getUint32(p, true); p += 4;
  const chunks = [];
  for (let i = 0; i < nChunks; i++) {
    p += 8; p += 4; p += 2; p += 2;
    const off = Number(dv.getBigUint64(p, true)); p += 8;
    p += 4;
    chunks.push(off);
  }
  let geom = 0, tex = 0;
  for (const off of chunks) {
    let q = off + 4 + 8 + 2 + 2 + 12 + 12;
    const blockCount = dv.getUint16(q, true); q += 2;
    for (let b = 0; b < blockCount; b++) {
      const type = buf[q]; q += 1 + 2 + 4;
      const plen = dv.getUint32(q, true); q += 4;
      if (type === 0 || type === 1) geom += plen;
      else if (type === 2) tex += plen;
    }
  }
  return { total: buf.byteLength, geom, tex, frameCount };
}

(async () => {
  const objs = folderBytes(CAPTURE, ".obj");
  const pngs = folderBytes(CAPTURE, ".png");
  const rawTotal = objs.bytes + pngs.bytes;
  console.log(`capture: ${CAPTURE}`);
  console.log(`Draco: encoding ${objs.n} frames (sample every ${SAMPLE})…`);
  const draco = await dracoGeom(CAPTURE);
  const ares = fs.existsSync(ARES) ? aresSplit(ARES) : null;
  const dur = objs.n / FPS;
  const fourdv = (MBPS_4DV / 8) * dur;                              // streaming-tier estimate
  const fourdvHR = MB(FDS_BYTES / FDS_DUR * dur);                   // real .4ds data rate → this clip
  const fourdvHRMbps = FDS_BYTES * 8 / FDS_DUR / 1e6;

  const R = (a, b) => (a / b).toFixed(1);
  console.log("\n=== SIZE BASELINES (all measured except 4DViews) ===");
  console.log(`frames ${objs.n}  ·  ${dur.toFixed(1)}s @ ${FPS}fps\n`);
  const line = (name, geom, tex, total, files, tag) =>
    console.log(`  ${name.padEnd(16)} geom ${geom.padStart(9)}  tex ${tex.padStart(9)}  total ${total.padStart(9)}  ${String(files).padStart(4)} files  ${tag}`);
  line("Raw OBJ+PNG", fmt(MB(objs.bytes)), fmt(MB(pngs.bytes)), fmt(MB(rawTotal)), objs.n + pngs.n, "measured");
  line("Draco-GLB seq", fmt(MB(draco.bytes)), fmt(MB(pngs.bytes)), fmt(MB(draco.bytes + pngs.bytes)), draco.frames, `measured (${draco.measured} enc)`);
  line("4DViews 720p", "—", "—", fmt(fourdv), 1, `est. @ ${MBPS_4DV} Mbps (streaming, lower fidelity)`);
  line(`4DViews ${FDS_LABEL}`, "—", "—", fmt(fourdvHR), 1, `MEASURED data rate ${fourdvHRMbps.toFixed(0)} Mbps (fidelity ≥ ARES)`);
  if (ares) line("ARES", fmt(MB(ares.geom)), fmt(MB(ares.tex)), fmt(MB(ares.total)), 1, "measured");

  if (ares) {
    console.log("\n=== HEADLINE RATIOS ===");
    console.log(`  ARES total vs Raw:        ${R(rawTotal, ares.total)}× smaller`);
    console.log(`  ARES total vs Draco-GLB:  ${R(draco.bytes + pngs.bytes, ares.total)}× smaller`);
    console.log(`  Geometry — Draco vs ARES: ${R(ares.geom, draco.bytes)}× smaller (Draco wins geometry)`);
    console.log(`  Texture  — ARES vs PNG:   ${R(pngs.bytes, ares.tex)}× smaller (ARES wins texture)`);
    console.log(`  4DViews 720p vs ARES:     ${R(MB(ares.total), fourdv)}× smaller than ARES (but lower fidelity, est.)`);
    console.log(`  ARES vs 4DViews ${FDS_LABEL}:  ${R(fourdvHR, MB(ares.total))}× smaller than 4DViews HR (fidelity-matched, MEASURED)`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
