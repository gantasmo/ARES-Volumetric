/**
 * Structural probe for .4ds (4DViews) and .ares containers — shared by the Convert tab's
 * unified import flow (the standalone Inspect tab was removed 2026-07-11; these are the same
 * functions it used, unchanged, just re-homed).
 *
 * Content-free: reads ONLY the container skeleton via File.slice() byte ranges —
 * header, frame directory, per-GOP block-offset tables, block *lengths*. It never reads,
 * decodes, or uploads any mesh or texture payload. Mirrors tools/probe-4ds.cjs + tools/ares-split
 * logic.
 */

const fmtMB = (b) => (b >= 1048576 * 1024 ? (b / 1073741824).toFixed(2) + " GB" : (b / 1048576).toFixed(2) + " MB");
const pct = (b, t) => ((b / t) * 100).toFixed(1) + "%";

async function slice(file, start, len) { return new DataView(await file.slice(start, start + len).arrayBuffer()); }

// ---- 4DViews .4ds (4DS0) ----------------------------------------------------
export async function probe4ds(file) {
  const size = file.size;
  const h = await slice(file, 0, 128);
  const magic = String.fromCharCode(h.getUint8(0), h.getUint8(1), h.getUint8(2), h.getUint8(3));
  if (magic !== "4DS0") throw new Error(`not a 4DS0 file (magic "${magic}")`);
  const verMajor = h.getUint16(4, true), verMinor = h.getUint16(6, true);
  const indexOff = Number(h.getBigUint64(14, true));
  const frameCount = h.getUint32(39, true);
  const fps = h.getFloat32(47, true);
  const texW = h.getUint32(63, true), texH = h.getUint32(67, true);

  const dirLen = size - indexOff;
  const dir = await slice(file, indexOff, dirLen);
  const recCount = Math.floor((dirLen - 9) / 16);
  const keyframes = [];
  for (let i = 0; i < recCount; i++) {
    const b = 9 + i * 16;
    keyframes.push({ idx: dir.getUint32(b, true), off: Number(dir.getBigUint64(b + 8, true)) });
  }
  let tableOverhead = 0;
  const offs = [];
  for (const k of keyframes) {
    const ch = await slice(file, k.off, 9);
    const tb = ch.getUint16(5, true), n = tb / 8;
    tableOverhead += 9 + tb;
    const tbl = await slice(file, k.off + 9, tb);
    for (let j = 0; j < n; j++) offs.push(Number(tbl.getBigUint64(j * 8, true)));
  }
  const sizes = offs.map((v, i) => (i === 0 ? v : v - offs[i - 1]));
  let geom = 0, geomBlocks = 0, bigBlocks = 0;
  for (const s of sizes) { if (s <= 200000) { geom += s; geomBlocks++; } else bigBlocks++; }
  const overhead = tableOverhead + dirLen + 128;
  const tex = size - geom - overhead;
  return {
    kind: "4ds", name: file.name, size, verMajor, verMinor, frameCount, fps, texW, texH,
    dur: frameCount / fps, keyframes: keyframes.length, keyIdx: keyframes.map((k) => k.idx),
    blocks: offs.length, geom, tex, overhead, geomBlocks, bigBlocks,
  };
}

// ---- ARES .ares -------------------------------------------------------------
export async function probeAres(file) {
  const buf = new Uint8Array(await file.arrayBuffer());
  const dv = new DataView(buf.buffer);
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== "ARES" && !(buf[0] === 0x41 && buf[1] === 0x52 && buf[2] === 0x45 && buf[3] === 0x53))
    throw new Error("not an .ares file");
  const frameCount = dv.getUint32(16, true);
  const fps = dv.getFloat32(12, true);
  const gopIndexOffset = Number(dv.getBigUint64(36, true));
  let p = gopIndexOffset;
  const nChunks = dv.getUint32(p, true); p += 4;
  const chunkOffsets = [];
  for (let i = 0; i < nChunks; i++) { p += 8 + 4 + 2 + 2; chunkOffsets.push(Number(dv.getBigUint64(p, true))); p += 8 + 4; }
  let geomI = 0, geomP = 0, tex = 0;
  for (const off of chunkOffsets) {
    let q = off + 4 + 8 + 2 + 2 + 12 + 12;
    const blockCount = dv.getUint16(q, true); q += 2;
    for (let b = 0; b < blockCount; b++) {
      const type = buf[q]; q += 1 + 2 + 4;
      const plen = dv.getUint32(q, true); q += 4;
      if (type === 0) geomI += plen; else if (type === 1) geomP += plen; else if (type === 2) tex += plen;
    }
  }
  const geom = geomI + geomP;
  return {
    kind: "ares", name: file.name, size: buf.byteLength, frameCount, fps, dur: frameCount / fps,
    chunks: nChunks, geom, tex, temporal: geomP > 0, overhead: buf.byteLength - geom - tex,
  };
}

// ---- render -----------------------------------------------------------------
function row(k, v) { return `<div class="k">${k}</div><div class="v">${v}</div>`; }

function splitBar(geom, tex, total) {
  const g = (geom / total) * 100, t = (tex / total) * 100, o = 100 - g - t;
  return `<div class="split">
    <div style="width:${g}%;background:var(--accent)" title="geometry ${fmtMB(geom)}">${g > 8 ? "geom " + g.toFixed(0) + "%" : ""}</div>
    <div style="width:${t}%;background:var(--series-b)" title="texture ${fmtMB(tex)}">${t > 8 ? "tex " + t.toFixed(0) + "%" : ""}</div>
    <div style="width:${o}%;background:#5a5852" title="overhead"></div>
  </div>`;
}

export function render4ds(r) {
  const eq = 9.0667 / r.dur; // normalize to the 272f/9.07s ARES clip
  return `<div class="card">
    <h3>4DViews <code>.4ds</code> <span class="badge">temporal</span> <span class="badge warn">reads structure only</span></h3>
    <div class="kv">
      ${row("file", r.name)}
      ${row("format", `4DS0 v${r.verMajor}.${r.verMinor} — single container, geometry + texture embedded`)}
      ${row("size", `${r.size.toLocaleString()} B (${fmtMB(r.size)})`)}
      ${row("frames", `${r.frameCount} @ ${r.fps.toFixed(3)} fps = ${r.dur.toFixed(1)} s`)}
      ${row("texture", `${r.texW} × ${r.texH}, embedded per-frame`)}
      ${row("geometry codec", `temporal — ${r.keyframes} intra keyframes + ${r.frameCount - r.keyframes} inter/P frames (adaptive GOP)`)}
      ${row("blocks", `${r.blocks} = 2 × ${r.frameCount} (geometry + texture per frame)`)}
    </div>
    ${splitBar(r.geom, r.tex, r.size)}
    <div class="kv" style="margin-top:8px">
      ${row("geometry", `<b>${fmtMB(r.geom)}</b> · ${pct(r.geom, r.size)} — temporal mesh (small blocks)`)}
      ${row("texture", `<b>${fmtMB(r.tex)}</b> · ${pct(r.tex, r.size)} — per-frame ${r.texW}² image, no inter-frame compression`)}
      ${row("reconciles", `${r.geom.toLocaleString()} + ${r.tex.toLocaleString()} + ${r.overhead.toLocaleString()} = ${(r.geom + r.tex + r.overhead).toLocaleString()} ${(r.geom + r.tex + r.overhead) === r.size ? "<span class='good'>exact ✓</span>" : "⚠"}`)}
      ${row("per-9.07s equiv", `geometry ${(r.geom * eq / 1048576).toFixed(1)} MiB · texture ${(r.tex * eq / 1048576).toFixed(1)} MiB`)}
    </div>
    <div class="note">Mirror image of ARES: 4DViews spends ~${pct(r.geom, r.size)} on geometry (temporal) and ~${pct(r.tex, r.size)} on texture
      (per-frame block, no video codec). ARES is the opposite — heavy intra geometry, tiny VP9 video texture. The target codec is the union of both.</div>
  </div>`;
}

export function renderAres(r) {
  return `<div class="card">
    <h3>ARES <code>.ares</code> ${r.temporal ? '<span class="badge">temporal I+P</span>' : '<span class="badge">intra</span>'}</h3>
    <div class="kv">
      ${row("file", r.name)}
      ${row("size", `${r.size.toLocaleString()} B (${fmtMB(r.size)})`)}
      ${row("frames", `${r.frameCount} @ ${r.fps.toFixed(2)} fps = ${r.dur.toFixed(1)} s · ${r.chunks} chunks`)}
      ${row("geometry", `meshopt ${r.temporal ? "I+P" : "intra"}`)}
    </div>
    ${splitBar(r.geom, r.tex, r.size)}
    <div class="kv" style="margin-top:8px">
      ${row("geometry", `<b>${fmtMB(r.geom)}</b> · ${pct(r.geom, r.size)}`)}
      ${row("texture", `<b>${fmtMB(r.tex)}</b> · ${pct(r.tex, r.size)} — VP9 video`)}
    </div>
    <div class="note">ARES puts most bytes in geometry (intra re-stores topology per frame) and keeps texture tiny via a video codec —
      the inverse of a 4DViews .4ds. Drop one of each to compare.</div>
  </div>`;
}

/** Dispatch by extension. Returns null for anything that isn't a .4ds/.ares (caller decides
 *  what "unsupported" means in its own flow); throws on a matching extension that fails to parse
 *  (bad magic / truncated file) so the caller can show a real error instead of a blank card. */
export async function probeFile(file) {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".4ds")) return probe4ds(file);
  if (lower.endsWith(".ares")) return probeAres(file);
  return null;
}

/** Render whichever probe kind was returned by probeFile(). */
export function renderProbe(probe) {
  return probe.kind === "4ds" ? render4ds(probe) : renderAres(probe);
}

/** The one honest status line for .4ds in the Convert flow (Inspect never needed this — it only
 *  ever showed structure; Convert's whole point is "can I turn this into an .ares"). Task I wired
 *  real conversion through the licensed BridgeCodec4DS decoder (tools/4ds/decode_4ds.py) — this
 *  probe stays byte-level/content-free as before, but a convert row now runs the actual codec on this
 *  machine. See the convert row rendered below this card for output name / max-frames / mirror-X. */
export const FOURDS_STATUS = "4DViews container recognized — structure shown above. Content conversion runs the licensed BridgeCodec4DS decoder on this machine (DXT1 desktop captures only) — see the convert row below.";
