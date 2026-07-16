/**
 * Convert tab — one unified import: drop/pick a folder of per-frame OBJ/PLY meshes + PNG atlases
 * (analysed, then run through the REAL encoder on this machine via the dev server's /encode
 * endpoint — streamed progress, no command line, produced .ares opens straight in the Viewer),
 * OR drop/pick a single .ares/.4ds file (structurally probed in-browser, findings shown inline —
 * this absorbs the former standalone Inspect tab, removed 2026-07-11: same probe functions,
 * now imported from ./probe.js instead of driving their own tab).
 */
import { initHistoryPanel, recordHistory } from "./history.js";
import { depStatus } from "./settings.js";
import { probeFile, renderProbe, FOURDS_STATUS } from "./probe.js";

const $ = (id) => document.getElementById(id);
const MB = (b) => (b >= 1073741824 ? (b / 1073741824).toFixed(2) + " GB" : (b / 1048576).toFixed(1) + " MB");
let history = null;

// Recursively gather File objects from a drop (handles folders via webkitGetAsEntry).
async function fromEntry(entry, out, depth) {
  if (depth > 8) return;
  if (entry.isFile) { await new Promise((res) => entry.file((f) => { out.push(f); res(); }, res)); return; }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    await new Promise((res) => {
      const readBatch = () => reader.readEntries(async (ents) => {
        if (!ents.length) return res();
        for (const e of ents) await fromEntry(e, out, depth + 1);
        readBatch();
      }, res);
      readBatch();
    });
  }
}
async function gather(dt) {
  const out = [];
  const items = dt.items ? [...dt.items] : [];
  const entries = items.map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (entries.length) { for (const e of entries) await fromEntry(e, out, 0); return out; }
  return [...dt.files];
}

async function pngDims(file) {
  try { const b = new DataView(await file.slice(0, 24).arrayBuffer()); return [b.getUint32(16), b.getUint32(20)]; }
  catch { return null; }
}
async function objVerts(file) {
  try { const t = await file.slice(0, Math.min(file.size, 3_000_000)).text(); return (t.match(/^v /gm) || []).length; }
  catch { return 0; }
}

let lastState = null;

// Normalise a drag-dropped File[] into the same stats shape /analyse returns (path unknown here —
// the browser can't see absolute paths, so drag-drop is preview-only until a folder is picked).
async function statsFromFiles(files) {
  const objs = files.filter((f) => /\.obj$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const plys = files.filter((f) => /\.ply$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const pngs = files.filter((f) => /\.png$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const meshes = objs.length ? objs : plys;
  if (!meshes.length) return null;
  const [sampleVerts, atlasDims] = await Promise.all([objs.length ? objVerts(meshes[0]) : 0, pngs.length ? pngDims(pngs[0]) : null]);
  const rel = (meshes[0].webkitRelativePath || "").split("/")[0] || "";
  return {
    meshes: meshes.length, kind: objs.length ? "OBJ" : "PLY", pngs: pngs.length, atlasDims,
    sampleVerts, rawBytes: files.reduce((s, f) => s + f.size, 0), files: files.length,
    folderHint: rel, name: (rel || "converted").replace(/[^a-z0-9._-]/gi, "_"), path: "",
  };
}

async function analyse(files) {
  const stats = await statsFromFiles(files);
  if (!stats) { $("convertOut").innerHTML = `<div class="card"><h3>No meshes found</h3><div class="note">Drop a folder containing per-frame <b>.obj</b> or <b>.ply</b> files (plus <b>atlas-*.png</b> textures) — or a single <b>.ares</b>/<b>.4ds</b> file to inspect it.</div></div>`; return; }
  renderConvertCard(stats);
}

// ---- unified import: one drop zone / picker routes to either the frames-folder analyse flow
// (unchanged, above) or the structural probe (.ares/.4ds — absorbed from the former Inspect tab).
// A folder drop always yields many files, so "exactly one file, and it's a container we probe"
// is an unambiguous signal — real frame folders never look like this.
function renderProbeOut(probe) {
  let html = renderProbe(probe);
  if (probe.kind === "4ds") html += `<div class="note" style="margin-top:8px;color:var(--warn)">${FOURDS_STATUS}</div>`;
  return html;
}

async function handleProbeFile(file) {
  const out = $("convertOut");
  out.innerHTML = `<div class="card"><h3>Reading ${file.name}…</h3><div class="note">structure only, on your machine</div></div>`;
  try {
    const probe = await probeFile(file);
    if (!probe) { out.innerHTML = `<div class="card"><h3>Unsupported</h3><div class="note">Drop a .4ds/.ares file to inspect it, or a frames folder to convert it.</div></div>`; return; }
    out.innerHTML = renderProbeOut(probe);
    if (probe.kind === "4ds") {
      const defaultName = file.name.replace(/\.4ds$/i, "").replace(/[^a-z0-9_-]/gi, "_").slice(0, 60) || "converted";
      out.insertAdjacentHTML("beforeend", fourdsConvertRowHtml(defaultName));
      wireFourdsConvertRow();
    }
    // Probes never leave the browser, so the history entry carries the full result — clicking
    // the entry later re-renders this card without re-reading the (path-less) local file.
    recordHistory({
      kind: "inspect", name: file.name,
      meta: { format: probe.kind === "4ds" ? "4DViews .4ds" : "ARES", frames: probe.frameCount, sizeMB: Math.round(probe.size / 1048576), probe },
    }).then(() => history && history.refresh());
  } catch (e) {
    out.innerHTML = `<div class="card"><h3 style="color:#f0a3a3">Could not parse ${file.name}</h3><div class="note">${e && e.message ? e.message : e}</div></div>`;
  }
}

// ---- .4ds → .ares conversion row (Task I): the structural probe above is browser-only (byte
// ranges via File.slice, no filesystem path — content never leaves the browser), but decode_4ds.py needs
// a real path on disk. So this row has its own native file picker (mirrors pickAndAnalyse's
// /pick?type=folder pattern, just type=file) to resolve one, then calls /probe-4ds to get real
// frame/fps/texture info straight from the codec and gate the Convert button on it. The path
// input is plain and editable (like #cvPath above) so it can also be filled without the native
// dialog — e.g. by automation.
const inputCss = "flex:1;min-width:0;padding:7px 9px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:7px;color:var(--text);font:12px ui-monospace,monospace";
function fourdsConvertRowHtml(defaultName) {
  return `
    <div class="card" id="cv4dsCard">
      <h3>Convert this .4ds on this machine</h3>
      <div class="kv" style="grid-template-columns:120px 1fr;gap:8px 12px">
        <div class="k">source path</div><div style="display:flex;gap:6px;align-items:center">
          <input id="cv4dsPath" placeholder="click Locate… to pick this file on disk (needed to decode it)" style="${inputCss}">
          <button class="btn ghost" id="cv4dsPickBtn" style="margin:0;padding:6px 10px;white-space:nowrap">📁 Locate…</button>
        </div>
        <div class="k">output name</div><div><input id="cv4dsName" value="${defaultName}" style="width:200px;padding:7px 9px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:7px;color:var(--text);font:12px system-ui">.ares</div>
        <div class="k">max frames</div><div><input id="cv4dsMax" type="number" min="1" placeholder="all" style="width:90px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)"> <small style="color:var(--text-faint)">full clip decodes at ~2 fps — a 455-frame capture takes ~4 min</small></div>
        <div class="k">mirror X</div><div><label style="font:12px system-ui;color:var(--text-mid);cursor:pointer"><input type="checkbox" id="cv4dsMirror"> <span title="left-handed → right-handed: negates X and reverses triangle winding together">fix left/right-mirrored bakes</span></label></div>
      </div>
      <div class="note" id="cv4dsCodecNote" style="margin-top:6px">Pick the file on disk to check codec status and real frame count.</div>
      <div style="display:flex;gap:8px;align-items:center;margin-top:10px">
        <button class="btn" id="cv4dsGo" disabled>Convert on this machine</button>
      </div>
      <div class="prog" id="cv4dsProg"><div></div></div>
      <pre id="cv4dsLog" style="display:none;margin-top:12px;max-height:220px;overflow:auto;background:rgba(0,0,0,.35);border-radius:8px;padding:10px;font:11px ui-monospace,monospace;color:var(--text-mid);white-space:pre-wrap"></pre>
      <div id="cv4dsDone"></div>
    </div>`;
}

async function probe4dsPath(p) {
  const note = $("cv4dsCodecNote"), go = $("cv4dsGo"), maxInp = $("cv4dsMax");
  if (!note) return; // row was replaced by a newer import in the meantime
  note.innerHTML = "Checking codec status…"; note.style.color = "";
  go.disabled = true;
  let info;
  try { info = await fetch("/probe-4ds?path=" + encodeURIComponent(p)).then((r) => r.json()); }
  catch { note.textContent = "Probe failed — is the ARES dev server running?"; note.style.color = "#f0a3a3"; return; }
  if (info.error) {
    note.innerHTML = `⚠ ${info.error}${info.missing ? ": " + info.missing.join("; ") : ""} <a href="#" id="cv4dsSettingsLink">— ⚙ Settings</a>`;
    note.style.color = "var(--warn)";
    const link = $("cv4dsSettingsLink");
    if (link) link.onclick = (e) => { e.preventDefault(); const b = document.querySelector('#tabs button[data-tab="settings"]'); if (b) b.click(); };
    return;
  }
  note.innerHTML = `✓ ${info.nbFrames} frames @ ${info.framerate.toFixed(2)}fps · ${info.textureSize}² ${info.textureEncoding} — ready to convert` +
    (info.nbFrames >= 200 ? ` <small style="color:var(--text-faint)">(full clip ≈ ${(info.nbFrames / 2 / 60).toFixed(1)} min to decode at ~2 fps — use max frames to bound a test run)</small>` : "");
  note.style.color = "#63d68a";
  maxInp.placeholder = `all ${info.nbFrames}`;
  maxInp.max = String(info.nbFrames);
  go.disabled = false;
}

function wireFourdsConvertRow() {
  $("cv4dsPickBtn").onclick = async () => {
    let picked;
    try { picked = await fetch("/pick?type=file&filter=" + encodeURIComponent("4DViews captures (*.4ds)|*.4ds") + "&for=convert4ds").then((r) => r.json()); }
    catch { $("cv4dsCodecNote").innerHTML = `<span style="color:#f0a3a3">Picker needs the ARES dev server running.</span>`; return; }
    if (!picked || !picked.path) return; // cancelled
    $("cv4dsPath").value = picked.path;
    probe4dsPath(picked.path);
  };
  // Escape hatch (also used by automated tests): typing/pasting a path and leaving the field
  // probes it too, same as clicking Locate… — no native dialog required.
  $("cv4dsPath").addEventListener("change", () => { const p = $("cv4dsPath").value.trim(); if (p) probe4dsPath(p); });
  $("cv4dsGo").onclick = run4dsConvert;
}

async function run4dsConvert() {
  const p = $("cv4dsPath").value.trim();
  if (!p) return;
  const name = ($("cv4dsName").value.trim() || "converted").replace(/[^a-z0-9_-]/gi, "_");
  const maxFrames = $("cv4dsMax").value.trim();
  const mirrorX = $("cv4dsMirror").checked;
  const q = new URLSearchParams({ path: p, name });
  if (maxFrames) q.set("maxFrames", maxFrames);
  if (mirrorX) q.set("mirrorX", "1");

  // Overwrite guard: EventSource can't read an error body, so check the target name up front
  // and ask; the server 409s as the hard backstop (overwrite=1 bypasses both, deliberately).
  try {
    const existing = await fetch("/list-ares").then((r) => r.json());
    if (Array.isArray(existing) && existing.some((f) => f.src === name + ".ares")) {
      if (!confirm(`${name}.ares already exists in apps/demo — overwrite it?`)) return;
      q.set("overwrite", "1");
    }
  } catch { /* server will 409 if it exists */ }

  const log = $("cv4dsLog"), prog = $("cv4dsProg"), done = $("cv4dsDone"), go = $("cv4dsGo");
  log.style.display = "block"; prog.style.display = "block"; prog.firstChild.style.width = "4%"; prog.firstChild.style.background = "#d98a3a";
  done.innerHTML = ""; go.disabled = true; go.textContent = "Converting…";
  const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  line(`\n=== ${name}.ares ← ${p} (max ${maxFrames || "all"} frames${mirrorX ? ", mirror-X" : ""}) ===`);

  const es = new EventSource("/convert-4ds?" + q.toString());
  const finish = (ok, msg) => {
    es.close(); go.disabled = false; go.textContent = "Convert on this machine";
    if (ok) prog.firstChild.style.width = "100%";
    else { prog.firstChild.style.background = "#f0a3a3"; line("✗ " + msg); done.innerHTML = `<div class="note" style="color:#f0a3a3;margin-top:10px">✗ ${msg}</div>`; }
  };
  es.addEventListener("start", (e) => { const d = JSON.parse(e.data); line(`▶ decoding ${d.path.split(/[\\/]/).pop()} → ${d.out}`); });
  es.addEventListener("log", (e) => { line(JSON.parse(e.data)); });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    if (d.stage === "decode" && d.of) prog.firstChild.style.width = Math.max(4, Math.round((d.frame / d.of) * 70)) + "%";
    else if (d.stage === "encode") prog.firstChild.style.width = "80%";
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    finish(true);
    done.innerHTML = `<div class="note" style="color:#63d68a;margin-top:12px">✓ wrote ${d.out} (${d.frames ?? "?"} frames @ ${d.fps}fps)</div>
      <button class="btn" id="cv4dsOpen">Open in Viewer</button>
      <button class="btn ghost" id="cv4dsShowcase" style="margin-left:6px">★ Add to source bar</button>`;
    $("cv4dsOpen").onclick = () => { location.search = "?src=" + name + ".ares"; };
    $("cv4dsShowcase").onclick = async (ev) => { const ok = await addToShowcase(name + ".ares", name); ev.target.textContent = ok ? "★ Added" : "✗ failed"; ev.target.disabled = ok; };
    if (history) history.refresh();
  });
  es.addEventListener("error", (e) => {
    let msg = "conversion failed — is the ARES dev server running and the codec available? (check ⚙ Settings)";
    try { const d = JSON.parse(e.data); msg = d.message || msg; } catch { /* connection close */ }
    finish(false, msg);
  });
  es.onerror = () => { /* SSE stream closed by server */ };
}

/** Entry point for both drag-drop and the click-to-choose input. */
async function handleImport(files) {
  if (files.length === 1) {
    const lower = files[0].name.toLowerCase();
    if (lower.endsWith(".ares") || lower.endsWith(".4ds")) { await handleProbeFile(files[0]); return; }
  }
  analyse(files);
}

// Pick a folder natively (no typing), analyse it server-side, and render the card pre-filled.
// for=convert → the dialog reopens in the last folder picked here (server remembers it).
async function pickAndAnalyse() {
  let picked;
  try { picked = await fetch("/pick?type=folder&for=convert").then((r) => r.json()); }
  catch { $("convertOut").innerHTML = `<div class="card"><div class="note" style="color:#f0a3a3">Picker needs the ARES dev server running.</div></div>`; return; }
  if (!picked || !picked.path) return; // cancelled
  await analyseServer(picked.path);
}
async function analyseServer(dir) {
  const out = $("convertOut");
  out.innerHTML = `<div class="card"><div class="note">Analysing ${dir}…</div></div>`;
  let info;
  try { info = await fetch("/analyse?dir=" + encodeURIComponent(dir)).then((r) => r.json()); }
  catch (e) { out.innerHTML = `<div class="card"><h3>Analyse failed</h3><div class="note" style="color:#f0a3a3">${e.message || e}</div></div>`; return; }
  if (info.error) { out.innerHTML = `<div class="card"><h3>Couldn't read that folder</h3><div class="note" style="color:#f0a3a3">${info.error}</div></div>`; return; }
  if (!info.meshes) { out.innerHTML = `<div class="card"><h3>No meshes found</h3><div class="note">${info.dir} has no <b>.obj</b>/<b>.ply</b> frames.</div></div>`; return; }
  const base = info.dir.split(/[\\/]/).pop() || "converted";
  renderConvertCard({
    meshes: info.meshes, kind: info.kind, pngs: info.pngs, atlasDims: info.atlasDims,
    sampleVerts: info.verts, rawBytes: info.rawBytes, files: undefined,
    folderHint: base, name: base.replace(/[^a-z0-9._-]/gi, "_"), path: info.dir,
  });
  if (history) history.refresh();   // the server recorded this analyse
}

function renderConvertCard(stats) {
  const out = $("convertOut");
  const { meshes, kind, pngs, atlasDims, sampleVerts, rawBytes, folderHint, path } = stats;
  // Rough .ares estimate from measured daniel ratios: meshopt geom ~17.6 B/vert/frame (+normals),
  // VP9 texture ~55 KB/frame @1024². Clearly labelled as an estimate; real number comes from encoding.
  const geomEstMB = sampleVerts ? (sampleVerts * 17.6 * meshes) / 1048576 : (meshes * 0.19);
  const texEstMB = pngs >= meshes ? (meshes * 55) / 1024 : 0;
  const estMB = geomEstMB + texEstMB;
  lastState = { meshes, kind, pngs, folderHint, name: (folderHint || "converted").replace(/[^a-z0-9._-]/gi, "_"), path };

  out.innerHTML = `
    <div class="card">
      <h3>Detected sequence</h3>
      <div class="kv">
        <div class="k">meshes</div><div class="v">${meshes} × ${kind}${sampleVerts ? ` (~${(sampleVerts / 1000).toFixed(1)}k verts/frame)` : ""}</div>
        <div class="k">atlases</div><div class="v">${pngs} × PNG${atlasDims ? ` (${atlasDims[0]}×${atlasDims[1]})` : ""} ${pngs && pngs < meshes ? '<span class="badge warn">fewer than meshes</span>' : ""}</div>
        <div class="k">raw size</div><div class="v">${MB(rawBytes)}${stats.files ? ` · ${stats.files} files` : ""}</div>
        <div class="k">est. .ares</div><div class="v">~${estMB.toFixed(0)} MB <small style="color:var(--text-faint)">(geom ~${geomEstMB.toFixed(0)} + tex ~${texEstMB.toFixed(0)}; rough)</small></div>
      </div>
      <div class="note">Encoding runs the real pipeline on this machine (meshopt geometry + ${pngs >= meshes ? "VP9/AV1 video texture" : "geometry only"}).</div>

      <div style="margin-top:14px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
        <span class="k" style="color:var(--text-dim);font-size:12px;margin-right:4px">presets</span>
        <button class="btn ghost" style="margin:0;padding:6px 12px" data-preset="web">Web · VP9 1024² crf32</button>
        <button class="btn ghost" style="margin:0;padding:6px 12px" data-preset="balanced">Balanced ✨ · AV1 1024² crf30 · smooth 0</button>
        <button class="btn ghost" style="margin:0;padding:6px 12px" data-preset="hq">HQ · AV1 2048² crf26 <small>(~+2–3× texture)</small></button>
        <button class="btn" style="margin:0;padding:6px 12px" data-preset="cohA" title="coherent pre-pass with the exact per-texel bake + VP9 2048² crf28 smooth0 — the long-but-right one (~1–2 h for a full clip)">Coherent A ★ · exact bake · VP9 2048² crf28</button>
      </div>
      <div class="kv" style="grid-template-columns:168px 1fr;margin-top:12px;gap:8px 12px">
        <div class="k">folder</div><div style="display:flex;gap:6px;align-items:center">
          <input id="cvPath" value="${path || ""}" placeholder="${folderHint ? "…full path to " + folderHint : "click Choose… to pick a folder"}" style="flex:1;min-width:0;padding:7px 9px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:7px;color:var(--text);font:12px ui-monospace,monospace">
          <button class="btn ghost" id="cvPickBtn" style="margin:0;padding:6px 10px;white-space:nowrap">📁 Choose…</button>
        </div>
        <div class="k">output name</div><div><input id="cvName" value="${lastState.name}" style="width:180px;padding:7px 9px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:7px;color:var(--text);font:12px system-ui">.ares</div>
        <div class="k">coherent</div><div><label style="font:12px system-ui;color:var(--text-mid);cursor:pointer"><input type="checkbox" id="cvCoherent" checked> <span title="stable-template GOPs: registers one mesh per GOP and rebakes atlases into its UVs — kills texture boiling and cuts file size ~3.7× (default). Needs mesh-fNNNNN.obj + atlas-fNNNNN.png frames; adds a pre-pass (minutes on long clips).">stable-template GOPs — less boiling, much smaller (default)</span></label></div>
        <div class="k">texture</div><div>
          <select id="cvCodec" class="cvsel"><option value="av1">AV1 (smaller, HW)</option><option value="vp9">VP9</option></select>
          <select id="cvSize" class="cvsel"><option value="1024">1024²</option><option value="2048">2048² (sharper, bigger)</option></select>
          CRF <input id="cvCrf" type="number" value="30" min="10" max="50" style="width:56px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)">
        </div>
        <div class="k">smoothing</div><div><input id="cvSmooth" type="number" value="0" min="0" max="6" style="width:56px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)"> Taubin passes (weld-aware; 0 = keeper default)</div>
        <div class="k">decimate</div><div><input id="cvDecimate" type="number" placeholder="off" min="0.2" max="0.95" step="0.05" style="width:64px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)" title="keep this fraction of triangles; atlas seams stay locked (0.6 measured: geometry −35%)"> ratio of triangles kept (blank = off)</div>
        <div class="k">max frames</div><div><input id="cvMax" type="number" placeholder="all" min="1" style="width:80px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)"></div>
      </div>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,.08)">
        <div class="cap" style="font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;color:var(--text-dim);margin-bottom:6px">Enhance texture (AI) — optional pre-step</div>
        <div class="kv" style="grid-template-columns:168px 1fr;gap:8px 12px">
          <div class="k">tier</div><div>
            <select id="enTier" class="cvsel"><option value="ncnn">Fast — Real-ESRGAN (no server, ~few s/frame)</option><option value="sd">Generative — SD img2img (Forge, auto-starts; minutes/frame)</option></select>
          </div>
          <div class="k">strength</div><div><input id="enStrength" type="range" min="0" max="100" value="70" style="width:180px;vertical-align:middle"> <span id="enStrengthVal" style="font:12px ui-monospace,monospace;color:var(--text-mid)">70%</span></div>
          <div class="k">scale</div><div>
            <select id="enScale" class="cvsel"><option value="2">2× (2048→4096)</option><option value="1">1× (re-detail only)</option></select>
            <small style="color:var(--text-faint)">SD tier runs at source resolution (scale ignored)</small>
          </div>
          <div class="k">max frames</div><div><input id="enMax" type="number" placeholder="all" min="1" style="width:80px;padding:6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:6px;color:var(--text)"></div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap">
          <button class="btn ghost" id="enGo" style="margin:0;padding:6px 14px">Enhance frames</button>
          <button class="btn ghost" id="enForge" style="margin:0;padding:6px 12px;display:none">⚡ Pre-start Forge</button>
        </div>
        <div class="note" style="margin-top:6px" id="enNote"><b>Fast</b> uses a bundled Real-ESRGAN (Vulkan) — no server, no terminal, fits your GPU. Writes enhanced atlas PNGs to a sibling folder and repoints the folder above at them, so “Convert on this machine” encodes the enhanced frames.</div>
        <div class="prog" id="enProg"><div></div></div>
        <pre id="enLog" style="display:none;margin-top:10px;max-height:160px;overflow:auto;background:rgba(0,0,0,.35);border-radius:8px;padding:10px;font:11px ui-monospace,monospace;color:var(--text-mid);white-space:pre-wrap"></pre>
        <div id="enDone"></div>
      </div>

      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn" id="cvGo">Convert on this machine</button>
        <button class="btn ghost" id="cvQueueAdd" style="margin-top:14px">＋ Add to batch</button>
      </div>
      <div id="cvQueue" style="margin-top:10px"></div>
      <div class="prog" id="cvProg"><div></div></div>
      <pre id="cvLog" style="display:none;margin-top:12px;max-height:220px;overflow:auto;background:rgba(0,0,0,.35);border-radius:8px;padding:10px;font:11px ui-monospace,monospace;color:var(--text-mid);white-space:pre-wrap"></pre>
      <div id="cvDone"></div>
    </div>`;
  $("cvGo").onclick = () => runEncode();
  $("cvQueueAdd").onclick = addToQueue;
  $("cvPickBtn").onclick = pickAndAnalyse;
  $("enGo").onclick = runEnhance;
  $("enForge").onclick = prewarmForge;
  $("enStrength").oninput = () => { $("enStrengthVal").textContent = $("enStrength").value + "%"; };
  const enTier = $("enTier");
  const syncTier = () => {
    const sd = enTier.value === "sd";
    $("enForge").style.display = sd ? "inline-block" : "none";
    $("enNote").innerHTML = sd
      ? "<b>Generative</b> re-imagines each frame with SD img2img via Forge — Forge <b>auto-starts</b> the first time (headless, ~30–60 s cold start), no terminal. Needs a checkpoint on your Forge install."
      : "<b>Fast</b> uses a bundled Real-ESRGAN (Vulkan) — no server, no terminal, fits your GPU. Writes enhanced atlas PNGs to a sibling folder and repoints the folder above at them, so “Convert on this machine” encodes the enhanced frames.";
  };
  enTier.onchange = syncTier; syncTier();
  const PRESETS = {
    web: { codec: "vp9", size: "1024", crf: "32", smooth: "0" },
    balanced: { codec: "av1", size: "1024", crf: "30", smooth: "0" },   // keeper recipe (smooth 0)
    hq: { codec: "av1", size: "2048", crf: "26", smooth: "0" },
    // The Coherent A recipe, exact: vp9 2048² crf28 smooth0 + coherent pre-pass (exact bake is the
    // runner default; the smeary "fast" bake is CLI-only now).
    cohA: { codec: "vp9", size: "2048", crf: "28", smooth: "0", coherent: true },
  };
  for (const b of out.querySelectorAll("[data-preset]")) b.onclick = () => {
    const p = PRESETS[b.dataset.preset];
    $("cvCodec").value = p.codec; $("cvSize").value = p.size; $("cvCrf").value = p.crf; $("cvSmooth").value = p.smooth;
    if (p.coherent !== undefined && $("cvCoherent")) $("cvCoherent").checked = !!p.coherent;
  };
  // Settings copied from a History entry before any folder was picked apply as soon as the card exists.
  if (pendingSettings) { applySettingsToCard(pendingSettings); pendingSettings = null; }
}

// ---- copy/transpose conversion settings between clips -----------
// Every GUI encode records its full recipe in History meta; "⧉ Use settings" re-applies that
// recipe to the card so ANY clip can be converted/baked with another clip's exact settings.
let pendingSettings = null;
function applySettingsToCard(m) {
  const set = (id, v) => { const el = $(id); if (el && v != null && v !== "") el.value = v; };
  set("cvCodec", m.codec); set("cvSize", m.texSize); set("cvCrf", m.crf);
  if (m.smooth != null && $("cvSmooth")) $("cvSmooth").value = m.smooth;         // "0" is a real value (keeper)
  if ($("cvDecimate")) $("cvDecimate").value = m.decimate || "";                 // recipe match: absent = off
  if ($("cvMax")) $("cvMax").value = m.maxFrames || "";
  if ($("cvCoherent")) $("cvCoherent").checked = m.coherent === "1";            // pre-coherent entries → unchecked (accurate)
}
function useSettings(meta) {
  if ($("cvCodec")) { applySettingsToCard(meta); window.scrollTo(0, 0); }
  else {
    pendingSettings = meta;
    $("convertOut").innerHTML = `<div class="note" style="margin:8px 0">⧉ settings copied (${meta.coherent === "1" ? "coherent " : ""}${meta.codec || "?"} ${meta.texSize || "?"}² crf${meta.crf || "?"} smooth${meta.smooth ?? "?"}) — pick a folder above and they'll be applied.</div>`;
  }
}

// ---- batch queue: collect several folder/settings jobs, run them sequentially -------------
const queue = [];
function currentJob() {
  const path = $("cvPath").value.trim();
  if (!path) { $("cvPath").focus(); $("cvPath").style.borderColor = "#f0a3a3"; return null; }
  return {
    path,
    name: ($("cvName").value.trim() || "converted").replace(/[^a-z0-9._-]/gi, "_"),
    codec: $("cvCodec").value, size: $("cvSize").value, crf: $("cvCrf").value,
    smooth: $("cvSmooth").value, max: $("cvMax").value, decimate: $("cvDecimate").value,
    coherent: $("cvCoherent")?.checked ? "1" : "",
  };
}
function renderQueue() {
  const host = $("cvQueue");
  if (!queue.length) { host.innerHTML = ""; return; }
  host.innerHTML = `<div class="cap" style="font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;color:var(--text-dim);margin-bottom:5px">Batch (${queue.length})</div>` +
    queue.map((j, i) => `<div style="display:flex;gap:8px;align-items:center;font:12px ui-monospace,monospace;color:var(--text-mid);padding:3px 0">
      <span style="color:${j.state === "done" ? "#63d68a" : j.state === "running" ? "var(--warn)" : j.state === "failed" ? "#f0a3a3" : "var(--text-faint)"}">${j.state === "done" ? "✓" : j.state === "running" ? "▶" : j.state === "failed" ? "✗" : "·"}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.name}.ares ← ${j.path}</span>
      ${j.state === "pending" ? `<button data-rm="${i}" style="background:none;border:0;color:var(--text-faint);cursor:pointer">✕</button>` : ""}
    </div>`).join("") +
    (queue.some((j) => j.state === "pending") ? `<button class="btn" id="cvRunAll" style="margin-top:8px">Convert all (${queue.filter((j) => j.state === "pending").length})</button>` : "");
  for (const b of host.querySelectorAll("[data-rm]")) b.onclick = () => { queue.splice(Number(b.dataset.rm), 1); renderQueue(); };
  const run = host.querySelector("#cvRunAll");
  if (run) run.onclick = runBatch;
}
function addToQueue() {
  const j = currentJob();
  if (!j) return;
  j.state = "pending";
  queue.push(j);
  renderQueue();
}
async function runBatch() {
  for (const j of queue) {
    if (j.state !== "pending") continue;
    j.state = "running"; renderQueue();
    try { await runEncode(j); j.state = "done"; }
    catch { j.state = "failed"; }
    renderQueue();
  }
}

/** Enhance the atlas PNGs via the dev server's /enhance SSE endpoint (local SD-Forge), then
 *  point #cvPath at the enhanced sibling folder so the normal encode picks it up. */
async function runEnhance() {
  const path = $("cvPath").value.trim();
  if (!path) { $("cvPath").focus(); $("cvPath").style.borderColor = "#f0a3a3"; return; }
  // Graceful dependency gate: warn with a download pointer instead of failing mid-run.
  if ($("enTier").value === "ncnn") {
    const dep = await depStatus("realesrgan").catch(() => null);
    if (dep && !dep.present) {
      $("enDone").innerHTML = `<div class="note" style="color:var(--warn);margin-top:8px">⚠ The Fast tier needs Real-ESRGAN (~50 MB), which is not installed.
        <a href="${dep.action.url}" target="_blank" rel="noopener">Download it here</a> and unzip into <code>ares/tools/bin/realesrgan-ncnn-vulkan/</code> — or check ⚙ Settings.</div>`;
      return;
    }
  }
  const q = new URLSearchParams({
    dir: path,
    tier: $("enTier").value,
    strength: String(Number($("enStrength").value) / 100),
    scale: $("enScale").value,
  });
  if ($("enMax").value) q.set("maxFrames", $("enMax").value);

  const log = $("enLog"), prog = $("enProg"), done = $("enDone"), go = $("enGo");
  log.style.display = "block"; prog.style.display = "block";
  prog.firstChild.style.width = "4%"; prog.firstChild.style.background = "#d98a3a";
  done.innerHTML = ""; go.disabled = true; go.textContent = "Enhancing…";
  const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  line(`\n=== enhance ${path} (${$("enTier").value} · ${$("enStrength").value}% · ${$("enScale").value}×) ===`);

  const es = new EventSource("/enhance?" + q.toString());
  const finish = (ok, msg) => {
    es.close(); go.disabled = false; go.textContent = "Enhance frames";
    if (ok) prog.firstChild.style.width = "100%";
    else { prog.firstChild.style.background = "#f0a3a3"; line("✗ " + msg); done.innerHTML = `<div class="note" style="color:#f0a3a3;margin-top:10px">✗ ${msg}</div>`; }
  };
  es.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    line(`▶ ${d.frames} frame(s) → ${d.out} via ${d.via}`);
  });
  es.addEventListener("log", (e) => line("· " + JSON.parse(e.data)));
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    line(`✓ ${d.file} (${(d.ms / 1000).toFixed(1)} s) — ${d.frame}/${d.of}`);
    prog.firstChild.style.width = Math.max(4, Math.round((d.frame / d.of) * 100)) + "%";
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    finish(true);
    $("cvPath").value = d.out;
    done.innerHTML = `<div class="note" style="color:#63d68a;margin-top:10px">✓ enhanced ${d.frames} frame(s) → <b>${d.out}</b><br>Folder path above now points at the enhanced frames — “Convert on this machine” will encode them.</div>`;
    if (history) history.refresh();
  });
  es.addEventListener("error", (e) => {
    let msg = "enhance failed — is the ARES dev server running and the folder path correct?";
    try { const d = JSON.parse(e.data); msg = d.message + (d.hint ? " — " + d.hint : ""); } catch { /* connection/404/503: keep generic */ }
    finish(false, msg);
  });
  es.onerror = () => { /* SSE stream closed by server */ };
}

/** Pre-start the local Forge (generative tier) so the first hero-frame enhance skips cold start. */
function prewarmForge() {
  const btn = $("enForge"), done = $("enDone");
  btn.disabled = true; const orig = btn.textContent; btn.textContent = "Starting Forge…";
  done.innerHTML = `<div class="note" style="margin-top:8px">⏳ starting Forge (headless, ~30–60 s)…</div>`;
  const es = new EventSource("/forge/start");
  const stop = (html) => { es.close(); btn.disabled = false; btn.textContent = orig; done.innerHTML = html; };
  es.addEventListener("log", (e) => { done.innerHTML = `<div class="note" style="margin-top:8px">⏳ ${JSON.parse(e.data)}</div>`; });
  es.addEventListener("done", () => stop(`<div class="note" style="margin-top:8px;color:#63d68a">✓ Forge ready — the Generative tier will use it.</div>`));
  es.addEventListener("error", (e) => { let m = "could not start Forge"; try { m = JSON.parse(e.data).message || m; } catch { /* connection close */ } stop(`<div class="note" style="margin-top:8px;color:#f0a3a3">✗ ${m}</div>`); });
  es.onerror = () => { /* SSE stream closed by server */ };
}

/** Add a finished conversion to the persistent source bar (main.js re-renders on the change event).
 *  main.js seeds /showcase with the defaults at page load, so the GET here returns the real list. */
async function addToShowcase(src, label) {
  try {
    const cur = await fetch("/showcase").then((r) => r.json());
    const list = Array.isArray(cur) ? cur : [];
    if (!list.some((e) => e.src === src)) list.push({ label: (label || src).replace(/\.ares$/i, "").slice(0, 60), src });
    const r = await fetch("/showcase", { method: "POST", body: JSON.stringify(list) });
    window.dispatchEvent(new CustomEvent("ares:showcase-changed"));
    return r.ok;
  } catch { return false; }
}

/** Run one encode job (from the form or the batch queue). Resolves on done, rejects on failure. */
function runEncode(job) {
  return new Promise((resolve, reject) => {
    const j = job ?? currentJob();
    if (!j) return reject(new Error("no folder path"));
    const q = new URLSearchParams({ dir: j.path, name: j.name, textureCodec: j.codec, texSize: j.size, crf: j.crf });
    if (Number(j.smooth) > 0) q.set("smooth", j.smooth);
    if (j.max) q.set("maxFrames", j.max);
    if (j.crop) q.set("crop", j.crop);
    if (j.decimate && Number(j.decimate) > 0 && Number(j.decimate) < 1) q.set("decimate", j.decimate);
    if (j.coherent) q.set("coherent", "1");

    const log = $("cvLog"), prog = $("cvProg"), done = $("cvDone"), go = $("cvGo");
    log.style.display = "block"; prog.style.display = "block"; prog.firstChild.style.width = "12%"; prog.firstChild.style.background = "#d98a3a";
    done.innerHTML = ""; go.disabled = true; go.textContent = "Converting…";
    const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
    line(`\n=== ${j.name}.ares ← ${j.path} (${j.coherent ? "coherent " : ""}${j.codec} ${j.size}² crf${j.crf} smooth${j.smooth}) ===`);

    const es = new EventSource("/encode?" + q.toString());
    let pulse = 12;
    const finish = (ok, msg) => {
      es.close(); go.disabled = false; go.textContent = "Convert on this machine";
      if (ok) {
        prog.firstChild.style.width = "100%";
        done.innerHTML = `<div class="note" style="color:#63d68a;margin-top:12px">✓ wrote ${msg}</div>
          <button class="btn" id="cvOpen">Open in Viewer</button>
          <button class="btn ghost" id="cvShowcase" style="margin-left:6px">★ Add to source bar</button>`;
        $("cvOpen").onclick = () => { location.search = "?src=" + j.name + ".ares"; };
        $("cvShowcase").onclick = async (e) => { const ok = await addToShowcase(j.name + ".ares", j.name); e.target.textContent = ok ? "★ Added" : "✗ failed"; e.target.disabled = ok; };
        if (history) history.refresh();
        resolve(msg);
      } else {
        prog.firstChild.style.background = "#f0a3a3";
        line("✗ " + msg);
        reject(new Error(msg));
      }
    };
    es.addEventListener("start", (e) => { line("▶ encode " + JSON.parse(e.data).args); });
    es.addEventListener("log", (e) => { line(JSON.parse(e.data)); pulse = Math.min(92, pulse + 3); prog.firstChild.style.width = pulse + "%"; });
    es.addEventListener("done", (e) => finish(true, JSON.parse(e.data).out));
    es.addEventListener("error", (e) => {
      let msg = "encode failed"; try { msg = JSON.parse(e.data).message || `exit ${JSON.parse(e.data).code}`; } catch { /* connection error */ }
      finish(false, msg);
    });
    es.onerror = () => { /* SSE stream closed by server */ };
  });
}

export function initConvert() {
  const drop = $("convertDrop"), input = $("convertFile"), probeInput = $("convertProbeFile");
  $("convertPick").onclick = pickAndAnalyse;   // native folder dialog — the primary, no-typing path
  const fourdsLink = $("fourdsSettingsLink");
  if (fourdsLink) fourdsLink.onclick = (e) => { e.preventDefault(); const b = document.querySelector('#tabs button[data-tab="settings"]'); if (b) b.click(); };
  input.addEventListener("change", () => { if (input.files.length) handleImport([...input.files]); });
  // Secondary picker: a single .ares/.4ds file. Separate <input> because the primary one carries
  // `webkitdirectory` (folder-only in Chrome) — a picker attribute can't do both at once. Drag-drop
  // on the one drop zone handles both cases already (gather() doesn't care what was dropped).
  if (probeInput) probeInput.addEventListener("change", () => { if (probeInput.files[0]) handleImport([probeInput.files[0]]); });
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { stop(e); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { stop(e); drop.classList.remove("over"); }));
  drop.addEventListener("drop", async (e) => { const files = await gather(e.dataTransfer); if (files.length) handleImport(files); });
  history = initHistoryPanel({
    host: $("convertHistory"),
    kinds: ["analyse", "encode", "enhance", "inspect"],
    actions: (item) => {
      const acts = [];
      if (item.kind === "encode" && item.out) acts.push({ label: "Play", run: () => { location.search = "?src=" + item.out.split("/").pop(); } });
      if (item.kind === "encode" && item.meta) acts.push({ label: "⧉ Use settings", run: () => useSettings(item.meta) });
      if (item.kind === "enhance" && item.out) acts.push({ label: "Use output", run: () => analyseServer(item.out) });
      if (item.kind === "inspect" && item.meta && item.meta.probe) acts.push({ label: "View", run: () => { $("convertOut").innerHTML = renderProbeOut(item.meta.probe); window.scrollTo(0, 0); } });
      if (item.path) acts.push({ label: "Re-analyse", run: () => { analyseServer(item.path); window.scrollTo(0, 0); } });
      return acts;
    },
  });
}
