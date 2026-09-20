/**
 * Convert tab — one unified import (one Open button, one drop zone): a folder of per-frame OBJ/PLY meshes + PNG atlases
 * (analysed, then run through the REAL encoder on this machine via the dev server's /encode
 * endpoint — streamed progress, no command line, produced .ares opens straight in the Viewer),
 * OR drop/pick a single .ares/.4ds file (structurally probed in-browser, findings shown inline —
 * this absorbs the former standalone Inspect tab, removed 2026-07-11: same probe functions,
 * now imported from ./probe.js instead of driving their own tab).
 */
import { initHistoryPanel, recordHistory } from "./history.js";
import { accessPrompt, collectFile, fetchJsonEnsuring, sseErrorData } from "./ensure.js";
import { probeFile, renderProbe, FOURDS_STATUS } from "./probe.js";
// 2D video → 2.5D: its own module (the menu.js/track.js precedent), ported from VJ-9000's depth
// source. convert.js only routes a picked or dropped video to it.
import { initDepthCard, renderDepthCard, isVideoName } from "./depth-card.js";

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
/** Files of a drop, plus the dropped item itself when there is exactly one: its name and, for a
 *  folder, its direct entry count (what /resolve-drop matches a folder on). A drag-drop File has
 *  no webkitRelativePath, so the folder name has to come from the entry. */
async function gather(dt) {
  const out = [];
  const items = dt.items ? [...dt.items] : [];
  const entries = items.map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return { files: [...dt.files], root: null };
  for (const e of entries) await fromEntry(e, out, 0);
  let root = null;
  if (entries.length === 1) {
    const e = entries[0];
    root = { name: e.name, isDir: !!e.isDirectory, entries: 0 };
    if (e.isDirectory) root.entries = await new Promise((res) => {
      const reader = e.createReader(); let n = 0;
      const batch = () => reader.readEntries((ents) => { if (!ents.length) return res(n); n += ents.length; batch(); }, () => res(n));
      batch();
    });
  }
  return { files: out, root };
}

/** Where a dropped item is on disk (serve.mjs /resolve-drop): { path } or { candidates }. */
async function locate(q) {
  const p = new URLSearchParams({ name: q.name, kind: q.kind });
  if (q.size) p.set("size", String(q.size));
  if (q.mtime) p.set("mtime", String(q.mtime));
  if (q.files) p.set("files", String(q.files));
  try { return await fetch("/resolve-drop?" + p).then((r) => r.json()); }
  catch { return { candidates: [] }; }
}
function showLocating(name) {
  $("convertOut").innerHTML = `<div class="card"><div class="cap">Locating</div><div class="note2">${name}</div></div>`;
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

// Normalise a drag-dropped File[] into the same stats shape /analyse returns. The browser gives no
// path; analyse() below locates the folder on disk before this preview is ever shown.
async function statsFromFiles(files, rootName = "") {
  const objs = files.filter((f) => /\.obj$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const plys = files.filter((f) => /\.ply$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const pngs = files.filter((f) => /\.png$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  const splats = files.filter((f) => /\.(spz|splat|sog|glb|gltf)$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  let meshes = objs.length ? objs : plys;
  let kind = objs.length ? "OBJ" : "PLY", splat = false;
  // Splat sequences (spec §6.8): a 3DGS PLY (header carries f_dc_0) or one SPZ/.splat/SOG/glTF per frame.
  if (!objs.length && plys.length) {
    try { const head = await plys[0].slice(0, 16384).text(); if (/f_dc_0/.test(head.slice(0, head.indexOf("end_header") >>> 0 || head.length))) { splat = true; kind = "3DGS PLY"; } } catch { /* mesh */ }
  } else if (!objs.length && splats.length) {
    const ext = (splats[0].name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    kind = ext === ".spz" ? "SPZ" : ext === ".splat" ? ".splat" : ext === ".sog" ? "SOG" : "glTF splat";
    meshes = splats; splat = true;
  }
  if (!meshes.length) return null;
  const [sampleVerts, atlasDims] = await Promise.all([objs.length ? objVerts(meshes[0]) : 0, pngs.length ? pngDims(pngs[0]) : null]);
  const rel = rootName || (meshes[0].webkitRelativePath || "").split("/")[0] || "";
  return {
    meshes: meshes.length, kind, pngs: pngs.length, atlasDims, splat, splatCount: kind === ".splat" ? Math.floor(meshes[0].size / 32) : 0,
    sampleVerts, rawBytes: files.reduce((s, f) => s + f.size, 0), files: files.length,
    folderHint: rel, name: (rel || "converted").replace(/[^a-z0-9._-]/gi, "_"), path: "",
  };
}

/** A dropped folder: located on disk, it is analysed exactly as a picked one (path filled). When
 *  the lookup finds no single match, the browser-side preview renders with the candidates. */
async function analyse(files, root) {
  const stats = await statsFromFiles(files, root && root.isDir ? root.name : "");
  if (!stats) { $("convertOut").innerHTML = `<div class="card"><div class="cap">No mesh or splat frames</div><div class="note2">Accepted inputs: ⋯ menu.</div></div>`; return; }
  if (stats.folderHint) {
    showLocating(stats.folderHint);
    const r = await locate({ name: stats.folderHint, kind: "dir", files: root && root.isDir ? root.entries : 0 });
    if (r.path) { await analyseServer(r.path); return; }
    if (r.candidates && r.candidates.length) stats.candidates = r.candidates.map((p) => ({ path: p }));
  }
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

async function handleProbeFile(file, path = "") {
  const out = $("convertOut");
  out.innerHTML = `<div class="card"><div class="cap">Probing ${file.name}</div></div>`;
  try {
    const probe = await probeFile(file);
    if (!probe) { out.innerHTML = `<div class="card"><div class="cap">Unsupported container</div><div class="note2">Accepted inputs: ⋯ menu.</div></div>`; return; }
    out.innerHTML = renderProbeOut(probe);
    if (probe.kind === "4ds") {
      const defaultName = file.name.replace(/\.4ds$/i, "").replace(/[^a-z0-9_-]/gi, "_").slice(0, 60) || "converted";
      out.insertAdjacentHTML("beforeend", fourdsConvertRowHtml(defaultName));
      wireFourdsConvertRow();
      const where = path || (await locate({ name: file.name, kind: "file", size: file.size, mtime: file.lastModified })).path;
      if (where && $("cv4dsPath")) { $("cv4dsPath").value = where; probe4dsPath(where); }
    }
    // Probes never leave the browser, so the history entry carries the full result — clicking
    // the entry later re-renders this card without re-reading the (path-less) local file.
    recordHistory({
      kind: "inspect", name: file.name,
      meta: { format: probe.kind === "4ds" ? "4DViews .4ds" : "ARES", frames: probe.frameCount, sizeMB: Math.round(probe.size / 1048576), probe },
    }).then(() => history && history.refresh());
  } catch (e) {
    out.innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Parse failed: ${file.name}</div><div class="note2">${e && e.message ? e.message : e}</div></div>`;
  }
}

// ---- .4ds → .ares conversion row (Task I): the structural probe above is browser-only (byte
// ranges via File.slice, no filesystem path — content never leaves the browser), but decode_4ds.py needs
// a real path on disk. So this row has its own native file picker (mirrors pickAndAnalyse's
// /pick?type=folder pattern, just type=file) to resolve one, then calls /probe-4ds to get real
// frame/fps/texture info straight from the codec and gate the Convert button on it. The path
// input is plain and editable (like #cvPath above) so it can also be filled without the native
// dialog — e.g. by automation.
function fourdsConvertRowHtml(defaultName) {
  return `
    <div class="card cv2" id="cv4dsCard">
      <section class="cvcol">
        <div class="cap">Source</div>
        <div class="fld"><span class="k">path</span><span class="row">
          <input id="cv4dsPath" class="inp" placeholder="path to .4ds" style="flex:1">
          <button class="u" id="cv4dsPickBtn" title="Pick the .4ds with the native dialog">Browse…</button>
        </span></div>
        <div class="note2" id="cv4dsCodecNote" style="margin-top:4px">no file selected</div>
      </section>
      <section class="cvcol">
        <div class="cap">Output</div>
        <div class="fld"><span class="k">name</span><span class="row">
          <input id="cv4dsName" class="inp" value="${defaultName}" style="flex:1;max-width:230px"><span class="suf">.ares</span></span></div>
        <div class="fld"><span class="k">frames</span><span class="row">
          <input id="cv4dsMax" class="inp" type="number" min="1" placeholder="all" style="width:72px" title="Decode runs at ~2 fps; bound a test run here."></span></div>
        <div class="fld"><span class="k">mirror X</span><span class="row">
          <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Left-handed to right-handed: negates X and reverses triangle winding together. Use on left/right-mirrored bakes.">
            <input type="checkbox" id="cv4dsMirror"><span>negate X, flip winding</span></label></span></div>
      </section>
      <div class="full">
        <div class="cvfoot">
          <button class="u primary" id="cv4dsGo" disabled>Convert</button>
          <div class="prog" id="cv4dsProg"><div></div></div>
        </div>
        <details class="sec"><summary>Log</summary><div class="body"><pre id="cv4dsLog" class="cvlog"></pre></div></details>
        <div id="cv4dsDone"></div>
      </div>
    </div>`;
}

async function probe4dsPath(p) {
  const note = $("cv4dsCodecNote"), go = $("cv4dsGo"), maxInp = $("cv4dsMax");
  if (!note) return; // row was replaced by a newer import in the meantime
  note.innerHTML = "reading codec status…"; note.style.color = "";
  go.disabled = true;
  // The probe installs what it lacks by itself: the Python environment through /install, the
  // licensed codec DLL through the native file dialog. Progress shows in this status line.
  let info;
  const status = (t) => { const n = $("cv4dsCodecNote"); if (n) n.textContent = String(t).trim().slice(0, 140); };
  try { info = await fetchJsonEnsuring("/probe-4ds?path=" + encodeURIComponent(p), { onLog: status, onStep: (s) => status(`${s.label}: ${s.index + 1} of ${s.total}`) }); }
  catch { note.textContent = "Probe failed: ARES dev server not reachable"; note.style.color = "var(--bad)"; return; }
  if (!$("cv4dsCodecNote")) return;
  if (info.error) {
    note.style.color = "var(--warn)";
    if (info.needsFile) {
      // The dialog was dismissed. The codec is licensed and cannot be fetched, so the row keeps a
      // control that reopens the dialog.
      note.innerHTML = `${info.error} · <button class="u" id="cv4dsLocate" style="padding:1px 8px">Locate…</button>`;
      $("cv4dsLocate").onclick = () => probe4dsPath(p);   // the probe reopens the dialog itself
    } else note.textContent = info.error;
    return;
  }
  note.innerHTML = `${info.nbFrames} frames · ${info.framerate.toFixed(2)} fps · ${info.textureSize}² ${info.textureEncoding}` +
    (info.nbFrames >= 200 ? ` <span title="Decode runs at ~2 fps; bound a test run with the frames field.">· ≈${(info.nbFrames / 2 / 60).toFixed(1)} min decode</span>` : "");
  note.style.color = "var(--good)";
  maxInp.placeholder = `all ${info.nbFrames}`;
  maxInp.max = String(info.nbFrames);
  go.disabled = false;
}

function wireFourdsConvertRow() {
  $("cv4dsPickBtn").onclick = async () => {
    let picked;
    try { picked = await fetch("/pick?type=file&filter=" + encodeURIComponent("4DViews captures (*.4ds)|*.4ds") + "&for=convert4ds").then((r) => r.json()); }
    catch { $("cv4dsCodecNote").innerHTML = `<span style="color:var(--bad)">Picker requires the ARES dev server</span>`; return; }
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

  // Name collisions never overwrite: the server versions the output (-v2, -v3 …) and reports
  // the name it actually wrote, so there is nothing to confirm here.

  const log = $("cv4dsLog"), prog = $("cv4dsProg"), done = $("cv4dsDone"), go = $("cv4dsGo");
  revealLog(log); prog.style.display = "block"; prog.firstChild.style.width = "4%"; prog.firstChild.style.background = "";
  done.innerHTML = ""; go.disabled = true; go.textContent = "Converting…";
  const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  line(`\n=== ${name}.ares ← ${p} (max ${maxFrames || "all"} frames${mirrorX ? ", mirror-X" : ""}) ===`);

  const es = new EventSource("/convert-4ds?" + q.toString());
  const finish = (ok, msg) => {
    es.close(); go.disabled = false; go.textContent = "Convert";
    if (ok) prog.firstChild.style.width = "100%";
    else { prog.firstChild.style.background = "var(--bad)"; line("✗ " + msg); done.innerHTML = `<div class="note2" style="color:var(--bad);margin-top:8px">${msg}</div>`; }
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
    done.innerHTML = `<div class="note2" style="color:var(--good);margin-top:10px">wrote ${d.out} · ${d.frames ?? "?"} frames · ${d.fps} fps</div>
      <button class="u" id="cv4dsOpen">Open in Viewer</button>
      <button class="u" id="cv4dsShowcase">Add to source bar</button>`;
    $("cv4dsOpen").onclick = () => { location.search = "?src=" + name + ".ares"; };
    $("cv4dsShowcase").onclick = async (ev) => { const ok = await addToShowcase(name + ".ares", name); ev.target.textContent = ok ? "Added" : "✗ failed"; ev.target.disabled = ok; };
    if (history) history.refresh();
  });
  es.addEventListener("error", async (e) => {
    const d = sseErrorData(e);
    finish(false, (d && d.message) || "decode failed");
    // The licensed codec DLL is the one input the server cannot fetch: collect it and run again.
    if (d && d.needsFile && (await collectFile(d.needsFile, { onLog: line }))) run4dsConvert();
  });
  es.onerror = () => { /* SSE stream closed by server */ };
}

/** Entry point for a drop. The browser withholds a dropped item's location, so the server locates
 *  it by name, size and time (/resolve-drop) and the flow continues as for a native pick. */
async function handleImport(files, root = null) {
  if (files.length === 1 && !(root && root.isDir)) {
    const f = files[0], lower = f.name.toLowerCase();
    if (lower.endsWith(".ares") || lower.endsWith(".4ds")) { await handleProbeFile(f); return; }
    if (isVideoName(lower)) {
      showLocating(f.name);
      const r = await locate({ name: f.name, kind: "file", size: f.size, mtime: f.lastModified });
      await renderDepthCard(r.path || "", { hint: f.name, candidates: r.candidates || [] });
      return;
    }
  }
  analyse(files, root);
}

// Open and a click on the drop zone: the native dialog (type=any), which returns the full path of a
// file or a folder (a browser dialog returns neither). openPath routes by what came back; a frame
// file picked from a sequence opens its folder.
const ANY_FILTER = "Supported|*.mp4;*.m4v;*.mov;*.webm;*.mkv;*.avi;*.mpg;*.mpeg;*.wmv;*.ares;*.4ds;*.obj;*.ply;*.spz;*.splat;*.sog;*.glb;*.gltf|All files|*.*";
async function pickAndOpen() {
  let picked;
  try { picked = await fetch("/pick?type=any&for=convert&filter=" + encodeURIComponent(ANY_FILTER)).then((r) => r.json()); }
  catch { $("convertOut").innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Picker requires the ARES dev server</div></div>`; return; }
  if (picked && picked.path) await openPath(picked.path);
}

/** A local .ares/.4ds read through /local-bytes byte ranges: the File surface probe.js uses
 *  (name, size, slice().arrayBuffer()), for a container opened by path. */
function remoteFile(path, size) {
  const name = path.split(/[\\/]/).pop();
  const url = "/local-bytes?path=" + encodeURIComponent(path);
  return {
    name, size,
    slice(start = 0, end = size) {
      const a = Math.max(0, Math.min(size, start)), b = Math.max(a, Math.min(size, end));
      return { arrayBuffer: async () => {
        if (b <= a) return new ArrayBuffer(0);
        const r = await fetch(url, { headers: { Range: `bytes=${a}-${b - 1}` } });
        if (!r.ok) throw new Error(`read ${name}: HTTP ${r.status}`);
        return r.arrayBuffer();
      } };
    },
  };
}

// Pick a folder natively (no typing), analyse it server-side, and render the card pre-filled.
// for=convert → the dialog reopens in the last folder picked here (server remembers it).
async function pickAndAnalyse() {
  let picked;
  try { picked = await fetch("/pick?type=folder&for=convert").then((r) => r.json()); }
  catch { $("convertOut").innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Picker requires the ARES dev server</div></div>`; return; }
  if (!picked || !picked.path) return; // cancelled
  await analyseServer(picked.path);
}
async function analyseServer(dir) {
  const out = $("convertOut");
  out.innerHTML = `<div class="card"><div class="cap">Analysing</div><div class="note2">${dir}</div></div>`;
  let info;
  try { info = await fetch("/analyse?dir=" + encodeURIComponent(dir)).then((r) => r.json()); }
  catch (e) { out.innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Analyse failed</div><div class="note2">${e.message || e}</div></div>`; return; }
  if (info.error) { out.innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Folder unreadable</div><div class="note2">${info.error}</div></div>`; return; }
  if (!info.meshes) { out.innerHTML = `<div class="card"><div class="cap">No mesh or splat frames</div><div class="note2">${info.dir}</div></div>`; return; }
  const base = info.dir.split(/[\\/]/).pop() || "converted";
  renderConvertCard({
    meshes: info.meshes, kind: info.kind, pngs: info.pngs, atlasDims: info.atlasDims, splat: !!info.splat, splatCount: info.splatCount || 0,
    sampleVerts: info.verts, rawBytes: info.rawBytes, files: undefined,
    folderHint: base, name: base.replace(/[^a-z0-9._-]/gi, "_"), path: info.dir,
  });
  if (history) history.refresh();   // the server recorded this analyse
}

// ---- shared card fragments ---------------------------------------------------------------
// The mesh and splat cards differ only in their codec/quantization block, so the source path
// row, the audio rows and the action bar are written once. Every string here is the control's
// name; anything that explains a control lives in its title= tooltip.
const pathFieldHtml = (path, folderHint) => `
        <div class="fld"><span class="k">path</span><span class="row">
          <input id="cvPath" class="inp" value="${path || ""}" placeholder="${folderHint ? "full path to " + folderHint : "folder path"}" style="flex:1">
          <button class="u" id="cvPickBtn" title="Pick the capture folder with the native dialog">Browse…</button>
        </span></div>
        <div id="cvPathNote" class="note2"></div>`;

const audioRowsHtml = () => `
          <div class="fld"><span class="k">track</span><span class="row">
            <input id="cvAudio" class="inp" placeholder="optional" style="flex:1" title="Any audio or video file ffmpeg reads. Transcoded to Opus 48 kHz and muxed into the .ares (spec §11.5). Empty leaves the clip silent.">
            <button class="u" id="cvAudioPick">Browse…</button>
          </span></div>
          <div class="fld"><span class="k">offset</span><span class="row">
            <input id="cvAudioOffset" class="inp" type="number" step="0.01" value="0" style="width:70px" title="Shift the audio in seconds; positive starts it later"><span class="suf">s</span>
          </span></div>`;

const actionBarHtml = () => `
      <div class="full">
        <div class="cvfoot">
          <button class="u primary" id="cvGo">Convert</button>
          <button class="u" id="cvQueueAdd" title="Queue this folder with these settings; run the queue from the batch list">Add to batch</button>
          <div class="prog" id="cvProg"><div></div></div>
        </div>
        <div id="cvQueue"></div>
        <details class="sec"><summary>Log</summary><div class="body"><pre id="cvLog" class="cvlog"></pre></div></details>
        <div id="cvDone"></div>
      </div>`;

/** Show a log <pre> and open the <details> holding it, so output is never silently hidden. */
function revealLog(el) {
  if (!el) return;
  el.style.display = "block";
  const sec = el.closest("details");
  if (sec) sec.open = true;
}

function renderConvertCard(stats) {
  const out = $("convertOut");
  if (stats.splat) return renderSplatCard(stats);
  const { meshes, kind, pngs, atlasDims, sampleVerts, rawBytes, folderHint, path } = stats;
  // Rough .ares estimate from measured daniel ratios: meshopt geom ~17.6 B/vert/frame (+normals),
  // VP9 texture ~55 KB/frame @1024². Clearly labelled as an estimate; real number comes from encoding.
  const geomEstMB = sampleVerts ? (sampleVerts * 17.6 * meshes) / 1048576 : (meshes * 0.19);
  const texEstMB = pngs >= meshes ? (meshes * 55) / 1024 : 0;
  const estMB = geomEstMB + texEstMB;
  lastState = { meshes, kind, pngs, folderHint, name: (folderHint || "converted").replace(/[^a-z0-9._-]/gi, "_"), path };

  out.innerHTML = `
    <div class="card cv2">
      <section class="cvcol">
        <div class="cap">Source</div>
${pathFieldHtml(path, folderHint)}
        <dl class="stats">
          <dt>meshes</dt><dd>${meshes} × ${kind}${sampleVerts ? ` · ${(sampleVerts / 1000).toFixed(1)}k verts/frame` : ""}</dd>
          <dt>atlases</dt><dd>${pngs} × PNG${atlasDims ? ` ${atlasDims[0]}×${atlasDims[1]}` : ""}${pngs && pngs < meshes ? '<span class="badge warn">fewer than meshes</span>' : ""}</dd>
          <dt>raw</dt><dd>${MB(rawBytes)}${stats.files ? ` · ${stats.files} files` : ""}</dd>
          <dt>est. .ares</dt><dd title="Estimate only: meshopt geometry ~${geomEstMB.toFixed(0)} MB + ${pngs >= meshes ? `video texture ~${texEstMB.toFixed(0)} MB` : "no texture track"}. The encode reports the real size.">~${estMB.toFixed(0)} MB</dd>
        </dl>
        <details class="sec">
          <summary>Texture enhance</summary>
          <div class="body">
            <div class="fld"><span class="k">tier</span><span class="row">
              <select id="enTier" class="inp" style="flex:1;max-width:260px">
                <option value="fast">Compact x4v3: CUDA</option>
                <option value="quality">x4plus: CUDA</option>
                <option value="ncnn">x4plus: ncnn-vulkan</option>
                <option value="sd">SD img2img: Forge</option>
              </select></span></div>
            <div class="fld"><span class="k">strength</span><span class="row">
              <input id="enStrength" type="range" min="0" max="100" value="70" style="flex:1;min-width:70px">
              <span id="enStrengthVal" style="font:11px ui-monospace,monospace;color:var(--text-mid);min-width:32px">70%</span></span></div>
            <div class="fld"><span class="k">scale</span><span class="row">
              <select id="enScale" class="inp" title="Output ratio. The net always runs at its native 4×; this resamples the result. The SD tier runs at source resolution and ignores it.">
                <option value="2">2×</option><option value="1">1×: re-detail only</option></select>
              <span class="k">frames</span><input id="enMax" class="inp" type="number" placeholder="all" min="1" style="width:60px"></span></div>
            <div class="row" style="margin-top:6px">
              <button class="u primary" id="enGo">Enhance</button>
              <button class="u" id="enForge" style="display:none" title="Start Forge headless now so the first enhanced frame skips the 30–60 s cold start">Start Forge</button>
            </div>
            <div class="note2" id="enNote" style="margin-top:5px"></div>
            <div class="prog" id="enProg"><div></div></div>
            <pre id="enLog" class="cvlog" style="display:none;margin-top:8px;max-height:110px"></pre>
            <div id="enDone"></div>
          </div>
        </details>
      </section>

      <section class="cvcol">
        <div class="cap">Output</div>
        <div class="fld"><span class="k">name</span><span class="row">
          <input id="cvName" class="inp" value="${lastState.name}" style="flex:1;max-width:230px"><span class="suf">.ares</span></span></div>
        <div class="fld"><span class="k">preset</span><span class="row">
          <button class="u" data-preset="web" title="VP9 · 1024² · CRF 32 · smoothing 0">Web</button>
          <button class="u" data-preset="balanced" title="AV1 · 1024² · CRF 30 · smoothing 0">Balanced</button>
          <button class="u" data-preset="hq" title="AV1 · 2048² · CRF 26 · smoothing 0: texture 2–3× larger">HQ</button>
          <button class="u" data-preset="cohA" title="Coherent pre-pass with the exact per-texel bake · VP9 · 2048² · CRF 28 · smoothing 0: slowest, highest quality (1–2 h for a full clip)">Coherent A</button>
        </span></div>
        <div class="fld"><span class="k">texture</span><span class="row">
          <select id="cvCodec" class="inp" title="AV1 is smaller at equal quality and hardware-decoded on recent GPUs"><option value="av1">AV1</option><option value="vp9">VP9</option></select>
          <select id="cvSize" class="inp"><option value="1024">1024²</option><option value="2048">2048²</option></select>
          <span class="k">CRF</span><input id="cvCrf" class="inp" type="number" value="30" min="10" max="50" style="width:52px">
        </span></div>
        <div class="fld"><span class="k">frames</span><span class="row">
          <input id="cvMax" class="inp" type="number" placeholder="all" min="1" style="width:72px"></span></div>
        <div class="fld"><span class="k">coherent</span><span class="row">
          <label class="row" style="gap:5px;cursor:pointer;color:var(--text-mid)" title="Stable-template GOPs: registers one mesh per GOP and rebakes the atlases into its UVs. Suppresses texture shimmer and cuts file size ~3.7×. Requires mesh-fNNNNN.obj + atlas-fNNNNN.png frames and adds a pre-pass of minutes on long clips.">
            <input type="checkbox" id="cvCoherent" checked><span>stable-template GOPs</span></label></span></div>
        <details class="sec"><summary>Geometry</summary><div class="body">
          <div class="fld"><span class="k">smoothing</span><span class="row">
            <input id="cvSmooth" class="inp" type="number" value="0" min="0" max="6" style="width:52px" title="Weld-aware Taubin passes. 0 is the recommended value."><span class="suf">Taubin passes</span></span></div>
          <div class="fld"><span class="k">decimate</span><span class="row">
            <input id="cvDecimate" class="inp" type="number" placeholder="off" min="0.2" max="0.95" step="0.05" style="width:64px" title="Fraction of triangles kept; atlas seams stay locked. 0.6 measured: geometry −35%. Blank disables it."><span class="suf">triangle ratio</span></span></div>
        </div></details>
        <details class="sec"><summary>Audio</summary><div class="body">
${audioRowsHtml()}
        </div></details>
      </section>

${actionBarHtml()}
    </div>`;
  $("cvGo").onclick = () => runEncode();
  $("cvQueueAdd").onclick = addToQueue;
  wireAudioPick();
  renderQueue();   // a re-analysed folder keeps the batch visible (it lives in memory across cards)
  $("cvPickBtn").onclick = pickAndAnalyse;
  $("cvPath").addEventListener("input", () => { $("cvPath").style.borderColor = ""; }); // clear the validation border once the user edits the field
  // A dropped folder whose path the server recognised: say so, so the filled-in path is not a mystery.
  if (stats.resolved && $("cvPathNote")) $("cvPathNote").textContent = "resolved from history: verify before converting";
  // Several matches: offer them as one-click fills instead of demanding the path be typed.
  if (stats.candidates && stats.candidates.length && $("cvPathNote")) {
    const note = $("cvPathNote");
    note.textContent = stats.candidates.length + " matches: ";
    for (const c of stats.candidates) {
      const b = document.createElement("button");
      b.className = "u"; b.style.cssText = "margin:0 3px"; b.textContent = c.path;
      b.title = c.files + " files";
      b.onclick = () => { $("cvPath").value = c.path; note.textContent = ""; };
      note.append(b);
    }
  }
  $("enGo").onclick = runEnhance;
  $("enForge").onclick = prewarmForge;
  $("enStrength").oninput = () => { $("enStrengthVal").textContent = $("enStrength").value + "%"; };
  const enTier = $("enTier");
  // Status line, not prose: which net runs, on what runtime, at what measured rate.
  // Rates are per 2048² atlas on this machine's class of GPU; the net always runs at its
  // native ratio and the scale above is a resample of that result.
  const TIER_NOTE = {
    fast:    "RealESRGAN Compact x4v3 · CUDA fp16, all devices · ~0.7 s/frame · needs the Python environment",
    quality: "RealESRGAN x4plus · CUDA fp16, all devices · ~12 s/frame · needs the Python environment",
    ncnn:    "RealESRGAN x4plus · ncnn-vulkan, no Python · ~22 s/frame · works on AMD and Intel GPUs",
    sd:      "SD img2img via Forge · auto-start, 30–60 s cold · checkpoint required · minutes/frame",
  };
  const syncTier = () => {
    const sd = enTier.value === "sd";
    $("enForge").style.display = sd ? "inline-flex" : "none";
    $("enNote").textContent = TIER_NOTE[enTier.value] || "";
  };
  enTier.onchange = syncTier; syncTier();
  const PRESETS = {
    web: { codec: "vp9", size: "1024", crf: "32", smooth: "0" },
    balanced: { codec: "av1", size: "1024", crf: "30", smooth: "0" },   // recommended recipe (smooth 0)
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
// Every GUI encode records its full recipe in History meta; "Use settings" re-applies that
// recipe to the card so ANY clip can be converted/baked with another clip's exact settings.
let pendingSettings = null;
function applySettingsToCard(m) {
  const set = (id, v) => { const el = $(id); if (el && v != null && v !== "") el.value = v; };
  set("cvCodec", m.codec); set("cvSize", m.texSize); set("cvCrf", m.crf);
  if (m.smooth != null && $("cvSmooth")) $("cvSmooth").value = m.smooth;         // "0" is a real value (recommended)
  if ($("cvDecimate")) $("cvDecimate").value = m.decimate || "";                 // recipe match: absent = off
  if ($("cvMax")) $("cvMax").value = m.maxFrames || "";
  if ($("cvCoherent")) $("cvCoherent").checked = m.coherent === "1";            // pre-coherent entries → unchecked (accurate)
}
function useSettings(meta) {
  if ($("cvCodec")) { applySettingsToCard(meta); $("convertOut").scrollTop = 0; }
  else {
    pendingSettings = meta;
    $("convertOut").innerHTML = `<div class="card"><div class="cap">Settings copied</div><div class="note2">${meta.coherent === "1" ? "coherent · " : ""}${meta.codec || "?"} · ${meta.texSize || "?"}² · CRF ${meta.crf || "?"} · smoothing ${meta.smooth ?? "?"}: applied on the next folder</div></div>`;
  }
}

/**
 * Convert card for a splat sequence (spec §6.8). No atlases, no coherent pre-pass, no decimation:
 * the settings that exist are the SH cap, the outlier-alpha filter and the position quantization.
 */
function renderSplatCard(stats) {
  const out = $("convertOut");
  const { meshes, kind, rawBytes, folderHint, path, splatCount } = stats;
  lastState = { meshes, kind, pngs: 0, folderHint, name: (folderHint || "converted").replace(/[^a-z0-9._-]/gi, "_"), path, splat: true };
  // Measured on the synthetic splat clip: ~9–10 B/splat/frame at SH 0 after meshopt (positions 6 B
  // + attrs 12 B raw); SH 1 adds ~7 B, SH 3 ~34 B. Labelled rough; the encode reports the truth.
  const perSplat = { 0: 10, 1: 17, 2: 28, 3: 44 };
  const est = (deg) => splatCount ? (splatCount * perSplat[deg] * meshes) / 1048576 : 0;
  out.innerHTML = `
    <div class="card cv2">
      <section class="cvcol">
        <div class="cap">Source</div>
${pathFieldHtml(path, folderHint)}
        <dl class="stats">
          <dt>frames</dt><dd>${meshes} × ${kind}${splatCount ? ` · ${(splatCount / 1000).toFixed(0)}k splats/frame` : ""}</dd>
          <dt>raw</dt><dd>${MB(rawBytes)}${stats.files ? ` · ${stats.files} files` : ""}</dd>
          <dt>est. .ares</dt><dd id="cvSplatEst" title="Estimate only, the encode reports the real size.">${splatCount ? `~${est(0).toFixed(0)} MB at SH 0 · ~${est(3).toFixed(0)} MB at SH 3` : "; "}</dd>
          <dt>profile</dt><dd title="Per-chunk AABB quantization, meshopt-coded attribute streams, Morton-ordered splats. Colour is the base 8-bit colour plus the SH bands kept.">Gaussian splat</dd>
        </dl>
      </section>

      <section class="cvcol">
        <div class="cap">Output</div>
        <div class="fld"><span class="k">name</span><span class="row">
          <input id="cvName" class="inp" value="${lastState.name}" style="flex:1;max-width:230px"><span class="suf">.ares</span></span></div>
        <div class="fld"><span class="k">SH bands</span><span class="row">
          <select id="cvShDegree" class="inp" style="flex:1" title="Spherical-harmonic bands kept. 0 drops view dependence for the smallest file; 3 keeps it in full.">
            <option value="">as captured</option>
            <option value="0">0: base colour</option>
            <option value="1">1</option><option value="2">2</option><option value="3">3: full</option>
          </select></span></div>
        <div class="fld"><span class="k">frames</span><span class="row">
          <input id="cvMax" class="inp" type="number" placeholder="all" min="1" style="width:72px"></span></div>
        <details class="sec" open><summary>Quantization</summary><div class="body">
          <div class="fld"><span class="k">min opacity</span><span class="row">
            <input id="cvMinAlpha" class="inp" type="number" value="0" min="0" max="1" step="0.05" style="width:64px" title="Drop splats below this opacity before quantization. Generated captures carry a haze of near-transparent outliers that wastes precision on empty space. 0 keeps all."><span class="suf">0 = keep all</span></span></div>
          <div class="fld"><span class="k">position</span><span class="row">
            <input id="cvQuantBits" class="inp" type="number" value="14" min="8" max="16" style="width:64px" title="Fixed-point bits per axis over each chunk's bounding box: 14 bits over a 16 m room = 1 mm steps."><span class="suf">bits per axis</span></span></div>
        </div></details>
        <details class="sec"><summary>Audio</summary><div class="body">
${audioRowsHtml()}
        </div></details>
      </section>

${actionBarHtml()}
    </div>`;
  $("cvGo").onclick = () => runEncode();
  $("cvQueueAdd").onclick = addToQueue;
  $("cvPickBtn").onclick = pickAndAnalyse;
  wireAudioPick();
  renderQueue();
}

/** Native file dialog for the audio row (serve.mjs /pick?type=file, remembered under "audio"). */
function wireAudioPick() {
  const b = $("cvAudioPick");
  if (!b) return;
  b.onclick = async () => {
    try {
      const r = await fetch("/pick?type=file&for=audio&filter=" + encodeURIComponent("Audio/video|*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg;*.opus;*.mp4;*.mov;*.webm;*.mkv|All files|*.*")).then((r) => r.json());
      if (r && r.path) $("cvAudio").value = r.path;
    } catch { /* picker needs the dev server */ }
  };
}

// ---- batch queue: collect several folder/settings jobs, run them sequentially -------------
const queue = [];
function currentJob() {
  const path = $("cvPath").value.trim();
  if (!path) { $("cvPath").focus(); $("cvPath").style.borderColor = "var(--bad)"; return null; }
  const audio = ($("cvAudio")?.value || "").trim(), audioOffset = $("cvAudioOffset")?.value || "0";
  if (lastState?.splat) {
    return {
      path, splat: true,
      name: ($("cvName").value.trim() || "converted").replace(/[^a-z0-9._-]/gi, "_"),
      codec: "", size: "", crf: "", smooth: "0", decimate: "", coherent: "",
      max: $("cvMax").value, shDegree: $("cvShDegree").value, splatMinAlpha: $("cvMinAlpha").value, quantBits: $("cvQuantBits").value,
      audio, audioOffset,
    };
  }
  return {
    path,
    name: ($("cvName").value.trim() || "converted").replace(/[^a-z0-9._-]/gi, "_"),
    codec: $("cvCodec").value, size: $("cvSize").value, crf: $("cvCrf").value,
    smooth: $("cvSmooth").value, max: $("cvMax").value, decimate: $("cvDecimate").value,
    coherent: $("cvCoherent")?.checked ? "1" : "",
    audio, audioOffset,
  };
}
function renderQueue() {
  const host = $("cvQueue");
  if (!queue.length) { host.innerHTML = ""; return; }
  host.innerHTML = `<div class="cap">Batch: ${queue.length}</div>` +
    queue.map((j, i) => `<div style="display:flex;gap:8px;align-items:center;font:12px ui-monospace,monospace;color:var(--text-mid);padding:3px 0">
      <span style="color:${j.state === "done" ? "var(--good)" : j.state === "running" ? "var(--warn)" : j.state === "failed" ? "var(--bad)" : "var(--text-faint)"}">${j.state === "done" ? "✓" : j.state === "running" ? "▶" : j.state === "failed" ? "✗" : "·"}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.name}.ares ← ${j.path}</span>
      ${j.state === "pending" ? `<button class="u" data-rm="${i}">✕</button>` : ""}
    </div>`).join("") +
    (queue.some((j) => j.state === "pending") ? `<button class="u primary" id="cvRunAll" style="margin-top:8px">Convert all: ${queue.filter((j) => j.state === "pending").length}</button>` : "");
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

const TIER_NOTE_LOG = {
  fast:    "RealESRGAN Compact x4v3 · CUDA fp16 · batch worker over every CUDA device",
  quality: "RealESRGAN x4plus · CUDA fp16 · batch worker over every CUDA device",
  ncnn:    "RealESRGAN x4plus · ncnn-vulkan · one process per frame",
  sd:      "SD img2img · Forge",
};

/** Enhance the atlas PNGs via the dev server's /enhance SSE endpoint, then point #cvPath at
 *  the enhanced sibling folder so the normal encode picks it up. */
async function runEnhance() {
  const path = $("cvPath").value.trim();
  if (!path) { $("cvPath").focus(); $("cvPath").style.borderColor = "var(--bad)"; return; }
  // No dependency gate here: /enhance installs the selected tier's runtime and weights inside
  // its own stream ("[setup] …" lines in the log below) and then runs.
  const q = new URLSearchParams({
    dir: path,
    tier: $("enTier").value,
    strength: String(Number($("enStrength").value) / 100),
    scale: $("enScale").value,
  });
  if ($("enMax").value) q.set("maxFrames", $("enMax").value);

  const log = $("enLog"), prog = $("enProg"), done = $("enDone"), go = $("enGo");
  revealLog(log); prog.style.display = "block";
  prog.firstChild.style.width = "4%"; prog.firstChild.style.background = "";
  done.innerHTML = ""; go.disabled = true; go.textContent = "Enhancing…";
  const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  line(`\n=== enhance ${path} (${$("enTier").value} · ${$("enStrength").value}% · ${$("enScale").value}×) ===`);
  line(TIER_NOTE_LOG[$("enTier").value] || "");

  const es = new EventSource("/enhance?" + q.toString());
  const finish = (ok, msg) => {
    es.close(); go.disabled = false; go.textContent = "Enhance";
    if (ok) prog.firstChild.style.width = "100%";
    else { prog.firstChild.style.background = "var(--bad)"; line("✗ " + msg); done.innerHTML = `<div class="note2" style="color:var(--bad);margin-top:8px">${msg}</div>`; }
  };
  const wantTier = $("enTier").value;
  es.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    line(`▶ ${d.frames} frame(s) → ${d.out} via ${d.via}`);
    // A dev server started before the CUDA tiers existed silently answers every tier with ncnn.
    // The symptom is a 10x slowdown and nothing else, so name it the moment it is detectable.
    if (d.tier && d.tier !== wantTier) {
      line(`! asked for tier "${wantTier}", server ran "${d.tier}"`);
      done.innerHTML = `<div class="note2" style="color:var(--warn);margin-top:8px">Dev server build predates the ${wantTier} tier: ${d.tier} ran in its place (roughly 10× slower).</div>`;
    }
  });
  es.addEventListener("log", (e) => line("· " + JSON.parse(e.data)));
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    line(`✓ ${d.file} (${(d.ms / 1000).toFixed(1)} s): ${d.frame}/${d.of}`);
    prog.firstChild.style.width = Math.max(4, Math.round((d.frame / d.of) * 100)) + "%";
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    finish(true);
    $("cvPath").value = d.out;
    done.innerHTML = `<div class="note2" style="color:var(--good);margin-top:8px">${d.frames} frames → <b>${d.out}</b> · source path repointed</div>`;
    if (history) history.refresh();
  });
  es.addEventListener("error", (e) => {
    const d = sseErrorData(e);
    finish(false, d ? d.message + (d.hint ? " · " + d.hint : "") : "enhance failed: stream closed");
    if (d && d.gated) accessPrompt(done, d, runEnhance);
  });
  es.onerror = () => { /* SSE stream closed by server */ };
}

/** Pre-start the local Forge (generative tier) so the first hero-frame enhance skips cold start. */
function prewarmForge() {
  const btn = $("enForge"), done = $("enDone");
  btn.disabled = true; const orig = btn.textContent; btn.textContent = "Starting…";
  done.innerHTML = `<div class="note2" style="margin-top:8px">Forge: starting (headless API)</div>`;
  const es = new EventSource("/forge/start");
  const stop = (html) => { es.close(); btn.disabled = false; btn.textContent = orig; done.innerHTML = html; };
  es.addEventListener("log", (e) => { done.innerHTML = `<div class="note2" style="margin-top:8px">${JSON.parse(e.data)}</div>`; });
  es.addEventListener("done", () => stop(`<div class="note2" style="margin-top:8px;color:var(--good)">Forge ready</div>`));
  es.addEventListener("error", (e) => { const d = sseErrorData(e); stop(`<div class="note2" style="margin-top:8px;color:var(--bad)">${(d && d.message) || "Forge start failed"}</div>`); });
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
    const q = new URLSearchParams({ dir: j.path, name: j.name });
    if (j.splat) {
      if (j.shDegree !== "" && j.shDegree != null) q.set("shDegree", j.shDegree);
      if (Number(j.splatMinAlpha) > 0) q.set("splatMinAlpha", j.splatMinAlpha);
      if (j.quantBits && Number(j.quantBits) !== 14) q.set("quantBits", j.quantBits);
    } else { q.set("textureCodec", j.codec); q.set("texSize", j.size); q.set("crf", j.crf); }
    if (Number(j.smooth) > 0) q.set("smooth", j.smooth);
    if (j.audio) { q.set("audio", j.audio); if (Number(j.audioOffset)) q.set("audioOffset", j.audioOffset); }
    if (j.max) q.set("maxFrames", j.max);
    if (j.crop) q.set("crop", j.crop);
    if (j.decimate && Number(j.decimate) > 0 && Number(j.decimate) < 1) q.set("decimate", j.decimate);
    if (j.coherent) q.set("coherent", "1");

    const log = $("cvLog"), prog = $("cvProg"), done = $("cvDone"), go = $("cvGo");
    revealLog(log); prog.style.display = "block"; prog.firstChild.style.width = "12%"; prog.firstChild.style.background = "";
    done.innerHTML = ""; go.disabled = true; go.textContent = "Converting…";
    const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
    line(j.splat
      ? `\n=== ${j.name}.ares ← ${j.path} (splat profile · SH ${j.shDegree === "" ? "as captured" : j.shDegree} · min opacity ${j.splatMinAlpha || 0} · ${j.quantBits || 14} bits) ===`
      : `\n=== ${j.name}.ares ← ${j.path} (${j.coherent ? "coherent " : ""}${j.codec} ${j.size}² crf${j.crf} smooth${j.smooth}) ===`);

    const es = new EventSource("/encode?" + q.toString());
    let pulse = 12;
    const finish = (ok, msg) => {
      es.close(); go.disabled = false; go.textContent = "Convert";
      if (ok) {
        prog.firstChild.style.width = "100%";
        done.innerHTML = `<div class="note2" style="color:var(--good);margin-top:10px">wrote ${msg}</div>
          <button class="u" id="cvOpen">Open in Viewer</button>
          <button class="u" id="cvShowcase">Add to source bar</button>`;
        $("cvOpen").onclick = () => { location.search = "?src=" + j.name + ".ares"; };
        $("cvShowcase").onclick = async (e) => { const ok = await addToShowcase(j.name + ".ares", j.name); e.target.textContent = ok ? "Added" : "✗ failed"; e.target.disabled = ok; };
        if (history) history.refresh();
        resolve(msg);
      } else {
        prog.firstChild.style.background = "var(--bad)";
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

/**
 * Open whatever the Windows shell verb was pointed at. The browser cannot read a path, so the
 * server classifies it first (/open-info) and the right flow takes over:
 *   folder            -> analyse it as a frame sequence (the volumetric case)
 *   .ares             -> probe the container (byte ranges through /local-bytes)
 *   .4ds              -> probe, then the on-machine decode row with the path filled
 *   a lone mesh/splat -> analyse the folder that CONTAINS it, because a sequence is a folder
 */
export async function openPath(target) {
  const out = $("convertOut");
  out.innerHTML = `<div class="card"><div class="cap">Opening</div><div class="note2">${target}</div></div>`;
  let info;
  try { info = await fetch("/open-info?path=" + encodeURIComponent(target)).then((r) => r.json()); }
  catch { out.innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Dev server not reachable</div></div>`; return; }
  if (!info || info.error) {
    out.innerHTML = `<div class="card"><div class="cap" style="color:var(--bad)">Cannot open</div><div class="note2">${(info && info.error) || target}</div></div>`;
    return;
  }
  if (info.kind === "dir") { await analyseServer(info.path); return; }

  const ext = info.ext || "";
  if (isVideoName(ext)) { await renderDepthCard(info.path); return; }   // a 2D video: the depth card
  if (ext === ".4ds" || ext === ".ares") { await handleProbeFile(remoteFile(info.path, info.size || 0), info.path); return; }
  // A single mesh or splat file: a sequence lives in a folder, so analyse the folder.
  await analyseServer(info.parent);
  if ($("cvPathNote")) $("cvPathNote").textContent = `opened the folder holding ${target.split(/[\\/]/).pop()}`;
}

export function initConvert() {
  const drop = $("convertDrop");
  $("convertOpen").onclick = pickAndOpen;   // every source through one native dialog
  initDepthCard({ addToShowcase, refreshHistory: () => history && history.refresh() });
  // Format matrix lives behind the ⋯ button: it is reference material, not a control, and it
  // used to cost five lines of permanent vertical space above the drop zone.
  const info = $("convertInfo"), formats = $("convertFormats");
  if (info && formats) info.onclick = () => {
    formats.hidden = !formats.hidden;
    info.setAttribute("aria-expanded", String(!formats.hidden));
  };
  const fourdsLink = $("fourdsSettingsLink");
  if (fourdsLink) fourdsLink.onclick = (e) => { e.preventDefault(); const b = document.querySelector('#tabs button[data-tab="settings"]'); if (b) b.click(); };
  drop.addEventListener("click", pickAndOpen);
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickAndOpen(); } });
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { stop(e); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { stop(e); drop.classList.remove("over"); }));
  drop.addEventListener("drop", async (e) => { const { files, root } = await gather(e.dataTransfer); if (files.length) handleImport(files, root); });
  history = initHistoryPanel({
    host: $("convertHistory"),
    kinds: ["analyse", "encode", "enhance", "inspect"],
    actions: (item) => {
      const acts = [];
      if (item.kind === "encode" && item.out) acts.push({ label: "Play", run: () => { location.search = "?src=" + item.out.split("/").pop(); } });
      // A depth conversion's recipe re-opens its own card; the mesh card's fields do not apply to it.
      if (item.kind === "encode" && item.meta && item.meta.source === "depth") acts.push({ label: "Use settings", run: () => { renderDepthCard(item.path || "", { settings: item.meta }); $("convertOut").scrollTop = 0; } });
      else if (item.kind === "encode" && item.meta) acts.push({ label: "Use settings", run: () => useSettings(item.meta) });
      if (item.kind === "enhance" && item.out) acts.push({ label: "Use output", run: () => analyseServer(item.out) });
      if (item.kind === "inspect" && item.meta && item.meta.probe) acts.push({ label: "View", run: () => { $("convertOut").innerHTML = renderProbeOut(item.meta.probe); $("convertOut").scrollTop = 0; } });
      if (item.path && !(item.meta && item.meta.source === "depth")) acts.push({ label: "Re-analyse", run: () => { analyseServer(item.path); $("convertOut").scrollTop = 0; } });
      return acts;
    },
  });
}
