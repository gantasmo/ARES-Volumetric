/**
 * ARES P1 demo — loads one .ares, plays it via @ares/core (WebGPU), and shows the
 * exit-criteria metrics live (spec §14 P1): TTFF, main-thread CPU/frame, decode/frame,
 * render FPS, and the single-request win vs a Draco-GLB-per-frame sequence.
 */
// orbitViewProj is the RENDERER's own camera math — the crop guides project through the exact same
// matrix the pixels do, so a guide can never drift from the geometry it claims to cut.
import { AresPlayer, rleEncodeMask, keepPredicateAt, orbitViewProj } from "@ares/core";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
// Default = the current keeper recipe (user verdict 2026-07-10: smooth 0 — "much less warping").
const SRC = new URLSearchParams(location.search).get("src") || "daniel-s0.ares";

// A/B source switcher — the person with eyes decides which looks best.
// Smoothing sweep at the same recipe (oct16 normals + AV1): s0..s3 differ ONLY in Taubin passes.
// v1 = untouched original for reference. Switching sources carries your camera + timestamp so the
// comparison is the same instant from the same angle (auto-orbit + looping made solo A/Bs drift).
const SOURCES = [
  { label: "Daniel ✓", src: "daniel-s0.ares" },      // KEEPER: oct16+AV1+reorder, smooth 0 (user verdict)
  { label: "smooth 1", src: "daniel-s1.ares" },
  { label: "s0 HQ 2048", src: "daniel-s0hq.ares" },  // faint-seam-line test: 2048² texture tier
  { label: "v1 original", src: "daniel.ares" },      // 67 MB baseline (VP9 + i8 normals, no reorder)
  { label: "Synth", src: "demo.ares" },
];
export { SOURCES };
// The source bar is user-editable + persisted (apps/demo/showcase.json via /showcase). SOURCES is
// only the first-run seed. Users add any existing/new .ares as a showcase button and remove stale ones.
let showcase = SOURCES.slice();
let showcaseEdit = false;

// Navigate to a source, carrying viewpoint + time + pause (fair A/B) and backend flags.
function navTo(src) {
  const p = window.__ares;
  let q = "?src=" + src;
  if (p) {
    const st = p.getStats(), cam = p.getCamera();
    q += "&t=" + (st.frameIndex / 30).toFixed(3);
    q += "&cam=" + [cam.azimuth, cam.elevation, cam.distance, cam.target[0], cam.target[1], cam.target[2]].map((v) => v.toFixed(4)).join("_");
    q += "&paused=" + (p.isPlaying ? "0" : "1");
  }
  const cur = new URLSearchParams(location.search);
  for (const flag of ["gl2", "worker"]) if (cur.get(flag) === "1") q += "&" + flag + "=1";
  location.search = q;
}
const saveShowcase = () => fetch("/showcase", { method: "POST", body: JSON.stringify(showcase) }).catch(() => {});

// Delete a clip's .ares from the library (permanent) — the owner's "delete from library" option.
// Confirms, hits /delete-ares (server confines to apps/demo + drops it from showcase.json), then
// removes it from the in-memory bar and re-renders.
async function deleteAres(src, label) {
  const name = String(src).replace(/\.ares$/i, "");
  if (!confirm(`Delete "${label || name}" from the library?\n\nThis permanently removes ${name}.ares from disk.`)) return;
  try {
    const r = await fetch("/delete-ares", { method: "POST", body: JSON.stringify({ name }) }).then((r) => r.json());
    if (!r.ok) { alert("Delete failed: " + (r.error || "unknown")); return; }
    showcase = showcase.filter((z) => z.src !== src);
    await saveShowcase();
    renderSourceBar();
    window.dispatchEvent(new CustomEvent("ares:library-changed"));
  } catch (e) { alert("Delete failed: " + e.message); }
}

// ---- Library model -----------------------------------------------------------------------------
// showcase = [{src, label, fav, folder, addedAt}] (server-persisted, extended in place). clipInfo =
// per-file facts from /list-ares (bytes/mtime/meta/thumb). View prefs (sort/layout/favOnly/collapsed
// + empty folders) are per-browser in localStorage. clipMeta = the LOADED clip's provenance sidecar.
let clipInfo = {};    // src -> { bytes, mtime, meta, thumb }
let clipMeta = null;
const LIBVIEW_KEY = "ares.lib.view";
let libView = (() => {
  const d = { sort: "recent", layout: "list", favOnly: false, collapsed: {}, emptyFolders: [] };
  try { return { ...d, ...JSON.parse(localStorage.getItem(LIBVIEW_KEY) || "{}") }; } catch { return d; }
})();
const saveLibView = () => { try { localStorage.setItem(LIBVIEW_KEY, JSON.stringify(libView)); } catch { /* private mode */ } };
const clipBytes = (src) => (clipInfo[src] && clipInfo[src].bytes) || 0;
const fmtMB = (b) => (b ? (b / 1048576).toFixed(b / 1048576 < 100 ? 1 : 0) + "M" : "");
const safeName = (src) => src.replace(/\.ares$/i, "").replace(/[^a-z0-9._-]/gi, "_");
const mkIconBtn = (txt, title, onclick) => { const b = document.createElement("button"); b.className = "libbtn"; b.textContent = txt; b.title = title; b.onclick = onclick; return b; };

// Older entries (and convert.js's bare {label,src} adds) get the extended fields filled in.
function migrateShowcase() {
  let changed = false;
  for (const s of showcase) {
    if (s.fav === undefined) { s.fav = false; changed = true; }
    if (s.folder === undefined) { s.folder = null; changed = true; }
    if (s.addedAt === undefined) { s.addedAt = (clipInfo[s.src] && clipInfo[s.src].mtime) || 0; changed = true; } // 0, not now, so a dangling/unknown entry doesn't float to the top of Recent
  }
  if (changed) saveShowcase();
}
function allFolders() {
  const set = new Set(libView.emptyFolders || []);
  for (const s of showcase) if (s.folder) set.add(s.folder);
  return [...set].sort((a, b) => a.localeCompare(b));
}
function sortItems(arr) {
  const mode = libView.sort;
  return arr.sort((a, b) => {
    if (mode === "name") return a.label.localeCompare(b.label, undefined, { numeric: true });
    if (mode === "size") return clipBytes(b.src) - clipBytes(a.src);
    // "recent" = most recent made/imported/edited: file mtime OR library-add time, whichever is newer.
    // A missing file (no clipInfo) keys to 0 so dangling entries sink to the bottom, never float up.
    const key = (s) => (clipInfo[s.src] ? Math.max((clipInfo[s.src].mtime || 0), (s.addedAt || 0)) : 0);
    return key(b) - key(a);
  });
}

// Toolbar: sort select · list/grid toggle · favorites filter · edit · (edit:) add clip / new folder.
function renderMediaHead() {
  const h = $("mediaHead"); if (!h) return;
  h.replaceChildren();
  // id + name: a bare <select> trips the browser's autofill heuristics ("form field has neither an
  // id nor a name"). It's only a console warning, but it's noise in a console we read for real bugs.
  const sort = document.createElement("select"); sort.className = "libsort"; sort.title = "sort the library";
  sort.id = "libSort"; sort.name = "libSort"; sort.autocomplete = "off";
  for (const [v, t] of [["recent", "Recent"], ["name", "Name"], ["size", "Size"]]) { const o = document.createElement("option"); o.value = v; o.textContent = t; if (libView.sort === v) o.selected = true; sort.append(o); }
  sort.onchange = () => { libView.sort = sort.value; saveLibView(); renderSourceBar(); };
  const sp = document.createElement("span"); sp.className = "sp";
  const layout = mkIconBtn(libView.layout === "grid" ? "▤" : "▦", libView.layout === "grid" ? "switch to list layout" : "switch to icon layout", () => { libView.layout = libView.layout === "grid" ? "list" : "grid"; saveLibView(); renderSourceBar(); });
  const fav = mkIconBtn("★", "show favorites only", () => { libView.favOnly = !libView.favOnly; saveLibView(); renderSourceBar(); });
  fav.classList.toggle("on", libView.favOnly);
  const edit = mkIconBtn(showcaseEdit ? "Done" : "✎", showcaseEdit ? "finish editing" : "rename, favorite, group, delete", () => { showcaseEdit = !showcaseEdit; renderSourceBar(); });
  edit.classList.toggle("on", showcaseEdit);
  h.append(sort, sp, layout, fav, edit);
  if (showcaseEdit) {
    h.append(mkIconBtn("＋", "add an existing .ares to the library", openAddPicker));
    h.append(mkIconBtn("🗀", "new folder", () => {
      const name = (prompt("New folder name:") || "").trim().slice(0, 40);
      if (!name) return;
      if (!(libView.emptyFolders || []).includes(name)) { (libView.emptyFolders ||= []).push(name); saveLibView(); }
      renderSourceBar();
    }));
  }
}

// Library body: ungrouped items first, then each folder as a collapsible group. List or icon layout.
function renderSourceBar() {
  const host = $("mediaList"); if (!host) return;
  migrateShowcase();
  renderMediaHead();
  const cur = SRC.replace("./", "");
  host.replaceChildren();
  let items = showcase.slice();
  if (libView.favOnly) items = items.filter((s) => s.fav);
  const mkContainer = () => { const c = document.createElement("div"); c.className = "mediaItems " + libView.layout; return c; };
  const ung = sortItems(items.filter((s) => !s.folder));
  if (ung.length) { const c = mkContainer(); for (const s of ung) c.append(renderItem(s, cur)); host.append(c); }
  for (const fname of allFolders()) {
    const inF = sortItems(items.filter((s) => s.folder === fname));
    if (!inF.length && !showcaseEdit) continue;
    const collapsed = !!libView.collapsed[fname];
    const hd = document.createElement("div"); hd.className = "folderHd" + (collapsed ? " collapsed" : "");
    const tw = document.createElement("span"); tw.className = "tw"; tw.textContent = "▾";
    const nm = document.createElement("span"); nm.className = "fn"; nm.textContent = fname;
    const ct = document.createElement("span"); ct.className = "fc"; ct.textContent = inF.length;
    hd.append(tw, nm, ct);
    hd.onclick = () => { libView.collapsed[fname] = !collapsed; saveLibView(); renderSourceBar(); };
    if (showcaseEdit) {
      const rm = mkIconBtn("✕", "delete this folder (its clips move to Ungrouped)", async (e) => {
        e.stopPropagation();
        for (const s of showcase) if (s.folder === fname) s.folder = null;
        libView.emptyFolders = (libView.emptyFolders || []).filter((x) => x !== fname);
        await saveShowcase(); saveLibView(); renderSourceBar();
      });
      rm.classList.add("fx"); hd.append(rm);
    }
    host.append(hd);
    if (!collapsed && inF.length) { const c = mkContainer(); for (const s of inF) c.append(renderItem(s, cur)); host.append(c); }
  }
  if (!host.children.length) { const e = document.createElement("div"); e.className = "libEmpty"; e.textContent = libView.favOnly ? "no favorites yet — tap ☆ on a clip" : "no clips — Convert one, or ＋ add"; host.append(e); }
}

const renderItem = (s, cur) => (libView.layout === "grid" ? renderGridItem(s, cur) : renderListItem(s, cur));

function starBtn(s) {
  const b = document.createElement("button"); b.className = "star" + (s.fav ? " on" : "");
  b.textContent = s.fav ? "★" : "☆"; b.title = s.fav ? "remove from favorites" : "add to favorites";
  b.onclick = (e) => { e.stopPropagation(); s.fav = !s.fav; saveShowcase(); renderSourceBar(); };
  return b;
}
// Edit-mode per-item cluster: move-to-folder · thumbnail (upload / clear) · remove-from-list · delete.
function editControls(s) {
  const wrap = document.createElement("span"); wrap.className = "mctl";
  const fsel = document.createElement("select"); fsel.className = "fsel"; fsel.title = "move to folder";
  for (const [v, t] of [["", "— none"], ...allFolders().map((f) => [f, f]), ["__new__", "＋ New…"]]) {
    const o = document.createElement("option"); o.value = v; o.textContent = t; if ((s.folder || "") === v) o.selected = true; fsel.append(o);
  }
  fsel.onclick = (e) => e.stopPropagation();
  fsel.onchange = async (e) => {
    e.stopPropagation();
    let v = fsel.value;
    if (v === "__new__") { v = (prompt("New folder name:") || "").trim().slice(0, 40); if (!v) { renderSourceBar(); return; } }
    s.folder = v || null; await saveShowcase(); renderSourceBar();
  };
  wrap.append(fsel);
  wrap.append(mkIconBtn("🖼", "set thumbnail from an image file", (e) => { e.stopPropagation(); uploadThumb(s.src); }));
  if (clipInfo[s.src] && clipInfo[s.src].thumb) wrap.append(mkIconBtn("⌫", "clear thumbnail", (e) => { e.stopPropagation(); clearThumb(s.src); }));
  wrap.append(mkIconBtn("✕", "remove from the list (keeps the file on disk)", async (e) => { e.stopPropagation(); showcase = showcase.filter((z) => z !== s); await saveShowcase(); renderSourceBar(); }));
  wrap.append(mkIconBtn("🗑", "permanently delete this .ares from disk", (e) => { e.stopPropagation(); deleteAres(s.src, s.label); }));
  return wrap;
}
function renameInline(labelEl, s) {
  const inp = document.createElement("input"); inp.className = "renameIn"; inp.value = s.label; inp.maxLength = 80;
  inp.onclick = (e) => e.stopPropagation();
  let done = false;
  const commit = async () => { if (done) return; done = true; const v = inp.value.trim(); if (v && v !== s.label) { s.label = v; await saveShowcase(); } renderSourceBar(); };
  inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } else if (e.key === "Escape") { done = true; renderSourceBar(); } };
  inp.onblur = commit;
  labelEl.replaceWith(inp); inp.focus(); inp.select();
}

// ---- Rich hover detail: full provenance + timestamps for a library item. Lazily fetches+caches the
//      clip's .ares.meta.json sidecar; timestamps from showcase.addedAt + the file mtime.
const clipMetaCache = {};
let detailEl = null, detailTimer = 0;
function ensureDetailEl() {
  if (detailEl) return detailEl;
  detailEl = document.createElement("div"); detailEl.id = "clipDetail"; detailEl.style.display = "none";
  document.body.append(detailEl);
  return detailEl;
}
const hideDetail = () => { if (detailEl) detailEl.style.display = "none"; };
function fmtDateTime(ms) {
  if (!ms) return "—";
  try { return new Date(ms).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }); } catch { return "—"; }
}
const cdRow = (k, v) => (v == null || v === "" ? "" : `<div class="cdrow"><span class="cdk">${k}</span><span class="cdv">${v}</span></div>`);
function buildDetailHTML(s, meta) {
  const ci = clipInfo[s.src] || {}, missing = !clipInfo[s.src];
  let h = `<div class="cdttl">${s.label}</div>`;
  if (s.label !== s.src) h += `<div class="cdsub">${s.src}</div>`;
  h += cdRow("Size", missing ? "missing" : fmtMB(clipBytes(s.src)));
  h += cdRow("Added", fmtDateTime(s.addedAt));
  h += cdRow("Made / edited", fmtDateTime(ci.mtime));
  if (s.folder) h += cdRow("Folder", s.folder);
  if (s.fav) h += cdRow("", "★ favorite");
  if (meta && Object.keys(meta).length) {
    if (meta.pipeline) h += cdRow("Pipeline", meta.pipeline);
    const reg = meta.register || (meta.coherent && meta.coherent.register);
    if (reg) h += cdRow("Register", reg);
    for (const [k, v] of Object.entries(meta.encode || {})) { if (v != null && v !== "" && typeof v !== "object") h += cdRow(k, String(v)); }
    for (const [k, v] of Object.entries(meta.geometry || {})) { if (v != null && typeof v !== "object") h += cdRow("geom·" + k, String(v)); }
    if (meta.source && meta.source.dir) h += cdRow("Source", String(meta.source.dir).split(/[\\/]/).slice(-2).join("/"));
  } else h += `<div class="cdrow"><span class="cdv" style="opacity:.55">no provenance sidecar</span></div>`;
  return h;
}
async function showDetail(el, s) {
  const d = ensureDetailEl();
  let meta = clipMetaCache[s.src];
  if (meta === undefined) {
    try { meta = await fetch("/clip-meta?name=" + encodeURIComponent(s.src.replace("./", ""))).then((r) => r.json()); } catch { meta = {}; }
    clipMetaCache[s.src] = meta;
  }
  d.innerHTML = buildDetailHTML(s, meta);
  d.style.display = "block";
  const r = el.getBoundingClientRect();
  d.style.left = Math.max(6, Math.min(r.right + 8, window.innerWidth - d.offsetWidth - 8)) + "px";
  d.style.top = Math.max(6, Math.min(r.top, window.innerHeight - d.offsetHeight - 8)) + "px";
}
function attachDetailHover(el, s) {
  el.addEventListener("mouseenter", () => { clearTimeout(detailTimer); detailTimer = setTimeout(() => showDetail(el, s), 300); });
  el.addEventListener("mouseleave", () => { clearTimeout(detailTimer); hideDetail(); });
}

function renderListItem(s, cur) {
  const missing = !clipInfo[s.src];   // in the showcase but the .ares is gone from disk
  const row = document.createElement("div"); row.className = "mrow" + (missing ? " missing" : "");
  row.append(starBtn(s));
  const b = document.createElement("button"); b.className = "clip";
  const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = s.label; b.append(lab);
  if (!showcaseEdit) { const z = document.createElement("span"); z.className = "sz"; z.textContent = missing ? "missing" : fmtMB(clipBytes(s.src)); b.append(z); }
  b.setAttribute("aria-pressed", String(cur === s.src));
  b.onclick = () => { if (showcaseEdit) renameInline(lab, s); else if (missing) alert(`"${s.label}" (${s.src}) is no longer on disk — remove it in edit mode (✎).`); else navTo(s.src); };
  attachDetailHover(b, s);   // rich hover: full provenance + added/made timestamps
  row.append(b);
  if (showcaseEdit) row.append(editControls(s));
  return row;
}
function renderGridItem(s, cur) {
  const missing = !clipInfo[s.src];
  const card = document.createElement("div"); card.className = "gcard" + (cur === s.src ? " active" : "") + (missing ? " missing" : "");
  const th = document.createElement("div"); th.className = "gthumb";
  if (clipInfo[s.src] && clipInfo[s.src].thumb) {
    const img = document.createElement("img"); img.loading = "lazy";
    img.src = ".thumbs/" + safeName(s.src) + ".jpg?t=" + ((clipInfo[s.src].mtime | 0));   // cache-bust on re-capture
    th.append(img);
  } else { const ph = document.createElement("div"); ph.className = "ph"; ph.textContent = (s.label[0] || "?").toUpperCase(); th.append(ph); }
  th.append(starBtn(s));
  th.onclick = () => { if (showcaseEdit) return; if (missing) alert(`"${s.label}" (${s.src}) is no longer on disk — remove it in edit mode (✎).`); else navTo(s.src); };
  card.append(th);
  attachDetailHover(card, s);   // rich hover: full provenance + added/made timestamps
  const lab = document.createElement("div"); lab.className = "glab"; lab.textContent = s.label;
  lab.onclick = () => { if (showcaseEdit) renameInline(lab, s); else navTo(s.src); };
  card.append(lab);
  if (!showcaseEdit) { const z = document.createElement("div"); z.className = "gsz"; z.textContent = fmtMB(clipBytes(s.src)); card.append(z); }
  else card.append(editControls(s));
  return card;
}

// ---- Thumbnails: capture from the live view, upload a file, or clear. Owner-only; never inspected here.
async function saveThumb(src, dataURI) {
  try { await fetch("/save-thumb?name=" + encodeURIComponent(src), { method: "POST", body: JSON.stringify({ dataURI }) }); await refreshLibraryInfo(); renderSourceBar(); }
  catch (e) { alert("thumbnail save failed: " + e.message); }
}
async function clearThumb(src) {
  try { await fetch("/save-thumb?name=" + encodeURIComponent(src), { method: "POST", body: JSON.stringify({ clear: true }) }); await refreshLibraryInfo(); renderSourceBar(); } catch { /* ignore */ }
}
function uploadThumb(src) {
  const inp = document.createElement("input"); inp.type = "file"; inp.accept = "image/*";
  inp.onchange = () => { const f = inp.files && inp.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => downscaleToThumb(String(r.result)).then((du) => saveThumb(src, du)); r.readAsDataURL(f); };
  inp.click();
}
function downscaleToThumb(dataURL, max = 220) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, max / Math.max(img.width, img.height));
      const w = Math.max(1, Math.round(img.width * scale)), h = Math.max(1, Math.round(img.height * scale));
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      c.getContext("2d").drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL("image/jpeg", 0.72));
    };
    img.onerror = () => resolve(dataURL);
    img.src = dataURL;
  });
}
// Capture the LOADED clip's current viewport (draw the live canvas into a 2D canvas so it works for
// WebGPU + WebGL2 backends) and save it as this clip's library thumbnail.
function captureThumb() {
  const src = SRC.replace("./", "");
  try {
    const cw = canvas.width || canvas.clientWidth, ch = canvas.height || canvas.clientHeight;
    const scale = Math.min(1, 220 / Math.max(cw, ch));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(cw * scale)); c.height = Math.max(1, Math.round(ch * scale));
    c.getContext("2d").drawImage(canvas, 0, 0, c.width, c.height);
    const du = c.toDataURL("image/jpeg", 0.72);
    if (du.length < 200) { alert("capture came back blank — make sure the clip is visible, or use 🖼 upload in edit mode"); return; }
    saveThumb(src, du);
  } catch (e) { alert("capture failed (" + e.message + ") — use 🖼 upload in edit mode instead"); }
}

async function openAddPicker() {
  const host = $("mediaList");
  if ($("srcAddRow")) { $("srcAddRow").remove(); return; }   // toggle
  const list = await fetch("/list-ares").then((r) => r.json()).catch(() => []);
  if (Array.isArray(list)) clipInfo = Object.fromEntries(list.map((f) => [f.src, f]));
  const shown = new Set(showcase.map((s) => s.src));
  const avail = (Array.isArray(list) ? list : []).filter((f) => !shown.has(f.src));
  const row = document.createElement("div");
  row.id = "srcAddRow";
  row.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:5px 6px;border-bottom:1px solid rgba(255,255,255,.08)";
  const sel = document.createElement("select"); sel.className = "cvsel"; sel.style.cssText = "width:100%;box-sizing:border-box";
  if (!avail.length) { const o = document.createElement("option"); o.value = ""; o.textContent = "(no other .ares — Convert one first)"; sel.append(o); }
  for (const f of avail) { const o = document.createElement("option"); o.value = f.src; o.textContent = `${f.src} (${(f.bytes / 1048576).toFixed(1)} MB)`; sel.append(o); }
  const label = document.createElement("input");
  label.placeholder = "label";
  label.style.cssText = "width:100%;box-sizing:border-box;padding:3px 6px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:5px;color:var(--text);font:12px system-ui";
  const setDefault = () => { label.value = (sel.value || "").replace(/\.ares$/i, ""); };
  sel.onchange = setDefault; setDefault();
  const go = document.createElement("button"); go.textContent = "Add to library"; go.style.cssText = "padding:3px 8px";
  go.onclick = async () => {
    if (!sel.value) return;
    showcase.push({ label: (label.value.trim() || sel.value.replace(/\.ares$/i, "")).slice(0, 80), src: sel.value, fav: false, folder: null, addedAt: Date.now() });
    await saveShowcase();
    renderSourceBar();
  };
  row.append(sel, label, go);
  host.prepend(row);
}

// Per-file facts (size, mtime, thumb, meta) for the library rows/cards + sorting.
async function refreshLibraryInfo() {
  try { const list = await fetch("/list-ares").then((r) => r.json()); if (Array.isArray(list)) clipInfo = Object.fromEntries(list.map((f) => [f.src, f])); } catch { /* offline */ }
}
async function refreshClipMeta() {
  try { clipMeta = await fetch("/clip-meta?name=" + encodeURIComponent(SRC.replace("./", ""))).then((r) => r.json()); } catch { clipMeta = null; }
}
// ⧉ recipe: copy the loaded clip's full conversion recipe (the bound-by-blood sidecar) to the
// clipboard so it can be re-applied to another conversion; also logs it for inspection.
async function showRecipe() {
  if (!clipMeta || !Object.keys(clipMeta).length) return;
  try { await navigator.clipboard.writeText(JSON.stringify(clipMeta, null, 2)); } catch { /* clipboard may be blocked */ }
  console.log("[recipe]", clipMeta);
  const b = $("recipeBtn"); if (b) { const o = b.textContent; b.textContent = "✓"; setTimeout(() => (b.textContent = o), 1200); }
}

(async function initSourceBar() {
  try {
    const saved = await fetch("/showcase").then((r) => r.json());
    // An EMPTY array is a valid saved library — the user deleted everything. Only a missing/unseeded
    // showcase (GET returns null) gets the defaults. Testing `.length` here treated "deliberately
    // empty" as "unseeded" and wrote the 5 seed clips back over it on the next load.
    if (Array.isArray(saved)) showcase = saved;
    else await saveShowcase();   // seed the file with the defaults so it persists + edits stick
  } catch { /* offline: keep in-memory defaults */ }
  await Promise.all([refreshLibraryInfo(), refreshClipMeta()]);
  renderSourceBar();
})();
// convert.js dispatches this after "Add to source bar" so the (possibly hidden) viewer bar refreshes.
window.addEventListener("ares:showcase-changed", async () => {
  try { const s = await fetch("/showcase").then((r) => r.json()); if (Array.isArray(s)) showcase = s; } catch { /* ignore */ }
  renderSourceBar();
});

// --- Bottom diagnostics bar owns all live telemetry (bake ETA + GPU/RAM/disk/RunPod). See
// diagnostics.js. It fires "ares:bake-done" on completion; keep the auto-add-to-Source here so a
// finished clip shows up in the bar without digging. ---
window.addEventListener("ares:bake-done", (e) => {
  const name = e.detail && e.detail.name;
  if (!name || showcase.some((s) => s.src === name)) return;
  showcase.push({ label: name.replace(/\.ares$/i, ""), src: name, fav: false, folder: null, addedAt: Date.now() });
  saveShowcase().then(renderSourceBar);
});
// The bar's "now playing" segment is pushed from renderHUD via this handle (see setDiagClip). Held
// in a promise so renderHUD never has to care whether the module has loaded yet.
let diagMod = null;
import("./diagnostics.js").then((m) => { m.initDiagnostics(); diagMod = m; }).catch((err) => console.error("diagnostics:", err));
import("./gizmo.js").then((m) => m.initGizmo()).catch((err) => console.error("gizmo:", err));

// --- Tabs: Viewer | Compare | Convert | Settings ----------------------------
// Every tool tab is a lazy-loaded module; the viewer keeps rendering behind the overlay.
// A tab-module failure logs and leaves the other tabs working — switching tabs can never break.
// The standalone Inspect tab was removed 2026-07-11 (merged into Convert's unified import):
// its probe functions live on in ./probe.js.
let convertInited = false, compareInited = false, settingsInited = false, computeInited = false;
const TAB_KEY = "ares.tab";
const VALID_TABS = new Set(["viewer", "compare", "convert", "settings", "compute"]);
// Old localStorage values (from before the merge) still redirect cleanly instead of erroring
// or silently falling back to Viewer — "inspect" now means "convert".
const LEGACY_TAB_MAP = { inspect: "convert" };
function setTab(name) {
  name = LEGACY_TAB_MAP[name] || name;
  // [role="tab"] scope matters: the header bar also holds the viewport-shading segmented control,
  // whose buttons are aria-pressed, not aria-selected — an unscoped "#tabs button" would stamp them.
  for (const b of document.querySelectorAll('#tabs button[role="tab"]')) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  // .viewer-only means "hidden outside the Viewer tab" — it does NOT mean "shown whenever the
  // Viewer tab is up. #cropOverlay is owned by the crop toggle, so blanket-writing display:"" here
  // would force it on behind its own button. Self-managed elements get hidden on the way out and
  // restored by their owner on the way back in. (#editPanel used to be one of these, back when an
  // Edit button conjured it; the rail is now simply part of the Viewer.)
  for (const el of document.querySelectorAll(".viewer-only")) {
    if (el.dataset.selfManaged !== undefined) { if (name !== "viewer") el.style.display = "none"; continue; }
    el.style.display = name === "viewer" ? "" : "none";
  }
  if (name === "viewer") window.__aresRestoreViewerPanels?.();
  $("tab-convert").classList.toggle("active", name === "convert");
  $("tab-compare").classList.toggle("active", name === "compare");
  $("tab-settings").classList.toggle("active", name === "settings");
  $("tab-compute").classList.toggle("active", name === "compute");
  if (name === "convert" && !convertInited) { convertInited = true; import("./convert.js").then((m) => m.initConvert()).catch((e) => console.error(e)); }
  if (name === "compare" && !compareInited) { compareInited = true; import("./compare.js").then((m) => m.initCompare()).catch((e) => console.error(e)); }
  if (name === "settings" && !settingsInited) { settingsInited = true; import("./settings.js").then((m) => m.initSettings()).catch((e) => console.error(e)); }
  if (name === "compute" && !computeInited) { computeInited = true; import("./compute.js").then((m) => m.initCompute()).catch((e) => console.error(e)); }
  // Don't burn decode+render behind the Compare overlay — pause the main viewer there.
  const p = window.__ares;
  if (p) { if (name === "compare" && p.isPlaying) { p.pause(); $("play").textContent = "▶"; } }
  // Persist the active tab (tabs must survive a reload) — every switch overwrites it, so
  // the LAST tab the user was on is always what comes back. try/catch: localStorage can throw in
  // rare privacy-mode contexts; a switch must never fail just because persistence did.
  try { localStorage.setItem(TAB_KEY, name); } catch { /* ignore */ }
}
// [role="tab"] scope is load-bearing: the header bar also holds the shading segmented control and
// the theme toggle. Unscoped, clicking those called setTab(undefined) → every .viewer-only element
// (both rails, transport, gizmo) got display:none and the Viewer tab de-selected.
for (const b of document.querySelectorAll('#tabs button[role="tab"]')) b.onclick = () => setTab(b.dataset.tab);
// Restore the persisted tab on load. First run (nothing stored yet, or a stale/unknown value —
// e.g. localStorage was cleared) defaults to Viewer, which is already the HTML's own default
// markup state, so routing through setTab("viewer") here is a no-op beyond re-persisting the key.
{
  let initialTab = "viewer";
  try {
    const saved = localStorage.getItem(TAB_KEY);
    const mapped = LEGACY_TAB_MAP[saved] || saved;
    if (mapped && VALID_TABS.has(mapped)) initialTab = mapped;
  } catch { /* ignore */ }
  setTab(initialTab);
}

// --- Slim edge rails (Viewer): drag-resize from the inner edge + one-click collapse to a ~30px
// icon strip, persisted in localStorage. Overlay rails (position:fixed) — they never reflow the
// canvas, fit() keeps sizing the full window. Works independent of the player/clip loading.
function initRail(el, { wKey, cKey, defaultW, min, max, side }) {
  const handle = el.querySelector(".railHandle");
  const chevron = el.querySelector(".railChevron");
  const glyphs = side === "left" ? { open: "‹", closed: "›" } : { open: "›", closed: "‹" };
  const applyW = (w) => { el.style.width = Math.max(min, Math.min(max, w)) + "px"; };
  applyW(Number(localStorage.getItem(wKey)) || defaultW);
  const setCollapsed = (c) => {
    el.classList.toggle("collapsed", c);
    chevron.textContent = c ? glyphs.closed : glyphs.open;
    localStorage.setItem(cKey, c ? "1" : "0");
  };
  setCollapsed(localStorage.getItem(cKey) === "1");
  chevron.onclick = (e) => { e.stopPropagation(); setCollapsed(!el.classList.contains("collapsed")); };
  el.addEventListener("click", () => { if (el.classList.contains("collapsed")) setCollapsed(false); });
  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener("pointerdown", (e) => {
    if (el.classList.contains("collapsed")) return;
    dragging = true; el.classList.add("dragging");
    startX = e.clientX; startW = el.getBoundingClientRect().width;
    try { handle.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    e.preventDefault();
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    applyW(startW + (side === "left" ? dx : -dx));   // left rail: drag its right edge right = wider
  });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false; el.classList.remove("dragging");
    localStorage.setItem(wKey, parseInt(el.style.width, 10));
  };
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
  return { setCollapsed, isCollapsed: () => el.classList.contains("collapsed") };
}
const hudRail = initRail($("hud"), { wKey: "ares.hud.w", cKey: "ares.hud.collapsed", defaultW: 260, min: 200, max: 420, side: "left" });
const editRail = initRail($("editPanel"), { wKey: "ares.rail.w", cKey: "ares.rail.collapsed", defaultW: 270, min: 220, max: 480, side: "right" });
// Section twists inside the edit rail (Crop/Tools/Ranges/SAM/Bake) — transient, not persisted;
// all start open so nothing looks accidentally empty on first run.
for (const sec of document.querySelectorAll("#editPanel .railSec")) {
  sec.querySelector(".secHead").onclick = () => sec.classList.toggle("collapsed");
}

// MEASURED baselines for THIS 272-frame capture — independent of ARES's own bytes (nothing here
// is derived from how ARES compresses). See docs/size-comparison.md for the measurement method.
//  raw   : the source folder, measured — OBJ 485.7 MB geom + PNG 1135.8 MB tex = 1621.5 MB / 544 files.
//  draco : REAL draco3d encode of every frame (pos14/uv12/nrm10 edgebreaker) = 20.3 MB geom, measured;
//          a per-frame glTF/Draco sequence ships the capture's own PNG atlases (1135.8 MB) / 272 files.
//  4dv   : 4DViews .4ds — proprietary, TIER-DEPENDENT. Three reference points (highest MEASURED):
//          720p streaming ~2 Mbps → 2.3 MB/9s (est., lower fidelity); standard ~16 Mbps → 18 MB/9s
//          (est.); and DESKTOP_HR — a REAL .4ds measured by data rate: 238,222,244 B / 15.2 s @ 29.99
//          fps = 125 Mbps → ~135 MB for this 9 s clip (≈2× ARES). The HR tier is the one whose fidelity
//          is comparable-to-above ARES (1024² + full mesh); ARES beats it ~2× there. Cross-content
//          normalization (different capture), so ±. Band spans 720p→HR.
const BASE = {
  raw:   { geom: 485.7, tex: 1135.8, total: 1621.5, files: 544, tag: "measured" },
  draco: { geom: 20.3,  tex: 1135.8, total: 1156.1, files: 272, tag: "measured" },
  fourdviews: { low: 2.3, high: 135, hrMeasured: 135, files: 1, tag: "720p est → HR measured" },
};
/** The capture BASE was measured on — 272 frames, ~8.8k verts. Used to derive per-vertex-frame rates
 *  for the Draco estimate on OTHER clips; never quoted as another clip's own numbers. */
const BASE_FRAMES = 272, BASE_VERTS = 8800;

/**
 * The RAW source THIS clip was actually made from, read from its own provenance sidecar
 * (`<clip>.ares.meta.json`, written at encode time / backfilled by tools/measure-source.mjs).
 *
 * This exists because the comparison used to quote BASE — hardcoded constants measured once on the
 * 272-frame daniel capture — for EVERY clip. So a 4910-frame SVF conversion from a 23.9 GB source
 * claimed its origin was "1621.5 MB · 544 files": another clip's numbers, presented as this one's.
 * BASE is now only a fallback for clips with no provenance, and it says so on screen.
 *
 * UNITS: MEBIbytes, because that is what the rest of this readout already uses — `aresTotalMB` is
 * `fileKB/1024` = bytes/1024². Converting the sidecar's raw bytes with /1e6 instead would divide SI
 * MB by MiB and inflate every savings ratio by 4.9% (it read 27.8× where the truth is 26.5×). The
 * sidecar stores BYTES precisely so this conversion is the only place units are decided.
 *
 * For the record, BASE.raw's numbers were never WRONG — 485.7 + 1135.8 MiB reproduces exactly when
 * you measure that capture today. The only fault was scope: they described one capture and were
 * quoted for all of them.
 */
function rawSourceOf(meta) {
  const s = meta && meta.source;
  if (!s || !(s.totalBytes > 0)) return null;
  const MB = (b) => b / 1048576;
  return {
    geom: MB(s.meshBytes ?? 0), tex: MB(s.texBytes ?? 0), total: MB(s.totalBytes),
    files: s.fileCount ?? ((s.meshFrames ?? 0) + (s.atlasFrames ?? 0)),
    frames: s.meshFrames ?? 0, verts: s.vertexCount ?? 0,
    dir: s.dir || "", kind: s.kind || "source",
    measured: true,
  };
}
const BAR_COLORS = { raw: "#8a6d3b", fourdviews: "var(--series-b)", draco: "var(--text-mid)", ares: "var(--accent)" };

function fit() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  return [Math.max(1, (canvas.clientWidth * dpr) | 0), Math.max(1, (canvas.clientHeight * dpr) | 0)];
}

function fail(msg) {
  const el = $("err");
  el.style.display = "block";
  el.textContent = msg;
  console.error(msg);
}

// Draco decode is CPU-side and needs a Worker; meshopt dequant is a GPU shader. The ~5× is a
// rough, clearly-labelled qualitative note (not a size figure — the sizes above are all measured).
const DRACO_DECODE_MULT = 5;
const sizeMB = (m) => (m >= 1024 ? (m / 1024).toFixed(2) + " GB" : m.toFixed(m < 100 ? 1 : 0) + " MB");

function buildRow(metric, ares, draco) {
  const tr = document.createElement("tr");
  const m = document.createElement("td"); m.className = "metric"; m.textContent = metric;
  const a = document.createElement("td"); a.className = "ares"; a.innerHTML = ares;
  const d = document.createElement("td"); d.className = "draco"; d.innerHTML = draco;
  tr.append(m, a, d); return tr;
}

function renderHUD(s) {
  const hasVideo = /video/.test(s.textureLabel);
  // ARES's own split, measured live from the loaded container (geom/frame × frames; texture = remainder).
  const aresGeomMB = (s.avgFrameKB * s.frameCount) / 1024;
  const aresTotalMB = s.fileKB / 1024;
  const aresTexMB = Math.max(0, aresTotalMB - aresGeomMB);

  // Collapsed-strip readout: the two numbers worth glancing at without expanding the rail.
  $("hudMiniFps").textContent = s.fps.toFixed(0) + "fps";
  $("hudMiniMB").textContent = sizeMB(aresTotalMB);

  // Bottom status bar: what's actually PLAYING. Self-throttling — it early-outs unless the clip
  // changed, so calling it on every stats tick is free.
  diagMod?.setDiagClip({
    name: (SRC || "").replace("./", ""),
    sizeMB: aresTotalMB, frames: s.frameCount, fps: 30, durationS: s.frameCount / 30,
  });

  // THIS clip's real origin, from its own provenance. Everything below prefers it over BASE.
  const raw = rawSourceOf(clipMeta) || { ...BASE.raw, frames: BASE_FRAMES, verts: BASE_VERTS, measured: false, kind: "source", dir: "" };
  // Draco for THIS clip: its geometry scales with vertex-frames, so derive a rate from the one real
  // draco3d encode we have (BASE, measured) and scale it. Its TEXTURE is not an estimate at all — a
  // Draco-GLB sequence ships the source's own PNGs, which we now know exactly. Marked est. where est.
  const dracoRatePerVertFrame = BASE.draco.geom / (BASE_FRAMES * BASE_VERTS);   // MiB per vertex-frame
  const vertFrames = (raw.verts || s.vertexCount) * (raw.frames || s.frameCount);
  const dracoGeom = raw.measured ? dracoRatePerVertFrame * vertFrames : BASE.draco.geom;
  const dracoTex = raw.measured ? raw.tex : BASE.draco.tex;
  const dracoTotal = dracoGeom + dracoTex;
  const dracoFiles = raw.frames || BASE.draco.files;
  const est = raw.measured ? "<small>est.</small>" : "<small>measured ✓</small>";

  const rows = [
    ["Delivered size", "<span class='good'>" + sizeMB(aresTotalMB) + " · 1 file</span>",
      sizeMB(dracoTotal) + " · " + dracoFiles + " files " + est],
    ["Delivery", "<span class='good'>1 file · 1 request</span>", dracoFiles + " files · " + dracoFiles + " req"],
    ["Geometry", (s.geometryMode || "meshopt intra") + " · " + aresGeomMB.toFixed(1) + " MB",
      "Draco " + dracoGeom.toFixed(1) + " MB " + est],
    ["Texture", hasVideo ? s.textureLabel + " · " + aresTexMB.toFixed(1) + " MB · <span class='good'>HW</span>" : s.textureLabel,
      hasVideo ? "PNG ×" + dracoFiles + " · " + sizeMB(dracoTex) + (raw.measured ? " <small>measured ✓</small>" : "") : "—"],
    ["Geom decode / frame", s.decodeMsPerFrame.toFixed(2) + " ms <small>main thread</small>",
      "~" + (s.decodeMsPerFrame * DRACO_DECODE_MULT).toFixed(1) + " ms <small>needs Worker</small>"],
    ["Dequantize", "<span class='good'>GPU shader</span>", "CPU"],
    ["Time to first frame", "<span class='good'>" + s.ttffMs.toFixed(0) + " ms</span>", "1.5–4 s <small>est.</small>"],
    ["Render", s.fps.toFixed(0) + " fps · " + s.cpuMsPerFrame.toFixed(2) + " ms CPU/f", "per-frame fetch+parse"],
    ["Frame", (s.frameIndex + 1) + " / " + s.frameCount, "—"],
  ];
  $("cmpBody").replaceChildren(...rows.map((r) => buildRow(r[0], r[1], r[2])));

  // Size bars (log scale so 1.6 GB → 18 MB all read): raw / 4DViews / Draco / ARES. All measured
  // except 4DViews (native-bitrate est.). Each bar annotates its geom+tex split where it has one.
  if (hasVideo) {
    $("sizebars").hidden = false;
    const rows = [
      [raw.measured ? `Raw ${raw.kind}` : "Raw OBJ+PNG", raw.total, BAR_COLORS.raw,
        raw.measured ? `measured · ${raw.files} files` : "measured on a DIFFERENT capture",
        raw.measured
          ? `THIS clip's own source: ${raw.files} files, ${raw.frames} frames — geom ${raw.geom.toFixed(0)} + tex ${raw.tex.toFixed(0)} MB.${raw.dir ? "\n" + raw.dir : ""}`
          : `No provenance sidecar for this clip — these are the 272-frame Daniel capture's numbers, NOT this clip's. Run: node tools/measure-source.mjs <clip>.ares`],
      ["Draco-GLB", dracoTotal, BAR_COLORS.draco, raw.measured ? "geom est. · PNG measured" : BASE.draco.tag,
        `geom ${dracoGeom.toFixed(1)} + PNG ${dracoTex.toFixed(0)} · per-frame sequence. ` +
        (raw.measured
          ? `Geometry scaled from a real draco3d encode (${BASE.draco.geom} MB over ${BASE_FRAMES}f × ~${(BASE_VERTS/1000).toFixed(1)}k verts) to this clip's ${(vertFrames/1e6).toFixed(1)}M vertex-frames. The PNG side is not an estimate — a Draco-GLB sequence ships this source's OWN atlases.`
          : `Measured on the 272-frame Daniel capture.`)],
      // Cross-content by construction: measured on a DIFFERENT 4DViews capture and normalised to a
      // 9s equivalent. It's a fidelity-tier reference point, not this clip's origin — labelled so.
      ["4DViews", [BASE.fourdviews.low, BASE.fourdviews.high], BAR_COLORS.fourdviews, "other capture · 9s-equiv ref",
        "native temporal codec. 720p streaming ~2 Mbps → 2.3 MB (est.). DESKTOP_HR MEASURED (real .4ds, 125 Mbps → ~135 MB/9s ≈ 2× ARES). Byte-proven internal split (9s-equiv): geometry ~5.5 MiB (temporal mesh, 4%) + texture ~130 MiB (per-frame 1440² GPU-block, 96%, NO video compression) — mirror image of ARES."],
      ["ARES", aresTotalMB, BAR_COLORS.ares, s.geometryMode && s.geometryMode.includes("temporal") ? "measured · P2" : "measured · intra",
        `geom ${aresGeomMB.toFixed(1)} + VP9 ${aresTexMB.toFixed(1)}`],
    ];
    const maxL = Math.log10(BASE.raw.total * 1.1);
    const minL = Math.log10(2);   // floor at 2 MB so the ~2.3 MB 4DViews streaming tier stays visible
    const pos = (v) => Math.max(0, Math.min(100, ((Math.log10(Math.max(v, 2)) - minL) / (maxL - minL)) * 100));
    $("barsBody").replaceChildren(...rows.map(([name, val, color, tag, split]) => {
      const isRange = Array.isArray(val);
      const left = isRange ? pos(val[0]) : 0;
      const w = Math.max(2, (isRange ? pos(val[1]) : pos(val)) - left);
      const el = document.createElement("div"); el.className = "bar";
      const n = document.createElement("div"); n.className = "name"; n.textContent = name;
      const track = document.createElement("div"); track.className = "track"; track.title = split;
      const fill = document.createElement("div"); fill.className = "fill";
      fill.style.marginLeft = left + "%"; fill.style.width = w + "%"; fill.style.background = color;
      if (isRange) fill.style.opacity = "0.6"; // a range reads as a translucent band, not a solid value
      track.append(fill);
      const v = document.createElement("div"); v.className = "val";
      v.innerHTML = (isRange ? val[0].toFixed(1) + "–" + sizeMB(val[1]) : sizeMB(val)) +
        ' <small style="color:var(--text-faint)">' + tag + "</small>";
      el.append(n, track, v); return el;
    }));
  } else {
    $("sizebars").hidden = true;
  }

  // Condensed readouts pinned at the bottom of the Media rail — owner rule: NONE over one line.
  // The full ARES-vs-Draco table + size bars are still there, folded into the <details> above.
  {
    const host = $("readoutLines");
    if (host) {
      const m = clipMeta || {};
      const coh = m.coherent || {};
      const rl = (k, v) => `<div class="rl"><span class="k">${k}</span><span class="v">${v}</span></div>`;
      const texShort = hasVideo ? s.textureLabel.replace(/\s*·.*/, "").replace(/\s*video\s*/i, " ").trim() : s.textureLabel;
      let prov;
      if (m.pipeline === "coherent") prov = `coherent · gop ${coh.gop ?? "?"} · stretch ${coh.stretchCut ?? "?"}×`;
      else if (m.pipeline === "native") prov = "native per-frame · intra";
      else if (m.pipeline) prov = String(m.pipeline);
      else prov = `<span style="color:var(--text-faint)">no recipe sidecar</span>`;
      host.innerHTML =
        rl("size", `<b>${sizeMB(aresTotalMB)}</b> · ${s.frameCount}f · ${(s.frameCount / (s.fps || 30)).toFixed(1)}s`) +
        rl("split", `geom ${aresGeomMB.toFixed(1)} + tex ${aresTexMB.toFixed(1)} MB`) +
        (hasVideo ? rl("texture", `${texShort} · HW`) : "") +
        rl("render", `${s.fps.toFixed(0)}fps · ${s.decodeMsPerFrame.toFixed(2)}ms/f · TTFF ${s.ttffMs.toFixed(0)}ms`) +
        rl("frame", `${s.frameIndex + 1} / ${s.frameCount}`) +
        rl("recipe", `${prov} <button class="rlBtn" id="thumbBtn" title="set this clip's library thumbnail from the current view">📷</button>${m.pipeline ? '<button class="rlBtn" id="recipeBtn" title="copy this clip\'s full conversion recipe to re-use on another clip">⧉</button>' : ""}`);
      const rb = $("recipeBtn"); if (rb) rb.onclick = showRecipe;
      const tb = $("thumbBtn"); if (tb) tb.onclick = captureThumb;
    }
  }

  // Transport readout. The scrub SLIDER is gone — the timeline strip is the scrubber now, and it
  // moves itself (tlWatch's rAF), so there is nothing to push here but the numbers.
  const dur = s.frameCount / 30;
  $("time").textContent = `${s.frameIndex + 1}/${s.frameCount} · ${((s.frameIndex + 1) / 30).toFixed(2)}s / ${dur.toFixed(1)}s`;

  // The headline: THIS clip vs the raw source THIS clip came from. When we have provenance it names
  // the origin and the ratio is a real measurement ÷ a real measurement; when we don't, it says so
  // rather than quietly borrowing another capture's numbers.
  const head = `<b>${s.frameCount} frames · ${(s.vertexCount / 1000).toFixed(1)}k verts · ${dur.toFixed(1)}s.</b> `;
  const origin = raw.dir ? raw.dir.split(/[\\/]/).filter(Boolean).pop() : "";
  $("cmp").innerHTML = !hasVideo
    ? head + `Single request, GPU-side dequant, hardware-ready texture path.`
    : raw.measured
      ? head +
        `Converted from <b>${sizeMB(raw.total)}</b> of raw ${raw.kind} (<b>${raw.files} files</b>${origin ? `, <span title="${raw.dir}">${origin}</span>` : ""}) ` +
        `→ <b class="good">${(raw.total / aresTotalMB).toFixed(1)}× smaller</b> in 1 request. ` +
        `<small>Source split: geom ${raw.geom.toFixed(0)} + tex ${raw.tex.toFixed(0)} MB. ` +
        `Measured on this clip's own source at encode time — not a reference figure.</small>`
      : head +
        `<b class="warn">No provenance for this clip</b> — its real source size is unknown, so the bars below fall back to the ` +
        `272-frame Daniel capture's measurements, which are <b>not this clip's</b>. ` +
        `<small>Fix: <code>node tools/measure-source.mjs ${(SRC || "").replace("./", "")}</code> (needs the source folder to still exist).</small>`;
}

// --- Mesh editor v1: crop-box preview (GPU discard) + bake through the local encoder ---------
function initEditor(player) {
  // The Edit rail is always here on the Viewer tab — no button gates it, and there is no "edit
  // mode" to be in or out of. Collapsing it is the rail's own ‹ chevron (or E), same as Media.

  // SAM service status + in-app launch (serve.mjs /sam/start streams SSE progress; the
  // service itself is spawned hidden by samEnsure — no terminal, no separate launcher).
  const samStatus = $("samStatus"), samStart = $("samStart"), samDot = $("samDot");
  // Muted state colors in the app's own idiom (ui-design-notes.md: no neon, no defaults).
  const SAM_DOT = { ready: "#8fd694", loading: "#d9a13a", failed: "#d97a6a", off: "#5a5852" };
  let samPoll = 0;
  async function samRefresh() {
    clearTimeout(samPoll);
    let h = null;
    try { h = await fetch("/sam/health").then((r) => r.json()); } catch {}
    if (h && h.ok) { samDot.style.background = SAM_DOT.ready; samStatus.textContent = `ready · ${h.model} on ${h.device}`; samStart.style.display = "none"; }
    else if (h && h.loading) { samDot.style.background = SAM_DOT.loading; samStatus.textContent = "model loading…"; samStart.style.display = "none"; }
    else if (h && h.error) { samDot.style.background = SAM_DOT.failed; samStatus.textContent = "load failed"; samStatus.title = "The SAM service failed to load its model — see tools/sam-service/sam-service.log for the traceback. Press Start SAM to retry."; samStart.style.display = ""; }
    else { samDot.style.background = SAM_DOT.off; samStatus.textContent = "not running"; samStart.style.display = ""; }
    syncSamTextGate(h);
    // Keep polling while EITHER the tracker or the independent, later-loading text/concept model
    // is still coming up — SAM_TEXT's model loads AFTER the tracker, so h.ok can go true first.
    if (h && (h.loading || h.textLoading)) samPoll = setTimeout(samRefresh, 3000);
  }
  samStart.onclick = () => {
    samStart.disabled = true;
    samDot.style.background = SAM_DOT.loading;
    samStatus.textContent = "starting…";
    const es = new EventSource("/sam/start");
    es.addEventListener("log", (e) => { try { samStatus.textContent = JSON.parse(e.data); } catch {} });
    es.addEventListener("done", () => { es.close(); samStart.disabled = false; samRefresh(); });
    es.addEventListener("error", () => { es.close(); samStart.disabled = false; samRefresh(); });
  };

  // Viewport shading — ONE mutually-exclusive segmented control in the header (owner ask #15):
  // shaded (textured) / clay (untextured, judge FORM) / wire (topology). Replaces the old
  // Wireframe toggle that lived in the Edit rail. Keys 1/2/3; W still toggles wire<->shaded.
  const shadeSeg = $("shadeSeg");
  let shadeMode = "shaded";
  const applyShade = (mode) => {
    shadeMode = mode;
    player.setWireframe(mode === "wire");
    player.setTextured(mode !== "clay");
    if (shadeSeg) for (const b of shadeSeg.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.dataset.shade === mode));
    }
  };
  if (shadeSeg) shadeSeg.onclick = (e) => {
    const b = e.target.closest("button[data-shade]");
    if (b) applyShade(b.dataset.shade);
  };
  // Exposed so the keyboard handler and view-preset code can drive/read the mode.
  window.__aresShade = { set: applyShade, get: () => shadeMode };

  const aabb = player.getAabb();

  // ---- Ground grid (header bar, ON by default) -----------------------------------------------
  // Not a tool and never was: it's a spatial reference, like shading and projection, so it sits in
  // the header with them and starts on. The infinite/fading grid itself is core (overlay.ts +
  // both renderers); this is the control surface: an on/off, an increment, and a live readout.
  //
  // UNITS: the clips do not agree. daniel-*/Microsoft captures are MILLIMETRES, 4DViews bakes are
  // METRES, so a hardcoded "mm" would be a lie half the time.
  // Infer from the AABB instead — the subject is a person either way, so a metre-scale clip is ~2
  // tall and a mm-scale one ~2000. Everything the user sees is then real-world size, and the world
  // step handed to the player is converted through `worldPerMm`.
  const clipSpanY = Math.abs(aabb.max[1] - aabb.min[1]);
  const worldPerMm = clipSpanY > 100 ? 1 : 0.001;      // world units per millimetre
  // Pick the unit off the MAGNITUDE, not the signed value: a floor at −3526 mm is −3.53 m, and
  // testing `mm >= 1000` on a negative silently never promotes it.
  const fmtMm = (mm) => {
    const a = Math.abs(mm);
    if (a >= 1000) return `${+(mm / 1000).toFixed(2)} m`;
    if (a >= 10) return `${+mm.toFixed(0)} mm`;
    return `${+mm.toFixed(a < 1 ? 2 : 1)} mm`;
  };

  const GRID_STEPS_MM = [0, 1, 5, 10, 50, 100, 250, 500, 1000, 5000];
  const gridSel = $("gridStep"), gridToggle = $("gridToggle"), gridRead = $("gridRead");
  if (gridSel) {
    gridSel.innerHTML = GRID_STEPS_MM.map((mm) => `<option value="${mm}">${mm === 0 ? "auto" : fmtMm(mm)}</option>`).join("");
    const savedStep = Number(localStorage.getItem("ares.grid.step") || 0);
    gridSel.value = String(GRID_STEPS_MM.includes(savedStep) ? savedStep : 0);
    const applyStep = () => {
      const mm = Number(gridSel.value);
      player.setGridStep(mm * worldPerMm);            // 0 stays 0 = auto
      localStorage.setItem("ares.grid.step", String(mm));
    };
    applyStep();
    gridSel.onchange = applyStep;
  }
  // Grid ON by default (the player's own default too); the choice persists.
  const setGridOn = (on) => {
    gridToggle?.setAttribute("aria-pressed", String(on));
    if (gridSel) gridSel.disabled = !on;
    if (gridRead) gridRead.style.opacity = on ? "" : ".35";
    player.setGrid(on);
    localStorage.setItem("ares.grid.on", on ? "1" : "0");
  };
  setGridOn(localStorage.getItem("ares.grid.on") !== "0");
  if (gridToggle) gridToggle.onclick = () => setGridOn(gridToggle.getAttribute("aria-pressed") !== "true");
  // Live cell readout. On "auto" the drawn cell changes with the zoom, so a static label would be
  // wrong most of the time — report what the shader is actually drawing. When the fine level has
  // faded past halfway the cell you can still SEE is the coarse one, so that's the honest number.
  (function gridReadWatch() {
    requestAnimationFrame(gridReadWatch);
    if (!gridRead || gridToggle?.getAttribute("aria-pressed") !== "true") return;
    const lod = player.getGridLod();
    const mm = (lod.fineFade >= 0.5 ? lod.step0 : lod.step1) / worldPerMm;
    const txt = fmtMm(mm);
    if (gridRead.textContent !== txt) gridRead.textContent = txt;
  })();

  const AXES = [["X", 0], ["Y", 1], ["Z", 2]];
  const AXIS_NAME = ["X", "Y", "Z"];
  const AXIS_TIP = {
    X: "X crop — trim left/right.",
    Y: "Y crop — trim vertically: the low plane cuts the FLOOR away, the high plane cuts anything above the subject.",
    Z: "Z crop — trim front/back depth. Useful for stray background geometry behind the subject.",
  };

  // ---- Crop = rulers + viewport guides (like rulers in Photoshop / After Effects, not sliders
  // in the tool panel). Six sliders for a spatial task meant
  // reading numbers to do something your eyes should do directly.
  //
  // The six planes live HERE as percentages of the clip's AABB; the rail shows a readout only. Over
  // the viewport, four guide lines ARE the planes for the two axes currently facing the camera —
  // drag one and you drag that plane, in millimetres. The third axis points into the screen (you
  // cannot drag depth on a 2D screen without lying about it), so it's shown greyed; orbit or press
  // 1/3/7 to bring it round. Which axis is which is recomputed from the live camera every move.
  const cropPct = [[0, 100], [0, 100], [0, 100]];   // per axis: [minPct, maxPct]
  const cropSpan = (i) => aabb.max[i] - aabb.min[i];
  const cropMm = (i, pct) => aabb.min[i] + (pct / 100) * cropSpan(i);

  const cropBox = () => {
    const min = [0, 0, 0], max = [0, 0, 0];
    let active = false;
    for (const [, i] of AXES) {
      const lo = cropPct[i][0] / 100, hi = cropPct[i][1] / 100;
      min[i] = aabb.min[i] + Math.min(lo, hi) * cropSpan(i);
      max[i] = aabb.min[i] + Math.max(lo, hi) * cropSpan(i);
      if (lo > 0.001 || hi < 0.999) active = true;
    }
    return active ? { min, max } : null;
  };

  // ---- Segmentation palette (SAM-demo idiom): each range/selection gets its own vivid color,
  // cycling like Meta's SAM object colors. Used by the mask preview tint, the marquee, and the
  // timeline range bars; persisted on the range (cosmetic field) so colors survive reloads.
  const SEG_PALETTE = ["#1E90FF", "#FF40FF", "#39FF14", "#FFA01E", "#00E5FF", "#AA5AFF"];
  const nextSegColor = () => SEG_PALETTE[edits.ranges.length % SEG_PALETTE.length];
  const activeSegColor = () => (activeRange && activeRange.color) || nextSegColor();
  const hexRgb = (hex) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];

  // ---- Timeline-ranged deletion (editor v2 §6/§11): the box sliders author DELETE keyframes while
  // a range session is active; regions interpolate between keyframes; preview == bake (shared rule).
  const clipBase = SRC.replace("./", "").replace(/\.ares$/i, "");
  const edits = { aresEdits: 1, source: SRC.replace("./", ""), fps: 30, frameCount: 0, ranges: [] };
  let activeRange = null;
  let saveTimer = 0;
  const doSave = () => fetch("/edits/" + clipBase, { method: "POST", body: JSON.stringify(edits, null, 1) }).catch(() => {});
  const saveEdits = () => { clearTimeout(saveTimer); saveTimer = setTimeout(doSave, 600); };
  const flushSave = () => { clearTimeout(saveTimer); saveTimer = 0; doSave(); };
  const curFrame = () => player.getStats().frameIndex;
  const preview = () => { player.setEditPreview(edits.ranges.length ? edits : null); saveEdits(); renderRanges(); };

  // ---- Undo/redo. A snapshot is the WHOLE editable state — ranges + model transform + clip trim.
  //
  // It used to be `edits.ranges` alone, which made Ctrl+Z silently dead for any tool whose state
  // lived elsewhere: rotate/move/scale and the trim brackets all pushed a "before" that compared
  // EQUAL to the after (their change wasn't in ranges), so pushUndoIfChanged took the "no-op
  // gesture: skip" branch and nothing was ever recorded. Undo must cover every tool or it can't be
  // trusted by any of them — so the snapshot is the state, not one field of it.
  //
  // One snapshot per user GESTURE, not per input tick: discrete actions (button clicks, dropdown/
  // checkbox changes, marquee/brush/SAM apply, Delete) push immediately via doMutation(); continuous
  // drags (a rotate drag, strength slider, color scrub, dst-frames typing) coalesce into ONE step via
  // gestureBegin()/gestureCommit() — begin captures the pre-drag baseline once, commit (pointerup/
  // change/blur/Enter) is the only point that actually pushes it. Cap 50 entries each side; any push
  // clears the redo stack (standard undo law).
  const UNDO_CAP = 50;
  let undoStack = [], redoStack = [];
  let gestureBefore = null;
  /** The full editable state: every tool's state, so every tool is undoable. `xf`/`trimIn`/`trimOut`
   *  are declared BELOW this point, hence the typeof guards — but this is only ever CALLED from
   *  handlers, long after the whole of initEditor has run, so in practice they're always defined. */
  const snapshotState = () => JSON.parse(JSON.stringify({
    ranges: edits.ranges,
    crop: cropPct,
    transform: typeof xf === "undefined" ? null : xf,
    trim: typeof trimIn === "undefined" ? null : { in: trimIn, out: trimOut },
  }));
  function updateUndoUI() {
    const ub = $("undoBtn"), rb = $("redoBtn");
    if (ub) ub.disabled = !undoStack.length;
    if (rb) rb.disabled = !redoStack.length;
  }
  function pushUndoIfChanged(before) {
    if (JSON.stringify(before) === JSON.stringify(snapshotState())) return;   // no-op gesture: skip
    undoStack.push(before);
    if (undoStack.length > UNDO_CAP) undoStack.shift();
    redoStack = [];
    updateUndoUI();
  }
  /** Wrap a single discrete mutation (one click/change = one undo step). Flushes any pending
   *  coalesced gesture FIRST so an interrupted drag still lands as its own step, in order. */
  function doMutation(fn) {
    gestureCommit();
    const before = snapshotState();
    fn();
    pushUndoIfChanged(before);
  }
  /** Continuous-drag coalescing: call at the top of every input-tick handler (idempotent — only
   *  the FIRST tick of a gesture actually captures the baseline) and gestureCommit() on the
   *  gesture's natural end (change event for range/color inputs, blur/Enter for text/number). */
  function gestureBegin() { if (gestureBefore === null) gestureBefore = snapshotState(); }
  function gestureCommit() {
    if (gestureBefore === null) return;
    const before = gestureBefore;
    gestureBefore = null;
    pushUndoIfChanged(before);
  }
  function syncRangeButtons() {
    $("rangeStart").disabled = !!activeRange;
    $("rangeKey").disabled = !activeRange;
    $("rangeEnd").disabled = !activeRange;
  }
  /** Put the whole editable state back. Each part goes through its OWN apply path (applyXf /
   *  applyTrim / preview) rather than being poked into place, so the player, the rail controls and
   *  the sidecar can't drift out of agreement with the restored state. */
  function restoreState(snap) {
    edits.ranges = snap.ranges;
    if (activeRange) activeRange = edits.ranges.find((r) => r.id === activeRange.id) || null;
    // cropPct is a const array — copy INTO it rather than rebinding, since cropBox()/renderCropUi()
    // and the guide drag all close over this exact array.
    if (snap.crop) for (let i = 0; i < 3; i++) { cropPct[i][0] = snap.crop[i][0]; cropPct[i][1] = snap.crop[i][1]; }
    if (snap.transform) { xf = JSON.parse(JSON.stringify(snap.transform)); applyXf({ save: false }); }
    if (snap.trim) { trimIn = snap.trim.in; trimOut = snap.trim.out; applyTrim({ save: false }); }
    syncRangeButtons();
    updateUndoUI();
    applyCrop();   // re-applies the restored planes to the live preview + readout
    preview();     // setEditPreview + save + re-render (the save covers the writes above)
  }
  function undo() {
    gestureCommit();
    if (!undoStack.length) return;
    const prev = undoStack.pop();
    redoStack.push(snapshotState());
    if (redoStack.length > UNDO_CAP) redoStack.shift();
    restoreState(prev);
  }
  function redo() {
    gestureCommit();
    if (!redoStack.length) return;
    const next = redoStack.pop();
    undoStack.push(snapshotState());
    if (undoStack.length > UNDO_CAP) undoStack.shift();
    restoreState(next);
  }
  function deleteActiveRange() {
    if (!activeRange) return;
    gestureCommit();
    const before = snapshotState();
    const idx = edits.ranges.indexOf(activeRange);
    if (idx === -1) { activeRange = null; syncRangeButtons(); return; }
    edits.ranges.splice(idx, 1);
    activeRange = null;
    syncRangeButtons();
    pushUndoIfChanged(before);
    preview();
  }
  $("undoBtn").onclick = (e) => { e.stopPropagation(); undo(); };
  $("redoBtn").onclick = (e) => { e.stopPropagation(); redo(); };

  const captureKeyframe = () => {
    if (!activeRange) return;
    const box = cropBox();
    if (!box) return; // full box = nothing selected to delete
    const f = curFrame();
    const boxVol = { type: "box", min: box.min, max: box.max };
    const kf = activeRange.keyframes.find((k) => k.frame === f);
    // The sliders own exactly ONE box volume per keyframe; brush/marquee volumes are preserved.
    if (kf) kf.volumes = [boxVol, ...kf.volumes.filter((v) => v.type !== "box")];
    else { activeRange.keyframes.push({ frame: f, volumes: [boxVol] }); activeRange.keyframes.sort((a, b) => a.frame - b.frame); }
    if (f > activeRange.endFrame) activeRange.endFrame = f;
    preview();
  };

  const applyCrop = () => {
    const box = cropBox();
    renderCropRead(box);
    if (activeRange) { player.setCrop(null); captureKeyframe(); }
    else player.setCrop(box);   // classic crop preview (keep-box)
    cropSig = "";               // force the guides to redraw against the new planes
  };
  $("cropReset").onclick = () => {
    gestureCommit();   // flush any in-flight guide drag as its own step first
    const before = snapshotState();
    for (const [, i] of AXES) { cropPct[i][0] = 0; cropPct[i][1] = 100; }
    applyCrop();
    pushUndoIfChanged(before);
  };

  // ---- Model transform (Unity keys: W move · E rotate · R scale · F focus) --------------------
  // The import-time orientation fix. Captures do not agree on up or origin: Microsoft/SVF is Y-up
  // millimetres, most scanner/DCC exports are Z-up, 4DViews is metres. Until a clip is upright and
  // standing on the ground plane you can't read the grid against it or compare two captures.
  //
  // Preview here is the SAME core evaluator the bake uses (packages/core/src/transform.ts), so what
  // you see is what the encoder writes into the geometry — the preview == bake law the edit ranges
  // already follow. Persisted per clip in the sidecar as `edits.transform`.
  // An UNTOUCHED clip is a true identity — centre "none", nothing moves. That is deliberate and it
  // is NOT the same as the centring default: `bottom` (stand it on the ground) is the right default
  // for an import, and the encoder applies it the moment any transform flag is given. But making it
  // the app's resting state would silently shift every existing clip the first time it loaded —
  // daniel-s0's X centre is −150 mm, so the subject would jump sideways with no one having asked.
  // Never move a clip unasked; ⊥ is one click away.
  const XF_DEFAULT = () => ({ upAxis: "y", center: "none", scale: 1, rotate: [0, 0, 0], translate: [0, 0, 0] });
  let xf = XF_DEFAULT();
  let xTool = null;                       // "move" | "rotate" | "scale" | null

  const xfIsDefault = () => JSON.stringify(xf) === JSON.stringify(XF_DEFAULT());

  function applyXf({ save = true, syncInputs = true } = {}) {
    player.setModelTransform(xf);
    if (syncInputs) {
      for (const b of $("upSeg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.up === xf.upAxis));
      for (const b of $("ctrSeg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.ctr === xf.center));
      const set = (id, v) => { const e = $(id); if (e && document.activeElement !== e) e.value = String(+v.toFixed(4)); };
      set("xfScale", xf.scale);
      set("xfRotX", xf.rotate[0]); set("xfRotY", xf.rotate[1]); set("xfRotZ", xf.rotate[2]);
      set("xfPosX", xf.translate[0]); set("xfPosY", xf.translate[1]); set("xfPosZ", xf.translate[2]);
    }
    // Readout in the clip's real units — the same mm/m inference the grid and crop use.
    const b = player.getTransformedAabb();
    const u = (v) => v / worldPerMm;
    $("xfRead").textContent = `${fmtMm(u(b.max[0] - b.min[0]))} × ${fmtMm(u(b.max[1] - b.min[1]))} × ${fmtMm(u(b.max[2] - b.min[2]))} · floor ${fmtMm(u(b.min[1]))}`;
    // Only persist a transform that actually transforms — an untouched clip shouldn't grow the key.
    if (xfIsDefault()) delete edits.transform; else edits.transform = JSON.parse(JSON.stringify(xf));
    if (save) saveEdits();
  }

  $("upSeg").onclick = (e) => { const b = e.target.closest("button[data-up]"); if (b) { doMutation(() => { xf.upAxis = b.dataset.up; }); applyXf(); } };
  $("ctrSeg").onclick = (e) => { const b = e.target.closest("button[data-ctr]"); if (b) { doMutation(() => { xf.center = b.dataset.ctr; }); applyXf(); } };
  $("xfMm2M").onclick = () => { doMutation(() => { xf.scale = 0.001; }); applyXf(); };
  $("xfReset").onclick = () => { doMutation(() => { xf = XF_DEFAULT(); }); applyXf(); };
  $("xfFocus").onclick = () => player.focus();
  const bindXfNum = (id, read, write) => {
    const el = $(id);
    if (!el) return;
    el.oninput = () => {
      const v = Number(el.value);
      const ok = Number.isFinite(v) && (id !== "xfScale" || v !== 0);
      el.style.borderColor = ok ? "" : "#f0a3a3";
      if (!ok) return;                    // never write an unusable value through to the transform
      gestureBegin(); write(v); applyXf({ syncInputs: false });
    };
    el.onchange = () => gestureCommit();
    void read;
  };
  bindXfNum("xfScale", () => xf.scale, (v) => { xf.scale = v; });
  for (const [i, id] of ["xfRotX", "xfRotY", "xfRotZ"].entries()) bindXfNum(id, () => xf.rotate[i], (v) => { xf.rotate[i] = v; });
  for (const [i, id] of ["xfPosX", "xfPosY", "xfPosZ"].entries()) bindXfNum(id, () => xf.translate[i], (v) => { xf.translate[i] = v; });

  // Tool arming. A transform tool is mutually exclusive with the SELECTION tools (nav/box/brush/
  // SAM): both want the same left-drag, and silently letting a box-select drag also move the model
  // is exactly the kind of "two things on one gesture" bug that is impossible to diagnose later.
  function setXTool(name) {
    xTool = xTool === name ? null : name;
    for (const b of document.querySelectorAll("#editPanel .xtool")) b.setAttribute("aria-pressed", String(b.dataset.xtool === xTool));
    if (xTool) document.querySelector('#editPanel .tool[data-tool="nav"]')?.click();   // disarm selection
    $("xformOverlay").style.display = xTool ? "" : "none";
  }
  for (const b of document.querySelectorAll("#editPanel .xtool")) b.onclick = () => setXTool(b.dataset.xtool);

  // Viewport drag. Screen-space and deliberately simple: horizontal drag drives the value, an
  // axis key constrains it, Shift is fine. The numeric fields remain the exact-value path — a drag
  // is for finding the number, the field is for stating it.
  {
    const ov = $("xformOverlay");
    let held = null;
    addEventListener("keydown", (e) => { if (xTool && "xyzXYZ".includes(e.key)) held = e.key.toLowerCase(); });
    addEventListener("keyup", (e) => { if (held && e.key.toLowerCase() === held) held = null; });
    ov.addEventListener("pointerdown", (e) => {
      if (!xTool || e.button !== 0) return;
      e.preventDefault();
      try { ov.setPointerCapture(e.pointerId); } catch { /* ignore */ }
      const x0 = e.clientX, y0 = e.clientY;
      const start = JSON.parse(JSON.stringify(xf));
      // gestureBegin is the ONLY baseline capture. Taking a second one and pushing it separately at
      // pointerup would land the drag on the undo stack TWICE, so one Ctrl+Z would look like it did
      // nothing (it would be undoing the identical second step).
      gestureBegin();
      const span = Math.max(1e-6, player.getTransformedAabb().max[1] - player.getTransformedAabb().min[1]);
      const move = (ev) => {
        const dx = ev.clientX - x0, dy = ev.clientY - y0;
        const fine = ev.shiftKey ? 0.15 : 1;
        const axis = held ? { x: 0, y: 1, z: 2 }[held] : null;
        if (xTool === "rotate") {
          const deg = dx * 0.5 * fine;
          xf.rotate = [...start.rotate];
          xf.rotate[axis ?? 1] = start.rotate[axis ?? 1] + deg;   // no axis held -> yaw, the common case
        } else if (xTool === "scale") {
          // Exponential so drag-left shrinks symmetrically to drag-right growing, and it can never
          // cross zero (a zero scale collapses the model to a point and is unrecoverable by dragging).
          xf.scale = Math.max(1e-9, start.scale * Math.exp(dx * 0.005 * fine));
        } else {
          const k = span * 0.002 * fine;                          // drag distance scales with the clip
          xf.translate = [...start.translate];
          if (axis === null) { xf.translate[0] = start.translate[0] + dx * k; xf.translate[1] = start.translate[1] - dy * k; }
          else xf.translate[axis] = start.translate[axis] + dx * k;
        }
        applyXf({ save: false });
      };
      const up = () => {
        removeEventListener("pointermove", move); removeEventListener("pointerup", up);
        gestureCommit();   // one drag = one undo step
        applyXf();
      };
      addEventListener("pointermove", move); addEventListener("pointerup", up);
    });
  }
  window.__aresXform = {
    tool: (n) => setXTool(n), get: () => JSON.parse(JSON.stringify(xf)),
    set: (t) => { xf = { ...XF_DEFAULT(), ...t }; applyXf(); },
    focus: () => player.focus(), reset: () => $("xfReset").click(),
  };

  // ---- crop rulers + guides ------------------------------------------------------------------
  // Column-major mat4 (camera.ts): clip.c = m[c]*x + m[4+c]*y + m[8+c]*z + m[12+c].
  const projPoint = (vp, p) => {
    const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
    return [(vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12]) / w,
            (vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13]) / w];
  };
  const cropCentre = () => [(aabb.min[0] + aabb.max[0]) / 2, (aabb.min[1] + aabb.max[1]) / 2, (aabb.min[2] + aabb.max[2]) / 2];
  /** NDC component `comp` of the point that sits at `v` along `ax` (other axes at the AABB centre). */
  const axisNdc = (vp, ax, comp, v) => { const c = cropCentre(); c[ax] = v; return projPoint(vp, c)[comp]; };
  /** Exact inverse of axisNdc. The projection of an axis-aligned line is a line, so
   *  ndc = (A + B v)/(C + D v) and v solves in closed form — no search, no drift.
   *  (Verified: 4320 round-trips across azimuth/elevation/aspect, worst error 3e-12 mm.) */
  const ndcToAxis = (vp, ax, comp, target) => {
    const c = cropCentre();
    let A = vp[comp] * c[0] + vp[4 + comp] * c[1] + vp[8 + comp] * c[2] + vp[12 + comp];
    let C = vp[3] * c[0] + vp[7] * c[1] + vp[11] * c[2] + vp[15];
    const B = vp[4 * ax + comp], D = vp[4 * ax + 3];
    A -= B * c[ax]; C -= D * c[ax];                       // strip the centre's own contribution
    const den = B - target * D;
    if (Math.abs(den) < 1e-9) return null;                // axis is edge-on to this component
    return (target * C - A) / den;
  };
  /** Which world axis runs across the screen, which up it, and which into it — from the live camera. */
  const cropAxisMap = (vp) => {
    const c = cropCentre(), base = projPoint(vp, c), d = [];
    for (let ax = 0; ax < 3; ax++) {
      const p = c.slice(); p[ax] += cropSpan(ax) * 0.25 || 1;
      const q = projPoint(vp, p);
      d.push([Math.abs(q[0] - base[0]), Math.abs(q[1] - base[1])]);
    }
    const mag = d.map(([x, y]) => Math.hypot(x, y));
    let depth = 0; for (let i = 1; i < 3; i++) if (mag[i] < mag[depth]) depth = i;
    const rest = [0, 1, 2].filter((i) => i !== depth);
    const h = d[rest[0]][0] >= d[rest[1]][0] ? rest[0] : rest[1];
    // Is the camera looking straight down an axis? It matters, and it isn't cosmetic:
    // an axis-aligned PLANE only projects to a straight LINE when you see it edge-on. From a 3/4
    // view the "X = -1" plane projects to a region, not a vertical line — so a guide line is a
    // rough handle at best, and shading "everything outside the lines" is simply FALSE. We only
    // claim the crop visually when the claim is exact; otherwise say so and offer 1/3/7.
    const aligned = mag[depth] / Math.max(1e-6, Math.max(mag[h], mag[rest[0] === h ? rest[1] : rest[0]])) < 0.06;
    return { h, v: rest[0] === h ? rest[1] : rest[0], depth, aligned };
  };

  // ONE row of chips: X/Y/Z, each "full" or its cut range. The value formats in the clip's real
  // units (mm vs m — same AABB inference the grid uses), and an axis that is actually cut lights up
  // so you can see at a glance which planes are doing something.
  function renderCropRead(box) {
    const el = $("cropRead");
    if (!el) return;
    const map = cropGuidesOn ? cropAxisMap(orbitViewProj(player.getCamera(), player.getViewAspect())) : null;
    el.innerHTML = AXES.map(([n, i]) => {
      const full = cropPct[i][0] <= 0.001 && cropPct[i][1] >= 99.999;
      const isDepth = map && map.depth === i;
      const mm = (v) => cropMm(i, v) / worldPerMm;
      const txt = full ? "full" : `${mm(cropPct[i][0]).toFixed(0)}…${mm(cropPct[i][1]).toFixed(0)}`;
      const cls = `cr${isDepth ? " depth" : ""}${full ? "" : " set"}`;
      return `<div class="${cls}" title="${AXIS_TIP[n]}${full ? "" : " Values in millimetres."}${isDepth ? " Currently pointing into the screen — orbit (1/3/7) to drag it." : ""}"><u>${n}</u><span>${txt}</span></div>`;
    }).join("");
  }

  // Crop guides default OFF. Opening the editor is not the same as wanting to crop: with them on by
  // default the app came up in a cropping mode nobody asked for — rulers down two edges, guide lines
  // over the model, and a shade dimming the viewport. R (or ⌗) turns them on when you mean it.
  let cropGuidesOn = localStorage.getItem("ares.crop.guides") === "1", cropSig = "";
  function renderCropUi() {
    const ov = $("cropOverlay");
    const vp = orbitViewProj(player.getCamera(), player.getViewAspect());
    const map = cropAxisMap(vp);
    const W = innerWidth, H = innerHeight;
    const toSx = (n) => (n * 0.5 + 0.5) * W;
    const toSy = (n) => (1 - (n * 0.5 + 0.5)) * H;
    // Rulers hug the viewport but sit inside whichever rails are open (rails float above at z10).
    // offsetWidth, NOT offsetParent: the rails are position:fixed, and offsetParent is ALWAYS null
    // for fixed elements — that test silently returned 0 and buried the left ruler under the rail.
    const hud = $("hud"), ed = $("editPanel");
    const L = hud && hud.offsetWidth > 0 ? hud.getBoundingClientRect().right : 0;
    const R = ed && ed.offsetWidth > 0 && ed.style.display !== "none" ? W - ed.getBoundingClientRect().left : 0;
    const T = 52;

    // The bottom transport/timeline dock owns the foot of the window — the vertical ruler stops at
    // it rather than running underneath (same reason the rails do). Measure from the dock's TOP
    // rather than adding up heights: the diagnostics bar sits below it, and one measurement can't
    // drift out of sync with the CSS the way an arithmetic guess can.
    const ctl = $("controls");
    const B = ctl && ctl.offsetHeight > 0 ? Math.max(0, H - ctl.getBoundingClientRect().top) : 0;
    const rh = $("cropRulerH"), rv = $("cropRulerV");
    rh.style.cssText = `left:${L}px;right:${R}px;top:${T}px`;
    rv.style.cssText = `left:${L}px;top:${T + 13}px;bottom:${B}px`;

    // Ticks are placed by the EXACT projection (never lerped), so perspective stays honest; only
    // the tick DENSITY is chosen from the average pixels-per-mm.
    const ticks = (ax, comp, horizontal) => {
      const lo = aabb.min[ax], hi = aabb.max[ax];
      const p0 = horizontal ? toSx(axisNdc(vp, ax, comp, lo)) : toSy(axisNdc(vp, ax, comp, lo));
      const p1 = horizontal ? toSx(axisNdc(vp, ax, comp, hi)) : toSy(axisNdc(vp, ax, comp, hi));
      const pxPerMm = Math.abs(p1 - p0) / Math.max(1e-6, hi - lo);
      let iv = 5000;
      for (const c of [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]) if (c * pxPerMm >= 46) { iv = c; break; }
      let out = `<em style="${horizontal ? "left:2px;top:1px" : "left:1px;top:2px"}"${map.aligned ? "" : ' class="off"'}>${AXIS_NAME[ax]}</em>`;
      for (let v = Math.ceil(lo / iv) * iv; v <= hi; v += iv) {
        const n = axisNdc(vp, ax, comp, v);
        if (n === null || !isFinite(n)) continue;
        const p = (horizontal ? toSx(n) - L : toSy(n) - (T + 13));
        if (p < 12 || p > (horizontal ? W - L - R : H - T - 13 - B)) continue;
        out += horizontal
          ? `<i style="left:${p}px;bottom:0;width:1px;height:4px"></i><s style="left:${p + 2}px;top:1px">${v}</s>`
          : `<i style="top:${p}px;right:0;height:1px;width:4px"></i><s style="top:${p + 1}px;left:1px;writing-mode:vertical-rl">${v}</s>`;
      }
      return out;
    };
    rh.innerHTML = ticks(map.h, 0, true);
    rv.innerHTML = ticks(map.v, 1, false);

    // The four guides = the two planes of each visible axis.
    let g = "";
    const px = {};
    for (const [ax, comp, cls] of [[map.h, 0, "gv"], [map.v, 1, "gh"]]) {
      px[ax] = [];
      for (const e of [0, 1]) {
        const n = axisNdc(vp, ax, comp, cropMm(ax, cropPct[ax][e]));
        const p = comp === 0 ? toSx(n) : toSy(n);
        px[ax].push(p);
        if (!isFinite(p)) continue;
        const tip = map.aligned
          ? `${AXIS_NAME[ax]} ${e ? "max" : "min"} plane — drag to move it. ${AXIS_TIP[AXIS_NAME[ax]]}`
          : `${AXIS_NAME[ax]} ${e ? "max" : "min"} plane. The view is off-axis, so this line only marks where the plane crosses the clip's centre — it is not the cut itself. Press 1 / 3 / 7 for a straight-on view to place it exactly.`;
        g += `<div class="cropGuide ${cls}${map.aligned ? "" : " off"}" data-ax="${ax}" data-edge="${e}" data-comp="${comp}" style="${cls === "gv" ? "left" : "top"}:${p}px" title="${tip}"><i></i><b>${AXIS_NAME[ax]} ${cropMm(ax, cropPct[ax][e]).toFixed(0)}</b></div>`;
      }
    }
    $("cropGuides").innerHTML = g;

    // Dim what the crop throws away — a crop you can't see isn't a crop you can trust.
    // ONLY when the view is axis-aligned: off-axis, "outside the lines" is not what gets cut, and
    // drawing it anyway would state something false with total confidence.
    const [hx0, hx1] = (px[map.h] || [0, W]).slice().sort((a, b) => a - b);
    const [vy0, vy1] = (px[map.v] || [0, H]).slice().sort((a, b) => a - b);
    const any = cropPct[map.h][0] > 0.001 || cropPct[map.h][1] < 99.999 || cropPct[map.v][0] > 0.001 || cropPct[map.v][1] < 99.999;
    $("cropShade").innerHTML = !(any && map.aligned) ? "" :
      `<div style="left:0;top:0;bottom:0;width:${Math.max(0, hx0)}px"></div>
       <div style="right:0;top:0;bottom:0;width:${Math.max(0, W - hx1)}px"></div>
       <div style="left:${Math.max(0, hx0)}px;width:${Math.max(0, hx1 - hx0)}px;top:0;height:${Math.max(0, vy0)}px"></div>
       <div style="left:${Math.max(0, hx0)}px;width:${Math.max(0, hx1 - hx0)}px;bottom:0;height:${Math.max(0, H - vy1)}px"></div>`;
  }

  // Repaint the guides only when the camera, the viewport, or the planes actually change.
  (function cropWatch() {
    requestAnimationFrame(cropWatch);
    const ov = $("cropOverlay");
    if (!ov || ov.style.display === "none") return;
    const c = player.getCamera();
    // The dock height is part of the signature: the ruler is laid out against it, so resizing the
    // timeline has to repaint the guides even though the camera never moved.
    const sig = `${c.azimuth.toFixed(4)}|${c.elevation.toFixed(4)}|${c.distance.toFixed(3)}|${c.target}|${innerWidth}x${innerHeight}|${$("controls")?.offsetHeight}|${cropPct}`;
    if (sig === cropSig) return;
    cropSig = sig;
    renderCropUi();
  })();

  // Drag a guide = drag that crop plane. Screen -> exact world mm via ndcToAxis; the whole drag is
  // ONE undo step (gestureBegin/Commit), matching every other editor gesture.
  $("cropGuides").addEventListener("pointerdown", (e) => {
    const gd = e.target.closest(".cropGuide");
    if (!gd) return;
    e.preventDefault(); e.stopPropagation();
    gd.classList.add("drag");
    const ax = Number(gd.dataset.ax), edge = Number(gd.dataset.edge), comp = Number(gd.dataset.comp);
    // gestureBegin unconditionally: the crop planes are part of the snapshot whether or not a range
    // is being authored, and pairing it with a second explicit push (as this did) double-counted the
    // drag on the undo stack whenever a range WAS active.
    gestureBegin();
    const move = (ev) => {
      const vp = orbitViewProj(player.getCamera(), player.getViewAspect());
      const ndc = comp === 0 ? (ev.clientX / innerWidth) * 2 - 1 : 1 - (ev.clientY / innerHeight) * 2;
      const world = ndcToAxis(vp, ax, comp, ndc);
      if (world === null) return;
      let pct = ((world - aabb.min[ax]) / Math.max(1e-9, cropSpan(ax))) * 100;
      pct = Math.max(0, Math.min(100, pct));
      if (Math.abs(pct) < 1.2) pct = 0; else if (Math.abs(pct - 100) < 1.2) pct = 100;   // snap to the full extent
      // Planes can meet but never cross — a min past its max is a crop with no inside.
      if (edge === 0) pct = Math.min(pct, cropPct[ax][1]); else pct = Math.max(pct, cropPct[ax][0]);
      cropPct[ax][edge] = pct;
      applyCrop();
    };
    const up = () => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      gd.classList.remove("drag");
      gestureCommit();   // one drag = one undo step
    };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  });

  // The crop toggle is the overlay's only owner now (the Edit rail is always up, so gating on it
  // would mean the guides were always up too — which is exactly the "why am I in crop mode?" bug).
  function syncCropOverlay() {
    const show = cropGuidesOn && $("editPanel").style.display !== "none";   // still hidden off the Viewer tab
    $("cropOverlay").style.display = show ? "" : "none";
    if (show) { cropSig = ""; renderCropUi(); }
  }
  const setCropGuides = (on) => {
    cropGuidesOn = on;
    $("cropGuidesToggle").setAttribute("aria-pressed", String(on));
    localStorage.setItem("ares.crop.guides", on ? "1" : "0");
    syncCropOverlay();
    renderCropRead(cropBox());
  };
  $("cropGuidesToggle").onclick = () => setCropGuides(!cropGuidesOn);
  setCropGuides(cropGuidesOn);

  // ---- projection (ortho, so the viewport isn't fisheyed) -----------
  // Ortho isn't only comfort: under perspective an axis-aligned plane projects to a REGION, so a
  // straight guide line — and the shade drawn from it — can only ever approximate the real cut.
  // In ortho, seen edge-on, that plane IS a straight line and the crop preview becomes exact.
  const projSeg = $("projSeg");
  const applyProj = (mode) => {
    player.setOrtho(mode === "ortho");
    for (const b of projSeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.proj === mode));
    cropSig = "";                       // guides re-project through the new camera
    if ($("cropOverlay").style.display !== "none") renderCropUi();
    renderCropRead(cropBox());
  };
  projSeg.onclick = (e) => { const b = e.target.closest("button"); if (b) applyProj(b.dataset.proj); };
  window.__aresProj = { set: applyProj, get: () => (player.isOrtho() ? "ortho" : "persp") };

  // Coming back to the Viewer tab: rebuild the self-managed overlay from the control that OWNS it
  // (the crop toggle) rather than letting setTab guess. Single source of truth.
  window.__aresRestoreViewerPanels = () => syncCropOverlay();

  const frameTotal = () => player.getStats().frameCount || 1;

  // ---- Frame-list parsing for copy.dstFrames text ("40,41" / "40-45" / mixed, comma-separated;
  // editor v3 §1 — packages/encoder/src/frame-copy.ts collectCopyOps requires a non-empty int[]
  // with every entry in [0, frameCount)). Returns null on ANY malformed token — the caller must
  // never write a partially-parsed result to the range (guardrail: invalid input never touches disk).
  function parseFrameSpec(text, frameCount) {
    const parts = String(text || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (!parts.length) return null;
    const out = new Set();
    for (const p of parts) {
      const m = /^(\d+)\s*-\s*(\d+)$/.exec(p);
      if (m) { let a = Number(m[1]), b = Number(m[2]); if (a > b) [a, b] = [b, a]; for (let v = a; v <= b; v++) out.add(v); }
      else if (/^\d+$/.test(p)) out.add(Number(p));
      else return null;
    }
    const arr = [...out].sort((a, b) => a - b);
    if (!arr.length) return null;
    for (const v of arr) if (!Number.isInteger(v) || v < 0 || v >= frameCount) return null;
    return arr;
  }
  /** Compact display: consecutive runs collapse to "a-b" — a nicer round-trip than one-per-comma. */
  function formatFrameSpec(arr) {
    if (!arr || !arr.length) return "";
    const out = [];
    let s = arr[0], p = arr[0];
    for (let i = 1; i <= arr.length; i++) {
      const v = arr[i];
      if (v === p + 1) { p = v; continue; }
      out.push(s === p ? String(s) : `${s}-${p}`);
      s = p = v;
    }
    return out.join(",");
  }

  // ---- Bake-side action payloads (editor v3 §1/§2/§3 — packages/core/src/edits.ts is the exact
  // schema; the encoder validates these verbatim, so every default written here must already be
  // schema-valid on its own). Switching action never discards a previously-authored payload:
  // applyDefaultPayload only fills a field the range doesn't already have, so flipping back to a
  // prior action restores it exactly (round-trip through action switches, not just save/reload).
  function applyDefaultPayload(r, action) {
    if (action === "recolor" && !r.recolor) r.recolor = { color: (r.color || nextSegColor()).toLowerCase(), strength: 0.8, mode: "tint" };
    if (action === "copy" && !r.copy) r.copy = { srcFrame: curFrame(), dstFrames: [], what: "both" };
  }
  function setRangeAction(r, action) {
    if (action === "delete") delete r.action; else r.action = action;
    applyDefaultPayload(r, action);
  }

  // ---- NLE timeline strip. Lives in the bottom dock (#rangeTrack inside #controls), full width:
  // ruler + clip-trim brackets + one lane per range + full-height playhead, all sharing ONE
  // coordinate system (percent of the same container), so they can never desync. Frame-integer
  // snapping; both-edge range trim; handles only on the ▶ active range; click/drag anywhere (ruler
  // or empty lane) scrubs. No zoom/pan — a 272-frame clip fits a static strip.
  //
  // ---- Clip trim (in/out): which frames of the clip survive at all, as opposed to a range, which
  // is about geometry WITHIN frames. Three things must agree or the feature is a lie:
  //   1. the strip dims what's cut,
  //   2. playback loops inside it (player.setTrim),
  //   3. a bake encodes only it (sidecar `trim` -> encoder --trim-in/--trim-out, which also
  //      re-bases every range's frame numbers onto the surviving window).
  // Scrubbing INTO the dimmed region still works on purpose: you cannot judge a cut point without
  // looking at the frames on the far side of it.
  let trimIn = 0, trimOut = -1;                       // -1 = "the last frame", whatever it turns out to be
  const trimOutEff = () => (trimOut < 0 ? frameTotal() - 1 : Math.min(trimOut, frameTotal() - 1));
  const trimIsFull = () => trimIn <= 0 && trimOutEff() >= frameTotal() - 1;
  function applyTrim({ save = true } = {}) {
    const total = frameTotal();
    trimIn = Math.max(0, Math.min(total - 1, trimIn));
    if (trimOut >= 0) trimOut = Math.max(trimIn, Math.min(total - 1, trimOut));
    player.setTrim(trimIn, trimOut);
    // Only persist a trim that actually trims — an untouched clip should not grow a `trim` key, and
    // the encoder should not be handed a no-op window to re-base ranges through.
    if (trimIsFull()) delete edits.trim; else edits.trim = { in: trimIn, out: trimOutEff() };
    const kept = trimOutEff() - trimIn + 1;
    const el = $("trimRead");
    if (el) {
      el.textContent = trimIsFull() ? "full" : `${trimIn}–${trimOutEff()} · ${kept}f · ${(kept / 30).toFixed(2)}s`;
      el.classList.toggle("on", !trimIsFull());
      el.title = trimIsFull()
        ? "No trim — the whole clip plays and bakes."
        : `Trimmed to source frames ${trimIn}–${trimOutEff()} (${kept} of ${total} frames, ${(kept / 30).toFixed(2)}s). Playback loops inside this window and a bake encodes only it; the dimmed frames are dropped, and edit ranges shift onto the new numbering.`;
    }
    if (save) saveEdits();
    renderRanges();
  }

  let tlLastF = -1;
  function renderRuler() {
    const ruler = $("tlRuler");
    if (!ruler) return;
    const total = frameTotal();
    const pxf = (ruler.clientWidth || 0) / total;
    let iv = 60;
    for (const c of [1, 2, 5, 10, 30, 60]) if (c * pxf >= 45) { iv = c; break; }
    const minor = iv >= 5 ? iv / 5 : 0;             // unlabeled sub-ticks (5→1, 10→2, 30→6, 60→12)
    let h = "";
    for (let f = 0; f < total; f += (minor || iv)) {
      const major = f % iv === 0;
      h += `<span style="position:absolute;left:${(f / total) * 100}%;bottom:0;width:1px;height:${major ? 6 : 3}px;background:${major ? "var(--text-dim)" : "#55534d"}"></span>`;
      if (major && pxf * iv >= 30) h += `<span style="position:absolute;left:${(f / total) * 100}%;top:0;transform:translateX(-50%);font:9px ui-monospace,monospace;color:var(--text-faint)">${f}</span>`;
    }
    ruler.innerHTML = h;
  }
  /** clientX → clamped integer frame, in the lanes' box (the shared coordinate system). */
  function tlFrameAt(clientX) {
    const rect = ($("tlLanes") || $("rangeTrack")).getBoundingClientRect();
    const total = frameTotal();
    return Math.max(0, Math.min(total - 1, Math.round(((clientX - rect.left) / Math.max(1, rect.width)) * total - 0.5)));
  }
  const tlSeek = (f) => { player.pause(); $("play").textContent = "▶"; player.seek(f / 30); tlLastF = -1; };
  // Playhead follows the presented frame (rAF, cheap: one style write when the frame changes).
  (function tlWatch() {
    requestAnimationFrame(tlWatch);
    const ph = $("tlPlayhead");
    if (!ph) return;
    const f = curFrame();
    if (f === tlLastF) return;
    tlLastF = f;
    ph.style.left = ((f + 0.5) / frameTotal()) * 100 + "%";
  })();
  if (window.ResizeObserver) new ResizeObserver(renderRuler).observe($("rangeTrack")); // dock resize → re-density ticks

  // Trim buttons: set in/out at the playhead, or clear. The keyboard mirrors these on [ and ].
  // doMutation so each is its own undo step, like every other discrete action.
  $("trimIn").onclick = () => { doMutation(() => { trimIn = Math.min(curFrame(), trimOutEff()); }); applyTrim(); };
  $("trimOut").onclick = () => { doMutation(() => { trimOut = Math.max(curFrame(), trimIn); }); applyTrim(); };
  $("trimReset").onclick = () => { doMutation(() => { trimIn = 0; trimOut = -1; }); applyTrim(); };
  window.__aresTrim = {
    setIn: () => $("trimIn").click(), setOut: () => $("trimOut").click(), reset: () => $("trimReset").click(),
    get: () => ({ in: trimIn, out: trimOutEff() }),
  };

  // Timeline dock height: drag its top edge, same law as the side rails (slim, drag-resizable,
  // persisted). Writes the --tl-h custom property, which is also what holds the rails and the crop
  // ruler off the dock — one number, one source of truth, no element has to know about the others.
  {
    const TL_KEY = "ares.tl.h", MIN = 58, MAX = 340;
    const applyH = (h) => document.documentElement.style.setProperty("--tl-h", Math.max(MIN, Math.min(MAX, h)) + "px");
    const saved = Number(localStorage.getItem(TL_KEY));
    if (saved) applyH(saved);
    let dragging = false, startY = 0, startH = 0;
    const grip = $("tlResize");
    grip.addEventListener("pointerdown", (e) => {
      dragging = true; startY = e.clientY; startH = $("controls").getBoundingClientRect().height;
      try { grip.setPointerCapture(e.pointerId); } catch { /* ignore */ }
      e.preventDefault();
    });
    grip.addEventListener("pointermove", (e) => { if (dragging) applyH(startH - (e.clientY - startY)); });
    const end = () => {
      if (!dragging) return;
      dragging = false;
      localStorage.setItem(TL_KEY, String(Math.round($("controls").getBoundingClientRect().height)));
      cropSig = "";   // the crop ruler's bottom edge is pinned to the dock — re-project it
    };
    grip.addEventListener("pointerup", end);
    grip.addEventListener("pointercancel", end);
  }
  $("rangeTrack").addEventListener("pointerdown", (e) => {
    const kf = e.target.closest(".tlKf");
    if (kf) { tlSeek(Number(kf.dataset.f)); return; }                    // diamond click = jump to keyframe
    // Clip-trim bracket drag. Checked before the range/scrub branches: the brackets sit on top of
    // the lanes, so whichever the pointer landed on, the bracket wins.
    const th = e.target.closest(".tlTrimH");
    if (th) {
      e.preventDefault();
      th.classList.add("drag");
      gestureBegin();                      // one bracket drag = one undo step
      const edge = th.dataset.trim;
      const move = (ev) => {
        const f = tlFrameAt(ev.clientX);
        // The two brackets may meet on a single frame but never cross — an inverted trim is a clip
        // with no frames in it.
        if (edge === "in") trimIn = Math.min(f, trimOutEff());
        else trimOut = Math.max(f, trimIn);
        applyTrim({ save: false });                                       // save once, on release
      };
      const up = () => {
        window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
        th.classList.remove("drag");
        gestureCommit();
        applyTrim();
      };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
      return;
    }
    const handle = e.target.closest(".tlHandle");
    const bar = e.target.closest(".tlBar");
    e.preventDefault();
    $("rangeTrack").setPointerCapture(e.pointerId);
    const total = frameTotal();
    if (handle || bar) {
      const r = edits.ranges[Number((handle || bar).dataset.ridx)];
      if (!r) return;
      if (r !== activeRange) { activeRange = r; syncRangeButtons(); renderRanges(); if (!handle) return; } // first click selects
      gestureCommit();
      const before = snapshotState();
      const mode = handle ? handle.dataset.edge : "move";
      const f0 = tlFrameAt(e.clientX);
      const orig = { s: r.startFrame, e: r.endFrame, kfs: r.keyframes.map((k) => k.frame) };
      // Frame-integer snap to playhead / other ranges' edges / clip ends (~7px threshold).
      const rectW = ($("tlLanes") || $("rangeTrack")).getBoundingClientRect().width;
      const th = Math.max(1, Math.round((7 / Math.max(1, rectW)) * total));
      const cands = [curFrame(), 0, total - 1];
      for (const o of edits.ranges) if (o !== r) cands.push(o.startFrame, o.endFrame);
      const snap = (f) => { let best = f, bd = th + 1; for (const c of cands) { const d = Math.abs(c - f); if (d < bd) { bd = d; best = c; } } return best; };
      const move = (ev) => {
        if (mode === "move") {
          let d = tlFrameAt(ev.clientX) - f0;
          const s1 = snap(orig.s + d);                                   // snap whichever edge bites first
          if (s1 !== orig.s + d) d = s1 - orig.s;
          else { const s2 = snap(orig.e + d); if (s2 !== orig.e + d) d = s2 - orig.e; }
          d = Math.max(-orig.s, Math.min(total - 1 - orig.e, d));
          r.startFrame = orig.s + d; r.endFrame = orig.e + d;
          r.keyframes.forEach((k, ki) => { k.frame = Math.max(0, Math.min(total - 1, orig.kfs[ki] + d)); }); // keyframes ride along
        } else if (mode === "start") r.startFrame = Math.min(snap(tlFrameAt(ev.clientX)), r.endFrame);
        else r.endFrame = Math.max(snap(tlFrameAt(ev.clientX)), r.startFrame);
        renderRanges();
      };
      const up = () => {
        window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
        pushUndoIfChanged(before); preview();                            // one undo step per gesture; preview on release
      };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    } else {
      // ruler / empty-lane scrub gesture: pause + hold the exact frame under the pointer
      tlSeek(tlFrameAt(e.clientX));
      const move = (ev) => tlSeek(tlFrameAt(ev.clientX));
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
  });

  function renderRanges() {
    const total = frameTotal();
    edits.frameCount = total;
    // Clip trim overlay: the dimmed head/tail + the two draggable brackets. Rendered first so the
    // range lanes and the playhead sit above the shade and stay readable inside the kept window.
    const tIn = Math.max(0, Math.min(total - 1, trimIn)), tOut = trimOutEff();
    const trimHtml =
      (tIn > 0 ? `<div class="tlCut" style="left:0;width:${(tIn / total) * 100}%"></div>` : "") +
      (tOut < total - 1 ? `<div class="tlCut" style="left:${((tOut + 1) / total) * 100}%;right:0"></div>` : "") +
      `<div class="tlTrimH in" data-trim="in" style="left:${(tIn / total) * 100}%" title="clip IN — drag to trim the start ( [ sets it at the playhead )"><i></i><b>in ${tIn}</b></div>
       <div class="tlTrimH out" data-trim="out" style="left:${((tOut + 1) / total) * 100}%" title="clip OUT — drag to trim the end ( ] sets it at the playhead )"><i></i><b>out ${tOut}</b></div>`;
    // strip: ruler + one lane per range (bar spans [start,end], diamonds at keyframes, trim
    // handles on the active range) + the full-height playhead. All positions are % of the strip.
    $("rangeTrack").innerHTML = `<div id="tlRuler" style="position:relative;height:14px;margin-bottom:1px;cursor:ew-resize;user-select:none;touch-action:none"></div>
      <div id="tlLanes" style="position:relative;min-height:3px;cursor:ew-resize;touch-action:none">` + edits.ranges.map((r, i) => {
      const col = r.color || "var(--warn)";
      const act = r === activeRange;
      const l = (r.startFrame / total) * 100, w = Math.max(0.4, ((r.endFrame - r.startFrame + 1) / total) * 100);
      const dias = r.keyframes.map((k) =>
        `<span class="tlKf" data-f="${k.frame}" title="keyframe @ ${k.frame} — click to jump" style="position:absolute;left:${((k.frame - r.startFrame) / Math.max(1, r.endFrame - r.startFrame)) * 100}%;top:-3px;transform:translateX(-50%);font-size:9px;cursor:pointer;color:${k.derived ? "var(--text-faint)" : col}">${k.derived ? "◇" : "◆"}</span>`).join("");
      const handles = act ? `<span class="tlHandle" data-ridx="${i}" data-edge="start" title="drag to trim the range start (snaps to playhead/edges)" style="position:absolute;left:-3px;top:-2px;width:7px;height:10px;background:${col};border-radius:2px;cursor:ew-resize"></span>
        <span class="tlHandle" data-ridx="${i}" data-edge="end" title="drag to trim the range end (snaps to playhead/edges)" style="position:absolute;right:-3px;top:-2px;width:7px;height:10px;background:${col};border-radius:2px;cursor:ew-resize"></span>` : "";
      return `<div style="position:relative;height:12px;margin:2px 0"><div class="tlBar" data-ridx="${i}" title="${act ? "drag to move the range (keyframes ride along)" : "click to select this range"}" style="position:absolute;left:${l}%;width:${w}%;height:6px;top:3px;background:${col}${act ? "BB" : "55"};border-radius:3px;cursor:${act ? "grab" : "pointer"}">${dias}${handles}</div></div>`;
    }).join("") + `</div>` + trimHtml +
      `<div id="tlPlayhead" style="position:absolute;top:0;bottom:0;width:2px;background:#e8e6da;opacity:.85;pointer-events:none;z-index:5;left:0"></div>`;
    renderRuler();
    tlLastF = -1;                                    // force playhead reposition after re-render

    // range rows: swatch + label + action select + remove button, then a slim contextual sub-row
    // (editor v3 UI task §1). recolor/copy sub-rows carry a "bake-only" badge — keepPredicateAt
    // (core) deliberately excludes non-delete ranges from the live preview, so this is the only
    // place their payload is visible before Bake; that intentional gap is called out inline.
    const inputCss = "padding:2px 3px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:4px;color:var(--text);font-size:10.5px";
    $("rangeList").innerHTML = edits.ranges.map((r, i) => {
      const action = r.action || "delete";
      const col = r.color || "var(--warn)";
      const mark = r === activeRange ? "▶ " : "";
      const optSel = (v) => (v === action ? " selected" : "");
      const bakeTag = `<span class="badge warn" style="margin:0" title="bake-only — the live preview intentionally only shows delete ranges (keepPredicateAt skips copy/recolor); this is exactly what Bake will apply">bake-only</span>`;
      let sub;
      if (action === "recolor") {
        const rc = r.recolor || { color: col, strength: 0.8, mode: "tint" };
        sub = `<input type="color" class="rcColor" data-ridx="${i}" value="${(rc.color || col).toLowerCase()}" title="recolor target color" style="width:20px;height:18px;padding:0;border:0;background:none;cursor:pointer">
          <input type="range" class="rcStrength" data-ridx="${i}" min="0" max="1" step="0.01" value="${rc.strength}" style="width:52px;accent-color:${col}" title="strength 0-1">
          <span class="rcStrengthVal" style="min-width:24px">${Number(rc.strength).toFixed(2)}</span>
          <select class="rcMode" data-ridx="${i}" style="${inputCss}" title="tint = preserve texel luma; hue = rotate hue only">
            <option value="tint"${rc.mode === "tint" ? " selected" : ""}>tint</option>
            <option value="hue"${rc.mode === "hue" ? " selected" : ""}>hue</option>
          </select>${bakeTag}`;
      } else if (action === "copy") {
        const cp = r.copy || { srcFrame: r.startFrame, dstFrames: [], what: "both" };
        sub = `<span>src</span>
          <input type="number" class="rcpSrc" data-ridx="${i}" min="0" max="${total - 1}" value="${cp.srcFrame}" style="width:40px;${inputCss}">
          <span>→</span>
          <input type="text" class="rcpDst" data-ridx="${i}" placeholder="40,41 or 40-45" value="${formatFrameSpec(cp.dstFrames)}" style="width:66px;${inputCss}">
          <select class="rcpWhat" data-ridx="${i}" style="${inputCss}" title="what to paste: geometry, texels, or both (default)">
            <option value="both"${(cp.what || "both") === "both" ? " selected" : ""}>both</option>
            <option value="geo"${cp.what === "geo" ? " selected" : ""}>geo</option>
            <option value="texels"${cp.what === "texels" ? " selected" : ""}>texels</option>
          </select>${bakeTag}`;
      } else {
        const ph = r.patchHoles;
        sub = `<label style="display:flex;gap:4px;align-items:center;cursor:pointer">
            <input type="checkbox" class="rphChk" data-ridx="${i}" ${ph ? "checked" : ""} style="margin:0">patch holes
          </label>${ph ? `
          <input type="color" class="rphColor" data-ridx="${i}" value="${(ph.color || "#888888").toLowerCase()}" title="hole fill color — leave as-is to use the auto rim-average color" style="width:20px;height:18px;padding:0;border:0;background:none;cursor:pointer">
          <button class="rphAuto" data-ridx="${i}" title="clear the explicit color — use the auto rim-average fill" style="background:none;border:0;color:var(--text-faint);cursor:pointer;font-size:10px;padding:0">auto</button>` : ""}`;
      }
      return `<div data-ridx="${i}" style="margin-top:${i ? 5 : 0}px;cursor:pointer" title="click to make this range ▶ active (Delete/Backspace then removes it)">
        <div style="display:flex;gap:6px;align-items:center;font-size:11px;color:var(--text-mid)">
          <span style="width:8px;height:8px;border-radius:2px;background:${col};flex:none"></span>
          <span style="flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${mark}${action} ${r.startFrame}–${r.endFrame} · ${r.keyframes.length} kf</span>
          <select class="ractsel" name="range-action-${i}" data-ridx="${i}" style="${inputCss}" title="bake action for this range">
            <option value="delete"${optSel("delete")}>delete</option>
            <option value="recolor"${optSel("recolor")}>recolor</option>
            <option value="copy"${optSel("copy")}>copy</option>
          </select>
          <button data-rrm="${i}" style="background:none;border:0;color:var(--text-faint);cursor:pointer" title="remove this range">✕</button>
        </div>
        <div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;padding:2px 0 0 14px;font-size:10.5px;color:var(--text-dim)">${sub}</div>
      </div>`;
    }).join("");

    // Click a row (not one of its interactive controls) to make it the ▶ active range — the target
    // Delete/Backspace removes, and what Key/End sessions extend. Doesn't itself touch `edits`, so
    // no undo step (only the resulting button-state/mark change).
    for (const row of $("rangeList").children) {
      const idx = Number(row.dataset.ridx);
      row.onclick = (e) => {
        if (e.target.closest("select, input, button, label")) return;
        const rr = edits.ranges[idx];
        if (rr && rr !== activeRange) { activeRange = rr; syncRangeButtons(); renderRanges(); }
      };
    }

    for (const b of $("rangeList").querySelectorAll("[data-rrm]")) b.onclick = (e) => {
      e.stopPropagation();
      gestureCommit();
      const before = snapshotState();
      const r = edits.ranges.splice(Number(b.dataset.rrm), 1)[0];
      if (r === activeRange) { activeRange = null; $("rangeKey").disabled = $("rangeEnd").disabled = true; $("rangeStart").disabled = false; }
      pushUndoIfChanged(before);
      preview();
    };
    // Action switch changes what the live preview does (a range leaving/entering "delete" starts/
    // stops being deleted), so this is the one contextual control that goes through preview() (full
    // setEditPreview + save + rerender) rather than the lighter save-only path below.
    for (const s of $("rangeList").querySelectorAll(".ractsel")) s.onchange = () => {
      gestureCommit();
      const before = snapshotState();
      setRangeAction(edits.ranges[Number(s.dataset.ridx)], s.value);
      pushUndoIfChanged(before);
      preview();
    };
    // Everything below is bake-only payload (no live-preview effect) — mutate + debounced-save
    // directly, skipping setEditPreview/full-rerender so a continuous drag (strength slider, color
    // picker) or in-progress typing (dst-frames text) never rebuilds the DOM out from under focus.
    // Discrete controls (checkbox/selects/the "auto" button) push ONE undo step per change via
    // doMutation(); continuous ones (color/range/text) coalesce via gestureBegin/gestureCommit.
    for (const c of $("rangeList").querySelectorAll(".rphChk")) c.onchange = () => {
      doMutation(() => {
        const r = edits.ranges[Number(c.dataset.ridx)];
        if (c.checked) r.patchHoles = r.patchHoles || {}; else delete r.patchHoles;
      });
      saveEdits(); renderRanges();
    };
    for (const c of $("rangeList").querySelectorAll(".rphColor")) {
      c.oninput = () => { gestureBegin(); edits.ranges[Number(c.dataset.ridx)].patchHoles.color = c.value; saveEdits(); };
      c.onchange = () => gestureCommit();
    }
    for (const b of $("rangeList").querySelectorAll(".rphAuto")) b.onclick = () => {
      doMutation(() => { delete edits.ranges[Number(b.dataset.ridx)].patchHoles.color; });
      saveEdits(); renderRanges();
    };
    for (const c of $("rangeList").querySelectorAll(".rcColor")) {
      c.oninput = () => { gestureBegin(); edits.ranges[Number(c.dataset.ridx)].recolor.color = c.value; saveEdits(); };
      c.onchange = () => gestureCommit();
    }
    for (const c of $("rangeList").querySelectorAll(".rcStrength")) {
      c.oninput = () => {
        gestureBegin();
        const r = edits.ranges[Number(c.dataset.ridx)];
        r.recolor.strength = Number(c.value);
        c.parentElement.querySelector(".rcStrengthVal").textContent = r.recolor.strength.toFixed(2);
        saveEdits();
      };
      c.onchange = () => gestureCommit();
    }
    for (const s of $("rangeList").querySelectorAll(".rcMode")) s.onchange = () => {
      doMutation(() => { edits.ranges[Number(s.dataset.ridx)].recolor.mode = s.value; });
      saveEdits();
    };
    for (const inp of $("rangeList").querySelectorAll(".rcpSrc")) {
      inp.oninput = () => {
        gestureBegin();
        const r = edits.ranges[Number(inp.dataset.ridx)];
        const n = Number(inp.value);
        const ok = Number.isInteger(n) && n >= 0 && n < total;
        inp.style.borderColor = ok ? "" : "#f0a3a3";
        if (ok) { r.copy.srcFrame = n; saveEdits(); }
      };
      inp.onblur = () => gestureCommit();
      inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); inp.blur(); } };
    }
    for (const inp of $("rangeList").querySelectorAll(".rcpDst")) {
      inp.oninput = () => {
        gestureBegin();
        const r = edits.ranges[Number(inp.dataset.ridx)];
        const arr = parseFrameSpec(inp.value, total);
        inp.style.borderColor = arr ? "" : "#f0a3a3";
        if (arr) { r.copy.dstFrames = arr; saveEdits(); }
      };
      inp.onblur = () => gestureCommit();
      inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); inp.blur(); } };
    }
    for (const s of $("rangeList").querySelectorAll(".rcpWhat")) s.onchange = () => {
      doMutation(() => { edits.ranges[Number(s.dataset.ridx)].copy.what = s.value; });
      saveEdits();
    };
  }

  const ensureRange = () => {
    if (activeRange) return activeRange;
    const f = curFrame();
    activeRange = { id: "r" + (edits.ranges.length + 1), color: nextSegColor(), mode: "delete", startFrame: f, endFrame: frameTotal() - 1, keyframes: [] };
    edits.ranges.push(activeRange);
    $("rangeStart").disabled = true; $("rangeKey").disabled = false; $("rangeEnd").disabled = false;
    return activeRange;
  };
  /** Append a volume to the keyframe at the current frame (creates it; slider boxes replace, others add). */
  const addVolumeAtCurrent = (vol) => {
    const r = ensureRange();
    const f = curFrame();
    let kf = r.keyframes.find((k) => k.frame === f);
    if (!kf) { kf = { frame: f, volumes: [] }; r.keyframes.push(kf); r.keyframes.sort((a, b) => a.frame - b.frame); }
    kf.volumes.push(vol);
    if (f > r.endFrame) r.endFrame = f;
    preview();
  };

  // ---- Selection tools (design §4/§5): screen box + surface brush, Blender through/visible law.
  // X-ray ON  → the marquee selects THROUGH the mesh (camera frustum region, no depth band).
  // X-ray OFF → solid-mode semantics: only what's VISIBLE (depth band captured from the pick raster).
  let tool = "nav";
  const overlay = $("selectOverlay"), marquee = $("marquee");
  const xrayOn = () => $("xray").getAttribute("aria-pressed") === "true";
  $("xray").onclick = () => $("xray").setAttribute("aria-pressed", String(!xrayOn()));
  for (const b of document.querySelectorAll("#editPanel .tool")) b.onclick = () => {
    tool = b.dataset.tool;
    for (const o of document.querySelectorAll("#editPanel .tool")) o.setAttribute("aria-pressed", String(o === b));
    overlay.style.display = tool === "nav" ? "none" : "block";
    if (tool !== "nav") player.pause(), $("play").textContent = "▶"; // edit on a held frame
    if (tool !== "sam") samSelClear();                                  // pending SAM prompts die with the tool
  };

  const ndcOf = (clientX, clientY) => {
    const c = document.getElementById("view");
    const r = c.getBoundingClientRect();
    return [((clientX - r.left) / r.width) * 2 - 1, (1 - (clientY - r.top) / r.height) * 2 - 1];
  };

  // ---- Navigation pass-through: the camera stays live while a tool is active. Middle-drag
  // pans, right-drag orbits, Alt+left-drag orbits, wheel zooms — forwarded to the canvas so the
  // player's own pointer handlers do the work; left-click stays with the tool. A pending SAM
  // selection stays alive through camera moves (the watcher re-projects it, world-anchored).
  const fwd = (e) => canvas.dispatchEvent(new PointerEvent(e.type, {
    bubbles: false, clientX: e.clientX, clientY: e.clientY, button: e.button,
    buttons: e.buttons, pointerId: e.pointerId, pointerType: e.pointerType, altKey: e.altKey,
  }));
  let navForward = false;
  overlay.addEventListener("wheel", (e) => {
    e.preventDefault();
    canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: e.deltaY, clientX: e.clientX, clientY: e.clientY, cancelable: true }));
  }, { passive: false });
  overlay.addEventListener("contextmenu", (e) => e.preventDefault()); // right-drag = orbit

  let dragStart = null, brushPts = null, brushPick = null;
  overlay.addEventListener("pointerdown", (e) => {
    if (e.button === 1 || e.button === 2 || (e.button === 0 && e.altKey)) {
      // A pending SAM selection SURVIVES navigation: the watcher re-projects it onto the
      // moving view (world-anchored). Only a new click from a moved viewpoint resets it.
      navForward = true;
      try { overlay.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
      e.preventDefault();
      fwd(e);
      return;
    }
    if (e.button !== 0) return;
    try { overlay.setPointerCapture(e.pointerId); } catch { /* synthetic events have no active pointer */ }
    if (tool === "sbox") {
      dragStart = [e.clientX, e.clientY];
      const col = activeSegColor();
      marquee.style.borderColor = col;
      marquee.style.background = col + "1F";
      marquee.style.display = "block";
      marquee.style.left = e.clientX + "px"; marquee.style.top = e.clientY + "px";
      marquee.style.width = "0px"; marquee.style.height = "0px";
    } else if (tool === "brush") {
      brushPick = player.pickRaster(384, 384);           // one raster per stroke (camera is held)
      brushPts = [];
      brushSample(e);
    } else if (tool === "sam") {
      samClick(e);
    }
  });
  overlay.addEventListener("pointermove", (e) => {
    if (navForward) { fwd(e); return; }
    if (tool === "sbox" && dragStart) {
      const x = Math.min(dragStart[0], e.clientX), y = Math.min(dragStart[1], e.clientY);
      marquee.style.left = x + "px"; marquee.style.top = y + "px";
      marquee.style.width = Math.abs(e.clientX - dragStart[0]) + "px";
      marquee.style.height = Math.abs(e.clientY - dragStart[1]) + "px";
    } else if (tool === "brush" && brushPts) brushSample(e);
  });
  overlay.addEventListener("pointercancel", (e) => { if (navForward) { navForward = false; fwd(e); } });
  overlay.addEventListener("pointerup", (e) => {
    try { overlay.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    if (navForward) { navForward = false; fwd(e); return; }
    if (tool === "sbox" && dragStart) {
      const [ax, ay] = ndcOf(dragStart[0], dragStart[1]);
      const [bx, by] = ndcOf(e.clientX, e.clientY);
      dragStart = null; marquee.style.display = "none";
      const rect = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
      if (rect[2] - rect[0] < 0.01 || rect[3] - rect[1] < 0.01) return;   // a click, not a box
      const cam = player.getCamera();
      const vol = { type: "mask2d", kind: "rect", rect, camera: { ...cam, aspect: player.getViewAspect() } };
      if (!xrayOn()) {
        // solid mode: visible-only via the depth band of what the box actually covers on screen
        const pick = player.pickRaster(384, 384);
        if (pick) {
          const { buf } = pick;
          let zmin = Infinity, zmax = -Infinity;
          const x0 = Math.round(((rect[0] + 1) / 2) * buf.w), x1 = Math.round(((rect[2] + 1) / 2) * buf.w);
          const y0 = Math.round(((1 - rect[3]) / 2) * buf.h), y1 = Math.round(((1 - rect[1]) / 2) * buf.h);
          for (let y = Math.max(0, y0); y <= Math.min(buf.h - 1, y1); y++)
            for (let x = Math.max(0, x0); x <= Math.min(buf.w - 1, x1); x++) {
              const d = buf.depth[y * buf.w + x];
              if (d !== Infinity) { if (d < zmin) zmin = d; if (d > zmax) zmax = d; }
            }
          if (zmin === Infinity) return;                 // box covers only background
          vol.depth = { zmin: zmin - 0.002, zmax: zmax + 0.002 };  // ±eps: raster-vs-centroid z tolerance
        }
      }
      doMutation(() => addVolumeAtCurrent(vol));
    } else if (tool === "brush" && brushPts) {
      if (brushPts.length) {
        const radius = Number($("brushR").value);
        doMutation(() => addVolumeAtCurrent({ type: "brushStrokes", strokes: [{ op: "add", radius, points: brushPts }] }));
      }
      brushPts = null; brushPick = null;
    }
  });
  function brushSample(e) {
    if (!brushPick) return;
    const c = document.getElementById("view");
    const r = c.getBoundingClientRect();
    const bx = Math.round(((e.clientX - r.left) / r.width) * brushPick.buf.w);
    const by = Math.round(((e.clientY - r.top) / r.height) * brushPick.buf.h);
    if (bx < 0 || by < 0 || bx >= brushPick.buf.w || by >= brushPick.buf.h) return;
    const id = brushPick.buf.ids[by * brushPick.buf.w + bx];
    if (id < 0) return;                                   // background — brush only paints the surface
    const p = brushPick.triCentroid(id);
    const last = brushPts[brushPts.length - 1];
    const minStep = Number($("brushR").value) * 0.5;      // decimate: no denser than half a radius
    if (!last || Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]) > minStep) brushPts.push(p);
  }

  // ---- SAM click-to-select (editor v2 §8.2/§8.3): click → /sam/segment → bitmap mask2d volume.
  // Clicks accumulate as prompts (Shift-click excludes); the commit button keyframes the
  // delete/recolor/copy exactly like Box/Brush volumes. The frame is captured at the FIRST click;
  // the depth band is captured with it (camera still matches the capture view at that moment).
  //
  // WORLD-ANCHORED PREVIEW: the selection is a region of the SCENE, not of the screen, so the
  // tint must stick to the object when the camera moves (nav pass-through makes that routine).
  // While the camera matches the capture view, the exact 2D mask paints directly; once it
  // moves, a rAF watcher re-projects the selection through the same mask2d evaluator the bake
  // uses (pickRaster surface points -> inside test), so what highlights from any angle is
  // exactly what committing will act on. A click from a moved viewpoint starts a fresh selection
  // (SAM prompts live in one image; refinement clicks must match the captured view).
  const samMaskCanvas = $("samMask"), samSelRow = $("samSelRow"), samSelInfo = $("samSelInfo");
  const samTextInput = $("samTextInput"), samTextGo = $("samTextGo"), samTextHint = $("samTextHint"), samTextChipsEl = $("samTextChips");
  let samSel = null;   // { capture, cam, camKey, color, points, labels, maskObj, bits, depth, score, busy, instances?, selected? }
  let samRaf = 0, samLastCamKey = "", samLastRepaint = 0, samTextQuery = "";
  const camKeyNow = () => {
    const c = player.getCamera();
    return [c.azimuth, c.elevation, c.distance, c.target[0], c.target[1], c.target[2]].map((v) => v.toFixed(5)).join(",");
  };
  function samSelClear() {
    samSel = null;
    if (samRaf) { cancelAnimationFrame(samRaf); samRaf = 0; }
    samMaskCanvas.style.display = "none";
    samSelRow.style.display = "none";
    samTextChipsEl.style.display = "none";
    samTextChipsEl.innerHTML = "";
  }
  $("samClear").onclick = samSelClear;

  async function samClick(e) {
    if (samSel && samSel.busy) return;
    // A moved camera invalidates the pending prompts (they live in the captured image):
    // clicking from the new viewpoint starts a fresh selection there.
    if (samSel && samSel.camKey !== camKeyNow()) samSelClear();
    if (!samSel) {
      // Health gate: degrade with guidance, never break the tool. The dev server auto-starts the
      // service on refused POSTs, but a clear "start it below" beats a silent multi-second stall.
      let h = null;
      try { h = await fetch("/sam/health").then((r) => r.json()); } catch { /* server down */ }
      if (!h || !h.ok) {
        samSelRow.style.display = "flex";
        samSelInfo.textContent = h && h.loading ? "SAM model is loading — try again shortly" : "SAM service not running — press Start SAM below";
        samRefresh();
        return;
      }
      samSel = {
        capture: player.captureFrame(1024),
        cam: { ...player.getCamera(), aspect: player.getViewAspect() },
        camKey: camKeyNow(),
        color: activeSegColor(),
        points: [], labels: [], maskObj: null, bits: null, maskW: 0, maskH: 0, depth: null, score: 0, busy: false,
      };
    }
    const r = canvas.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
    samSel.points.push([fx * samSel.capture.width, fy * samSel.capture.height]);
    samSel.labels.push(e.shiftKey ? 0 : 1);
    samSel.busy = true;
    samSelRow.style.display = "flex";
    samSelInfo.textContent = "segmenting…";
    try {
      const res = await fetch("/sam/segment", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ image: samSel.capture.dataUrl, points: samSel.points, labels: samSel.labels }),
      });
      if (!res.ok) {
        const detail = await res.json().then((j) => j && j.detail).catch(() => null);
        throw new Error(detail || ("HTTP " + res.status));
      }
      await samShowMask(await res.json());
    } catch (err) {
      samSelInfo.textContent = "✗ segment failed: " + (err && err.message ? err.message : err);
    }
    if (samSel) samSel.busy = false;
  }

  /** Decode the mask PNG, capture the depth band, build the volume pieces, paint, start tracking. */
  async function samShowMask(j) {
    samSel.instances = null;               // a click-driven mask is single-instance — drop any
    samTextChipsEl.style.display = "none"; // stale chip UI left over from a prior text search.
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = "data:image/png;base64," + j.mask; });
    const w = img.naturalWidth, h = img.naturalHeight;
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const ctx = off.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const px = ctx.getImageData(0, 0, w, h).data;
    const bits = new Uint8Array(w * h);
    let count = 0;
    for (let i = 0; i < bits.length; i++) if (px[i * 4] > 127) { bits[i] = 1; count++; }
    samSel.bits = bits; samSel.maskW = w; samSel.maskH = h; samSel.score = j.score;
    // One stable mask object: the evaluator's RLE-decode cache is keyed on it (WeakMap), and
    // commit serializes it verbatim.
    samSel.maskObj = { width: w, height: h, rle: rleEncodeMask(bits) };
    // Depth band NOW, while the camera still matches the captured view (a raster from a moved
    // camera would band the wrong depths). X-ray at commit time decides whether it is used.
    samSel.depth = null;
    const pick = player.pickRaster(384, 384);
    if (pick) {
      let zmin = Infinity, zmax = -Infinity;
      const { buf } = pick;
      for (let y = 0; y < buf.h; y++) for (let x = 0; x < buf.w; x++) {
        const mx = Math.floor((x / buf.w) * w), my = Math.floor((y / buf.h) * h);
        if (!bits[my * w + mx]) continue;
        const d = buf.depth[y * buf.w + x];
        if (d !== Infinity) { if (d < zmin) zmin = d; if (d > zmax) zmax = d; }
      }
      if (zmin !== Infinity) samSel.depth = { zmin: zmin - 0.002, zmax: zmax + 0.002 };
    }
    samPaintDirect(count);
    samLastCamKey = samSel.camKey;
    if (!samRaf) samWatch();
    const pts = samSel.points.length;
    samSelInfo.textContent = `mask ${j.score.toFixed(2)} · ${j.backend} · ${j.ms} ms · ${pts} pt${pts > 1 ? "s" : ""} · ${((count / bits.length) * 100).toFixed(0)}% of view`;
  }

  /** Exact-mask paint for the captured viewpoint (full mask resolution). */
  function samPaintDirect() {
    const { bits, maskW: w, maskH: h } = samSel;
    const [cr, cg, cb] = hexRgb(samSel.color || activeSegColor());
    const r = canvas.getBoundingClientRect();
    samMaskCanvas.style.left = r.left + "px"; samMaskCanvas.style.top = r.top + "px";
    samMaskCanvas.style.width = r.width + "px"; samMaskCanvas.style.height = r.height + "px";
    samMaskCanvas.width = w; samMaskCanvas.height = h;
    const mctx = samMaskCanvas.getContext("2d");
    const pv = mctx.createImageData(w, h);
    for (let i = 0; i < bits.length; i++) if (bits[i]) { pv.data[i * 4] = cr; pv.data[i * 4 + 1] = cg; pv.data[i * 4 + 2] = cb; pv.data[i * 4 + 3] = 115; }
    mctx.putImageData(pv, 0, 0);
    samMaskCanvas.style.display = "block";
  }

  /** The pending volume as committing would build it (X-ray decides the depth band). */
  const samVolume = () => ({
    type: "mask2d", kind: "bitmap", mask: samSel.maskObj, camera: samSel.cam,
    ...(!xrayOn() && samSel.depth ? { depth: samSel.depth } : {}),
  });

  /** Re-project the selection onto the CURRENT view: surface points from the pick raster are
   *  tested against the same evaluator the bake runs, so the tint sticks to the object. */
  function samPaintProjected() {
    const f = curFrame();
    const vol = samVolume();
    const keep = keepPredicateAt({ aresEdits: 1, ranges: [{ mode: "delete", startFrame: f, endFrame: f, keyframes: [{ frame: f, volumes: [vol] }] }] }, f);
    if (!keep) return;
    const r = canvas.getBoundingClientRect();
    const RW = 320, RH = Math.max(32, Math.round((RW * r.height) / Math.max(1, r.width)));
    const pick = player.pickRaster(RW, RH);
    if (!pick) return;
    const [cr, cg, cb] = hexRgb(samSel.color || activeSegColor());
    samMaskCanvas.style.left = r.left + "px"; samMaskCanvas.style.top = r.top + "px";
    samMaskCanvas.style.width = r.width + "px"; samMaskCanvas.style.height = r.height + "px";
    samMaskCanvas.width = RW; samMaskCanvas.height = RH;
    const mctx = samMaskCanvas.getContext("2d");
    const pv = mctx.createImageData(RW, RH);
    const inside = new Map();  // triangle id -> inside? (centroid test once per triangle)
    const { buf, triCentroid } = pick;
    for (let i = 0; i < buf.ids.length; i++) {
      const id = buf.ids[i];
      if (id < 0) continue;
      let hit = inside.get(id);
      if (hit === undefined) { const p = triCentroid(id); hit = !keep(p[0], p[1], p[2]); inside.set(id, hit); }
      if (hit) { pv.data[i * 4] = cr; pv.data[i * 4 + 1] = cg; pv.data[i * 4 + 2] = cb; pv.data[i * 4 + 3] = 115; }
    }
    mctx.putImageData(pv, 0, 0);
    samMaskCanvas.style.display = "block";
  }

  /** Camera watcher: repaint the preview when the view moves (throttled; exact paint at rest). */
  function samWatch() {
    samRaf = requestAnimationFrame(samWatch);
    if (!samSel || !samSel.bits) return;
    const k = camKeyNow();
    if (k === samLastCamKey) return;
    const now = performance.now();
    if (now - samLastRepaint < 90) return;    // ~11 Hz while the camera is moving
    samLastCamKey = k; samLastRepaint = now;
    if (k === samSel.camKey) samPaintDirect();  // back at the captured view — exact mask
    else samPaintProjected();
  }

  // Commit button label ALWAYS names the action it performs (Apply/bake means "do it", not "delete
  // it", so a button reading "Apply" over a delete-producing control reads backwards). The select
  // next to it decides Delete/Recolor/Copy…; the button text mirrors whichever is picked.
  const SAM_COMMIT_LABEL = { delete: "Delete ✂", recolor: "Recolor…", copy: "Copy…" };
  const samActSel = $("samActSel"), samApplyBtn = $("samApply");
  function syncSamCommitLabel() { samApplyBtn.textContent = SAM_COMMIT_LABEL[samActSel.value] || "Commit"; }
  samActSel.onchange = syncSamCommitLabel;
  syncSamCommitLabel();

  /** Commit the pending SAM selection as `action` (delete/recolor/copy — defaults to whatever the
   *  select next to the button is currently showing). Split action (editor v3 UI task §2): the tiny
   *  select decides what this selection becomes — Delete, Recolor (prefilled with the palette
   *  color), or Copy (srcFrame prefilled to the captured frame). setRangeAction never clobbers a
   *  payload the range already has, so re-applying onto the SAME still-active range keeps prior
   *  values. One commit = one undo step, even though it can both create the range AND add a volume.
   *  Returns true if it actually committed something (false = nothing pending, or nothing to delete
   *  — caller decides the fallback, e.g. the Delete key falling through to "remove active row"). */
  function commitSamSelection(action) {
    if (!samSel || !samSel.bits) { samSelClear(); return false; }
    if (!xrayOn() && !samSel.depth) { samSelInfo.textContent = "mask covers only background — nothing to delete"; return false; }
    doMutation(() => {
      setRangeAction(ensureRange(), action);
      addVolumeAtCurrent(samVolume());
    });
    samSelClear();
    return true;
  }
  /** Is there a pending SAM selection ready to commit right now? Used by the Delete/Backspace key
   *  precedence rule: a pending selection wins over "remove the active range row". */
  const hasPendingSamSelection = () => !!(samSel && samSel.bits && !samSel.busy);

  samApplyBtn.onclick = () => commitSamSelection(samActSel.value);

  // ---- SAM TEXT/concept prompt: same held frame as the click flow above, but
  // POSTs to /segment_text, which returns N instances {masks[], scores[]} instead of one mask.
  // The union/instance-pick logic below is the only new part — it reduces whichever instance(s)
  // are chosen down to the EXACT samSel shape samClick/samShowMask already produce (bits/maskW/
  // maskH/maskObj/depth/cam/camKey/color/busy), so commitSamSelection, samVolume, samPaintDirect/
  // Projected, samWatch and hasPendingSamSelection all run completely unchanged below this point.

  /** Health-gates the text row independently of the tracker dot above — the concept model loads
   *  AFTER the tracker and can be off entirely (SAM_TEXT=0); degrade to a compact hint, never
   *  hard-fail (the click-to-select tool is unaffected either way). */
  function syncSamTextGate(h) {
    const ready = !!(h && h.textEnabled && h.textReady);
    samTextInput.disabled = !ready;
    samTextGo.disabled = !ready;
    samTextHint.style.display = ready ? "none" : "";
    if (ready) return;
    // Terse status + the WHY in a tooltip (rail carries no prose — design law).
    const s = !h ? ["offline", "The SAM service isn't running. Press Start SAM above; click-to-select needs it too."]
      : !h.textEnabled ? ["text: off", "Text-prompt segmentation is disabled on the service (SAM_TEXT=0). Click-to-select still works. Set SAM_TEXT=1 and restart the SAM service to enable it."]
      : h.textLoading ? ["loading…", "The text/concept model is still loading — it initialises after the click tracker. This takes a few seconds on first use."]
      : h.textError ? ["text: failed", "The text model failed to load — see tools/sam-service/sam-service.log. Click-to-select is unaffected."]
      : ["text: n/a", "Text-prompt segmentation is unavailable on this service build. Click-to-select is unaffected."];
    samTextHint.textContent = s[0];
    samTextHint.title = s[1];
  }

  /** Recompute samSel.bits/maskObj/depth from the current instance pick ("all" = union of every
   *  returned mask, or a single instance index) and repaint — same recipe samShowMask uses. */
  function samApplyTextSelection() {
    const { instances, selected, maskW: w, maskH: h } = samSel;
    let bits;
    if (selected === "all") {
      bits = new Uint8Array(w * h);
      for (const inst of instances) for (let i = 0; i < bits.length; i++) if (inst.bits[i]) bits[i] = 1;
    } else {
      bits = instances[selected].bits;
    }
    let count = 0;
    for (let i = 0; i < bits.length; i++) if (bits[i]) count++;
    samSel.bits = bits;
    samSel.maskObj = { width: w, height: h, rle: rleEncodeMask(bits) };
    // Depth band NOW, while the camera still matches the captured view — identical recipe to
    // samShowMask (X-ray at commit time decides whether it is actually used).
    samSel.depth = null;
    const pick = player.pickRaster(384, 384);
    if (pick) {
      let zmin = Infinity, zmax = -Infinity;
      const { buf } = pick;
      for (let y = 0; y < buf.h; y++) for (let x = 0; x < buf.w; x++) {
        const mx = Math.floor((x / buf.w) * w), my = Math.floor((y / buf.h) * h);
        if (!bits[my * w + mx]) continue;
        const d = buf.depth[y * buf.w + x];
        if (d !== Infinity) { if (d < zmin) zmin = d; if (d > zmax) zmax = d; }
      }
      if (zmin !== Infinity) samSel.depth = { zmin: zmin - 0.002, zmax: zmax + 0.002 };
    }
    samPaintDirect();
    samLastCamKey = samSel.camKey;
    if (!samRaf) samWatch();
    const n = instances.length;
    const label = selected === "all" ? `${n} instance${n > 1 ? "s" : ""} (union)` : `#${selected + 1}/${n} · score ${instances[selected].score.toFixed(2)}`;
    samSelInfo.textContent = `"${samTextQuery}" · ${label} · ${samSel.ms} ms · ${((count / bits.length) * 100).toFixed(0)}% of view`;
  }

  /** [All | 1 | 2 | …] chips, SEG_PALETTE-colored per instance (Meta SAM idiom) — click one to
   *  isolate it, or All to go back to the union. Only rendered when there is more than one
   *  instance to choose between (a single result makes "All" and "1" the same selection). */
  function renderSamTextChips() {
    const n = samSel.instances.length;
    if (n <= 1) { samTextChipsEl.style.display = "none"; samTextChipsEl.innerHTML = ""; return; }
    const mk = (label, idx) => {
      const b = document.createElement("button");
      b.textContent = label;
      const active = samSel.selected === idx;
      const col = idx === "all" ? (samSel.color || activeSegColor()) : SEG_PALETTE[idx % SEG_PALETTE.length];
      b.style.cssText = "margin:0;padding:2px 7px;font-size:10.5px;border-radius:5px;cursor:pointer;" +
        `border:1px solid ${col};background:${active ? col + "3D" : "rgba(255,255,255,.05)"};color:${active ? "var(--text)" : "var(--text-dim)"}`;
      b.onclick = () => { samSel.selected = idx; renderSamTextChips(); samApplyTextSelection(); };
      return b;
    };
    samTextChipsEl.replaceChildren(mk("All", "all"), ...samSel.instances.map((_, i) => mk(String(i + 1), i)));
    samTextChipsEl.style.display = "flex";
  }

  /** Decode every returned instance mask PNG into bits, default-select the union, render chips. */
  async function samShowTextInstances(j) {
    const masksB64 = j.masks || [], scores = j.scores || [];
    samSel.ms = j.ms;
    if (!masksB64.length) {
      samSelInfo.textContent = `no matches for "${samTextQuery}"`;
      samSel.busy = false;
      return;
    }
    const instances = [];
    let w = 0, h = 0;
    for (let i = 0; i < masksB64.length; i++) {
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = "data:image/png;base64," + masksB64[i]; });
      w = img.naturalWidth; h = img.naturalHeight;
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      const ctx = off.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const px = ctx.getImageData(0, 0, w, h).data;
      const bits = new Uint8Array(w * h);
      for (let p = 0; p < bits.length; p++) if (px[p * 4] > 127) bits[p] = 1;
      instances.push({ bits, score: scores[i] ?? 0 });
    }
    samSel.instances = instances;
    samSel.selected = "all";
    samSel.maskW = w; samSel.maskH = h;
    renderSamTextChips();
    samApplyTextSelection();
    samSel.busy = false;
  }

  /** Enter/Go entry point: activates the SAM tool (same precondition the click flow has — held
   *  frame, paused playback), captures it, and starts a FRESH selection (a new query is a
   *  different concept entirely, unlike click refinement, so any pending selection is dropped). */
  async function samTextSearch() {
    const text = samTextInput.value.trim();
    if (!text || samTextGo.disabled) return;
    if (samSel && samSel.busy) return;
    if (tool !== "sam") document.querySelector('#editPanel .tool[data-tool="sam"]')?.click();
    samSelClear();
    samTextQuery = text;
    samSel = {
      capture: player.captureFrame(1024),
      cam: { ...player.getCamera(), aspect: player.getViewAspect() },
      camKey: camKeyNow(),
      color: activeSegColor(),
      points: [], labels: [], maskObj: null, bits: null, maskW: 0, maskH: 0, depth: null, score: 0,
      busy: true, instances: null, selected: "all",
    };
    samSelRow.style.display = "flex";
    samSelInfo.textContent = "segmenting…";
    try {
      const res = await fetch("/sam/segment_text", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ image: samSel.capture.dataUrl, text }),
      });
      if (!res.ok) {
        const detail = await res.json().then((j) => j && j.detail).catch(() => null);
        throw new Error(detail || ("HTTP " + res.status));
      }
      await samShowTextInstances(await res.json());
    } catch (err) {
      samSelInfo.textContent = "✗ text segment failed: " + (err && err.message ? err.message : err);
      if (samSel) samSel.busy = false;
    }
  }
  samTextGo.onclick = samTextSearch;
  samTextInput.addEventListener("keydown", (e) => {
    // Own Enter/Escape handling — the global shortcut listener already no-ops for any INPUT
    // target, so this never fights it, and typing here never triggers tool-letter hotkeys.
    if (e.key === "Enter") { e.preventDefault(); samTextSearch(); }
    else if (e.key === "Escape") { e.preventDefault(); samTextInput.blur(); samSelClear(); }
  });

  $("rangeStart").onclick = () => {
    doMutation(() => {
      ensureRange();
      captureKeyframe();     // sliders as they stand seed the first keyframe (if a box is set)
    });
    renderRanges();
  };
  $("rangeKey").onclick = () => doMutation(captureKeyframe);
  $("rangeEnd").onclick = () => {
    if (!activeRange) return;
    doMutation(() => {
      activeRange.endFrame = Math.max(curFrame(), activeRange.startFrame);
      activeRange = null;
    });
    $("rangeStart").disabled = false; $("rangeKey").disabled = true; $("rangeEnd").disabled = true;
    preview();
  };

  // Draw the strip once, immediately. The timeline is the ONLY scrubber now — when it lived in the
  // Edit rail it could wait for a sidecar or a first edit, because it was hidden until then.
  applyTrim({ save: false });
  applyXf({ save: false });   // seeds the readout + segmented controls from the default transform

  // Load an existing sidecar for this clip (non-destructive edits survive reloads).
  fetch("/edits/" + clipBase).then((r) => r.json()).then((j) => {
    if (!j || j.aresEdits !== 1) return;
    if (Array.isArray(j.ranges) && j.ranges.length) {
      edits.ranges = j.ranges;
      player.setEditPreview(edits);
    }
    // Restore the clip trim. save:false — reloading a clip must not rewrite its own sidecar.
    if (j.trim && Number.isInteger(j.trim.in) && Number.isInteger(j.trim.out) && j.trim.out >= j.trim.in) {
      trimIn = j.trim.in; trimOut = j.trim.out;
      applyTrim({ save: false });
      // Park the playhead on the in point: a trimmed clip's first frame IS its in point, and
      // opening on a frame the trim says is gone would contradict the dimmed head.
      tlSeek(trimIn);
    }
    if (j.transform && typeof j.transform === "object") { xf = { ...XF_DEFAULT(), ...j.transform }; applyXf({ save: false }); }
    renderRanges();
  }).catch(() => {});

  // Native folder picker for the bake source (no typing).
  $("bakePick").onclick = async () => {
    try {
      const r = await fetch("/pick?type=folder&for=bake").then((r) => r.json());
      if (r && r.path) { $("bakePath").value = r.path; $("bakePath").style.borderColor = ""; }
    } catch { $("bakeLog").textContent = "⚠ picker needs the ARES dev server running"; }
  };

  // Bake: re-encode the SOURCE frames with the edit list (+ optional crop) — the .ares is a
  // delivery artifact; edits are non-destructive and live in the sidecar.
  $("bakeGo").onclick = async () => {
    const dir = $("bakePath").value.trim();
    const log = $("bakeLog");
    if (!dir) { log.textContent = "⚠ enter the source frames folder (the OBJ/PNG directory this .ares came from)"; return; }
    const box = activeRange ? null : cropBox();          // the crop belongs to the range while authoring
    const trimmed = !trimIsFull();
    const xformed = !xfIsDefault();
    if (!box && !edits.ranges.length && !trimmed && !xformed) { log.textContent = "⚠ nothing to bake — transform, crop, trim the clip, or author a range first"; return; }
    const name = ($("bakeName").value.trim() || "cropped").replace(/[^a-z0-9._-]/gi, "_");
    const q = new URLSearchParams({ dir, name, textureCodec: "av1", texSize: "1024", crf: "30", smooth: "0" });
    if (box) q.set("crop", [...box.min, ...box.max].map((v) => v.toFixed(1)).join(","));
    // The model transform bakes into the geometry (encoder --up-axis/--center/...). Same core
    // evaluator as the live preview, so the baked clip lands exactly where the viewport showed it.
    if (xformed) {
      q.set("upAxis", xf.upAxis); q.set("center", xf.center); q.set("scale", String(xf.scale));
      q.set("rotate", xf.rotate.join(",")); q.set("translate", xf.translate.join(","));
    }
    // The sidecar carries the trim as well as the ranges, so a trim-only bake still needs it sent —
    // gating on ranges alone would have silently baked the untrimmed clip.
    if (edits.ranges.length || trimmed) {
      // flush the sidecar synchronously so the encoder reads the latest state
      clearTimeout(saveTimer);
      await fetch("/edits/" + clipBase, { method: "POST", body: JSON.stringify(edits, null, 1) }).catch(() => {});
      q.set("editsName", clipBase);
    }
    log.textContent = "▶ baking…\n";
    const es = new EventSource("/encode?" + q.toString());
    es.addEventListener("log", (e) => { log.textContent += JSON.parse(e.data) + "\n"; log.scrollTop = log.scrollHeight; });
    es.addEventListener("done", (e) => {
      es.close();
      log.textContent += "✓ done\n";
      const open = document.createElement("button");
      open.className = "btn"; open.style.cssText = "margin-top:6px;padding:5px 10px"; open.textContent = "Open " + name + ".ares";
      open.onclick = () => { location.search = "?src=" + name + ".ares"; };
      log.after(open);
    });
    es.addEventListener("error", () => { es.close(); log.textContent += "✗ bake failed (see server log)\n"; });
  };

  /** Del/Backspace precedence rule (Del applies the deletion): a pending SAM
   *  selection wins and commits as a DELETE range (regardless of what the Delete/Recolor/Copy…
   *  select is currently showing — Del is always "delete", the mouse-driven commit button is where
   *  Recolor/Copy are chosen explicitly). Only when there is NO pending selection does Del fall back
   *  to its prior meaning: remove the active (▶) range row. Returns true if it handled the key (so
   *  the caller doesn't ALSO run the fallback).
   */
  function commitPendingSelectionOrRemoveRow() {
    if (hasPendingSamSelection()) { commitSamSelection("delete"); return true; }
    deleteActiveRange();
    return true;
  }

  // Exposed to the global keyboard handler in main() — that's the one place with `player` in scope
  // for Space/arrows/tool letters too, so it stays the single keydown listener for the whole app.
  return { undo, redo, deleteActiveRange, flushSave, hasPendingSamSelection, commitSamSelection, commitPendingSelectionOrRemoveRow };
}

async function main() {
  if (!navigator.gpu) console.warn("[ares] WebGPU unavailable — using the WebGL2 fallback renderer (spec §10.4)");

  const [w, h] = fit();
  canvas.width = w; canvas.height = h;

  let player;
  try {
    const q0 = new URLSearchParams(location.search);
    player = await AresPlayer.create({
      canvas, src: SRC, loop: true, autoOrbit: false, onStats: renderHUD,  // orbit is opt-in via the Orbit button
      onEnded: () => { $("play").textContent = "▶"; },   // "once" mode auto-pauses on the last frame
      useWorker: q0.get("worker") === "1",   // §10.7 worker-thread geometry decode
      forceGL2: q0.get("gl2") === "1",       // §10.4 WebGL2 fallback (testing)
    });
  } catch (e) {
    // The clip failing never takes the app down: tabs are already wired, so Convert/Compare/
    // Settings all still work. Point at Settings, which can generate the synth clip one-click.
    const el = $("err");
    el.style.display = "block";
    el.innerHTML = `Failed to load <b>${SRC.split("/").pop()}</b>: ${(e && e.message ? e.message : e)}<br><br>
      Open <a href="#" id="errSettings">⚙ Settings</a> to check components — the synthetic demo clip
      can be generated there with one click — or pick another source from the Convert tab's history.`;
    document.getElementById("errSettings").onclick = (ev) => { ev.preventDefault(); setTab("settings"); };
    console.error(e);
    return;
  }

  window.__ares = player; // debug handle
  const backend = (new URLSearchParams(location.search).get("gl2") === "1" || !navigator.gpu) ? "WebGL2 fallback" : "WebGPU";
  $("title").textContent = "playing " + SRC.split("/").pop() + " — meshopt geometry + WebCodecs texture, " + backend;

  // Restore viewpoint/time carried over from a source switch (fair A/B: same instant, same angle).
  // SCALE GUARD: a carried camera is only meaningful if the new clip lives at a comparable world
  // scale — daniel-* captures are in millimeters, 4DViews bakes are in meters, so a mm-scale
  // distance/target applied to a meters clip parks the camera ~1000× too far out and the viewer
  // looks COMPLETELY BLANK (measured: 0 pickRaster hits vs 730 framed). Out-of-family carry →
  // keep the player's own AABB auto-framing instead of applying it.
  const qs = new URLSearchParams(location.search);
  if (qs.get("cam")) {
    const [az, el, d, tx, ty, tz] = qs.get("cam").split("_").map(Number);
    let camOk = [az, el, d, tx, ty, tz].every(Number.isFinite);
    const bb = camOk && player.getAabb ? player.getAabb() : null;
    if (bb) {
      const diag = Math.hypot(bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]) || 1;
      const toCenter = Math.hypot(
        tx - (bb.min[0] + bb.max[0]) / 2, ty - (bb.min[1] + bb.max[1]) / 2, tz - (bb.min[2] + bb.max[2]) / 2);
      camOk = d > diag * 0.02 && d < diag * 50 && toCenter < diag * 5;
    }
    if (camOk) {
      player.setCamera({ azimuth: az, elevation: el, distance: d, target: [tx, ty, tz] });
      player.autoOrbit = false;
    } else {
      console.warn("[ares] carried camera is out of scale for this clip — auto-framing instead");
    }
  }
  if (qs.get("t")) player.seek(Number(qs.get("t")));
  const startPaused = qs.get("paused") === "1";
  if (!startPaused) player.play();
  $("play").textContent = startPaused ? "▶" : "⏸";
  const orbitBtn = $("orbit");
  const syncOrbit = () => orbitBtn.setAttribute("aria-pressed", String(player.autoOrbit));
  syncOrbit();

  // --- View presets (Blender numbers) + view lock ("2D mode") — camera-only, never touch
  // `edits`/ranges/the sidecar, so they work in AND out of edit mode. Presets snap exactly
  // (no animation — this is an editing tool, not a demo) and keep distance/target untouched.
  //
  // ELEV_CLAMP mirrors packages/core/src/player.ts attachPointer's manual-drag elevation clamp
  // (±1.4 rad ≈ ±80.2°). A literal ±89.9° "top"/"bottom" would exceed that clamp: setCamera()
  // itself doesn't clamp, but the FIRST manual drag afterward would immediately snap the value
  // back down to ±1.4, a visible jump. Landing exactly ON the clamp boundary means a follow-up
  // drag continues smoothly from there instead.
  const ELEV_CLAMP = 1.4;
  const VIEW_PRESETS = {
    1: [0, 0], 3: [Math.PI / 2, 0], 7: [0, ELEV_CLAMP],               // front / right / top
  };
  const VIEW_PRESETS_OPP = {
    1: [Math.PI, 0], 3: [-Math.PI / 2, 0], 7: [0, -ELEV_CLAMP],       // back / left / bottom
  };
  function applyViewPreset(num, opposite) {
    const p = (opposite ? VIEW_PRESETS_OPP : VIEW_PRESETS)[num];
    if (!p) return;
    const cam = player.getCamera();
    player.setCamera({ azimuth: p[0], elevation: p[1], distance: cam.distance, target: cam.target });
  }

  let viewLocked = false;
  const lockBtn = $("viewLock"), lockTag = $("lockTag");
  function setViewLock(on) {
    viewLocked = on;
    lockBtn.setAttribute("aria-pressed", String(on));
    lockTag.style.display = on ? "" : "none";
    if (on && player.autoOrbit) { player.autoOrbit = false; syncOrbit(); }   // lock forces auto-orbit off
  }
  lockBtn.onclick = () => setViewLock(!viewLocked);
  // View lock is per-SESSION only (deliberately not localStorage/sessionStorage) — `viewLocked` is
  // a plain in-memory variable, so a reload always starts unlocked; a stale lock surviving a reload
  // would silently block rotation with no visible cause.

  // Suppress ONLY rotation while locked. Core's left-drag-orbit lives in packages/core/src/
  // player.ts attachPointer (a plain pointerdown/pointermove pair on the canvas) with no
  // suppression hook, so this is a demo-side interception: a capture-phase listener on `window`
  // runs during the capture leg, strictly BEFORE the event reaches canvas's own bubble-phase
  // listeners (core's included) — stopPropagation() there means core's `dragging` flag never
  // becomes true for a left-button press, so its pointermove handler (which mutates
  // orbit.azimuth/elevation) never runs at all; there is no per-frame "correct it back" flicker.
  // Only button 0 (left) is blocked — middle-drag pan (button 1) and wheel zoom are separate code
  // paths and are untouched, and this also catches the editor overlay's Alt+left-drag orbit
  // pass-through (same synthetic-dispatch target: canvas), so a tool-mode Alt-drag can't rotate
  // either. View presets call setCamera() directly (not a pointer gesture) so they still work
  // while locked. DECISION: left-drag in Nav is a true no-op while locked, not repurposed as pan —
  // middle-drag pan + wheel zoom remain the way to reframe without rotating.
  window.addEventListener("pointerdown", (e) => {
    if (viewLocked && e.target === canvas && e.button === 0) e.stopPropagation();
  }, true);

  orbitBtn.onclick = () => {
    if (viewLocked) return;   // no-op while locked — the visible lock tag/tooltip explains why
    player.autoOrbit = !player.autoOrbit; syncOrbit();
  };

  $("play").onclick = () => {
    if (player.isPlaying) { player.pause(); $("play").textContent = "▶"; }
    else { player.play(); $("play").textContent = "⏸"; }
  };

  // Loop-mode transport: cycle Loop → Ping-pong → Once (persisted). The player owns the frame math
  // (frameIndexForClock); ping-pong is a triangle wave, once clamps + auto-pauses (flips Play above).
  const LOOP_MODES = [
    { mode: "loop", label: "⟳", name: "Loop" },
    { mode: "pingpong", label: "⇄", name: "Ping-pong" },
    { mode: "once", label: "→", name: "Once" },
  ];
  const loopBtn = $("loopMode");
  let loopIdx = LOOP_MODES.findIndex((m) => m.mode === (localStorage.getItem("ares.loopMode") || "loop"));
  if (loopIdx < 0) loopIdx = 0;
  const applyLoopMode = () => {
    const m = LOOP_MODES[loopIdx];
    player.loopMode = m.mode;
    loopBtn.textContent = m.label;
    loopBtn.title = `at the clip end: ${m.name} — click to cycle Loop → Ping-pong → Once`;
    localStorage.setItem("ares.loopMode", m.mode);
  };
  applyLoopMode();
  loopBtn.onclick = () => { loopIdx = (loopIdx + 1) % LOOP_MODES.length; applyLoopMode(); };

  const editorApi = initEditor(player);

  // --- Keyboard shortcuts (viewer tab; inactive when a tool tab overlays) ---------------------
  //  Ctrl+S save-flush (always, any focus) · 1/3/7 (+ Ctrl = opposite face) view presets, Digit
  //  or Numpad · Ctrl+Z undo · Ctrl+Shift+Z / Ctrl+Y redo · Space play/pause · arrows step 1 frame
  //  (Shift = 10) · Home/End first/last · E edit panel · L view lock · V/M/B/S tools · X x-ray ·
  //  W wireframe · O orbit (no-op while locked) · Esc back to Nav · Delete/Backspace: with a
  //  pending SAM selection, commits it as a delete range; otherwise removes the active (▶) range
  //  row (only when not typing in an input)
  const holdAt = (sec) => { player.pause(); $("play").textContent = "▶"; player.seek(sec); };
  window.addEventListener("keydown", (e) => {
    const ctrl = e.ctrlKey || e.metaKey;

    // Ctrl+S: ALWAYS flush the sidecar save + block the browser's native Save-page dialog —
    // regardless of focus or active tab, since that dialog would otherwise pop up anywhere.
    if (ctrl && !e.altKey && (e.key === "s" || e.key === "S")) { e.preventDefault(); editorApi.flushSave(); return; }

    if (document.querySelector("div.tool.active")) return;   // Convert/Compare/Settings front-most

    // View presets (Blender numbers): Digit1/3/7 OR Numpad1/3/7 via e.code, so NumLock state and
    // keyboard layout can't break them (e.key for a numpad digit changes with NumLock; e.code
    // doesn't). Camera-only — never touches `edits`/ranges/the sidecar — so these work in AND out
    // of edit mode, and are deliberately NOT gated by "typing in an input" below.
    const presetDigit = /^(?:Digit|Numpad)([137])$/.exec(e.code);
    if (presetDigit && !e.shiftKey && !e.altKey) { e.preventDefault(); applyViewPreset(Number(presetDigit[1]), ctrl); return; }

    // Undo/redo: also not gated by "typing" — the rail's own controls ARE inputs, and undo should
    // work no matter which one last had focus.
    if (ctrl && !e.altKey && (e.key === "z" || e.key === "Z")) { e.preventDefault(); if (e.shiftKey) editorApi.redo(); else editorApi.undo(); return; }
    if (ctrl && !e.altKey && !e.shiftKey && (e.key === "y" || e.key === "Y")) { e.preventDefault(); editorApi.redo(); return; }

    if (ctrl) return;   // any OTHER ctrl combo: leave it to the browser/native behavior

    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.tagName === "BUTTON" || t.isContentEditable)) return;

    const st = player.getStats();
    const n = Math.max(1, st.frameCount);
    const step = (d) => holdAt((((st.frameIndex + d) % n + n) % n) / 30);
    const rail = $("editPanel");
    // A collapsed rail hides its buttons behind a ~30px strip (display:none on .railBody) — a
    // shortcut that changes tool state must expand it first, or the change is invisible until the
    // user notices and expands manually (the "clickable but not visible" trap).
    const expandRail = () => { if (rail.classList.contains("collapsed")) $("railCollapse").click(); };
    const clickTool = (name) => {
      expandRail();
      document.querySelector(`#editPanel .tool[data-tool="${name}"]`)?.click();
    };
    switch (e.key) {
      case " ": e.preventDefault(); $("play").click(); break;
      case "ArrowLeft": e.preventDefault(); step(e.shiftKey ? -10 : -1); break;
      case "ArrowRight": e.preventDefault(); step(e.shiftKey ? 10 : 1); break;
      case "Home": e.preventDefault(); holdAt(0); break;
      case "End": e.preventDefault(); holdAt((n - 1) / 30); break;
      // Clip trim in/out at the playhead — the NLE convention, and the two glyphs on the buttons.
      case "[": e.preventDefault(); window.__aresTrim?.setIn(); break;
      case "]": e.preventDefault(); window.__aresTrim?.setOut(); break;
      // ---- TRANSFORM TOOLS — Unity's W/E/R, plus F to focus. These are
      // the muscle memory, so they take the BARE keys; the three bindings they displaced move to
      // Shift+ (below). Note X/Y/Z constrain an armed transform DRAG — that's read inside the drag
      // handler, not here, so x-ray and shading keep their bare keys and G/S never had to move.
      case "w": expandRail(); window.__aresXform?.tool("move"); break;
      case "e": expandRail(); window.__aresXform?.tool("rotate"); break;
      case "r": expandRail(); window.__aresXform?.tool("scale"); break;
      case "f": case "F": window.__aresXform?.focus(); break;
      case "q": case "Q": clickTool("nav"); break;   // Unity's hand tool ≈ Nav (camera only)
      // ---- the three displaced by W/E/R above:
      case "W": window.__aresShade?.set(window.__aresShade.get() === "wire" ? "shaded" : "wire"); break;
      case "E": $("railCollapse").click(); break;                    // rail collapse
      case "R": expandRail(); $("cropGuidesToggle").click(); break;  // crop guides
      case "o": case "O": $("orbit").click(); break;
      case "l": case "L": $("viewLock").click(); break;
      case "v": case "V": clickTool("nav"); break;
      case "m": case "M": clickTool("sbox"); break;
      case "b": case "B": clickTool("brush"); break;
      case "s": case "S": clickTool("sam"); break;
      case "x": case "X": expandRail(); $("xray").click(); break;
      // Shading is a VIEWPORT mode (header bar), so it works with the Edit rail closed too. Z cycles.
      // NOT 1/2/3 — Digit1/3/7 are the view presets and return earlier in this handler.
      case "z": case "Z": {
        const order = ["shaded", "clay", "wire"];
        window.__aresShade?.set(order[(order.indexOf(window.__aresShade.get()) + 1) % order.length]);
        break;
      }
      // G is a viewport reference now (header bar), like Z/W/P — it must not need the rail open.
      case "g": case "G": $("gridToggle").click(); break;
      case "r": case "R": expandRail(); $("cropGuidesToggle").click(); break;
      case "p": case "P": window.__aresProj.set(window.__aresProj.get() === "ortho" ? "persp" : "ortho"); break;
      case "Escape": expandRail(); document.querySelector('#editPanel .tool[data-tool="nav"]')?.click(); break;
      case "Delete": case "Backspace": e.preventDefault(); editorApi.commitPendingSelectionOrRemoveRow(); break;
    }
  });

  let rt = 0;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { const [w, h] = fit(); player.resize(w, h); }, 100);
  });
}

main();
