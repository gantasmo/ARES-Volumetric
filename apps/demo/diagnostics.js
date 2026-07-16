// Diagnostics status bar — a slim bottom rail (Blender/Apollo-style) that surfaces live system +
// pipeline telemetry from /diagnostics (SSE, 2s): coherent-bake stage+ETA, GPU util/temp/VRAM, RAM,
// disk, RunPod on/off. UI law: dark greys (never black), offwhite HIGH-CONTRAST text, compact
// monospace readouts, minimal padding, one-click collapse, tooltips on every segment, no glow.
import { logSubscribe, logMemory, logRestore, logClear, alog } from "./log.js";

const $ = (id) => document.getElementById(id);
const COLLAPSE_KEY = "ares.diagbar.collapsed";
const LOGOPEN_KEY = "ares.diagbar.logopen";
/** How long a finished bake stays on the bar before it goes idle. The server reports `bake.done`
 *  forever, so without this the bar advertises the last bake as though it were the current state —
 *  it sat on "✓ daniel-coherent-v3.ares" for days, next to whatever clip was actually playing. */
const BAKE_DONE_TTL_MS = 90_000;

const fmtT = (s) => { s = Math.max(0, Math.round(s || 0)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : (m ? `${m}m ${s % 60}s` : `${s}s`); };
const gb = (mb) => (mb / 1024).toFixed(1);
const stageLabel = (s) => ({ register: "Registering", "bake-gpu": "GPU baking", bake: "Baking", encode: "Encoding" }[s] || s || "…");

/** Set by initDiagnostics; a no-op before the bar exists so callers never have to care about order. */
let setClipSeg = () => {};

/**
 * Tell the bar what's loaded. Called by main.js from the player's stats — the bar itself owns no
 * knowledge of clips, and deliberately does not poll for one.
 *
 * Cheap to call every stats tick: it early-outs unless something actually changed.
 */
let lastClipKey = "";
export function setDiagClip(c) {
  const key = c ? `${c.name}|${(c.sizeMB || 0).toFixed(1)}|${c.frames}` : "";
  if (key === lastClipKey) return;
  lastClipKey = key;
  setClipSeg(c);
}

/**
 * The activity-log panel: a scrollback that expands UP from the diagnostics bar.
 *
 * Deliberately NOT a card and NOT new chrome — it grows out of the rail that is already there
 * (UI law: slim rails, minimal padding, one toggle, no whitespace waste). Monospace, dense rows,
 * severity by colour only. Filter chips are toggles, not a dropdown, so state is visible at a glance.
 */
function buildLogPanel(toggleBtn) {
  const panel = document.createElement("div");
  panel.id = "diagLogPanel";
  panel.style.cssText = [
    "position:fixed", "left:0", "right:0", "bottom:22px", "z-index:39",
    "height:min(46vh,420px)", "display:none", "flex-direction:column",
    "background:#1b1b1f", "border-top:1px solid #34343a",
    "font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace", "color:#cbc8ba",
  ].join(";");

  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:8px;padding:4px 8px;border-bottom:1px solid #2c2c32;background:#202024;flex:0 0 auto";
  const title = document.createElement("span");
  title.textContent = "ACTIVITY";
  title.style.cssText = "color:#8f8c81;letter-spacing:.1em;font-size:10px";

  // Source filters. Everything on by default — the point is a complete record.
  const show = { app: true, bake: true, claude: true, error: true };
  const COLOR = { app: "#cbc8ba", bake: "#9ec98f", claude: "#8fa9c9", error: "#d99a9a" };
  const chips = {};
  const chipRow = document.createElement("span");
  chipRow.style.cssText = "display:flex;gap:4px";
  for (const k of ["app", "bake", "claude", "error"]) {
    const c = document.createElement("button");
    c.textContent = k;
    c.title = `show / hide ${k} lines`;
    c.style.cssText = `all:unset;cursor:pointer;padding:1px 5px;border:1px solid ${COLOR[k]};border-radius:2px;font-size:10px;color:${COLOR[k]}`;
    c.onclick = () => { show[k] = !show[k]; c.style.opacity = show[k] ? "1" : ".32"; render(); };
    chips[k] = c; chipRow.append(c);
  }

  const search = document.createElement("input");
  search.type = "text"; search.placeholder = "filter…";
  search.id = "diagLogFilter"; search.name = "diagLogFilter";   // a11y: form fields need id+name
  search.style.cssText = "all:unset;flex:1;min-width:60px;padding:1px 5px;background:#26262b;border:1px solid #34343a;border-radius:2px;color:#e8e6da;font:11px ui-monospace,monospace";
  search.oninput = render;

  const mkBtn = (txt, title, fn) => { const b = document.createElement("button"); b.textContent = txt; b.title = title; b.style.cssText = "all:unset;cursor:pointer;color:#8f8c81;padding:1px 5px;font-size:10px;border:1px solid #34343a;border-radius:2px"; b.onclick = fn; return b; };
  const copyBtn = mkBtn("copy", "copy the visible log to the clipboard", async () => {
    const txt = visible().map(fmtLine).join("\n");
    try { await navigator.clipboard.writeText(txt); copyBtn.textContent = "copied"; setTimeout(() => (copyBtn.textContent = "copy"), 1200); } catch { /* */ }
  });
  const clearBtn = mkBtn("clear", "erase the log on the server — this is a permanent audit trail, so it asks first", async () => {
    if (!confirm("Clear the activity log?\n\nThis erases the server-side history permanently.")) return;
    await logClear();
  });
  const closeBtn = mkBtn("✕", "close (the log keeps recording)", () => setOpen(false));

  head.append(title, chipRow, search, copyBtn, clearBtn, closeBtn);

  const body = document.createElement("div");
  body.style.cssText = "flex:1;overflow:auto;padding:3px 0";

  panel.append(head, body);
  document.body.append(panel);

  const ts = (t) => { const d = new Date(t); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`; };
  const fmtLine = (e) => `${ts(e.t)} ${String(e.src).padEnd(6)} ${e.lvl === "act" ? "*" : " "} ${e.msg}${e.data ? "  " + e.data : ""}`;
  const visible = () => {
    const q = search.value.trim().toLowerCase();
    return logMemory().filter((e) => show[e.src] !== false && (!q || (e.msg + " " + (e.data || "")).toLowerCase().includes(q)));
  };

  let pinned = true;   // follow the tail unless the owner has scrolled up to read something
  body.addEventListener("scroll", () => { pinned = body.scrollTop + body.clientHeight >= body.scrollHeight - 24; });

  function rowEl(e) {
    const r = document.createElement("div");
    r.style.cssText = "display:flex;gap:7px;padding:1px 8px;white-space:pre-wrap;word-break:break-word";
    if (e.lvl === "error") r.style.background = "rgba(217,154,154,.09)";
    else if (e.lvl === "act") r.style.background = "rgba(127,143,110,.10)";
    const a = document.createElement("span"); a.textContent = ts(e.t); a.style.cssText = "color:#6f6d64;flex:0 0 auto";
    const b = document.createElement("span"); b.textContent = e.src; b.style.cssText = `color:${COLOR[e.src] || "#8f8c81"};flex:0 0 44px`;
    const c = document.createElement("span");
    c.textContent = e.msg + (e.data ? "  " + e.data : "");
    c.style.cssText = `flex:1;color:${e.lvl === "error" ? "#d99a9a" : e.lvl === "warn" ? "#d6bd8a" : "#cbc8ba"}`;
    r.append(a, b, c);
    return r;
  }
  function render() {
    const rows = visible();
    body.replaceChildren(...rows.map(rowEl));
    if (pinned) body.scrollTop = body.scrollHeight;
    count.textContent = String(rows.length);
  }
  const count = document.createElement("span");
  count.style.cssText = "color:#6f6d64;font-size:10px;flex:0 0 auto";
  head.append(count);

  let open = false;
  try { open = localStorage.getItem(LOGOPEN_KEY) === "1"; } catch { /* */ }
  function setOpen(v) {
    open = v;
    panel.style.display = open ? "flex" : "none";
    toggleBtn.textContent = open ? "▤ log ▾" : "▤ log";
    toggleBtn.style.color = open ? "#e8e6da" : "#8f8c81";
    try { localStorage.setItem(LOGOPEN_KEY, open ? "1" : "0"); } catch { /* */ }
    if (open) { pinned = true; render(); }
  }
  toggleBtn.onclick = () => setOpen(!open);

  // Incremental: append one row on a new entry; full rerender when the ring was replaced (null).
  logSubscribe((e) => {
    if (!open) return;
    if (e === null) { render(); return; }
    const q = search.value.trim().toLowerCase();
    if (show[e.src] === false) return;
    if (q && !(e.msg + " " + (e.data || "")).toLowerCase().includes(q)) return;
    body.append(rowEl(e));
    while (body.childElementCount > 3000) body.firstElementChild.remove();
    if (pinned) body.scrollTop = body.scrollHeight;
    count.textContent = String(body.childElementCount);
  });

  // Restore the server-side history so a fresh tab shows everything that happened before it opened.
  logRestore(800).then(() => { if (open) render(); });
  setOpen(open);
  return { setOpen, render };
}

export function initDiagnostics() {
  if ($("diagBar")) return;
  const bar = document.createElement("div");
  bar.id = "diagBar";
  bar.style.cssText = [
    "position:fixed", "left:0", "right:0", "bottom:0", "z-index:40",
    "display:flex", "align-items:center", "gap:14px",
    "height:22px", "padding:0 8px", "box-sizing:border-box",
    "font:11px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    "color:#cbc8ba", "background:#202024", "border-top:1px solid #34343a",
    "user-select:none",
  ].join(";");

  const seg = (title) => { const s = document.createElement("span"); s.style.cssText = "display:inline-flex;align-items:center;gap:5px;white-space:nowrap"; if (title) s.title = title; return s; };
  // collapse chevron (uniform small button)
  const chev = document.createElement("button");
  chev.style.cssText = "all:unset;cursor:pointer;color:#8f8c81;padding:0 4px;font-size:11px";
  chev.title = "Collapse / expand the diagnostics bar";

  // The clip actually loaded, live. First segment because it's the thing the bar is most often read
  // for — "what am I looking at". Fed by main.js via setDiagClip() from the player's own stats.
  const clipSeg = seg("The clip currently loaded — name · delivered size · frames · duration. Live, from the player.");
  const bakeSeg = seg("Coherent bake — stage, frame, and measured ETA (from actual throughput, not a guess). Goes idle shortly after a bake finishes; it is NOT the loaded clip.");
  const gpuSeg = seg("GPU — utilization · temperature · VRAM used / total");
  const ramSeg = seg("System RAM — used / total");
  const diskSeg = seg("Disk (project drive) — free space");
  const rpSeg = seg("RunPod — on/off, active pod GPU + hourly cost, account balance");

  // ---- Activity log toggle. Lives ON the bar (no new chrome, one toggle). The audit trail of
  // what actually happened lives behind this door.
  const logBtn = document.createElement("button");
  logBtn.id = "diagLogBtn";
  logBtn.style.cssText = "all:unset;cursor:pointer;color:#8f8c81;padding:0 5px;font-size:11px;white-space:nowrap";
  logBtn.title = "Activity log — everything the app, the bakes, and the tooling did. Server-persisted: survives tab changes and reloads.";

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;align-items:center;gap:14px;flex:1;overflow:hidden";
  wrap.append(clipSeg, bakeSeg, gpuSeg, ramSeg, diskSeg, rpSeg);
  bar.append(chev, wrap, logBtn);
  document.body.append(bar);
  // keep the viewport clear of the bar
  document.body.style.paddingBottom = "22px";

  const logPanel = buildLogPanel(logBtn);

  let collapsed = false;
  try { collapsed = localStorage.getItem(COLLAPSE_KEY) === "1"; } catch { /* */ }
  const applyCollapse = () => {
    chev.textContent = collapsed ? "▸" : "▾";
    wrap.style.display = collapsed ? "none" : "flex";
    bar.style.width = collapsed ? "auto" : "";
    bar.style.right = collapsed ? "auto" : "0";
  };
  chev.onclick = () => { collapsed = !collapsed; try { localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0"); } catch { /* */ } applyCollapse(); };
  applyCollapse();

  const dim = (el, on) => { el.style.opacity = on ? "1" : ".45"; };
  const barEl = (pct) => `<span style="display:inline-block;width:64px;height:5px;background:#37373d;border-radius:2px;overflow:hidden;vertical-align:middle"><span style="display:block;height:100%;width:${pct}%;background:#7f8f6e"></span></span>`;

  // The server has no timestamp on bake.done, so age it from when it was FIRST SEEN — persisted, or
  // a bake from days ago would re-announce itself for 90s on every single page load. Keyed by the
  // bake's own identity, so a genuinely new bake is always fresh.
  const SEEN_KEY = "ares.diagbar.bakeSeen";
  const loadSeen = () => { try { return new Map(Object.entries(JSON.parse(localStorage.getItem(SEEN_KEY) || "{}"))); } catch { return new Map(); } };
  const doneSeenAt = loadSeen();
  /** Bakes carried over from an EARLIER session. The server tails the bake log and reports its last
   *  [DONE] forever, while the in-memory guard below dies with the page — so every reload used to
   *  re-fire the auto-add and resurrect a clip the user had deliberately deleted from the library.
   *  A bake we have already announced once is never announced again: the user's delete wins. */
  const alreadyAnnounced = new Set(doneSeenAt.keys());
  let addedKey = null;
  // Change-detectors so the 2s SSE tick can't spam the activity log with identical lines.
  let lastBakeStage = null, lastBakePct = -1, lastBakeDone = null;
  /** Announce a finished bake exactly once, ever — not once per page load. */
  const announceOnce = (d, key) => {
    if (!d.ok || !d.name || !key) return;
    if (alreadyAnnounced.has(key) || addedKey === key) return;
    addedKey = key;
    window.dispatchEvent(new CustomEvent("ares:bake-done", { detail: d }));
  };
  const saveSeen = () => {
    try {
      // Keep only the most recent handful — this is a display cache, not a log.
      const recent = [...doneSeenAt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
      localStorage.setItem(SEEN_KEY, JSON.stringify(Object.fromEntries(recent)));
    } catch { /* private mode: the TTL just falls back to per-session */ }
  };
  const render = (d) => {
    // ---- Bake ----
    const bake = d.bake || {};
    const doneKey = bake.done ? `${bake.done.name}|${bake.done.sizeMB}|${bake.done.totalS}` : null;
    if (doneKey && !doneSeenAt.has(doneKey)) { doneSeenAt.set(doneKey, Date.now()); saveSeen(); }
    const doneFresh = doneKey && (Date.now() - doneSeenAt.get(doneKey)) < BAKE_DONE_TTL_MS;
    if (bake.done && doneFresh) {
      const ok = bake.done.ok;
      bakeSeg.innerHTML = "";
      const t = document.createElement("span");
      // "baked" spelled out: this line used to read as the loaded clip's name and size.
      t.textContent = ok ? `✓ baked ${bake.done.name} (${bake.done.sizeMB}MB, ${fmtT(bake.done.totalS)})` : `✗ bake failed`;
      t.style.color = ok ? "#9ec98f" : "#d99a9a";
      bakeSeg.append(t);
      dim(bakeSeg, true);
      // Fire the auto-add exactly once per bake, regardless of the display TTL above.
      announceOnce(bake.done, doneKey);
    } else if (bake.done && !doneFresh) {
      // A bake that finished while this tab was closed still auto-adds — but only the FIRST time we
      // ever see it. Re-announcing a long-finished bake is what used to undo the user's deletes.
      announceOnce(bake.done, doneKey);
      bakeSeg.textContent = "idle"; dim(bakeSeg, false);
    } else if (bake.prog && bake.prog.total) {
      const p = bake.prog;
      bakeSeg.innerHTML = `<span style="color:#b7b3a4">⏳ ${stageLabel(p.stage)}</span> <span>${p.done}/${p.total}</span> ${barEl(p.pct || 0)} <span>${p.pct || 0}%</span> <span style="color:#8f8c81">ETA ${fmtT(p.etaS)}</span>`;
      dim(bakeSeg, true);
    } else { bakeSeg.textContent = "idle"; dim(bakeSeg, false); }

    // ---- Forward pipeline milestones into the activity log. Only on CHANGE: the SSE ticks every 2s
    // and an unconditional log would bury everything else under thousands of identical progress lines.
    const stageKey = bake.prog ? `${bake.prog.stage}|${bake.prog.total}` : null;
    if (stageKey && stageKey !== lastBakeStage) { lastBakeStage = stageKey; alog(`bake stage: ${stageLabel(bake.prog.stage)} (${bake.prog.total} items)`, { src: "bake", lvl: "act" }); }
    if (bake.prog && bake.prog.pct != null) {
      const decade = Math.floor(bake.prog.pct / 10);
      if (decade !== lastBakePct) { lastBakePct = decade; alog(`bake ${stageLabel(bake.prog.stage)} ${bake.prog.pct}% (${bake.prog.done}/${bake.prog.total}, ETA ${fmtT(bake.prog.etaS)})`, { src: "bake" }); }
    }
    if (doneKey && doneKey !== lastBakeDone) {
      lastBakeDone = doneKey;
      alog(bake.done.ok ? `bake DONE: ${bake.done.name} (${bake.done.sizeMB}MB in ${fmtT(bake.done.totalS)})` : `bake FAILED: ${bake.done.error || "unknown"}`,
        { src: "bake", lvl: bake.done.ok ? "act" : "error" });
    }

    // ---- GPU ----
    if (d.gpu) {
      const g = d.gpu; const hot = g.temp >= 84;
      gpuSeg.innerHTML = `<span style="color:#8f8c81">GPU</span> <span>${g.util}%</span> <span style="color:${hot ? "#d99a9a" : "#cbc8ba"}">${g.temp}°</span> <span>${gb(g.vramUsedMB)}/${gb(g.vramTotalMB)}G</span>`;
      dim(gpuSeg, true);
    } else { gpuSeg.innerHTML = `<span style="color:#8f8c81">GPU —</span>`; dim(gpuSeg, false); }

    // ---- RAM ----
    if (d.ram) { ramSeg.innerHTML = `<span style="color:#8f8c81">RAM</span> <span>${gb(d.ram.usedMB)}/${gb(d.ram.totalMB)}G</span>`; }

    // ---- Disk ----
    if (d.disk) { const low = d.disk.freeGB < 20; diskSeg.innerHTML = `<span style="color:#8f8c81">DISK</span> <span style="color:${low ? "#d99a9a" : "#cbc8ba"}">${d.disk.freeGB}G free</span>`; }

    // ---- RunPod ----
    const rp = d.runpod;
    if (!rp || rp.error) { rpSeg.innerHTML = `<span style="color:#8f8c81">☁ —</span>`; dim(rpSeg, false); }
    else {
      const on = rp.pods && rp.pods.length;
      const dot = on ? "#9ec98f" : "#6f6d64";
      rpSeg.innerHTML = `<span style="color:${dot}">●</span> <span>${on ? `${rp.pods[0].gpu || "pod"} $${(rp.pods[0].costPerHr || 0).toFixed(2)}/hr` : "off"}</span> <span style="color:#8f8c81">$${(rp.balance || 0).toFixed(2)}</span>`;
      dim(rpSeg, true);
    }
  };

  // ---- Now playing. Pushed by main.js from the player's live stats; nothing here polls.
  clipSeg.innerHTML = `<span style="color:#8f8c81">▶ —</span>`;
  dim(clipSeg, false);
  setClipSeg = (c) => {
    if (!c || !c.name) { clipSeg.innerHTML = `<span style="color:#8f8c81">▶ —</span>`; dim(clipSeg, false); return; }
    const bits = [`<span style="color:#8f8c81">▶</span>`, `<span style="color:#e8e6da">${c.name}</span>`];
    if (c.sizeMB) bits.push(`<span>${c.sizeMB >= 1024 ? (c.sizeMB / 1024).toFixed(2) + "G" : c.sizeMB.toFixed(0) + "M"}</span>`);
    if (c.frames) bits.push(`<span style="color:#8f8c81">${c.frames}f · ${fmtT(c.durationS)}</span>`);
    clipSeg.innerHTML = bits.join(" ");
    clipSeg.title = `Loaded clip: ${c.name}\n${c.sizeMB ? c.sizeMB.toFixed(1) + " MB delivered · " : ""}${c.frames || "?"} frames @ ${c.fps || 30}fps · ${(c.durationS || 0).toFixed(1)}s`
      + `\n\nThis is what's PLAYING. The segment beside it is the last bake, which is a different thing.`;
    dim(clipSeg, true);
  };

  let es = null, finished = false;
  const connect = () => {
    es = new EventSource("/diagnostics");
    es.addEventListener("diag", (e) => { try { render(JSON.parse(e.data)); } catch { /* */ } });
    es.onerror = () => { es.close(); if (!finished) setTimeout(connect, 4000); };
  };
  connect();
}
