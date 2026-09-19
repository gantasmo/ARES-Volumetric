/**
 * Settings tab — hardware-aware component installer.
 *
 * The server (/deps, backed by tools/installer.mjs) reports the real GPU, every component's
 * detected state, and three profiles costed against that GPU. This renders them and drives
 * /install, which resolves each component's `requires` graph before downloading anything.
 *
 * The shape of the tab follows the decision a person actually makes:
 *   1. "what have I got"      — the hardware line, with the dtype and CUDA build it implies
 *   2. "just set it up"       — three profiles, the affordable one badged, one click
 *   3. "no, I want THAT one"  — the component list, tick what you want, install the selection
 * A gated model is never a dead end: the licence button sits next to it, and once accepted the
 * same Install button works, because the download runs through the stored HF token.
 */

import { accessPrompt } from "./ensure.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const gb = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(mb % 1024 && mb < 10240 ? 1 : 0) + " GB" : (mb || 0) + " MB");

let deps = [], gpu = null, profs = [], recommended = null, pre = null;
let selected = new Set();
let busy = false;

/** Other modules ask "is X available?" before starting work that needs it (graceful warnings).
 *  Aliases are honoured so callers can keep using the id they already know. */
export async function depStatus(id) {
  if (!deps.length) { try { await load(); } catch { return null; } }
  return deps.find((d) => d.id === id || (d.aliases || []).includes(id)) || null;
}

async function load() {
  const d = await fetch("/deps").then((r) => r.json());
  deps = d.deps || []; gpu = d.gpu; profs = d.profiles || []; recommended = d.recommended; pre = d.preflight;
  try { shell = await fetch("/shell/status").then((r) => r.json()); } catch { shell = null; }
  return d;
}

// ---------------------------------------------------------------- pieces ----

function hardwareCard() {
  if (!gpu) return "";
  const blockers = [];
  if (!pre?.python && !pre?.envPresent) blockers.push("CPython 3.11 to 3.13: none detected. The Python environment component unpacks a private interpreter.");
  const gpuRows = gpu.gpus.map((g, i) =>
    `<div class="k">GPU ${i}</div><div class="v">${esc(g.name)} · ${gb(g.vramMB)} · sm_${String(g.cc).replace(".", "")}</div>`).join("");
  return `
    <div class="card">
      <div class="row" style="margin-bottom:6px"><h3 style="margin:0;flex:1">Your machine</h3>
        <span class="note" style="margin:0">${esc(gpu.label)}</span></div>
      <div class="kv">
        ${gpuRows || `<div class="k">GPU</div><div class="v">none detected: models will run on the CPU</div>`}
        <div class="k">precision</div><div class="v">${esc(gpu.dtype)} <small style="color:var(--text-faint)">· ${esc(gpu.dtypeWhy)}</small></div>
        ${gpu.cudaIndex ? `<div class="k">PyTorch build</div><div class="v">${esc(gpu.cudaIndex)} <small style="color:var(--text-faint)">· ${esc(gpu.cudaWhy)}</small></div>` : ""}
        <div class="k">Hugging Face</div><div class="v" id="hfState">${pre?.hfToken ? "token stored: gated models can download" : "not signed in"}</div>
      </div>
      ${blockers.map((b) => `<div class="note2" style="color:var(--warn);margin-top:6px">${b}</div>`).join("")}
      ${tokenFormHtml(!!pre?.hfToken)}
    </div>`;
}

/** Hugging Face sign-in, inline. Gated repos used to dead-end here: the licence button opened a
 *  browser and the token still had to be created with `hf auth login` in a terminal. The token is
 *  posted once to /hf-token, validated server-side, and never rendered back. */
function tokenFormHtml(haveToken) {
  if (haveToken) return `<div class="note2" style="margin-top:6px">Signed in. <a href="#" id="hfChange">Use a different token</a></div>`;
  return `
    <div id="hfForm" style="margin-top:8px">
      <div class="row" style="flex-wrap:wrap;gap:6px">
        <input id="hfToken" class="inp" type="password" autocomplete="off" spellcheck="false"
               placeholder="hf_…" style="flex:1;min-width:200px">
        <button class="u primary" id="hfSave">Sign in</button>
        <a class="u gatebtn" href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Create a token ↗</a>
      </div>
      <div class="note2" id="hfMsg" style="margin-top:4px">A read token is enough. Stored by huggingface_hub, the same place the CLI keeps it.</div>
    </div>`;
}

function wireTokenForm() {
  const change = $("hfChange");
  if (change) change.onclick = (e) => {
    e.preventDefault();
    change.closest(".note2").outerHTML = tokenFormHtml(false);
    wireTokenForm();
  };
  const btn = $("hfSave");
  if (!btn) return;
  const submit = async () => {
    const input = $("hfToken"), msg = $("hfMsg");
    const token = input.value.trim();
    if (!token) { input.focus(); return; }
    btn.disabled = true; msg.textContent = "checking…"; msg.style.color = "";
    let r;
    try { r = await fetch("/hf-token", { method: "POST", body: JSON.stringify({ token }) }).then((x) => x.json()); }
    catch { r = { ok: false, error: "dev server not reachable" }; }
    input.value = "";                                     // never leave the token in the DOM
    btn.disabled = false;
    if (r.ok) { msg.style.color = "var(--good)"; msg.textContent = r.user ? `signed in as ${r.user}` : "token stored"; setTimeout(() => render(), 700); }
    else { msg.style.color = "var(--bad)"; msg.textContent = r.error || "rejected"; }
  };
  btn.onclick = submit;
  $("hfToken").onkeydown = (e) => { if (e.key === "Enter") submit(); };
}

function profileCards() {
  const byId = Object.fromEntries(deps.map((d) => [d.id, d]));
  const card = (p) => {
    const isRec = p.id === recommended;
    const names = p.included.map((id) => byId[id]?.label).filter(Boolean);
    const dropped = p.dropped.length
      ? `<div class="note2" style="color:var(--warn)">Left out: this GPU cannot hold ${p.dropped.map((d) => `${esc(d.label)} (needs ${gb(d.needMB)})`).join(", ")}.</div>`
      : "";
    return `
      <div class="prof${isRec ? " rec" : ""}${p.complete ? " done" : ""}">
        <div class="ptop">
          <span class="pname">${esc(p.label)}</span>
          ${isRec ? `<span class="badge">recommended for your GPU</span>` : ""}
        </div>
        <div class="pblurb">${esc(p.blurb)}</div>
        <div class="pstat">
          <b>${p.complete ? "already complete" : gb(p.downloadMB) + " to download"}</b>
          ${p.peakVramMB ? ` · needs ${gb(p.peakVramMB)} VRAM at peak` : ""}
        </div>
        <div class="plist">${names.map((n) => esc(n)).join(" · ")}</div>
        ${dropped}
        <button class="u pgo" data-profile="${esc(p.id)}"${p.complete ? " disabled" : ""}>
          ${p.complete ? "Installed" : "Install " + esc(p.label)}</button>
      </div>`;
  };
  return `<div class="card">
      <div class="row" style="margin-bottom:2px"><h3 style="margin:0;flex:1">One click</h3></div>
      <div class="note" style="margin:0 0 8px">Each one installs everything it needs, in order: Python environment included. Anything you already have is skipped.</div>
      <div class="profs">${profs.map(card).join("")}</div>
    </div>`;
}

/** Windows shell integration. Registering writes only under HKCU\Software\Classes, so it needs
 *  no elevation and the same card turns it off again. */
let shell = null;
function shellCard() {
  if (!shell || !shell.supported) return "";
  const on = shell.registered;
  return `
    <div class="card">
      <div class="row" style="margin-bottom:2px"><h3 style="margin:0;flex:1">Windows context menu</h3>
        <button class="u${on ? "" : " primary"}" id="shellToggle">${on ? "Remove" : "Add"}</button></div>
      <div class="note" style="margin:0 0 6px">“Convert folder to .ares” on a folder, “Convert to .ares” on ${shell.exts} file types. Written under HKCU, no administrator, and Remove takes it back out.</div>
      <div class="kv">
        <div class="k">folders</div><div class="v">${on && shell.folders ? "registered" : "not registered"}</div>
        <div class="k">file types</div><div class="v">${on && shell.files ? shell.exts + " registered" : "not registered"}</div>
        <div class="k">every file type</div><div class="v"><label style="cursor:pointer"><input type="checkbox" id="shellAll"${shell.allFiles ? " checked" : ""}> also add it to all files</label></div>
      </div>
      <div class="note2" style="margin-top:6px">${esc(shell.note || "")}</div>
      <div class="note2" id="shellMsg" style="margin-top:4px"></div>
      ${msixRows()}
    </div>`;
}

/** The Windows 11 short menu. Separate from the registry verbs above because it is a different
 *  mechanism with a different failure mode: a packaged IExplorerCommand handler, which has to be
 *  built and signed, and whose certificate needs one elevated command to trust. */
function msixRows() {
  const m = shell && shell.msix;
  if (!m || !m.supported) return "";
  const on = m.installed;
  // Nothing here blocks the button: Install adds MSVC Build Tools through /install when the
  // compiler is absent, and the server switches Developer Mode and trusts the certificate itself,
  // each behind one Windows elevation prompt.
  const pending = [
    !m.buildable ? "MSVC Build Tools + Windows SDK: installed first (2.5 GB)" : "",
    !m.devMode ? "Developer Mode: enabled during install (elevation prompt)" : "",
  ].filter(Boolean).join(" · ");
  return `
    <div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">
      <div class="row" style="margin-bottom:2px">
        <span class="cap" style="flex:1">Windows 11 short menu</span>
        <button class="u${on ? "" : " primary"}" id="msixToggle">${on ? "Remove" : "Install"}</button>
      </div>
      <div class="note2">The entries above live under “Show more options”. This packaged handler puts the same verb in the short menu Windows 11 opens first.</div>
      <div class="kv" style="margin-top:4px">
        <div class="k">package</div><div class="v">${on ? esc(m.packageFullName || "installed") : "not installed"}</div>
        <div class="k">certificate</div><div class="v">${m.certTrusted ? "trusted" : "not trusted: imported during install (elevation prompt)"}</div>
      </div>
      ${pending && !on ? `<div class="note2" style="margin-top:4px">${pending}</div>` : ""}
      <div class="note2" id="msixMsg" style="margin-top:4px"></div>
    </div>`;
}

function wireShellCard() {
  const btn = $("shellToggle");
  if (!btn) return;
  const mbtn = $("msixToggle");
  if (mbtn) mbtn.onclick = async () => {
    const installed = !!(shell.msix && shell.msix.installed);
    // The compiler is a catalog component: install it through the normal progress panel first.
    // runInstall re-renders the tab, so every element is looked up again afterwards.
    if (!installed && !shell.msix.buildable) {
      if (busy) return;
      if (!(await runInstall(["msvc-build-tools"]))) return;
    }
    const mbtn2 = $("msixToggle") || mbtn, msg = $("msixMsg");
    mbtn2.disabled = true;
    msg.style.color = ""; msg.textContent = installed ? "removing…" : "building, signing, registering: about a minute on the first build…";
    let r;
    try { r = await fetch(installed ? "/shell/msix-uninstall" : "/shell/msix-install", { method: "POST", body: "{}" }).then((x) => x.json()); }
    catch { r = { ok: false, error: "dev server not reachable" }; }
    mbtn2.disabled = false;
    if (r.ok) { msg.style.color = "var(--good)"; msg.textContent = (r.log || []).slice(-1)[0] || "done"; render(); return; }
    msg.style.color = "var(--bad)";
    msg.textContent = r.error || "failed";
  };

  btn.onclick = async () => {
    const msg = $("shellMsg");
    btn.disabled = true;
    msg.style.color = ""; msg.textContent = shell.registered ? "removing…" : "registering…";
    const route = shell.registered ? "/shell/unregister" : "/shell/register";
    let r;
    try { r = await fetch(route, { method: "POST", body: JSON.stringify({ allFiles: !!$("shellAll")?.checked }) }).then((x) => x.json()); }
    catch { r = { ok: false, error: "dev server not reachable" }; }
    btn.disabled = false;
    if (r.ok) { msg.style.color = "var(--good)"; msg.textContent = (r.log || []).slice(-1)[0] || "done"; render(); }
    else { msg.style.color = "var(--bad)"; msg.textContent = r.error || "failed"; }
  };
}

function componentRows() {
  const installable = deps.filter((d) => !d.statusOnly);
  const groups = [...new Set(installable.map((d) => d.group))];
  const row = (d) => {
    const on = selected.has(d.id);
    const size = d.present
      ? `installed${d.onDiskMB ? " · " + gb(d.onDiskMB) : ""}`
      : `${gb(d.sizeMB)}${d.vramMB ? " · " + gb(d.vramMB) + " VRAM" : ""}`;
    const gate = d.gated && !d.present
      ? `<a class="u gatebtn" href="${esc(d.gated.url)}" target="_blank" rel="noopener">Accept licence ↗</a>`
        + (pre?.hfToken ? "" : `<span class="csize" style="color:var(--warn)">token needed</span>`)
      : "";
    return `
      <label class="comp${d.present ? " have" : ""}">
        <input type="checkbox" data-id="${esc(d.id)}"${on ? " checked" : ""}${d.present ? " disabled" : ""}>
        <span class="cmain">
          <span class="clbl">${esc(d.label)}${d.present ? ' <span class="tick">✓</span>' : ""}</span>
          <span class="cen">${esc(d.enables)}</span>
          ${d.why ? `<span class="cwhy">${esc(d.why)}</span>` : ""}
          ${d.diskNote ? `<span class="cwhy" style="color:var(--warn)">${esc(d.diskNote)}</span>` : ""}
          ${d.present && d.path ? `<span class="cpath" title="${esc(d.path)}">${esc(d.path)}</span>` : ""}
        </span>
        <span class="cside">${gate}<span class="csize">${size}</span></span>
      </label>`;
  };
  const chosen = [...selected];
  const totalMB = chosen.reduce((s, id) => s + (deps.find((d) => d.id === id)?.sizeMB || 0), 0);
  return `<div class="card">
      <div class="row" style="margin-bottom:2px"><h3 style="margin:0;flex:1">Or pick your own</h3>
        <button class="u" id="cmpInstall"${chosen.length ? "" : " disabled"}>
          ${chosen.length ? `Install ${chosen.length} · ${gb(totalMB)}` : "Install selected"}</button></div>
      <div class="note" style="margin:0 0 6px">Dependencies come along automatically: ticking a model pulls in the Python environment if it is missing.</div>
      ${groups.map((g) => `<div class="cgroup">${esc(g)}</div>` + installable.filter((d) => d.group === g).map(row).join("")).join("")}
    </div>`;
}

function statusRows() {
  const rows = deps.filter((d) => d.statusOnly);
  if (!rows.length) return "";
  return `<details class="card"><summary style="cursor:pointer;font-weight:600;font-size:13px">Other components (${rows.filter((r) => r.present).length}/${rows.length} present)</summary>
      <div class="note" style="margin:6px 0">Not downloadable from here: your own capture data, or a licensed SDK no installer may fetch.</div>
      ${rows.map((d) => `
        <label class="comp status${d.present ? " have" : ""}">
          <span style="width:16px;text-align:center;color:var(--text-faint)">${d.present ? "✓" : "·"}</span>
          <span class="cmain">
            <span class="clbl">${esc(d.label)}</span>
            <span class="cen">${esc(d.enables)}</span>
            <span class="cwhy">${esc(d.why || "")}</span>
            ${d.present && d.path ? `<span class="cpath" title="${esc(d.path)}">${esc(d.path)}</span>` : ""}
          </span>
          <span class="cside">${d.link ? `<a class="u gatebtn" href="${esc(d.link)}" target="_blank" rel="noopener">Get it ↗</a>` : ""}<span class="csize">${d.present ? "present" : "missing"}</span></span>
        </label>`).join("")}
    </details>`;
}

// ---------------------------------------------------------------- install ----

/** Stream one /install run into the log panel. Resolves true when it completes cleanly. */
function runInstall(ids) {
  return new Promise((done) => {
    busy = true;
    const panel = $("instPanel"), log = $("instLog"), title = $("instTitle"), bar = $("instBar");
    panel.style.display = "block";
    log.textContent = "";
    title.textContent = "Starting…";
    bar.style.width = "0%";
    for (const b of document.querySelectorAll("#depsOut .u")) b.disabled = true;
    const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
    const es = new EventSource("/install?ids=" + encodeURIComponent(ids.join(",")));
    let total = 0;
    es.addEventListener("log", (e) => { try { line(JSON.parse(e.data)); } catch { /* ignore */ } });
    es.addEventListener("plan", (e) => { try { total = JSON.parse(e.data).todo.length; } catch { /* ignore */ } });
    es.addEventListener("step", (e) => {
      try { const s = JSON.parse(e.data); title.textContent = `${s.label}: ${s.index + 1} of ${s.total}`; bar.style.width = ((s.index / s.total) * 100).toFixed(0) + "%"; } catch { /* ignore */ }
    });
    es.addEventListener("stepDone", (e) => {
      try { const s = JSON.parse(e.data); bar.style.width = ((s.index / s.total) * 100).toFixed(0) + "%"; } catch { /* ignore */ }
    });
    es.addEventListener("done", (e) => {
      es.close(); busy = false;
      bar.style.width = "100%";
      try { title.textContent = "✓ " + JSON.parse(e.data).message; } catch { title.textContent = "✓ done"; }
      selected.clear();
      setTimeout(() => render().then(() => done(true)), 800);
    });
    es.addEventListener("error", (e) => {
      es.close(); busy = false;
      let msg = "failed (connection lost)", gate = null;
      try { const d = JSON.parse(e.data); msg = d.message || msg; gate = d.gated ? d : null; } catch { /* no payload */ }
      title.textContent = "✗ " + msg;
      line("✗ " + msg);
      // Re-render so the buttons come back, but keep the log on screen to be read. A gated
      // repository raises the access prompt under the log; Resume repeats this same install.
      render({ keepLog: true }).then(() => {
        if (gate) {
          const host = document.createElement("div");
          $("instPanel").append(host);
          accessPrompt(host, gate, () => { runInstall(ids).then(done); });
        } else done(false);
      });
    });
  });
}

// ----------------------------------------------------------------- render ----

async function render({ keepLog = false } = {}) {
  const out = $("depsOut");
  const priorLog = keepLog ? $("instLog")?.textContent : null;
  const priorTitle = keepLog ? $("instTitle")?.textContent : null;
  try { await load(); }
  catch {
    out.innerHTML = `<div class="card"><div class="note" style="color:var(--bad)">Dev server not reachable: component status unavailable.</div></div>`;
    return;
  }

  const ready = deps.filter((d) => d.present).length;
  out.innerHTML = `
    ${hardwareCard()}
    ${profileCards()}
    <div class="card" id="instPanel" style="display:${keepLog ? "block" : "none"}">
      <div class="row" style="margin-bottom:6px"><h3 style="margin:0;flex:1" id="instTitle">Installing…</h3></div>
      <div class="ibar"><div class="ifill" id="instBar" style="width:0%"></div></div>
      <div id="instLog" class="ilog"></div>
    </div>
    ${componentRows()}
    ${statusRows()}
    ${shellCard()}
    <div class="card">
      <div class="row" style="margin-bottom:4px"><h3 style="margin:0;flex:1">Services</h3>
        <span class="note" style="margin:0">${ready}/${deps.length} components present</span>
        <button class="u" id="depRefresh">Refresh</button></div>
      <div class="kv">
        <div class="k">SAM segmentation</div><div class="v" id="depSam">checking…</div>
        <div class="k">dev server</div><div class="v">this page: encode, enhance, pickers, history</div>
      </div>
    </div>`;

  if (keepLog && priorLog != null) { $("instLog").textContent = priorLog; $("instTitle").textContent = priorTitle; }

  $("depRefresh").onclick = () => render();
  wireTokenForm();
  wireShellCard();
  for (const b of out.querySelectorAll("[data-profile]")) b.onclick = () => {
    if (busy) return;
    const p = profs.find((x) => x.id === b.dataset.profile);
    if (p) runInstall(p.included);
  };
  wirePicker();   // owns the checkboxes and the Install-selected button

  fetch("/sam/health").then((r) => r.json()).then((h) => {
    $("depSam").textContent = h && h.ok
      ? `running · ${h.backend || h.model} in ${h.dtype || "?"} on ${h.device}`
      : h && h.loading ? "loading model…" : "not running (starts on demand)";
  }).catch(() => { $("depSam").textContent = "not running (starts on demand)"; });
}

/** (Re)attach handlers inside the pick-your-own card after it is replaced. */
function wirePicker() {
  const btn = $("cmpInstall");
  if (btn) btn.onclick = () => { if (!busy && selected.size) runInstall([...selected]); };
  for (const c of document.querySelectorAll("#depsOut .comp input[type=checkbox]")) c.onchange = () => {
    if (c.checked) selected.add(c.dataset.id); else selected.delete(c.dataset.id);
    // Re-render just this card so the running total and the button label stay honest.
    const host = $("cmpInstall")?.closest(".card");
    if (host) { host.outerHTML = componentRows(); wirePicker(); }
  };
}

export function initSettings() { render(); }
