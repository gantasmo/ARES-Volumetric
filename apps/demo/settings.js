/**
 * Settings tab — dependency status + guided installs.
 *
 * The dev server's /deps reports every optional model, tool, and dataset: present or not, what
 * it enables, and how to get it. Actions are either external links (gated/manual downloads) or
 * SSE-streamed local installs (/setup/*) that refuse to touch anything already present. Nothing
 * here gates the app itself — missing pieces only disable their feature, which warns at use.
 */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let deps = [];

/** Other modules ask "is X available?" before starting work that needs it (graceful warnings). */
export async function depStatus(id) {
  if (!deps.length) { try { deps = (await fetch("/deps").then((r) => r.json())).deps || []; } catch { return null; } }
  return deps.find((d) => d.id === id) || null;
}

async function render() {
  const out = $("depsOut");
  try { deps = (await fetch("/deps").then((r) => r.json())).deps || []; }
  catch { out.innerHTML = `<div class="card"><div class="note" style="color:#f0a3a3">Dependency check needs the ARES dev server (launch via Play ARES Demo.vbs).</div></div>`; return; }

  const row = (d, i) => {
    const color = d.present ? "#63d68a" : d.optional ? "var(--text-faint)" : "var(--warn)";
    const size = d.sizeMB ? d.sizeMB + " MB" : (d.sizeNote || "");
    let action = "";
    if (!d.present && d.action) {
      if (d.action.kind === "link") action = `<a class="btn ghost" style="margin:0;padding:5px 12px;font-size:12px;text-decoration:none" href="${esc(d.action.url)}" target="_blank" rel="noopener">Download page ↗</a>`;
      else if (d.action.kind === "sse") action = `<button class="btn ghost" style="margin:0;padding:5px 12px;font-size:12px" data-sse="${esc(d.action.route)}" data-i="${i}">${esc(d.action.label || "Install")}</button>`;
    }
    const note = d.action && d.action.note ? `<div class="note2">${d.present ? "" : esc(d.action.note)}</div>` : "";
    return `<div class="dep">
      <span class="dot" style="background:${color}"></span>
      <span class="lbl">${esc(d.label)}${d.optional ? ' <small style="color:var(--text-faint);font-weight:400">optional</small>' : ""}</span>
      <span class="en">${esc(d.enables)}</span>
      <span class="side">${action}<span class="size">${d.present ? "installed" + (size ? " · " + size : "") : size}</span></span>
      ${d.present && d.path ? `<div class="pth" title="${esc(d.path)}">${esc(d.path)}</div>` : note}
      <div id="depLog${i}" class="note2" style="display:none;font:10.5px ui-monospace,monospace;white-space:pre-wrap;max-height:120px;overflow:auto"></div>
    </div>`;
  };

  const ready = deps.filter((d) => d.present).length;
  out.innerHTML = `
    <div class="card">
      <div style="display:flex;gap:10px;align-items:baseline;margin-bottom:4px">
        <h3 style="margin:0;flex:1">Components</h3>
        <span class="note" style="margin:0">${ready}/${deps.length} present</span>
        <button class="btn ghost" id="depRefresh" style="margin:0;padding:5px 12px;font-size:12px">Refresh</button>
      </div>
      ${deps.map(row).join("")}
    </div>
    <div class="card">
      <h3>Service endpoints</h3>
      <div class="kv">
        <div class="k">SAM segmentation</div><div class="v" id="depSam">checking…</div>
        <div class="k">dev server</div><div class="v">this page — encode, enhance, pickers, history</div>
      </div>
      <div class="note">The SAM service starts from the editor panel (Start SAM) or automatically on first use.</div>
    </div>`;

  $("depRefresh").onclick = render;
  fetch("/sam/health").then((r) => r.json()).then((h) => {
    $("depSam").textContent = h && h.ok ? `running · ${h.model} on ${h.device}` : h && h.loading ? "loading model…" : "not running (starts on demand)";
  }).catch(() => { $("depSam").textContent = "not running (starts on demand)"; });

  for (const b of out.querySelectorAll("[data-sse]")) b.onclick = () => {
    const log = $("depLog" + b.dataset.i);
    b.disabled = true;
    log.style.display = "block";
    log.textContent = "starting…\n";
    const es = new EventSource(b.dataset.sse);
    const line = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
    es.addEventListener("log", (e) => { try { line(JSON.parse(e.data)); } catch { /* ignore */ } });
    es.addEventListener("done", (e) => { es.close(); try { line("✓ " + (JSON.parse(e.data).message || "done")); } catch { line("✓ done"); } setTimeout(render, 1200); });
    es.addEventListener("error", (e) => {
      es.close(); b.disabled = false;
      try { line("✗ " + (JSON.parse(e.data).message || "failed")); } catch { line("✗ failed (connection lost)"); }
    });
  };
}

export function initSettings() { render(); }
