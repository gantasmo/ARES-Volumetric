/**
 * Compute tab — RunPod control + watch (the shared "family" view).
 *
 * The dev server proxies the RunPod API and drives pods over SSH; the API key + private SSH key
 * stay SERVER-SIDE (the browser only calls /runpod/*). This panel: live balance/spend/pods,
 * launch (confirm-gated — the only spend), terminate, and a live log console that streams either
 * a pod's run log (Logs) or a named action (Probe / Setup) straight from the pod over SSH.
 */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtDur = (s) => {
  if (s == null) return "—";
  s = Math.floor(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
};

let timer = null, es = null;

function logLine(t) {
  const box = $("rpLog");
  if (!box) return;
  box.textContent += t + "\n";
  box.scrollTop = box.scrollHeight;
}
function stopStream() { if (es) { es.close(); es = null; } }
function openStream(url, header) {
  stopStream();
  const box = $("rpLog");
  if (box) box.textContent = (header ? header + "\n" : "");
  es = new EventSource(url);
  es.addEventListener("log", (e) => { try { logLine(JSON.parse(e.data)); } catch { /* */ } });
  es.addEventListener("done", (e) => { let c = 0; try { c = JSON.parse(e.data).code; } catch { /* */ } logLine(c === 0 ? "✓ done" : "— finished (exit " + c + ")"); stopStream(); });
  es.addEventListener("error", (e) => { try { logLine("✗ " + (JSON.parse(e.data).message || "stream error")); } catch { logLine("✗ stream closed"); } stopStream(); });
}

async function render() {
  const out = $("rpOut");
  if (!out) return;
  let d;
  try { d = await fetch("/runpod/status").then((r) => r.json()); }
  catch { out.innerHTML = `<div class="card"><div class="note" style="color:var(--bad)">Compute needs the ARES dev server running.</div></div>`; return; }
  if (!d.ok) { out.innerHTML = `<div class="card"><div class="note" style="color:var(--bad)">RunPod: ${esc(d.error)}</div></div>`; return; }

  const bal = Number(d.balance || 0);
  const spend = Number(d.spendPerHr || 0);
  const pods = d.pods || [];
  const podRows = pods.length ? pods.map((p) => `
    <div class="dep" style="align-items:center">
      <span class="lbl">${esc(p.name || p.id)}</span>
      <span class="en">${esc(p.gpu || "?")}${p.gpuCount > 1 ? " ×" + p.gpuCount : ""} · ${p.status === "RUNNING" ? `<span style="color:var(--good)">RUNNING</span>` : esc(p.status || "?")} · $${Number(p.costPerHr || 0).toFixed(2)}/hr · up ${fmtDur(p.uptimeS)}</span>
      <span class="side" style="display:flex;gap:5px">
        <button class="u" data-logs="${esc(p.id)}">Logs</button>
        <button class="u" data-probe="${esc(p.id)}">Probe</button>
        <button class="u" data-setup="${esc(p.id)}">Setup</button>
        <button class="u danger" data-stop="${esc(p.id)}">Terminate</button>
      </span>
    </div>`).join("") : `<div class="note">No pods running.</div>`;

  out.innerHTML = `
    <div class="card">
      <div style="display:flex;gap:14px;align-items:center;margin-bottom:4px">
        <h3 style="margin:0;flex:1">Account</h3>
        <button class="u primary" id="rpLaunch">Launch RTX 3090 · $0.22/hr</button>
        <button class="u" id="rpRefresh">Refresh</button>
      </div>
      <div class="kv">
        <div class="k">Balance</div><div class="v" style="font-weight:600">$${bal.toFixed(2)}</div>
        <div class="k">Spending now</div><div class="v">${spend > 0.01 ? "$" + spend.toFixed(2) + "/hr" : "idle ($" + spend.toFixed(3) + "/hr)"}</div>
      </div>
    </div>
    <div class="card">
      <h3 style="margin:0 0 4px">Pods</h3>
      ${podRows}
    </div>
    <div class="card" style="flex:1 1 auto;min-height:0;display:flex;flex-direction:column">
      <div style="display:flex;gap:10px;align-items:baseline;margin-bottom:4px">
        <h3 style="margin:0;flex:1">Console</h3>
        <button class="u" id="rpClear">Clear</button>
      </div>
      <pre id="rpLog" style="margin:0;font:10.5px ui-monospace,monospace;white-space:pre-wrap;flex:1 1 auto;min-height:80px;overflow:auto;background:var(--bg-rail);border-radius:var(--r);padding:6px 8px;color:var(--text-mid)"></pre>
    </div>`;

  $("rpRefresh").onclick = render;
  $("rpClear").onclick = () => { const b = $("rpLog"); if (b) b.textContent = ""; };

  $("rpLaunch").onclick = async () => {
    if (!confirm("Launch a community RTX 3090 pod (~$0.22/hr)?\nBilling starts now and runs until you Terminate it.")) return;
    logLine("launching RTX 3090…");
    try {
      const r = await fetch("/runpod/launch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: "ares-sam3d" }) }).then((x) => x.json());
      if (!r.ok) { logLine("✗ launch failed: " + r.error); return; }
      logLine("✓ launched pod " + r.pod.id + " — booting; opening logs…");
      setTimeout(render, 1500);
      setTimeout(() => openStream("/runpod/logs?id=" + encodeURIComponent(r.pod.id), "▶︎ logs for " + r.pod.id), 1600);
    } catch (e) { logLine("✗ launch error: " + e.message); }
  };

  for (const b of out.querySelectorAll("[data-logs]")) b.onclick = () => openStream("/runpod/logs?id=" + encodeURIComponent(b.dataset.logs), "▶︎ logs for " + b.dataset.logs);
  for (const b of out.querySelectorAll("[data-probe]")) b.onclick = () => openStream("/runpod/action?script=probe&id=" + encodeURIComponent(b.dataset.probe), "▶︎ probe " + b.dataset.probe);
  for (const b of out.querySelectorAll("[data-setup]")) b.onclick = () => openStream("/runpod/action?script=setup&id=" + encodeURIComponent(b.dataset.setup), "▶︎ setup " + b.dataset.setup);
  for (const b of out.querySelectorAll("[data-stop]")) b.onclick = async () => {
    if (!confirm("Terminate pod " + b.dataset.stop + "? This deletes it and stops billing.")) return;
    logLine("terminating " + b.dataset.stop + "…");
    try {
      const r = await fetch("/runpod/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: b.dataset.stop }) }).then((x) => x.json());
      logLine(r.ok ? "✓ terminated" : "✗ " + r.error);
      setTimeout(render, 1200);
    } catch (e) { logLine("✗ " + e.message); }
  };
}

export function initCompute() {
  render();
  if (timer) clearInterval(timer);
  // Live "watch together" refresh while the tab is open; guarded so it idles on other tabs.
  timer = setInterval(() => {
    const t = $("tab-compute");
    if (t && t.classList.contains("active") && !es) render(); // don't re-render mid-stream (would drop the console)
  }, 12000);
}
