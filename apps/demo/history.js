/**
 * Unified volcap history — one searchable list, rendered by the Convert tab (the standalone
 * Inspect tab that used to share this panel was merged into Convert 2026-07-11;
 * "inspect" survives as a history entry KIND, not a tab).
 *
 * The dev server auto-records folder analyses, encodes, and enhance runs (serve.mjs);
 * probe results (kind "inspect") are posted client-side because file probes never leave the
 * browser. Entries live in apps/demo/history.json (server-side, survives reloads).
 */

const KINDS = {
  analyse: { label: "analysed", color: "var(--accent)" },
  encode: { label: "encoded", color: "#63d68a" },
  enhance: { label: "enhanced", color: "var(--series-b)" },
  inspect: { label: "inspected", color: "var(--warn)" },
};

/** Post one history entry (best-effort — history must never block the primary action). */
export async function recordHistory(entry) {
  try { await fetch("/history/add", { method: "POST", body: JSON.stringify(entry) }); } catch { /* server down */ }
}

const fmtWhen = (iso) => {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const sameDay = new Date().toDateString() === d.toDateString();
  return sameDay ? d.toTimeString().slice(0, 5) : d.toISOString().slice(0, 10);
};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const metaSummary = (e) => {
  const m = e.meta || {};
  if (e.kind === "analyse") return [m.meshes && `${m.meshes} ${m.kind || "mesh"}`, m.rawBytes && `${(m.rawBytes / 1048576).toFixed(0)} MB raw`].filter(Boolean).join(" · ");
  if (e.kind === "encode") return [m.source === "4ds" && "4DS→", m.codec && m.codec.toUpperCase(), m.texSize && m.texSize + "²", m.crf && "crf" + m.crf, m.frames && `${m.frames}f`, m.mirrorX && "mirror-X"].filter(Boolean).join(" ");
  if (e.kind === "enhance") return [m.frames && `${m.frames} frames`, m.tier].filter(Boolean).join(" · ");
  if (e.kind === "inspect") return [m.format, m.frames && `${m.frames}f`, m.sizeMB && m.sizeMB + " MB"].filter(Boolean).join(" · ");
  return "";
};

/**
 * Render a searchable history list into `host`, filtered to `kinds`.
 * `actions(item)` returns [{label, run}] row buttons. Returns { refresh }.
 */
export function initHistoryPanel({ host, kinds, actions }) {
  host.innerHTML = `
    <div class="card hist">
      <div style="display:flex;gap:10px;align-items:baseline;margin-bottom:8px">
        <h3 style="margin:0;flex:1">History</h3>
        <input type="search" class="histSearch" name="history-filter" placeholder="filter by name or path…"
          style="width:220px;padding:6px 9px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.14);border-radius:7px;color:var(--text);font:12px system-ui">
      </div>
      <div class="histList"></div>
    </div>`;
  const list = host.querySelector(".histList");
  const search = host.querySelector(".histSearch");
  let all = [];

  const render = () => {
    const q = search.value.trim().toLowerCase();
    const items = all
      .filter((e) => kinds.includes(e.kind))
      .filter((e) => !q || (e.name || "").toLowerCase().includes(q) || (e.path || "").toLowerCase().includes(q));
    if (!items.length) {
      list.innerHTML = `<div class="note" style="margin:2px 0 4px">${all.some((e) => kinds.includes(e.kind)) ? "no matches" : "nothing yet — items appear here as folders are analysed, files inspected, and encodes finish"}</div>`;
      return;
    }
    list.innerHTML = items.slice(0, 40).map((e, i) => {
      const k = KINDS[e.kind] || { label: e.kind, color: "var(--text-dim)" };
      return `<div class="hrow" data-h="${i}">
        <span class="chip" style="background:${k.color}22;color:${k.color}">${k.label}</span>
        <span class="nm" title="${esc(e.name)}">${esc(e.name || "—")}</span>
        <span class="pth" title="${esc(e.path)}">${esc(e.path || "")}</span>
        <span class="mt">${esc(metaSummary(e))}</span>
        <span class="dt">${fmtWhen(e.at)}</span>
        <span class="act"></span>
      </div>`;
    }).join("");
    const shown = items.slice(0, 40);
    for (const row of list.querySelectorAll(".hrow")) {
      const item = shown[Number(row.dataset.h)];
      const act = row.querySelector(".act");
      for (const a of (actions ? actions(item) : [])) {
        const b = document.createElement("button");
        b.textContent = a.label;
        b.onclick = (ev) => { ev.stopPropagation(); a.run(item); };
        act.append(b);
      }
    }
  };

  const refresh = async () => {
    try { all = (await fetch("/history").then((r) => r.json())).items || []; } catch { all = []; }
    render();
  };
  search.oninput = render;
  refresh();
  return { refresh };
}
