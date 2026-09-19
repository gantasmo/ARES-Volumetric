/**
 * Context menus — one implementation for every right-click in the app.
 *
 * The rails are ~270 px of controls pinned to the screen edges, so acting on something in the
 * viewport meant: click the thing, travel to the rail, click the control, travel back. A context
 * menu puts the operations that apply to what is under the cursor AT the cursor.
 *
 * Deliberately small: a flat list with separators and captions, no submenus. A submenu is a
 * second travel problem, and nothing here needs one.
 *
 *   showMenu(event, [
 *     { cap: "Selection" },
 *     { label: "Isolate", hint: "Enter", run: () => …, primary: true },
 *     "-",
 *     { label: "X-ray", hint: "X", checked: true, run: () => … },
 *     { label: "Delete", hint: "Del", run: () => …, danger: true, disabled: false },
 *   ]);
 *
 * Items with no `run` and no `cap` are ignored, so callers can build a list with `&&` guards
 * and drop the falsy entries without filtering.
 */

let host = null;      // the open menu element, or null
let onClose = null;

const STYLE = `
#ctxMenu { position: fixed; z-index: 60; min-width: 190px; padding: 4px;
  background: var(--bg-panel); border: 1px solid var(--border-st); border-radius: var(--r);
  box-shadow: 0 8px 28px rgba(0,0,0,.5); font: 12px system-ui; user-select: none; }
#ctxMenu .mi { display: flex; align-items: center; gap: 12px; padding: 5px 8px; border-radius: var(--r);
  color: var(--text); cursor: pointer; white-space: nowrap; }
#ctxMenu .mi .lbl { flex: 1; }
#ctxMenu .mi .hint { font: 10px ui-monospace, monospace; color: var(--text-faint); }
#ctxMenu .mi:hover, #ctxMenu .mi.on { background: var(--accent); color: var(--accent-ink); }
#ctxMenu .mi:hover .hint, #ctxMenu .mi.on .hint { color: var(--accent-ink); opacity: .75; }
#ctxMenu .mi.primary .lbl { font-weight: 600; }
#ctxMenu .mi.danger { color: var(--bad); }
#ctxMenu .mi.danger:hover, #ctxMenu .mi.danger.on { background: var(--bad); color: #fff; }
#ctxMenu .mi.dis { color: var(--text-faint); cursor: default; }
#ctxMenu .mi.dis:hover { background: none; color: var(--text-faint); }
#ctxMenu .mi .tick { width: 10px; font-size: 10px; color: var(--accent); }
#ctxMenu .mi:hover .tick, #ctxMenu .mi.on .tick { color: var(--accent-ink); }
#ctxMenu .sep { height: 1px; margin: 4px 6px; background: var(--border); }
#ctxMenu .cap { padding: 5px 8px 3px; font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .4px; color: var(--text-faint); }
`;

function ensureStyle() {
  if (document.getElementById("ctxMenuStyle")) return;
  const s = document.createElement("style");
  s.id = "ctxMenuStyle";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

export function closeMenu() {
  if (!host) return;
  host.remove();
  host = null;
  const cb = onClose; onClose = null;
  if (cb) cb();
}

/** True while a menu is open — key handlers use this to stay out of the way. */
export const menuOpen = () => !!host;

/**
 * Open a menu at the event's position. Returns nothing; the item's `run` does the work.
 * `items` entries: "-" separator · {cap} caption · {label, hint?, run, disabled?, checked?,
 * primary?, danger?}.
 */
export function showMenu(ev, items, opts = {}) {
  ensureStyle();
  closeMenu();
  const usable = items.filter((it) => it === "-" || (it && (it.cap || it.run)));
  if (!usable.length) return;

  host = document.createElement("div");
  host.id = "ctxMenu";
  host.setAttribute("role", "menu");

  const rows = [];
  for (const it of usable) {
    if (it === "-") { const d = document.createElement("div"); d.className = "sep"; host.appendChild(d); continue; }
    if (it.cap) { const d = document.createElement("div"); d.className = "cap"; d.textContent = it.cap; host.appendChild(d); continue; }
    const d = document.createElement("div");
    d.className = "mi" + (it.disabled ? " dis" : "") + (it.primary ? " primary" : "") + (it.danger ? " danger" : "");
    d.setAttribute("role", "menuitem");
    d.innerHTML = `<span class="tick">${it.checked ? "✓" : ""}</span><span class="lbl"></span>` +
      (it.hint ? `<span class="hint"></span>` : "");
    d.querySelector(".lbl").textContent = it.label;
    if (it.hint) d.querySelector(".hint").textContent = it.hint;
    if (!it.disabled) {
      rows.push(d);
      d.onclick = () => { closeMenu(); it.run(); };
    }
    host.appendChild(d);
  }
  document.body.appendChild(host);

  // Place at the cursor, then pull back inside the viewport rather than letting the menu run off
  // the edge — a right-click near the bottom of the window is the common case, not the odd one.
  const pad = 6, r = host.getBoundingClientRect();
  let x = ev.clientX, y = ev.clientY;
  if (x + r.width + pad > innerWidth) x = Math.max(pad, innerWidth - r.width - pad);
  if (y + r.height + pad > innerHeight) y = Math.max(pad, innerHeight - r.height - pad);
  host.style.left = x + "px";
  host.style.top = y + "px";

  onClose = opts.onClose || null;

  // Keyboard: arrows move, Enter runs, Esc closes. Captured so the app's own shortcut handler
  // never sees these while a menu is up.
  let idx = -1;
  const mark = () => rows.forEach((r2, i) => r2.classList.toggle("on", i === idx));
  const onKey = (e) => {
    if (!host) return;
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeMenu(); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); e.stopPropagation();
      if (!rows.length) return;
      idx = (idx + (e.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length;
      mark();
      return;
    }
    if (e.key === "Enter" && idx >= 0) { e.preventDefault(); e.stopPropagation(); rows[idx].click(); }
  };
  addEventListener("keydown", onKey, true);

  const dismiss = (e) => { if (host && !host.contains(e.target)) closeMenu(); };
  // pointerdown, not click: a drag that starts outside should dismiss immediately.
  addEventListener("pointerdown", dismiss, true);
  addEventListener("blur", closeMenu);
  addEventListener("resize", closeMenu);

  const prev = onClose;
  onClose = () => {
    removeEventListener("keydown", onKey, true);
    removeEventListener("pointerdown", dismiss, true);
    removeEventListener("blur", closeMenu);
    removeEventListener("resize", closeMenu);
    if (prev) prev();
  };
}
