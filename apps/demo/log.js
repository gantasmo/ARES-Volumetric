// GLOBAL ACTIVITY LOG — client side.
//
// The log is an audit trail, not a summary. It records what ACTUALLY happened — timestamped,
// server-persisted (apps/demo/.ares-activity.jsonl via /log), and readable after the fact.
// Three producers write to the same sink: the app (here), the pipeline (bake stages, forwarded
// from the /diagnostics SSE).
//
// Design notes:
// - SERVER-side persistence, not localStorage: it must survive tab switches, reloads, a different
//   browser, and a crash. localStorage would also silently cap out at ~5MB.
// - Console + errors are HOOKED, not replaced: the original console still fires, so devtools is
//   unchanged and nothing that logs today has to be rewritten to appear here.
// - Batched POSTs (250ms) so a chatty frame can't turn into 60 requests/sec.
// - Never throws. A logger that breaks the app it observes is worse than no logger.
const BUF = [];
let flushTimer = null;
const MAX_MEM = 3000;        // in-memory ring for instant panel render; the server holds the full history
const mem = [];
const subs = new Set();

function emit(entry) {
  mem.push(entry);
  if (mem.length > MAX_MEM) mem.shift();
  for (const fn of subs) { try { fn(entry); } catch { /* a bad subscriber must not kill logging */ } }
  BUF.push(entry);
  if (!flushTimer) flushTimer = setTimeout(flush, 250);
}

async function flush() {
  flushTimer = null;
  if (!BUF.length) return;
  const batch = BUF.splice(0, BUF.length);
  try {
    await fetch("/log", { method: "POST", body: JSON.stringify(batch), keepalive: true });
  } catch { /* offline: the in-memory ring still shows it; dropping is better than blocking the app */ }
}

/** Log a line. src: app|bake|claude|error. lvl: info|warn|error|act ("act" = a state change). */
export function alog(msg, { src = "app", lvl = "info", data } = {}) {
  emit({ t: Date.now(), src, lvl, msg: String(msg), data });
}
/** A state-changing action — the lines that matter most when auditing "what did it just do". */
export const aact = (msg, data) => alog(msg, { lvl: "act", data });

export const logSubscribe = (fn) => { subs.add(fn); return () => subs.delete(fn); };
export const logMemory = () => mem.slice();

/** Pull the server's history into this tab (a fresh tab starts with an empty ring). */
export async function logRestore(tail = 800) {
  try {
    const past = await fetch(`/log?tail=${tail}`).then((r) => r.json());
    if (Array.isArray(past) && past.length) {
      mem.length = 0;
      for (const e of past.slice(-MAX_MEM)) mem.push(e);
      for (const fn of subs) { try { fn(null); } catch { /* */ } }   // null = "rerender everything"
    }
  } catch { /* server down: keep whatever this tab has */ }
}

export async function logClear() {
  mem.length = 0; BUF.length = 0;
  try { await fetch("/log", { method: "DELETE" }); } catch { /* */ }
  for (const fn of subs) { try { fn(null); } catch { /* */ } }
}

let installed = false;
/** Hook console + global errors. Idempotent. Call once, early. */
export function installLogCapture() {
  if (installed) return;
  installed = true;

  for (const level of ["log", "info", "warn", "error"]) {
    const orig = console[level].bind(console);
    console[level] = (...args) => {
      orig(...args);                                    // devtools behaviour is unchanged
      try {
        const msg = args.map((a) =>
          typeof a === "string" ? a
            : a instanceof Error ? `${a.name}: ${a.message}`
              : (() => { try { return JSON.stringify(a); } catch { return String(a); } })()
        ).join(" ");
        if (msg.startsWith("[log]")) return;            // don't log the logger
        emit({ t: Date.now(), src: level === "error" ? "error" : "app", lvl: level === "log" ? "info" : level, msg: msg.slice(0, 2000) });
      } catch { /* never let logging break a console call */ }
    };
  }

  // Uncaught errors + rejected promises: the two things that silently eat a session.
  window.addEventListener("error", (e) => {
    emit({ t: Date.now(), src: "error", lvl: "error", msg: `uncaught: ${e.message}`, data: `${e.filename}:${e.lineno}:${e.colno}` });
  });
  window.addEventListener("unhandledrejection", (e) => {
    const r = e.reason;
    emit({ t: Date.now(), src: "error", lvl: "error", msg: `unhandled rejection: ${r && r.message ? r.message : String(r)}` });
  });

  // App-wide events already broadcast by the app — free signal, no call sites to touch.
  for (const ev of ["ares:bake-done", "ares:library-changed", "ares:showcase-changed"]) {
    window.addEventListener(ev, (e) => {
      emit({ t: Date.now(), src: "app", lvl: "act", msg: ev, data: e.detail ? JSON.stringify(e.detail).slice(0, 500) : undefined });
    });
  }

  window.addEventListener("beforeunload", () => { if (BUF.length) flush(); });
  emit({ t: Date.now(), src: "app", lvl: "info", msg: `session start — ${location.pathname}${location.search}` });
}

// Global handle so anything (including the console, and me) can write a line without an import.
window.aresLog = { alog, aact, logRestore, logClear, logMemory, logSubscribe };
