#!/usr/bin/env node
/**
 * ARES launcher — the single entry point for the app.
 *
 * One implementation for every way of starting ARES, replacing the four root .vbs launchers
 * and their three near-identical PowerShell workers (2026-09-07 audit, "project hygiene":
 * "three PS1 launchers repeat the same discovery blocks"). Node runs everywhere the repo does,
 * so the launcher is Node; `ARES.vbs` -> `tools/launch.ps1` is only the Windows double-click
 * bootstrap (it finds or installs Node, then runs this file).
 *
 *   node tools/launch.mjs [app|probe|bench|sam] [options]   any OS
 *   npm start                                               same, mode app
 *   ARES.vbs [mode]                                         Windows, windowless double-click
 *
 * Modes
 *   app    (default) the demo app: Viewer | Compare | Inspect | Convert, plus the editor
 *   probe  Phase 0 capability probe (WebGPU adapters, WebCodecs HW decode, isolation)
 *   bench  runs the intra geometry bench (~1-2 min), then opens its report page
 *   sam    starts the local SAM segmentation service (Windows; the app also starts it itself)
 *
 * Options
 *   --port N      base port for the dev server (default 8137; serve.mjs walks up 10 if busy)
 *   --src clip    .ares under apps/demo to open in the app (default: first clip present)
 *   --detach      start the server in the background and exit (what the Windows path uses)
 *   --no-open     do not open a browser
 *   --no-build    skip the TypeScript build
 *   --no-install  skip npm install even when node_modules looks stale
 *   -h, --help
 *
 * Every step is logged to tools/launch.log — the only place the windowless Windows path can
 * report from. Exit code 0 means the app is up.
 */
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const LOG_FILE = join(ROOT, "tools", "launch.log");
const SERVE_JS = join(ROOT, "tools", "serve.mjs");
const CLI_JS = join(ROOT, "packages", "encoder", "dist", "cli.js");
const BENCH_JS = join(ROOT, "bench", "dist", "run.js");
const SAM_PS1 = join(ROOT, "tools", "sam-service", "run-sam-service.ps1");
const DEFAULT_PORT = 8137;
const PORT_SPAN = 11; // serve.mjs listen(port, 10): the base port plus ten walk-ups
const NPM = process.platform === "win32" ? "npm.cmd" : "npm";
const MODES = new Set(["app", "probe", "bench", "sam"]);
const PAGE = { app: "/apps/demo/", probe: "/apps/phase0-probe/", bench: "/bench/report/" };
/** Clips the app opens without being told which. .ares files are git-ignored, so a fresh clone
 *  has none of them and mode app synthesizes demo.ares instead. */
const CLIP_CANDIDATES = ["daniel-s0.ares", "daniel.ares", "demo.ares"];
const OPTIONS = ["--port", "--src", "--detach", "--no-open", "--no-build", "--no-install", "--help"];

let ownsServer = false; // true once this process started the server itself (foreground mode)

function log(line) {
  const s = String(line).replace(/\s+$/, "");
  if (!s) return;
  console.log(s);
  try { appendFileSync(LOG_FILE, `${new Date().toISOString()} ${s}\n`); } catch { /* the log is a convenience */ }
}

function fail(msg) {
  log(`FATAL: ${msg}`);
  process.exit(1);
}

/** Spawn and wait, mirroring output into the log. Resolves with the exit code (1 on spawn error). */
function run(cmd, args, opts = {}) {
  return new Promise((done) => {
    log(`$ ${cmd} ${args.join(" ")}`);
    const p = spawn(cmd, args, { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"], windowsHide: true, ...opts });
    p.stdout?.on("data", (d) => log(d));
    p.stderr?.on("data", (d) => log(d));
    p.on("error", (e) => { log(`ERROR: ${cmd}: ${e.message}`); done(1); });
    p.on("close", (code) => done(code ?? 1));
  });
}

// --- arguments ---------------------------------------------------------------
const HELP = `ARES launcher

  node tools/launch.mjs [app|probe|bench|sam] [options]

Modes
  app     (default) demo app: Viewer | Compare | Inspect | Convert
  probe   Phase 0 capability probe
  bench   run the intra geometry bench (~1-2 min), then open its report
  sam     start the local SAM segmentation service (Windows only)

Options
  --port N       base port for the dev server (default ${DEFAULT_PORT})
  --src clip     .ares under apps/demo to open (default: first clip present, else synth)
  --detach       start the server in the background and exit
  --no-open      do not open a browser
  --no-build     skip the TypeScript build
  --no-install   skip npm install even when node_modules looks stale
  -h, --help     this text

Log: tools/launch.log`;

function flagValue(argv, name) {
  const i = argv.indexOf(name);
  if (i < 0) return undefined;
  const v = argv[i + 1];
  // A value flag followed by another flag (or by nothing) is a usage error, not a value.
  if (v === undefined || v.startsWith("-")) fail(`${name}: expected a value`);
  return v;
}

function parseArgs(argv) {
  if (argv.includes("-h") || argv.includes("--help")) { console.log(HELP); process.exit(0); }
  for (const a of argv) {
    if (a.startsWith("-") && !OPTIONS.includes(a)) fail(`unknown option ${a} — run with --help`);
  }
  const port = flagValue(argv, "--port");
  const src = flagValue(argv, "--src");
  const taken = new Set([port, src].filter((v) => v !== undefined));
  const mode = argv.find((a) => !a.startsWith("-") && !taken.has(a)) ?? "app";
  if (!MODES.has(mode)) fail(`unknown mode ${JSON.stringify(mode)} — expected one of ${[...MODES].join(", ")}`);
  const portNum = port === undefined ? DEFAULT_PORT : Number(port);
  if (!Number.isInteger(portNum) || portNum < 1024 || portNum > 65535) {
    fail(`--port: expected an integer 1024-65535, got ${JSON.stringify(port)}`);
  }
  return {
    mode, src,
    port: portNum,
    detach: argv.includes("--detach"),
    open: !argv.includes("--no-open"),
    build: !argv.includes("--no-build"),
    install: !argv.includes("--no-install"),
  };
}

// --- steps -------------------------------------------------------------------
async function ensureDeps(enabled) {
  const stamp = join(ROOT, "node_modules", ".package-lock.json");
  const stale = !existsSync(stamp) || statSync(join(ROOT, "package.json")).mtimeMs > statSync(stamp).mtimeMs;
  if (!stale) { log("dependencies up to date"); return; }
  if (!enabled) { log("dependencies look stale; --no-install given, continuing"); return; }
  log("installing dependencies (npm install)...");
  const code = await run(NPM, ["install", "--no-fund", "--no-audit"], { shell: process.platform === "win32" });
  if (code !== 0) fail(`npm install failed (exit ${code}). See ${LOG_FILE}.`);
}

/** tsc directly when it is installed: one less wrapper process, and no npm script indirection. */
async function build(enabled, fatal) {
  if (!enabled) { log("build skipped (--no-build)"); return; }
  log("building (tsc -b)...");
  const tsc = join(ROOT, "node_modules", "typescript", "bin", "tsc");
  const code = existsSync(tsc)
    ? await run(process.execPath, [tsc, "-b"])
    : await run(NPM, ["run", "build"], { shell: process.platform === "win32" });
  if (code === 0) return;
  if (fatal) fail(`build failed (exit ${code}). See ${LOG_FILE}.`);
  log(`WARNING: build failed (exit ${code}) — this page does not need the packages; see ${LOG_FILE}`);
}

async function ensureClip(src) {
  if (src) return src; // named explicitly; the app reports it if the name is wrong
  const demoDir = join(ROOT, "apps", "demo");
  const present = CLIP_CANDIDATES.find((c) => existsSync(join(demoDir, c)));
  if (present) return present;
  log("no clip in apps/demo — synthesizing demo.ares...");
  if (!existsSync(CLI_JS)) fail("the encoder is not built, so no demo clip can be generated. Run without --no-build.");
  const code = await run(process.execPath, [CLI_JS, "synth", "-o", "apps/demo/demo.ares", "--shape", "object", "--frames", "60", "--fps", "30"]);
  if (code !== 0) fail(`demo clip generation failed (exit ${code}). See ${LOG_FILE}.`);
  return "demo.ares";
}

const samePath = (a, b) =>
  process.platform === "win32"
    ? resolvePath(a).toLowerCase() === resolvePath(b).toLowerCase()
    : resolvePath(a) === resolvePath(b);

/** A dev server for THIS checkout, or null. A server from another checkout on the port is
 *  skipped rather than reused, so the launcher never opens a different tree. */
async function findServer(basePort) {
  for (let p = basePort; p < basePort + PORT_SPAN; p++) {
    try {
      const r = await fetch(`http://127.0.0.1:${p}/__ares`, { signal: AbortSignal.timeout(600) });
      if (!r.ok) continue;
      const info = await r.json();
      if (info.server === "ares-dev" && samePath(info.root ?? "", ROOT)) return p;
    } catch { /* nothing of ours listening there */ }
  }
  return null;
}

async function ensureServer(basePort, detach) {
  const running = await findServer(basePort);
  if (running) { log(`reusing the dev server on port ${running}`); return running; }
  log("starting the COOP/COEP dev server...");
  const child = spawn(process.execPath, [SERVE_JS, String(basePort)], {
    cwd: ROOT,
    detached: detach,
    windowsHide: true,
    stdio: detach ? "ignore" : "inherit",
  });
  if (detach) {
    child.unref();
  } else {
    ownsServer = true;
    child.on("exit", (code) => { if (code) log(`the dev server exited (${code})`); process.exit(code ?? 0); });
    const stop = () => { try { child.kill(); } catch { /* already gone */ } };
    process.on("SIGINT", () => { stop(); process.exit(0); });
    process.on("SIGTERM", () => { stop(); process.exit(0); });
  }
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 250));
    const port = await findServer(basePort);
    if (port) { log(`server up on port ${port}`); return port; }
  }
  fail(`the dev server did not answer on ports ${basePort}-${basePort + PORT_SPAN - 1}. See ${LOG_FILE}.`);
}

/** The URLs built here contain no & or spaces, so cmd's start needs no extra quoting. */
function openBrowser(url) {
  const [cmd, args] =
    process.platform === "win32" ? ["cmd", ["/c", "start", "", url]]
    : process.platform === "darwin" ? ["open", [url]]
    : ["xdg-open", [url]];
  try { spawn(cmd, args, { detached: true, stdio: "ignore", windowsHide: true }).unref(); }
  catch (e) { log(`could not open a browser (${e.message}) — go to ${url}`); }
}

function startSamService() {
  if (process.platform !== "win32") fail("mode sam is Windows-only (the service runs from a local Python env); start it from the app's Edit panel instead.");
  if (!existsSync(SAM_PS1)) fail(`missing ${SAM_PS1}`);
  log("starting the SAM segmentation service (the first start loads weights, 10-20 s)...");
  spawn("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", SAM_PS1], {
    detached: true, stdio: "ignore", windowsHide: true,
  }).unref();
  log("SAM service starting on http://127.0.0.1:7263 — log: tools/sam-service/sam-service.log");
}

// --- main --------------------------------------------------------------------
const opts = parseArgs(process.argv.slice(2));
log(`--- ARES launcher: mode ${opts.mode}, node ${process.version}, ${ROOT}`);

if (opts.mode === "sam") {
  startSamService();
  process.exit(0);
}

await ensureDeps(opts.install);
// The probe is plain HTML served over COOP/COEP: it reports capabilities even when the packages
// do not compile, which is exactly the situation someone reaches for it in.
await build(opts.build, opts.mode !== "probe");

let query = "";
if (opts.mode === "app") query = `?src=${await ensureClip(opts.src)}`;

if (opts.mode === "bench") {
  if (!existsSync(BENCH_JS)) fail("the bench is not built. Run without --no-build.");
  log("running the intra bench (a minute or two)...");
  const code = await run(process.execPath, [BENCH_JS]);
  if (code !== 0) fail(`bench run failed (exit ${code}). See ${LOG_FILE}.`);
}

const port = await ensureServer(opts.port, opts.detach);
const url = `http://127.0.0.1:${port}${PAGE[opts.mode]}${query}`;
if (opts.open) { openBrowser(url); log(`opened ${url}`); }
else log(`ready at ${url}`);

if (opts.detach || !ownsServer) process.exit(0); // the server outlives us; foreground waits on it
log("serving — Ctrl-C to stop");
