#!/usr/bin/env node
/**
 * Screenshot a clip in the real player, headless, over the DevTools protocol.
 *
 *   node tools/relief-shot.mjs <server-origin> <out.png> "src=/apps/demo/x.ares&frame=40&az=0.6&el=0.1"
 *   node tools/relief-shot.mjs <server-origin> <out.json> "src=...&from=100&seconds=6" playback-trace.html
 *
 * Opens apps/demo/verify/<page> (relief-shot.html by default) with the given query in headless
 * Chrome, waits for the page to title itself "ready", and writes the canvas as a PNG, or, for an
 * out path ending in .json, the page's `window.__result`. Exits 1 with the page's own error when it
 * titles itself "error: ...". Needs a running dev server (tools/serve.mjs) for the import map and
 * the clip; needs no npm dependency (Node 22 has WebSocket and fetch).
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [origin, outPng, query = "", page = "relief-shot.html"] = process.argv.slice(2);
if (!origin || !outPng) { console.error("usage: relief-shot.mjs <origin> <out.png|out.json> [query] [page]"); process.exit(2); }

const CHROME = process.env.CHROME || [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium",
].find((p) => existsSync(p));
if (!CHROME) { console.error("no Chrome or Edge found"); process.exit(2); }

const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), "ares-shot-"));
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--window-size=1280,720", "--hide-scrollbars",
  "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--disable-accelerated-video-decode",
  "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 1;
try {
  let target = null;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch { /* not up yet */ }
  }
  if (!target) throw new Error("Chrome did not open a debugging port");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("devtools socket failed")); });
  let id = 0;
  const pending = new Map();
  const logs = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
    else if (m.method === "Runtime.exceptionThrown") logs.push("exception: " + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
  };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${origin}/apps/demo/verify/${page}?${query}` });
  let title = "";
  for (let i = 0; i < 600; i++) {
    await sleep(250);
    title = (await send("Runtime.evaluate", { expression: "document.title" })).result?.result?.value ?? "";
    if (title === "ready" || title.startsWith("error")) break;
  }
  if (title !== "ready") throw new Error(`page did not become ready (${title || "no title"})\n${logs.join("\n")}`);
  if (/\.json$/i.test(outPng)) {
    const res = await send("Runtime.evaluate", { expression: "JSON.stringify(window.__result ?? null)", returnByValue: true });
    writeFileSync(outPng, res.result?.result?.value ?? "null");
  } else {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(outPng, Buffer.from(shot.result.data, "base64"));
  }
  const stats = (await send("Runtime.evaluate", { expression: "JSON.stringify({texApplied: window.player?.texApplied, texIdx: window.player?.texAppliedIdx, label: window.player?.textureLabel, tvErr: String(window.player?.textureVideo?.lastError ?? \"\"), tvState: window.player?.textureVideo?.decoder?.state, fed: window.player?.textureVideo?.fedNext, ready: window.player?.textureVideo?.ready?.size, gpu: !!navigator.gpu})" })).result?.result?.value;
  console.log(`wrote ${outPng} ${stats ?? ""}`);
  if (logs.length) console.log(logs.slice(-6).join("\n"));
  ws.close();
  code = 0;
} catch (e) {
  console.error(String(e?.message || e));
} finally {
  try { chrome.kill(); } catch { /* gone */ }
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold it */ }
}
process.exit(code);
