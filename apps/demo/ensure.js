/**
 * Component ensure, client side.
 *
 * The rule (owner's, verbatim intent): a job never stops to tell the person that something is
 * absent. SSE work routes install what they lack inside their own stream (tools/serve.mjs
 * ensureComponents), so their clients only have to print the "[setup] …" log lines they already
 * print. This module covers the three cases a stream cannot:
 *
 *   installStream(ids)      a plain JSON route answered { needs: [ids] }: run /install for those
 *                           ids, relay its progress, and let the caller repeat the request
 *   fetchJsonEnsuring(url)  the whole loop for a JSON GET: fetch, install `needs`, collect
 *                           `needsFile` with the native file dialog, fetch again
 *   accessPrompt(host, …)   a gated Hugging Face repository: the one input only the person has.
 *                           Token field + licence page + Resume, inline where the job failed;
 *                           Resume re-issues the same job
 */

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Parse the payload of a named SSE `error` event. Null for a bare connection error. */
export function sseErrorData(e) {
  try { return e && e.data ? JSON.parse(e.data) : null; } catch { return null; }
}

/**
 * Run /install?ids=… to completion.
 *   onLog(line)                 every progress line
 *   onStep({label,index,total}) component boundaries
 * Resolves { ok:true } or { ok:false, message, gated, needsToken, url, label, component }.
 */
export function installStream(ids, { onLog = () => {}, onStep = () => {} } = {}) {
  return new Promise((done) => {
    const es = new EventSource("/install?ids=" + encodeURIComponent(ids.join(",")));
    es.addEventListener("log", (e) => { try { onLog(JSON.parse(e.data)); } catch { /* ignore */ } });
    es.addEventListener("step", (e) => { try { onStep(JSON.parse(e.data)); } catch { /* ignore */ } });
    es.addEventListener("done", () => { es.close(); done({ ok: true }); });
    es.addEventListener("error", (e) => {
      es.close();
      const d = sseErrorData(e);
      done({ ok: false, message: "install stream closed", ...(d || {}) });
    });
  });
}

/** Collect a file only the person has (a licensed SDK binary) with the native dialog and hand its
 *  path to the server route that copies it into place. Resolves true when it landed. */
export async function collectFile(spec, { onLog = () => {} } = {}) {
  onLog(`${spec.label}: file dialog open`);
  let picked = null;
  try { picked = await fetch("/pick?type=file&for=" + encodeURIComponent(spec.id) + "&filter=" + encodeURIComponent(spec.filter || "")).then((r) => r.json()); }
  catch { return false; }
  if (!picked || !picked.path) { onLog(`${spec.label}: no file chosen`); return false; }
  let r = null;
  try { r = await fetch(spec.route, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ path: picked.path }) }).then((x) => x.json()); }
  catch { r = null; }
  onLog(r && r.ok ? `${spec.label}: stored` : `${spec.label}: ${(r && r.error) || "copy failed"}`);
  return !!(r && r.ok);
}

/**
 * GET a JSON route, satisfying whatever it says it lacks, then GET it again.
 * Returns the final JSON. `gate` is set on the result when a gated repository stopped the install,
 * so the caller can raise accessPrompt() with its own resume.
 */
export async function fetchJsonEnsuring(url, { onLog = () => {}, onStep = () => {}, rounds = 3 } = {}) {
  let j = null;
  for (let i = 0; i < rounds; i++) {
    j = await fetch(url).then((r) => r.json());
    const needs = j && Array.isArray(j.needs) ? j.needs : [];
    if (!needs.length && !(j && j.needsFile)) return j;
    if (needs.length) {
      onLog(`installing: ${needs.join(", ")}`);
      const r = await installStream(needs, { onLog, onStep });
      if (!r.ok) return { ...j, error: r.message || j.error, gate: r.gated ? r : null };
    }
    if (j.needsFile && !(await collectFile(j.needsFile, { onLog }))) return j;
  }
  return j;
}

/**
 * Inline access prompt for a gated Hugging Face repository.
 *   host    element the prompt renders into (replaces its content)
 *   info    { label, url, needsToken }  from the job's error event
 *   resume  called once access is in place; re-issue the same job from it
 */
export function accessPrompt(host, info, resume) {
  const needsToken = !!info.needsToken;
  host.innerHTML = `
    <div class="card" style="margin-top:8px;border-color:var(--warn)">
      <div class="cap">Gated repository${info.label ? ": " + esc(info.label) : ""}</div>
      <div class="note2" style="margin:4px 0 6px">${needsToken ? "Hugging Face access token required." : "Licence acceptance required for the stored token's account."}</div>
      <div class="row" style="flex-wrap:wrap;gap:6px">
        ${needsToken ? `<input class="inp" type="password" autocomplete="off" spellcheck="false" placeholder="hf_…" data-ap="token" style="flex:1;min-width:180px">` : ""}
        ${info.url ? `<a class="u gatebtn" href="${esc(info.url)}" target="_blank" rel="noopener">Licence page ↗</a>` : ""}
        ${needsToken ? `<a class="u gatebtn" href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Token page ↗</a>` : ""}
        <button class="u primary" data-ap="go">Resume</button>
      </div>
      <div class="note2" data-ap="msg" style="margin-top:4px"></div>
    </div>`;
  const q = (k) => host.querySelector(`[data-ap="${k}"]`);
  const go = q("go"), msg = q("msg"), input = q("token");
  const submit = async () => {
    go.disabled = true; msg.style.color = "";
    if (input) {
      const token = input.value.trim();
      if (!token) { go.disabled = false; input.focus(); return; }
      msg.textContent = "validating token…";
      let r;
      try { r = await fetch("/hf-token", { method: "POST", body: JSON.stringify({ token }) }).then((x) => x.json()); }
      catch { r = { ok: false, error: "dev server not reachable" }; }
      input.value = "";                                   // never leave the token in the DOM
      if (!r.ok) { msg.style.color = "var(--bad)"; msg.textContent = r.error || "token rejected"; go.disabled = false; return; }
    }
    host.innerHTML = "";
    resume();
  };
  go.onclick = submit;
  if (input) input.onkeydown = (e) => { if (e.key === "Enter") submit(); };
}
