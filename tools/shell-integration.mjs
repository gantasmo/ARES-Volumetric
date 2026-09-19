/**
 * Windows shell integration — "Convert to .ares" on the right-click menu.
 *
 * Everything lives under HKEY_CURRENT_USER\Software\Classes, so registering needs no
 * administrator and touches nothing outside this user's profile. Unregistering removes exactly
 * the keys this file wrote and nothing else.
 *
 * Written as a .reg file and applied with `reg import` rather than a series of `reg add` calls:
 * the command value contains both quotes and backslashes, and reg.exe re-parses its own command
 * line, so building it as argv is a quoting minefield. A .reg file has one escaping rule
 * (double the backslashes, backslash the quotes) and is auditable before it is applied.
 *
 * The FOLDER verb is the important one. A volumetric capture is a directory of per-frame meshes
 * and atlases, not a single file, so "right-click the folder → Convert to .ares" is the gesture
 * that matches the data. The per-extension verbs are for the single-file cases the app can
 * genuinely do something with.
 *
 * Windows 11 note: HKCU shell verbs appear in the classic context menu — the one behind
 * "Show more options" (Shift+F10 opens it directly). Getting into the short modern menu requires
 * a packaged IExplorerCommand handler, which would mean shipping an MSIX. Said plainly here
 * because a verb the user cannot find is not integration.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const KEY_NAME = "ARES.Convert";
const HKCU_CLASSES = "HKEY_CURRENT_USER\\Software\\Classes";
const VERB_FILE = "Convert to .ares";
const VERB_DIR = "Convert folder to .ares";

/** Formats the app can actually act on. A verb offered over something we cannot open is a lie,
 *  so this list is the app's real reach, not a wish list. Folders cover frame sequences. */
export const EXTS = [
  ".ares",                                            // open / inspect
  ".4ds",                                             // 4DViews container (needs the licensed codec)
  ".obj", ".ply", ".stl", ".glb", ".gltf",            // meshes — treated as one frame of a sequence
  ".spz", ".splat", ".sog",                           // gaussian splats
  ".abc", ".usd", ".usdz", ".usda", ".usdc", ".fbx",  // scene/geometry caches
  ".drc",                                             // Draco
  ".e57", ".pcd", ".las", ".laz", ".xyz", ".pts",     // point clouds
];

const isWindows = () => process.platform === "win32";

function run(cmd, args) {
  return new Promise((done) => {
    const p = spawn(cmd, args, { windowsHide: true });
    let out = "";
    p.stdout?.on("data", (d) => { out += d; });
    p.stderr?.on("data", (d) => { out += d; });
    p.on("error", () => done({ code: 1, out: "spawn failed" }));
    p.on("close", (code) => done({ code: code ?? 1, out }));
  });
}

/** .reg string escaping: backslashes double, quotes get a backslash. */
const regStr = (s) => s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

/** The command each verb runs. `%1` is the clicked file, `%V` the folder for a background click.
 *  wscript.exe keeps it windowless: ARES.vbs forwards to the launcher, which starts (or reuses)
 *  the dev server and opens the app pointed at the path. */
function commandFor(root, arg) {
  const vbs = join(root, "ARES.vbs");
  return `wscript.exe "${vbs}" app --open "${arg}"`;
}

/** Build the .reg body. `remove` produces the same key list with `-` prefixes, so register and
 *  unregister can never drift out of step. */
export function regScript(root, { allFiles = false, remove = false } = {}) {
  const lines = ["Windows Registry Editor Version 5.00", ""];
  const key = (path) => `[${remove ? "-" : ""}${path}]`;
  const verb = (base, label, arg) => {
    lines.push(key(`${base}\\shell\\${KEY_NAME}`));
    if (!remove) {
      lines.push(`@="${regStr(label)}"`);
      // Icon: the launcher has none of its own, so borrow the shell's generic 3D icon slot by
      // leaving it unset rather than pointing at a file that may not exist.
      lines.push(`"Position"="Bottom"`);
    }
    lines.push("");
    lines.push(key(`${base}\\shell\\${KEY_NAME}\\command`));
    if (!remove) lines.push(`@="${regStr(commandFor(root, arg))}"`);
    lines.push("");
  };

  for (const ext of EXTS) verb(`${HKCU_CLASSES}\\SystemFileAssociations\\${ext}`, VERB_FILE, "%1");
  verb(`${HKCU_CLASSES}\\Directory`, VERB_DIR, "%1");
  verb(`${HKCU_CLASSES}\\Directory\\Background`, VERB_DIR, "%V");
  // Opt-in only. A verb on every file in the system is the kind of thing installers get hated
  // for, so it is a separate choice, and the app says so when it cannot read what it was given.
  if (allFiles) verb(`${HKCU_CLASSES}\\*`, VERB_FILE, "%1");

  return lines.join("\r\n");
}

/** Is the integration currently registered, and for what? */
export async function shellStatus(root) {
  if (!isWindows()) return { supported: false, registered: false, why: "Windows only" };
  const q = async (path) => (await run("reg", ["query", path])).code === 0;
  const dir = await q(`HKCU\\Software\\Classes\\Directory\\shell\\${KEY_NAME}\\command`);
  const file = await q(`HKCU\\Software\\Classes\\SystemFileAssociations\\.ares\\shell\\${KEY_NAME}\\command`);
  const allFiles = await q(`HKCU\\Software\\Classes\\*\\shell\\${KEY_NAME}\\command`);
  return {
    supported: true,
    registered: dir || file,
    folders: dir,
    files: file,
    allFiles,
    exts: EXTS.length,
    launcherPresent: existsSync(join(root, "ARES.vbs")),
    note: "Windows 11 shows these under “Show more options” (Shift+F10).",
  };
}

async function applyReg(body, onLine = () => {}) {
  const dir = join(tmpdir(), "ares-shell");
  await mkdir(dir, { recursive: true });
  const file = join(dir, `ares-${Date.now()}.reg`);
  // reg.exe wants UTF-16LE for a "Version 5.00" file.
  await writeFile(file, "﻿" + body, "utf16le");
  const r = await run("reg", ["import", file]);
  try { await unlink(file); } catch { /* leave it in temp */ }
  if (r.out.trim()) onLine(r.out.trim().split(/\r?\n/).slice(-2).join(" "));
  return r.code === 0;
}

export async function shellRegister(root, { allFiles = false } = {}, onLine = () => {}) {
  if (!isWindows()) return { ok: false, error: "shell integration is Windows-only" };
  if (!existsSync(join(root, "ARES.vbs"))) return { ok: false, error: "ARES.vbs not found at the repo root" };
  onLine(`registering ${EXTS.length} file types, folders, and folder backgrounds…`);
  const ok = await applyReg(regScript(root, { allFiles }), onLine);
  if (!ok) return { ok: false, error: "reg import failed" };
  onLine("registered: right-click a capture folder and pick “Convert folder to .ares”");
  return { ok: true, ...(await shellStatus(root)) };
}

export async function shellUnregister(root, onLine = () => {}) {
  if (!isWindows()) return { ok: false, error: "shell integration is Windows-only" };
  onLine("removing the context-menu entries…");
  // allFiles: true so the `*` key is removed too, whether or not it was ever added — deleting a
  // key that is not there is not an error, and leaving it behind would be.
  const ok = await applyReg(regScript(root, { allFiles: true, remove: true }), onLine);
  if (!ok) return { ok: false, error: "reg import failed" };
  onLine("removed");
  return { ok: true, ...(await shellStatus(root)) };
}
