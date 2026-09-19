/**
 * Build, install and remove the ARES sparse MSIX package — the Windows 11 SHORT context menu.
 *
 *   node tools/msix/build.mjs status      what is present and what is registered
 *   node tools/msix/build.mjs build       compile the DLL + stub, then pack the .msix
 *   node tools/msix/build.mjs install     build if needed, then register it for this user
 *   node tools/msix/build.mjs uninstall   remove the package
 *
 * Three things make this reachable without ceremony:
 *
 *  - SPARSE + SIGNED. The DLL and the launcher stay in the repo and are referenced through
 *    -ExternalLocation, so rebuilding the handler needs no repack and no reinstall.
 *    Unsigned was tried first and does not work for a package like this — measured on Windows 11
 *    26200: sparse + unsigned is 0x80073D2C, and packaged + unsigned is 0x80080204 even with the
 *    unsigned-namespace OID taken verbatim from Microsoft's docs.
 *  - SELF-SIGNED, minted here into the current user's store. Installing it needs the certificate
 *    trusted in LocalMachine, which is the ONE elevated command in the whole flow; install()
 *    detects that state and hands back the exact command rather than failing obscurely.
 *  - The repo root is compiled into the binaries (generated AresRoot.h), so MOVING THE CHECKOUT
 *    MEANS REBUILDING — install() checks for that.
 *  - DISCOVERED, NOT CONFIGURED. MSVC, the Windows SDK, makeappx and signtool are located by
 *    probing the standard install roots, so no Developer Command Prompt and no environment setup.
 */
import { spawn } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(dirname(HERE));
const OUT = join(HERE, "ARES.msix");
const PKG_NAME = "ARES.Volumetric";
export const CLSID = "7107EFE2-8605-4F5D-8757-7472171D3327";   // must match AppxManifest + AresShell.cpp

const isWindows = () => process.platform === "win32";

function sh(cmd, args, opts = {}) {
  return new Promise((done) => {
    const p = spawn(cmd, args, { windowsHide: true, ...opts });
    let out = "";
    p.stdout?.on("data", (d) => { out += d; });
    p.stderr?.on("data", (d) => { out += d; });
    p.on("error", (e) => done({ code: 1, out: out + "\n" + e.message }));
    p.on("close", (code) => done({ code: code ?? 1, out }));
  });
}

const ps = (script) => sh("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]);

/** Run a batch script from a temp .cmd file.
 *  `cmd /c "…"` strips the outermost quotes when the string holds several quoted paths, and every
 *  compile step here is `"<vcvars>" && cl …` — two quoted paths minimum. A file has no such rule. */
async function bat(lines, label) {
  const file = join(tmpdir(), `ares-msix-${label}-${process.pid}.cmd`);
  await writeFile(file, ["@echo off", ...lines].join("\r\n"), "utf8");
  const r = await sh("cmd.exe", ["/c", file]);
  try { await rm(file, { force: true }); } catch { /* temp file */ }
  return r;
}

/** Newest subdirectory of `base` whose name looks like a version, or null. */
function newestVersioned(base, pick = (p) => p) {
  if (!existsSync(base)) return null;
  const dirs = readdirSync(base).filter((d) => /^\d+(\.\d+)+$/.test(d)).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }));
  for (let i = dirs.length - 1; i >= 0; i--) {
    const cand = pick(join(base, dirs[i]));
    if (cand && existsSync(cand)) return cand;
  }
  return null;
}

/** A Windows SDK tool (makeappx.exe, signtool.exe) from any installed kit. */
export function findSdkTool(exe) {
  for (const kit of ["C:\\Program Files (x86)\\Windows Kits\\10\\bin", "C:\\Program Files\\Windows Kits\\10\\bin"]) {
    const hit = newestVersioned(kit, (d) => join(d, "x64", exe));
    if (hit) return hit;
  }
  return null;
}
export const findMakeAppx = () => findSdkTool("makeappx.exe");
export const findSignTool = () => findSdkTool("signtool.exe");

const CERT_SUBJECT = "CN=ARES Volumetric Dev";   // must equal AppxManifest's Publisher
export const CER_PATH = join(HERE, "ARES-dev.cer");

/** Find or mint the self-signed signing certificate in the CURRENT USER's store (no admin), and
 *  export its public half so the trust step has something to import. */
async function ensureCert(onLine) {
  const r = await ps(
    `$s='${CERT_SUBJECT}'; ` +
    `$c = Get-ChildItem Cert:\\CurrentUser\\My | Where-Object { $_.Subject -eq $s -and $_.NotAfter -gt (Get-Date) } | Select-Object -First 1; ` +
    `if (-not $c) { $c = New-SelfSignedCertificate -Type Custom -Subject $s -KeyUsage DigitalSignature ` +
    `-FriendlyName 'ARES MSIX signing' -CertStoreLocation 'Cert:\\CurrentUser\\My' ` +
    `-NotAfter (Get-Date).AddYears(3) ` +
    `-TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3','2.5.29.19={text}Subject Type:End Entity') }; ` +
    `Export-Certificate -Cert $c -FilePath '${CER_PATH}' -Force | Out-Null; ` +
    `$c.Thumbprint`);
  const thumb = ((r.out || "").trim().split(/\r?\n/).filter((l) => /^[0-9A-F]{40}$/i.test(l.trim()))[0] || "").trim();
  if (!thumb) return { ok: false, error: "could not create a signing certificate", log: (r.out || "").slice(-800) };
  onLine(`signing certificate ${thumb.slice(0, 12)}…`);
  return { ok: true, thumb };
}

/** Is that certificate trusted machine-wide? Deployment checks the machine stores, not the
 *  user's — CurrentUser\\Root is not enough, verified: it still fails with 0x800B0109. */
async function certTrusted() {
  const r = await ps(
    `$t = (Get-ChildItem Cert:\\LocalMachine\\TrustedPeople, Cert:\\LocalMachine\\Root -ErrorAction SilentlyContinue | ` +
    `Where-Object { $_.Subject -eq '${CERT_SUBJECT}' }); if ($t) { 'YES' } else { 'NO' }`);
  return /YES/.test(r.out || "");
}

/** The single elevated command the whole feature needs. */
export const trustCommand = () =>
  `powershell -Command "Import-Certificate -FilePath '${CER_PATH}' -CertStoreLocation Cert:\\LocalMachine\\TrustedPeople"`;

/** vcvars64.bat from any Visual Studio edition, newest first. */
export function findVcvars() {
  const roots = ["C:\\Program Files\\Microsoft Visual Studio", "C:\\Program Files (x86)\\Microsoft Visual Studio"];
  const editions = ["BuildTools", "Community", "Professional", "Enterprise"];
  const found = [];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    for (const year of readdirSync(root)) {
      for (const ed of editions) {
        const p = join(root, year, ed, "VC", "Auxiliary", "Build", "vcvars64.bat");
        if (existsSync(p)) found.push({ year, p });
      }
    }
  }
  found.sort((a, b) => b.year.localeCompare(a.year, undefined, { numeric: true }));
  return found.length ? found[0].p : null;
}

export function toolchain() {
  return {
    platform: process.platform,
    vcvars: isWindows() ? findVcvars() : null,
    makeappx: isWindows() ? findMakeAppx() : null,
    dll: existsSync(join(HERE, "AresShell.dll")) ? join(HERE, "AresShell.dll") : null,
    exe: existsSync(join(HERE, "Ares.exe")) ? join(HERE, "Ares.exe") : null,
    msix: existsSync(OUT) ? OUT : null,
  };
}

/** Is the package registered for this user, and does it point at THIS checkout? */
export async function msixStatus() {
  if (!isWindows()) return { supported: false, installed: false, why: "Windows only" };
  const t = toolchain();
  const r = await ps(
    `$p = Get-AppxPackage -Name '${PKG_NAME}' | Select-Object -First 1; ` +
    `if ($p) { "$($p.PackageFullName)|$($p.InstallLocation)" } else { "" }`);
  const line = (r.out || "").trim().split(/\r?\n/).filter(Boolean).pop() || "";
  const [fullName, installLoc] = line.split("|");
  return {
    supported: true,
    installed: !!fullName,
    packageFullName: fullName || null,
    // This is the package's own staging folder under WindowsApps, NOT the external location:
    // a sparse package holds only its manifest and tiles there. The checkout it actually drives
    // is the one compiled into the binaries, checked against ROOT before every install.
    installLocation: installLoc || null,
    buildable: !!(t.vcvars && t.makeappx),
    vcvars: t.vcvars, makeappx: t.makeappx,
    built: !!t.msix, dll: !!t.dll, exe: !!t.exe,
    devMode: await devModeOn(),
    // The one thing that needs elevation, surfaced so the UI can say so before anything runs.
    certTrusted: await certTrusted(),
    cerPath: existsSync(CER_PATH) ? CER_PATH : null,
    trustCommand: trustCommand(),
  };
}

async function devModeOn() {
  const r = await ps("(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock' " +
    "-ErrorAction SilentlyContinue).AllowDevelopmentWithoutDevLicense");
  return /1/.test((r.out || "").trim());
}

export async function build(onLine = () => {}) {
  if (!isWindows()) return { ok: false, error: "Windows only" };
  const vcvars = findVcvars();
  if (!vcvars) return { ok: false, error: "MSVC not found: component msvc-build-tools is not installed" };
  const makeappx = findMakeAppx();
  if (!makeappx) return { ok: false, error: "makeappx.exe not found, the Windows SDK is missing" };

  // The binaries ship inside the package and run from WindowsApps, so the checkout's location has
  // to be compiled in. A generated header keeps the path out of the command line, where the
  // backslashes and quotes would need escaping twice.
  await writeFile(join(HERE, "AresRoot.h"),
    "// Generated by tools/msix/build.mjs: do not edit. Rebuild after moving the checkout.\r\n" +
    "#pragma once\r\n" +
    `#define ARES_ROOT L"${ROOT.replace(/\\/g, "\\\\")}"\r\n`, "utf8");

  // vcvars sets ~40 environment variables and only applies to the shell it ran in, so it has to
  // run in the same script as the compiler. Both binaries build in one pass.
  onLine("compiling AresShell.dll and Ares.exe…");
  const r0 = await bat([
    `cd /d "${HERE}"`,
    `call "${vcvars}" >nul`,
    `cl /nologo /std:c++17 /EHsc /O2 /W3 /LD /DUNICODE /D_UNICODE AresShell.cpp /link /DEF:AresShell.def /OUT:AresShell.dll`,
    `if errorlevel 1 exit /b 1`,
    `cl /nologo /std:c++17 /EHsc /O2 /W3 /DUNICODE /D_UNICODE AresLaunch.cpp /link /SUBSYSTEM:WINDOWS /OUT:Ares.exe`,
    `if errorlevel 1 exit /b 1`,
  ], "cc");
  if (r0.code !== 0 || !existsSync(join(HERE, "AresShell.dll")) || !existsSync(join(HERE, "Ares.exe"))) {
    return { ok: false, error: "compile failed", log: r0.out.slice(-1500) };
  }

  // Sparse: only the manifest and the tiles go inside. The DLL and the launcher are referenced
  // in place through -ExternalLocation, so rebuilding them needs no repack.
  onLine("packing ARES.msix…");
  const stage = join(HERE, ".stage");
  await rm(stage, { recursive: true, force: true });
  await mkdir(join(stage, "Assets"), { recursive: true });
  // Stamp the version rather than editing the tracked manifest, so a rebuild does not show up as
  // a source change. Build = days since 2026-01-01, revision = minutes since midnight: both stay
  // inside the 0-65535 each component allows, and the pair only ever increases.
  const now = new Date();
  const days = Math.floor((now - new Date(2026, 0, 1)) / 86400000);
  const mins = now.getHours() * 60 + now.getMinutes();
  const version = `1.0.${days}.${mins}`;
  const manifest = (await readFile(join(HERE, "AppxManifest.xml"), "utf8"))
    .replace(/(<Identity[^>]*?\sVersion=")[^"]*"/s, `$1${version}"`);
  if (!manifest.includes(`Version="${version}"`)) return { ok: false, error: "could not stamp the package version" };
  await writeFile(join(stage, "AppxManifest.xml"), manifest, "utf8");
  onLine(`version ${version}`);
  for (const a of ["Square44x44Logo.png", "Square150x150Logo.png", "StoreLogo.png"]) {
    await copyFile(join(HERE, "Assets", a), join(stage, "Assets", a));
  }

  await rm(OUT, { force: true });
  const r = await sh(makeappx, ["pack", "/d", stage, "/p", OUT, "/nv", "/o"]);
  await rm(stage, { recursive: true, force: true });
  if (r.code !== 0 || !existsSync(OUT)) return { ok: false, error: "makeappx failed", log: r.out.slice(-1200) };

  const signtool = findSignTool();
  if (!signtool) return { ok: false, error: "signtool.exe not found, the Windows SDK is missing" };
  const cert = await ensureCert(onLine);
  if (!cert.ok) return cert;
  onLine("signing…");
  // No /t timestamp: it needs the network, and the signature only has to outlive the cert on a
  // machine that already trusts it.
  const sg = await sh(signtool, ["sign", "/fd", "SHA256", "/sha1", cert.thumb, OUT]);
  if (sg.code !== 0) return { ok: false, error: "signing failed", log: sg.out.slice(-1200) };

  onLine(`built and signed ${OUT} (${(statSync(OUT).size / 1024).toFixed(0)} KB)`);
  return { ok: true, msix: OUT, thumb: cert.thumb };
}

export async function install(onLine = () => {}) {
  if (!isWindows()) return { ok: false, error: "Windows only" };
  if (!existsSync(OUT)) {
    const b = await build(onLine);
    if (!b.ok) return b;
  }
  if (!(await devModeOn())) {
    return { ok: false, error: "Developer Mode is off (Settings → System → For developers)" };
  }
  // The compiled-in root must still be this checkout, or the verb would drive a directory that
  // has moved. Cheap to check, and the failure is otherwise silent and confusing.
  const baked = existsSync(join(HERE, "AresRoot.h"))
    ? (/#define ARES_ROOT L"(.*)"/.exec(await readFile(join(HERE, "AresRoot.h"), "utf8")) || [])[1]
    : null;
  if (baked && baked.replace(/\\\\/g, "\\") !== ROOT) {
    onLine("the checkout moved since the package was built: rebuilding");
    const b = await build(onLine);
    if (!b.ok) return b;
  }

  // Deployment validates against the MACHINE trust stores, so this is the one step that needs
  // elevation. Detect it up front: the alternative is a 0x800B0109 nobody can act on.
  if (!(await certTrusted())) {
    return {
      ok: false, needsTrust: true, cerPath: CER_PATH, command: trustCommand(),
      error: "the signing certificate is not trusted: the command below imports it, in an elevated terminal",
    };
  }

  onLine("registering the package for this user…");
  const r = await ps(
    `try { Add-AppxPackage -Path '${OUT}' -ExternalLocation '${ROOT}' -ForceUpdateFromAnyVersion -ErrorAction Stop; 'OK' } ` +
    `catch { 'ERR ' + $_.Exception.Message }`);
  const out = (r.out || "").trim();
  if (!/(^|\n)OK\s*$/.test(out)) return { ok: false, error: out.replace(/^ERR /m, "").split(/\r?\n/)[0] || "Add-AppxPackage failed", log: out.slice(-1200) };
  onLine("installed: right-click a capture folder; the entry is in the short menu");
  return { ok: true, ...(await msixStatus()) };
}

export async function uninstall(onLine = () => {}) {
  if (!isWindows()) return { ok: false, error: "Windows only" };
  onLine("removing the package…");
  const r = await ps(
    `try { $p = Get-AppxPackage -Name '${PKG_NAME}'; if ($p) { Remove-AppxPackage -Package $p.PackageFullName -ErrorAction Stop }; 'OK' } ` +
    `catch { 'ERR ' + $_.Exception.Message }`);
  const out = (r.out || "").trim();
  if (!/(^|\n)OK\s*$/.test(out)) return { ok: false, error: out.replace(/^ERR /m, "").split(/\r?\n/)[0] || "Remove-AppxPackage failed" };
  onLine("removed");
  return { ok: true, ...(await msixStatus()) };
}

// ---- CLI ---------------------------------------------------------------------------------------
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const cmd = process.argv[2] || "status";
  const say = (l) => console.log(l);
  const run = { status: msixStatus, build: () => build(say), install: () => install(say), uninstall: () => uninstall(say) }[cmd];
  if (!run) { console.error(`unknown command: ${cmd}`); process.exit(2); }
  const r = await run();
  console.log(JSON.stringify(r, null, 1));
  process.exit(r && r.ok === false ? 1 : 0);
}
