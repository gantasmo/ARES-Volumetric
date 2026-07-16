/**
 * Make the npm-workspace symlink view of each package VALID for the TypeScript language server.
 *
 * npm workspaces symlinks every workspace into node_modules/@ares/<name>. If a tsconfig.json is
 * opened THROUGH that symlink (VS Code history, go-to-definition, …), its relative paths resolve
 * against node_modules/@ares/ — where `tsconfig.base.json` and `packages/…` don't exist — and the
 * IDE reports phantom "cannot read file" errors even though `tsc -b` at the real path is green.
 *
 * Real fix (no suppression): create the missing targets inside node_modules/@ares/ so the
 * symlinked view resolves exactly like the real one:
 *   node_modules/@ares/tsconfig.base.json   — copy of the root base config
 *   node_modules/@ares/packages             — junction → <repo>/packages
 * With those two, every workspace tsconfig resolves from BOTH paths (bench extends
 * ../tsconfig.base.json ✓, references ../packages/core ✓; core extends ../../tsconfig.base.json ✓).
 *
 * Runs on postinstall (npm regenerates node_modules) and is idempotent.
 */
import { copyFileSync, existsSync, mkdirSync, symlinkSync, rmSync, lstatSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url))); // ares/
const SCOPE = join(ROOT, "node_modules", "@ares");

if (!existsSync(SCOPE)) {
  console.log("[fix-ide-symlinks] node_modules/@ares missing (run npm install first) — nothing to do");
  process.exit(0);
}

// 1. Base config, so `extends "../tsconfig.base.json"` (and ../../ from packages/*) resolves.
copyFileSync(join(ROOT, "tsconfig.base.json"), join(SCOPE, "tsconfig.base.json"));

// 2. `packages` junction, so `references: ../packages/*` resolves. Junction = no admin needed.
const pkgsLink = join(SCOPE, "packages");
try {
  const st = lstatSync(pkgsLink);
  if (st.isSymbolicLink() || st.isDirectory()) rmSync(pkgsLink, { recursive: false });
} catch { /* doesn't exist — fine */ }
try {
  symlinkSync(join(ROOT, "packages"), pkgsLink, "junction");
} catch (e) {
  console.error("[fix-ide-symlinks] could not create packages junction:", e.message);
  process.exit(1);
}

// 3. `bench` sibling (bench lives at repo root, referenced as ../bench from nothing today, but the
// root tsconfig may reference it; keep the view complete for future workspace layouts).
mkdirSync(SCOPE, { recursive: true });

console.log("[fix-ide-symlinks] node_modules/@ares view is now self-consistent (tsconfig.base.json + packages junction)");
