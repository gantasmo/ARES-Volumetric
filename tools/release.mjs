#!/usr/bin/env node
/**
 * Release package builder — assembles one downloadable archive from what the repo already
 * produces, so a GitHub release can carry the runtime without a clone or a registry.
 *
 *   npm run release            build + bundle, then assemble
 *   node tools/release.mjs     assemble from existing output (fails if it is missing)
 *
 * Output (git-ignored, `dist/` at the repo root):
 *   dist/release/ares-volumetric-<version>/       the staged tree
 *   dist/release/ares-volumetric-<version>.zip    the same tree, zipped (store + deflate,
 *                                                 written here because Node ships no zip writer)
 *
 * In the box: the single-file browser bundles, npm tarballs for the four packages
 * (`npm i ./npm/ares-core-<v>.tgz`), the spec, and the licence/notice files. The demo app is
 * not included: it needs the dev server and the encoder, which means the repo.
 */
import { spawn } from "node:child_process";
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { deflateRawSync } from "node:zlib";
import { basename, dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const PACKAGES = ["@ares/core", "@ares/encoder", "@ares/three", "@ares/react"];
const DOCS = ["README.md", "CHANGELOG.md", "LICENSE", "THIRD-PARTY-NOTICES.md", "CONTRIBUTING.md",
              "ARES-Runtime-Specification.md", "ARES-Runtime-Specification.html"];

const fail = (msg) => { console.error(`[release] ${msg}`); process.exit(1); };

function run(cmd, args, opts = {}) {
  return new Promise((done) => {
    const p = spawn(cmd, args, { cwd: ROOT, stdio: "inherit", windowsHide: true, ...opts });
    p.on("error", (e) => { console.error(`[release] ${cmd}: ${e.message}`); done(1); });
    p.on("close", (code) => done(code ?? 1));
  });
}

// --- zip (store + deflate, no ZIP64: nothing here approaches 4 GB) ------------
const CRC_TABLE = Int32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});
function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}
/** DOS date/time; the epoch is 1980, and seconds have 2-second resolution. */
function dosStamp(d) {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}
async function zipDir(dir, outFile, prefix) {
  const now = dosStamp(new Date());
  const files = [];
  const walk = async (abs) => {
    for (const e of await readdir(abs, { withFileTypes: true })) {
      const p = join(abs, e.name);
      if (e.isDirectory()) await walk(p);
      else files.push(p);
    }
  };
  await walk(dir);
  files.sort();

  const local = [], central = [];
  let offset = 0;
  for (const abs of files) {
    const name = `${prefix}/${relative(dir, abs).split(sep).join("/")}`;
    const nameBuf = Buffer.from(name, "utf8");
    const raw = await readFile(abs);
    const deflated = deflateRawSync(raw, { level: 9 });
    const store = deflated.length >= raw.length;
    const data = store ? raw : deflated;
    const method = store ? 0 : 8;
    const crc = crc32(raw);

    const lfh = Buffer.alloc(30);
    lfh.writeUInt32LE(0x04034b50, 0); lfh.writeUInt16LE(20, 4); lfh.writeUInt16LE(0, 6);
    lfh.writeUInt16LE(method, 8); lfh.writeUInt16LE(now.time, 10); lfh.writeUInt16LE(now.date, 12);
    lfh.writeUInt32LE(crc, 14); lfh.writeUInt32LE(data.length, 18); lfh.writeUInt32LE(raw.length, 22);
    lfh.writeUInt16LE(nameBuf.length, 26); lfh.writeUInt16LE(0, 28);
    local.push(lfh, nameBuf, data);

    const cdh = Buffer.alloc(46);
    cdh.writeUInt32LE(0x02014b50, 0); cdh.writeUInt16LE(20, 4); cdh.writeUInt16LE(20, 6);
    cdh.writeUInt16LE(0, 8); cdh.writeUInt16LE(method, 10);
    cdh.writeUInt16LE(now.time, 12); cdh.writeUInt16LE(now.date, 14);
    cdh.writeUInt32LE(crc, 16); cdh.writeUInt32LE(data.length, 20); cdh.writeUInt32LE(raw.length, 24);
    cdh.writeUInt16LE(nameBuf.length, 28); cdh.writeUInt32LE(0, 30); cdh.writeUInt32LE(0, 34);
    cdh.writeUInt32LE(0, 38); cdh.writeUInt32LE(offset, 42);
    central.push(cdh, nameBuf);

    offset += lfh.length + nameBuf.length + data.length;
  }
  const cd = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(files.length, 8); eocd.writeUInt16LE(files.length, 10);
  eocd.writeUInt32LE(cd.length, 12); eocd.writeUInt32LE(offset, 16);
  await writeFile(outFile, Buffer.concat([...local, cd, eocd]));
  return files.length;
}

// --- assemble ----------------------------------------------------------------
const pkg = JSON.parse(await readFile(join(ROOT, "package.json"), "utf8"));
const name = `ares-volumetric-${pkg.version}`;
const stage = join(ROOT, "dist", "release", name);
const coreBundle = join(ROOT, "packages/core/dist/bundle");
if (!existsSync(coreBundle)) fail("packages/core/dist/bundle missing — run `npm run build && npm run bundle` first");

await rm(stage, { recursive: true, force: true });
await mkdir(join(stage, "bundles"), { recursive: true });
await mkdir(join(stage, "npm"), { recursive: true });
await mkdir(join(stage, "docs"), { recursive: true });

await cp(coreBundle, join(stage, "bundles"), { recursive: true });
const threeBundle = join(ROOT, "packages/three/dist/bundle");
if (existsSync(threeBundle)) await cp(threeBundle, join(stage, "bundles"), { recursive: true });
else console.warn("[release] no @ares/three bundle — skipping");

// npm through its own cli.js, never through a shell: the repo path may contain spaces, and
// `shell: true` concatenates arguments without quoting them (Node DEP0190).
function npmCli() {
  const here = dirname(process.execPath);
  const candidates = [
    process.env.npm_execpath,                                              // set inside npm run
    join(here, "node_modules/npm/bin/npm-cli.js"),                         // Windows
    join(here, "../lib/node_modules/npm/bin/npm-cli.js"),                  // POSIX
  ];
  return candidates.find((c) => c && c.endsWith(".js") && existsSync(c));
}

console.log(`[release] packing ${PACKAGES.length} packages...`);
const cli = npmCli();
if (!cli) fail("could not locate npm-cli.js next to this Node install");
for (const p of PACKAGES) {
  const code = await run(process.execPath, [cli, "pack", "--workspace", p, "--pack-destination", join(stage, "npm"), "--silent"]);
  if (code !== 0) fail(`npm pack ${p} failed (exit ${code})`);
}

for (const d of DOCS) {
  const src = join(ROOT, d);
  if (!existsSync(src)) { console.warn(`[release] missing ${d} — skipping`); continue; }
  if ((await stat(src)).size === 0) fail(`${d} is empty — regenerate it before cutting a release`);
  await cp(src, join(stage, "docs", basename(d)));
}

const tarballs = (await readdir(join(stage, "npm"))).sort();
const repoUrl = pkg.repository?.url?.replace(/^git\+/, "").replace(/\.git$/, "") ?? "the repository";

/** Nobody downloading this has read the changelog, and for a first release there is no "before" to
 *  diff against — so every generated document opens by saying what the thing IS. */
const ABOUT = `**ARES Volumetric plays volumetric video in a browser, from a single file.**

A volumetric capture — an animated person or object, one mesh plus one texture per frame, or a
Gaussian splat cloud — normally arrives as thousands of files and gigabytes of PNGs. ARES packs a
whole clip into one \`.ares\` file: quantized, meshopt-compressed geometry interleaved with a
hardware-decodable AV1/VP9 video texture in one GOP-aligned stream, plus an optional Opus audio
track. The player fetches that one file, keeps vertex positions quantized until the vertex shader
dequantizes them on the GPU, and hands each decoded video frame to the GPU without a CPU pixel copy.

A real 272-frame capture (11.3k vertices per frame, 9.1 s at 30 fps) is **49.7 MB in one file and
one request** — against 1.58 GB across 544 files as raw OBJ+PNG, or 1.13 GB as a Draco-GLB sequence.

This ${pkg.version} release is the format, the runtime and the tools that produce it: mesh clips with
video texture, static and dynamic Gaussian splats (SPZ, 3DGS PLY, \`.splat\`, glTF
\`KHR_gaussian_splatting\` and SOG in; SPZ, PLY, GLB and \`.splat\` out), an Opus audio track,
playback effects, WebGPU with a WebGL2 fallback, optional worker decode, a Three.js object and a
React component, and an \`ares\` CLI that encodes, inspects and exports. The repository additionally
carries the demo app (viewer, side-by-side compare, container inspector, converter and mesh editor),
a capability probe and a benchmark harness.`;

await writeFile(join(ROOT, "dist", "release", `${name}-notes.md`), `# ARES Volumetric ${pkg.version}

${ABOUT}

Source, demo app and encoder CLI: ${repoUrl}

## Download

\`${name}.zip\` — the browser bundles, npm tarballs for the four packages, and the specification.
Unzip and open \`README.md\` inside for the file-by-file guide. Everything else (demo app, encoder
CLI, dev server) comes from a clone: on Windows double-click \`ARES.vbs\`, anywhere else run
\`npm install && npm start\`.

## What is in this release

See \`docs/CHANGELOG.md\` in the archive for the itemised list.
`);

await writeFile(join(stage, "README.md"), `# ARES Volumetric ${pkg.version}

${ABOUT}

Source, demo app and encoder CLI: ${repoUrl}

## bundles/
Single-file browser builds of \`@ares/core\` (meshoptimizer inlined, nothing else fetched).
Each has a \`.min.js\` and a source map.

    <canvas id="stage"></canvas>
    <script type="module">
      import { AresPlayer } from "./ares-core.esm.js";

      const player = await AresPlayer.create({
        canvas: document.getElementById("stage"),
        src: "clip.ares",
        loop: true,
        useWorker: true,                            // optional: decode geometry off the main thread
        workerUrl: "./ares-decode-worker.js",       // required whenever useWorker is set
      });
      player.play();
    </script>

\`AresPlayer.create\` is the only entry point — the constructor is internal. \`useWorker\`
needs \`workerUrl\` to point at \`ares-decode-worker.js\`; neither bundle can resolve it on its
own, and without a usable worker the player decodes on the main thread. \`ares-core.iife.js\` is
the classic-script variant (\`window.ARES\`), and \`ares-three.esm.js\` keeps \`three\` external
as a peer.

Serve the files over HTTP (not \`file://\`) — a module worker and WebCodecs both need an origin.
Cross-origin isolation is not required: the worker path transfers ArrayBuffers rather than sharing
memory.

## npm/
Tarballs of the four packages:

${tarballs.map((t) => `    npm i ./npm/${t}`).join("\n")}

\`@ares/core\` installs from its tarball alone. The other three declare \`@ares/*\` dependencies
that npm resolves by name, so install them against a registry that has those packages, or add
\`overrides\` pointing at the tarballs beside them.

## docs/
The specification (Markdown and printable HTML), the project README, changelog, contributing
guide, licence and third-party notices. ARES is MIT; \`THIRD-PARTY-NOTICES.md\` covers what the
bundles and tools carry.
`);

const zip = join(ROOT, "dist", "release", `${name}.zip`);
const count = await zipDir(stage, zip, name);
const size = (await stat(zip)).size;
console.log(`[release] ${stage}`);
console.log(`[release] ${zip} (${count} files, ${(size / 1024 / 1024).toFixed(1)} MB)`);
console.log(`[release] ${join(ROOT, "dist", "release", `${name}-notes.md`)}  (release notes: what ARES is + the download)`);
