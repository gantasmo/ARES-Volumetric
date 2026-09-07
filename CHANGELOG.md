# Changelog

## 0.1.0 — 2026-09-07

**ARES Volumetric plays volumetric video in a browser, from a single file.** A capture that would
normally arrive as thousands of meshes and gigabytes of PNGs becomes one `.ares` file: quantized,
meshopt-compressed geometry interleaved with a hardware-decodable AV1/VP9 video texture in one
GOP-aligned stream, with an optional Opus audio track. The player fetches that one file, dequantizes
vertex positions in the vertex shader, and uploads each decoded video frame straight to the GPU. A
real 272-frame capture is 49.7 MB in one request, against 1.58 GB across 544 files as raw OBJ+PNG.

This is the first versioned cut — the format, the browser runtime, the `ares` CLI, the Three.js and
React wrappers, the demo app and the specification, as they stand after the 2026-09-07 audit
([AUDIT.md](AUDIT.md)). **Added** below is what this release contains rather than a delta from an
earlier version; **Fixed** is relative to the unversioned initial public drop (commit `0907ee5`).

### Added
- Gaussian splat profile (spec §6.8, §11.6.3): `SPLT` geometry track, meshopt-coded splat streams
  quantized over the chunk AABB, SH degree 0–3, Morton-ordered frames. Both renderers draw it
  (EWA-projected instanced quads, premultiplied over, CPU counting sort gated on view change).
- Importers: Niantic SPZ v1–v4, 3DGS PLY, `.splat`, glTF/GLB with `KHR_gaussian_splatting`,
  PlayCanvas SOG (bundle or directory). Exporters: SPZ, 3DGS PLY, GLB, `.splat`; OBJ/PLY for meshes.
- `ares export <file.ares> -o <out> [--frame N]`, `ares synth --shape splat`, splat flags on
  `ares encode` (`--sh-degree`, `--splat-min-alpha`, `--splat-box-alpha`, `--splat-order`,
  `--quant-bits`).
- Convert tab recognises splat sequences; the viewer gates mesh-only tools for splat clips.
- `npm test` (node --test), GitHub Actions CI (Ubuntu/Windows × Node 22/24), esbuild bundles
  (`npm run bundle`), `AresPlayerOptions.workerUrl`, publish-ready package manifests.

- Editor: lasso and measure tools, grow/shrink/invert/mirror of the active range, camera bookmarks,
  per-range mute/name/interpolation (linear/hold/smooth), a bake-side sculpt action (move, inflate,
  smooth, flatten, pinch), analysis views (normals, UV checker, depth, points with size), an Export
  section (frame → OBJ, still → PNG, turntable → WebM), and a `?` shortcuts panel.
- Runtime: `AresPlayer.exportFrame()`, `setShadeMode()`, `setPointSize()`, `orbitSpeed`.
- Playback effects (both renderers, meshes and splats): clip plane, dissolve, tint, rim, scanlines,
  wobble, splat jitter/size/opacity; keyframed in the sidecar; audio-reactive binding; FX rail section.
- Dynamic splat profile: P-frames with position/attribute/SH deltas, births and deaths;
  `--splat-temporal auto|index|nn|off`.
- Audio: an Opus track per chunk (`ares encode --audio file`), WebCodecs decode into Web Audio,
  audio-led clock, mute/volume in the player and the demo transport, Convert-card audio row.

- Dev server: host pinning and a same-origin guard on every route that does work or spends,
  JSON-only `/runpod/launch`, separator-aware static guard, SSH host keys pinned on first use.
- Demo: Compare pauses off-tab and answers Space, batch queue survives re-analyse, Inspect reads
  only the container skeleton and shows profile + audio.

### Changed
- One launcher for the whole app, at the repo root: `ARES.mjs` (`npm start`, or `ARES.vbs` to
  double-click on Windows, `ARES-console.cmd` for a visible console) with modes
  `app | probe | bench | sam`. It replaces the four root `.vbs` launchers and their three
  PowerShell workers; `npm run serve` still starts the bare dev server.
- Package manifests point at the real repository (`gantasmo/ARES-Volumetric`) and carry
  `homepage`, `bugs` and a monorepo `repository.directory`. Each publishable package now ships its
  own README and a copy of the MIT licence, `@webgpu/types` moved to `dependencies` (its types
  appear in `@ares/core`'s public declarations), the browser packages no longer demand Node 22.15,
  and `@ares/three` publishes its bundle deterministically.
- `npm run release` builds the downloadable archive for a GitHub release: browser bundles, the four
  npm tarballs, the spec and the notices, as a directory, a zip and a release-notes file.

### Fixed
- WebGL2 renderer: crop preview and wireframe were silent no-ops.
- Geometry decoder caps vertex/index/splat counts before allocating (untrusted input).
- `ByteWriter(0)` looped forever; superblock `quant_bits_uv` misreported 14 (UVs are 16-bit).
- CLI: value flags followed by another flag are errors instead of NaN; numeric ranges validated;
  usage text complete; `ares info` reads its argument; mesh/atlas frame pairing uses natural sort.
- PLY mesh import keeps UVs, normals and colours.
