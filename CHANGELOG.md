# Changelog

## 0.1.0 — 2026-09-07 (unreleased)

First versioned cut, following the 2026-09-07 audit ([AUDIT.md](AUDIT.md)).

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
- One launcher for the whole app: `tools/launch.mjs` (`npm start`, or `ARES.vbs` on Windows) with
  modes `app | probe | bench | sam`. It replaces the four root `.vbs` launchers and their three
  PowerShell workers; `npm run serve` still starts the bare dev server.

### Fixed
- WebGL2 renderer: crop preview and wireframe were silent no-ops.
- Geometry decoder caps vertex/index/splat counts before allocating (untrusted input).
- `ByteWriter(0)` looped forever; superblock `quant_bits_uv` misreported 14 (UVs are 16-bit).
- CLI: value flags followed by another flag are errors instead of NaN; numeric ranges validated;
  usage text complete; `ares info` reads its argument; mesh/atlas frame pairing uses natural sort.
- PLY mesh import keeps UVs, normals and colours.
