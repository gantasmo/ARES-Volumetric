# Contributing

## Setup

```
npm install          # workspaces + postinstall link fixer (tools/fix-ide-symlinks.mjs)
npm run build        # tsc -b across packages/core, encoder, three, react, bench
npm test             # build + node --test (packages/*/test/*.test.mjs)
npm run bundle       # esbuild single-file browser bundles into packages/*/dist/bundle/
npm start            # launcher: build if stale, COOP/COEP dev server, opens the app
npm run serve        # just the dev server, no build and no browser
```

Node 22.15 or newer (zstd in `node:zlib` is used by the SPZ importer). ffmpeg on PATH (or
`FFMPEG=`) for video textures and the SOG importer. No GPU is needed for the test suite; the
renderers are exercised in the browser on real hardware.

If the workspace links in `node_modules/@ares/*` point at a path that no longer exists (the repo
moved drives), `npm install` recreates them.

## Layout and conventions

- `packages/core` is the browser runtime and must stay dependency-free except `meshoptimizer`.
  Everything the demuxer reads is untrusted: bounds-check offsets, cap counts before allocating
  (`geometry.ts` guardCount), never `eval`.
- `packages/encoder` is Node-only. Importers live in `src/importers/` and produce either an
  `EncodeMeshFrame` (meshes) or a `SplatFrame` (Gaussian splats); the muxer never sees file
  formats. To add a format: write `importers/<fmt>.ts` with `parse<Fmt>(bytes)` and, if it can be
  written, `write<Fmt>(frame)`; register the extension in `cli.ts` (`SPLAT_EXTS` / `meshesIn`),
  export it from `src/index.ts`, and add a round-trip test under `packages/encoder/test/`.
- Byte layouts are shared law: a change to a block layout touches `@ares/core` (decoder),
  `@ares/encoder` (encoder), `spec/11-file-format.md`, and a round-trip test, in one change.
- The demo app (`apps/demo`) has no build step; it loads `packages/*/dist` through an import map.
  UI rules, in full: use the CSS custom-property tokens and the shared `.u` / `.inp` classes rather
  than new one-off styles; no neon or saturated accents, no indicator lights, no emoji icons; data
  colours stay muted; panels fit their space instead of introducing scrollbars.
- Do not launch the browser to verify UI in a shared session; verify statically (`node --check`,
  reading diffs) and let the owner reload.

## Tests

`node --test` with the built `dist/` (the suite imports `../dist/index.js`). Keep tests
deterministic: seeded generators, `synthClip` / `synthSplatClip`, temp dirs cleaned in `finally`.
A test that needs ffmpeg must skip cleanly when it is absent.

## Releasing

Packages are versioned together (`0.1.0` today). `npm run release:check` builds, tests, bundles
and dry-runs `npm pack` for every workspace. Publishing is `npm publish --workspace @ares/core`
(then encoder, three, react) once the `@ares` npm scope is confirmed available to the owner.

`npm run release` builds the downloadable archive for a GitHub release:
`dist/release/ares-volumetric-<version>/` (browser bundles, the four npm tarballs, the spec and
the licence/notice files), the same tree as `ares-volumetric-<version>.zip`, and
`ares-volumetric-<version>-notes.md` to pass to `gh release create --notes-file`. `dist/` is
git-ignored; the archive is an attachment, never a commit.

## Spec

`spec/*.md` is the source; `python spec/build.py` assembles `ARES-Runtime-Specification.md`
(and `.html` when the `markdown` pip module is installed). Edit the chapter, not the assembled file.
