# @ares/core

The ARES Volumetric browser runtime: it plays a whole volumetric capture — animated mesh with a
video texture, or a Gaussian splat cloud — from a single `.ares` file.

One file carries quantized, meshopt-compressed geometry interleaved with a hardware-decodable
AV1/VP9 video texture in one GOP-aligned stream, plus an optional Opus audio track. This package
demuxes it, decodes geometry (main thread or a worker), decodes the texture through WebCodecs, and
draws it with WebGPU, falling back to WebGL2.

Part of [ARES Volumetric](https://github.com/gantasmo/ARES-Volumetric); the format is defined in
[ARES-Runtime-Specification.md](https://github.com/gantasmo/ARES-Volumetric/blob/master/ARES-Runtime-Specification.md).

## Install

```
npm i @ares/core
```

## Use

```js
import { AresPlayer } from "@ares/core";

const player = await AresPlayer.create({
  canvas: document.querySelector("canvas"),
  src: "clip.ares",
  loop: true,
});
player.play();
```

`AresPlayer.create` is the only entry point — the constructor is internal. Useful options:
`useWorker` + `workerUrl` (geometry decode off the main thread), `audio` / `volume` / `muted`,
`autoOrbit`, `forceGL2`, and the `onFrame` / `onStats` / `onEnded` callbacks.

Beyond playback the player exposes `pause`, `seek`, transport and trim controls, `setShadeMode`,
`setPointSize`, `setFx` / `setFxTrack` for playback effects, `exportFrame()` and `dispose()`.

## Single-file bundles

`dist/bundle/` ships browser builds with `meshoptimizer` inlined, for a plain `<script>` with no
bundler:

```html
<script type="module">
  import { AresPlayer } from "@ares/core/bundle/ares-core.esm.js";
</script>
```

`ares-core.iife.js` is the classic-script variant (`window.ARES`). Both need an explicit
`workerUrl` pointing at `ares-decode-worker.js` if you want worker decode.

## Requirements

A browser with WebGPU (or WebGL2 for the fallback path) and WebCodecs for video-textured clips.
This package has no Node runtime requirement of its own — it is browser code.

MIT. See `THIRD-PARTY-NOTICES.md` in the repository for what the bundles carry.
