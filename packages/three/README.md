# @ares/three

[ARES Volumetric](https://github.com/gantasmo/ARES-Volumetric) as a `THREE.Object3D`: drop a
volumetric capture — one `.ares` file carrying animated geometry plus a hardware-decoded video
texture — into a Three.js scene and drive it from your own render loop.

## Install

```
npm i @ares/three three
```

`three` is a peer dependency (>=0.160).

## Use

```js
import { AresObject } from "@ares/three";

const ares = new AresObject({ canvas, src: "clip.ares", loop: true });
scene.add(ares);

renderer.setAnimationLoop((t, dt) => {
  ares.tick(dt / 1000);   // advances playback and renders one ARES frame
  renderer.render(scene, camera);
});

// later
ares.dispose();
```

`whenReady()` resolves to the underlying [`AresPlayer`](https://www.npmjs.com/package/@ares/core)
for transport, effects and stats; `isReady` is the synchronous check.

Note for this release: the ARES renderer draws to its own WebGPU canvas, decoupled from the host
renderer's — pass that canvas as `canvas`. Presenting into a shared `BufferGeometry` under a
Three.js WebGPURenderer is planned, not shipped.

## Bundles

`dist/bundle/ares-three.esm.js` is a single-file build with `three` left external.

MIT.
