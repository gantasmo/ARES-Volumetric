# @ares/react

[ARES Volumetric](https://github.com/gantasmo/ARES-Volumetric) for
[@react-three/fiber](https://github.com/pmndrs/react-three-fiber): an `<Ares/>` component that plays
a volumetric capture from a single `.ares` file and advances it from r3f's frame loop.

## Install

```
npm i @ares/react react three @react-three/fiber
```

`react` (>=18), `three` (>=0.160) and `@react-three/fiber` (>=8) are peer dependencies.

## Use

```jsx
import { Canvas } from "@react-three/fiber";
import { Ares } from "@ares/react";

function Scene({ canvas }) {
  return <Ares src="clip.ares" canvas={canvas} loop />;
}

<Canvas>
  <Scene canvas={aresCanvas} />
</Canvas>
```

Props: `src` (URL or a preloaded `Uint8Array`), `canvas` (the surface the ARES WebGPU renderer draws
into — in this release it is its own canvas, not the r3f one), `loop`, `autoOrbit`. The component
renders nothing itself; it owns an [`AresObject`](https://www.npmjs.com/package/@ares/three), ticks
it from `useFrame`, and disposes it on unmount.

MIT.
