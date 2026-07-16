## 12. JavaScript / WebGPU implementation

This section sketches the reference runtime's integration surface. Code is illustrative (TypeScript
+ WGSL), not final API. It shows the three things that make ARES fast: **WebCodecs texture decode**,
**persistent GPU buffers with per-frame delta upload**, and **GPU-side dequantization**.

### 12.1 Package layout

```
@ares/core     demuxer, scheduler, workers, WebGPU/WebGL renderers, WASM geo decoder
@ares/three    THREE.Object3D wrapper (drives a BufferGeometry / points from @ares/core)
@ares/react    <Ares src="..."/> component for @react-three/fiber
@ares/encoder  (Node/Rust) importers + coders + muxer (offline)
```

Runtime dependencies are minimal by design (goal N1): `@ares/core` has no Three.js dependency; the
Three.js/React wrappers are thin and optional.

### 12.2 Texture decode via WebCodecs (the correct path)

```ts
// One VideoDecoder per active texture track. Frames are pulled, not played.
const decoder = new VideoDecoder({
  output: (frame: VideoFrame) => {
    // Import directly into WebGPU — no CPU pixel copy.
    const tex = device.importExternalTexture({ source: frame });
    uploader.attachTexture(frame.timestamp, tex, frame); // frame.close() after use
  },
  error: (e) => scheduler.onDecodeError('texture', e),
});
const cfg = { codec: 'av01.0.05M.08', description: track.codecConfig /* from track dir */ };
if ((await VideoDecoder.isConfigSupported(cfg)).supported) decoder.configure(cfg);
else decoder.configure(vp9FallbackConfig);       // §7.2 capability probe

// Per chunk: feed EncodedVideoChunks demuxed from the .ares texture block.
for (const ec of demux.textureChunks(gopIndex)) {
  decoder.decode(new EncodedVideoChunk({
    type: ec.isKeyframe ? 'key' : 'delta',
    timestamp: ec.ptsMicros, data: ec.bytes,
  }));
}
```

> This is the concrete form of the [§7.1](#71-the-decision-one-video-track-decoded-via-webcodecs)
> correction: `VideoDecoder`, not `<video>`. Frame-accurate, pull-based, GPU-importable.

### 12.3 Geometry: persistent buffers + delta upload

```ts
// Allocated ONCE per capture (max size for the profile). Reused every frame. (§10.6)
const posBuf = device.createBuffer({ size: maxVerts*3*2, usage: STORAGE|COPY_DST }); // u16 x3
const idxBuf = device.createBuffer({ size: maxIdx*4,    usage: INDEX|COPY_DST });

function onIFrame(f: DecodedIFrame) {
  device.queue.writeBuffer(idxBuf, 0, f.indices);     // topology: once per GOP
  device.queue.writeBuffer(posBuf, 0, f.positions);   // quantized u16, dequant on GPU
}
function onPFrame(f: DecodedPFrame) {
  // Sparse: only changed vertices. Scatter via a small compute pass or ranged writes.
  applyDeltaCompute(posBuf, f.changedIndices, f.residuals); // §12.5
}
```

Only the position buffer changes per frame; indices persist for the whole GOP (persistent topology,
§6.5). That is the byte-level payoff of the core bet.

### 12.4 GPU-side dequantization (WGSL vertex shader)

```wgsl
struct GopParams { aabbMin: vec3f, aabbMax: vec3f, invMax: f32 };
@group(0) @binding(0) var<uniform> gop: GopParams;
@group(0) @binding(1) var<storage> qpos: array<u32>; // packed u16x3

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let q = unpackU16x3(qpos, vi);                 // 0..65535 per axis
  let n = vec3f(q) * gop.invMax;                 // 0..1
  let world = mix(gop.aabbMin, gop.aabbMax, n);  // dequantize — on the GPU, not the CPU
  return camera.viewProj * vec4f(world, 1.0);
}
```

Dequantization, normal derivation, and (splat profile) attribute unpacking run on the GPU, so the
CPU only ever moves compact quantized bytes (goal 3/4).

### 12.5 Applying sparse deltas

A tiny compute pass scatters `changedIndices[i] → posBuf[idx] += residual[i]`, so P-frame upload
cost scales with *changed* vertices, not total vertices. For the WebGL2 fallback, deltas apply on the
CPU into a staging typed array followed by `bufferSubData` over the changed range (§10.4).

### 12.6 Three.js wrapper

```ts
class AresObject extends THREE.Object3D {
  constructor(private player: AresPlayer) { super(); }
  // Presents into a BufferGeometry whose attributes alias the core's GPU buffers where possible;
  // on WebGL2 it updates a DynamicDrawUsage position attribute per frame.
}
```

### 12.7 React / react-three-fiber component

```tsx
export function Ares({ src, tier = 'auto', ...props }: AresProps) {
  const { gl } = useThree();
  const ref = useRef<AresObject>(null);
  useEffect(() => {
    let p: AresPlayer;
    AresPlayer.create({ src, renderer: gl.isWebGPURenderer ? 'webgpu' : 'webgl2' })
      .then((player) => { p = player; ref.current = new AresObject(player); player.play(); });
    return () => p?.dispose();               // deterministic teardown (§10.6)
  }, [src]);
  useFrame((_, dt) => ref.current?.player.tick(dt)); // advance clock; render is decoupled
  return <primitive object={ref.current} {...props} />;
}
```

Usage is a one-liner, matching the "drag-in-and-play" ergonomics lesson from 4DViews
([§3.2](#32-4dviews-holosys)):

```tsx
<Canvas><Ares src="/captures/dancer.ares" position={[0,0,0]} /></Canvas>
```

### 12.8 What the integrator must provide

- WebGPU (or accept WebGL2 fallback).
- For the `SharedArrayBuffer` fast path: COOP/COEP headers (else the compatible path is used
  automatically, §10.5).
- A CDN/host supporting `Range` requests (single-file mode) or serving the segmented chunks.
