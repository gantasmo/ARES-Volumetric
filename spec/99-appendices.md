## Appendix A — Binary layouts

Consolidated, byte-exact reference for implementers. Little-endian; `varint` = LEB128.

### A.1 File header (repeat of §11.2, canonical)

```
struct AresHeader {          // 64 bytes, fixed
  u8   magic[4];             // 'A','R','E','S'
  u8   version_major;        // 0
  u8   version_minor;        // 1
  u16  header_flags;         // bit0 live | bit1 has_audio | bit2 coi_hint | bit3 has_aux
  u8   geometry_profile;     // 0 mesh-IPB | 1 splat-IPB | 2 video-geometry(exp)
  u8   texture_codec;        // 0 none|1 AV1|2 VP9|3 HEVC|4 AVC
  u8   intra_codec;          // 0 meshopt|1 draco|2 raw
  u8   entropy_codec;        // 0 range/ANS|1 none
  f32  fps;
  u32  frame_count;
  u64  duration_us;
  u64  superblock_offset;
  u64  gop_index_offset;
  u64  track_dir_offset;
  u64  first_chunk_offset;
  u32  header_crc32;         // CRC-32 of bytes [0,60)
}
```

### A.2 GOP index record

```
struct GopRecord {
  u64  start_pts_us;
  u32  frame_start;
  u16  frame_count;
  u16  flags;                // bit0 forced_keyframe | bit1 has_aux
  u64  byte_offset;          // single-file mode
  u32  byte_length;
  u32  tier_offset[tier_count];
}
```

### A.3 Chunk header

```
struct ChunkHeader {
  u8   magic[4];             // 'C','N','K','0'
  u64  pts_start_us;
  u16  frame_count;
  u16  flags;
  f32  gop_aabb_min[3];
  f32  gop_aabb_max[3];
  u16  block_count;
  BlockDir block_dir[block_count];   // {u8 type; u16 track_id; u32 offset; u32 length}
}
```

### A.4 Mesh P-frame residual stream (sparse)

```
struct PFrame {
  u8    frame_type;          // 1 P | 2 B
  u8    ref_lo, ref_hi;      // anchor indices (B uses both)
  u8    predictor;           // 0 prev | 1 prev+velocity
  u32   changed_count;
  // then, delta-coded ascending vertex indices + quantized residuals:
  repeat changed_count {
    varint index_delta;      // gap from previous changed index
    svarint dqx, dqy, dqz;   // zig-zag quantized position residual
  }
}
```

### A.5 Splat I-frame record

```
struct SplatIFrame {
  u32   splat_count;
  repeat splat_count {
    f16  pos[3];
    f16  scale[3];
    i8   rot_quat[4];        // normalized
    u8   opacity;
    f16  sh[k];              // k = f(sh_degree): 1,4,9,16 coeff sets × 3 channels
  }
}
```

---

## Appendix B — Pseudocode

### B.1 Encoder: GOP segmentation via tracking error

```
canonical = reconstruct_mesh(frame[0])          # I-frame base
gop_start = 0
emit_I(canonical)
for t in 1..N-1:
    target = reconstruct_or_pointcloud(frame[t])
    fitted, err = nonrigid_register(canonical, target)   # §6.5.1
    if err > TAU or topology_changed(fitted, target):
        canonical = reconstruct_mesh(frame[t])           # new GOP
        gop_start = t
        emit_I(canonical)
    else:
        residual = quantize(fitted.pos - predict(canonical, history))
        emit_P(sparse(residual))                         # §A.4
        canonical.pos = fitted.pos                       # advance reference
```

### B.2 Runtime: seek

```
function seek(t_seconds):
    pts = t_seconds * 1e6
    gop = binary_search(gop_index, pts)     # O(log n), §9.2
    chunk = fetch_range(gop.byte_offset, gop.byte_length[tier])
    Iframe = decode_intra(chunk.geometry_I)
    upload_persistent(Iframe)               # indices once, positions once
    texdec.configure_if_needed(chunk.texture.codec_config)
    feed_texture_chunks(chunk, from=pts)
    present_when_ready(pts)                  # §8.6 sync
```

### B.3 Runtime: steady-state tick

```
function tick(now):
    ensure_prefetch_window(now, PREFETCH_SECONDS)   # §9.4 fetch+decode ahead
    abr_maybe_switch(estimate_bw(), buffer_level()) # §9.3 at GOP boundary only
    slot = ready_slot_for(clock.pts(now))           # triple buffer, §10.2
    if slot: renderer.draw(slot)                     # else hold last (§N4)
    recycle_old_frames(memory_budget)                # ring buffer, §10.6
```

### B.4 Runtime: apply sparse P-frame (compute)

```
# GPU compute, one invocation per changed vertex (§12.5)
i = changed_index[gid]
pos_buf[i] = pos_buf[i] + residual[gid]   # in quantized space; dequant in vertex shader
```

---

## Appendix C — Core runtime data structures

```ts
interface GopIndexEntry {
  startPtsUs: number; frameStart: number; frameCount: number;
  byteOffset: number; byteLength: number; tierOffset: number[];
  forcedKeyframe: boolean; hasAux: boolean;
}
interface DecodedFrame {
  pts: number;
  positions?: Uint16Array;        // quantized; dequant on GPU
  changedIndices?: Uint32Array;   // P/B sparse
  residuals?: Int16Array;         // P/B sparse
  splat?: SplatBuffers;           // splat profile
  texture?: VideoFrame;           // WebCodecs output, close() after import
}
interface RingBuffer<T> { capacity: number; push(f: T): void; get(pts: number): T|null;
                          evictOlderThan(pts: number): void; }
interface Scheduler { ensurePrefetch(now: number): void; onThroughput(bps: number): void;
                      selectTier(): number; }
```

---

## Appendix D — Glossary

| Term | Definition |
|---|---|
| **ARES** | This runtime + container format (working title). Container extension `.ares`. |
| **GOP** | Group of pictures/frames: an I-frame plus its dependent P/B frames; the unit of chunking and seeking. |
| **I / P / B frame** | Intra (keyframe, standalone) / Predicted (delta from prior) / Bidirectional (interpolated between two anchors). |
| **Persistent topology** | A stable vertex set + connectivity held across a GOP, so only positions change per frame. |
| **Geometry image** | A 2D parameterization of a surface where pixels encode geometry attributes, enabling video coding. |
| **3DGS / splat** | 3D Gaussian Splatting; scene as anisotropic Gaussians rendered by rasterization. |
| **WebCodecs** | Browser API (`VideoDecoder`/`VideoEncoder`) for frame-accurate, pull-based codec access. |
| **ABR** | Adaptive bitrate: switching quality tiers by network/on-screen conditions. |
| **TTFF** | Time-to-first-frame. |
| **meshopt / Draco** | Mesh compression codecs; meshopt favors fast SIMD decode, Draco favors ratio. |
| **PTS** | Presentation timestamp on the shared timeline. |
| **COOP/COEP** | HTTP headers enabling cross-origin isolation → `SharedArrayBuffer`. |

---

## Appendix E — References

Sources informing this specification (carried from the project research brief; bracket numbers match
in-text citations). Retrieval as of mid-2026.

1. Fraunhofer HHI — volumetric capture bitrate figures ("Dimitri" sequence, ~110 Gbps uncompressed).
4. Universal Volumetric (UVOL) — per-frame Draco mesh + KTX2/Basis textures + JSON manifest.
7. Khronos — glTF Volumetric subgroup launch (2026); notes Gaussian Splatting.
13. Arcturus AVV / HoloSuite (2024) — near-lossless compression, multi-resolution textures, per-vertex
    motion vectors; ~25% of original size claims.
16. VVglTF (2025) — streaming glTF segments over HTTP with frame-rate adaptation.
25. Google — WebP vs PNG size comparison (~26% smaller lossless; ~3× smaller lossy at similar SSIM).
29. Three.js docs — DRACOLoader / Web Worker decoding guidance.
32. glTF-Transform / Pixyz — Draco makes `.glb` "much lighter" (~10–20% of original geometry).
33. KTX2 / Basis Universal — GPU-compressed texture transcoding (UASTC/ETC1S).
36. Depthkit — capture toolkit; combined color+depth video layout.
37. 4DViews HOLOSYS — lightweight textured mesh capture; engine-ready runtimes.
39. MPEG — V-PCC (video-atlas) and G-PCC (octree/predictive) point-cloud coding standards.
43. Brown University "PackUV" (CVPR 2026) — mapping 3D Gaussian-splat frames into 2D video tracks;
    chunking long sequences to reset stream state and handle object entry/exit.

> Citation numbers are inherited from the source research brief and are intentionally
> non-contiguous. A future revision SHOULD normalize them and add DOIs/URLs verified at publication
> time. Any figure above without a first-principles derivation is a third-party claim, not an ARES
> measurement.
