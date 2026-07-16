## 11. File format specification

This section defines the `.ares` container. It is a chunked binary format: a fixed magic, a
superblock header, a GOP index, then time-ordered chunks. All integers little-endian;
offsets/lengths in bytes; strings UTF-8 with `u16` length prefix. This is a **draft**; field widths
marked *reserved* exist for forward compatibility.

### 11.1 Top-level layout

```
+-----------------------------+  offset 0
| File header (fixed 64 B)    |
+-----------------------------+
| Superblock (metadata)       |  variable, offset in header
+-----------------------------+
| GOP index (seek table)      |  variable, offset in header
+-----------------------------+
| Track directory             |  describes tracks: geometry, texture(s), audio, aux
+-----------------------------+
| Chunk 0 | Chunk 1 | ... | N  |  time-ordered, each = one GOP
+-----------------------------+
| Extension blocks (optional) |
+-----------------------------+
```

### 11.2 File header (64 bytes, fixed)

| Offset | Size | Type | Field | Notes |
|---:|---:|---|---|---|
| 0 | 4 | u8[4] | `magic` | `0x41 0x52 0x45 0x53` = "ARES" |
| 4 | 1 | u8 | `version_major` | `0` for this draft |
| 5 | 1 | u8 | `version_minor` | `1` |
| 6 | 2 | u16 | `header_flags` | bit0 live, bit1 has_audio, bit2 crossorigin_hint, … |
| 8 | 1 | u8 | `geometry_profile` | 0=mesh-IPB, 1=splat-IPB, 2=video-geometry (exp.) |
| 9 | 1 | u8 | `texture_codec` | 0=none,1=AV1,2=VP9,3=HEVC(opt),4=AVC |
| 10 | 1 | u8 | `intra_codec` | 0=meshopt,1=draco,2=raw |
| 11 | 1 | u8 | `entropy_codec` | 0=range/ANS,1=none |
| 12 | 4 | f32 | `fps` | nominal frame rate |
| 16 | 4 | u32 | `frame_count` | total frames |
| 20 | 8 | u64 | `duration_us` | microseconds |
| 28 | 8 | u64 | `superblock_offset` | |
| 36 | 8 | u64 | `gop_index_offset` | |
| 44 | 8 | u64 | `track_dir_offset` | |
| 52 | 8 | u64 | `first_chunk_offset` | |
| 60 | 4 | u32 | `header_crc32` | CRC of bytes [0,60) |

### 11.3 Superblock (global metadata)

Key–value metadata + global geometry/scene parameters:

| Field | Type | Notes |
|---|---|---|
| `aabb_min[3]`, `aabb_max[3]` | f32 | global bounding box (also per-GOP, §11.6) |
| `quant_bits_pos` | u8 | e.g. 14 |
| `quant_bits_uv` | u8 | e.g. 14 |
| `normal_encoding` | u8 | 0=none/derived,1=octahedral |
| `sh_degree` | u8 | splat profile only (0–3) |
| `gop_length` | u16 | nominal frames per GOP |
| `tier_count` | u8 | ABR tiers available |
| `kv_count` | u16 | user metadata pairs |
| `kv[]` | (str,str) | title, author, capture rig, color space, license, … |

### 11.4 GOP index (seek table)

One record per GOP, enabling O(log n) seek and range requests (§9.2):

| Field | Type | Notes |
|---|---|---|
| `start_pts_us` | u64 | presentation time of the GOP's I-frame |
| `frame_start` | u32 | first frame index in GOP |
| `frame_count` | u16 | frames in GOP |
| `byte_offset` | u64 | offset of chunk in file (single-file mode) |
| `byte_length` | u32 | chunk length (all tiers) or base tier |
| `tier_offsets[tier_count]` | u32 | per-tier sub-offsets within chunk |
| `flags` | u16 | bit0 forced_keyframe(topology change), bit1 has_aux |

### 11.5 Track directory

Describes each elementary stream so the runtime wires the right decoder:

| Field | Type | Notes |
|---|---|---|
| `track_count` | u16 | |
| per track: `track_id` | u16 | |
| `track_type` | u8 | 0=geometry,1=texture-color,2=texture-aux,3=audio,4=metadata |
| `codec_fourcc` | u8[4] | e.g. `AV01`,`VP09`,`MSHO`(meshopt),`OPUS` |
| `tier` | u8 | which ABR rung (texture) |
| `codec_config_len` | u16 | |
| `codec_config[]` | u8 | e.g. AV1 sequence header / `VideoDecoder.configure` description |

### 11.6 Chunk layout (one GOP)

Each chunk is self-contained and independently decodable:

```
Chunk
├─ Chunk header
│    magic 'CNK0' | pts_start_us(u64) | frame_count(u16) | flags(u16)
│    gop_aabb_min[3],gop_aabb_max[3] (f32)   # local quant range
│    block_count(u16) | block_dir[block_count]  # {type,u8; track_id,u16; offset,u32; length,u32}
├─ Geometry block(s)
│    I-frame: intra mesh/splat (meshopt/draco payload) — the keyframe
│    P/B-frames: [frame_type(u8)][ref(u8)][residual_stream …]
├─ Texture block(s)  (per tier present in this chunk)
│    one closed video GOP: [EncodedVideoChunk headers][coded bitstream]
├─ Aux block(s)      (confidence/segmentation/depth mono video; optional)
├─ Audio block       (Opus packets for this span; optional)
└─ Metadata block    (markers, subtitles, per-frame bounding volumes; optional)
```

#### 11.6.1 Geometry block — mesh I-frame

| Field | Type | Notes |
|---|---|---|
| `vertex_count` | u32 | |
| `index_count` | u32 | |
| `attr_mask` | u16 | bit0 pos, bit1 normal, bit2 uv, bit3 color |
| `positions` | meshopt(u16×3 quantized) | dequant via chunk AABB + `quant_bits_pos` |
| `normals` | meshopt(oct) | optional |
| `uvs` | meshopt(u16×2) | optional |
| `indices` | meshopt(u32) | triangle list |

#### 11.6.2 Geometry block — mesh P/B-frame

| Field | Type | Notes |
|---|---|---|
| `frame_type` | u8 | 1=P, 2=B |
| `ref_lo`,`ref_hi` | u8 | anchor frame indices (B uses two) |
| `predictor` | u8 | 0=prev,1=prev+velocity |
| `changed_count` | u32 | vertices with non-zero residual (sparse) |
| `residuals` | entropy(Δquantized) | per changed vertex: index varint + Δxyz |

> P/B frames are **sparse**: only vertices whose residual exceeds the dead-zone are listed. Static
> regions cost near-zero bytes. Topology (indices) is **not** repeated — it persists from the I-frame.

#### 11.6.3 Geometry block — splat profile

I-frame: `splat_count(u32)` then packed `pos(3×f16)|scale(3×f16)|rot(4×i8 quat)|opacity(u8)|SH(k×
f16)`. P/B-frames: sparse attribute residuals + a birth/death list (`added[]`, `removed[]`).

### 11.7 Versioning and extensions

- **Version.** `version_major` breaks compatibility; `version_minor` is additive. A decoder MUST
  refuse a higher `version_major` and MUST tolerate a higher `version_minor`.
- **Extension blocks.** Any block whose `type` a decoder does not recognize MUST be **skipped** using
  its `length` (F10). This is how new block types (live hints, new aux tracks, neural residuals) ship
  without breaking old decoders. Unknown *required* extensions are signalled by a bit in
  `header_flags` so a decoder can fail loudly when it genuinely cannot play.
- **FourCC registry.** Codec/track FourCCs are centrally registered in this spec's appendix to avoid
  collisions.

### 11.8 Integrity and security

- `header_crc32` guards the header; each chunk MAY carry a `crc32` in its header for corruption
  detection.
- The demuxer MUST bounds-check every offset/length against the file/chunk size before use (N6) and
  MUST treat all fields as untrusted. No offset may be dereferenced without validation. No code path
  uses `eval` or constructs code from container data.

### 11.9 Why not just wrap MP4/Matroska?

MP4/Matroska could carry these as custom tracks, and a future bridge MAY do so for tooling interop.
But a purpose-built container keeps the header/index tiny, avoids box-parsing overhead on the
critical path, and lets geometry and texture share one GOP-aligned chunk with one PTS domain. The
spec stays simple enough to parse in a few KB of JS. [ASSERTED] A Matroska mapping is tracked as
optional interop work ([§16](#16-open-research-questions-and-risks)).
