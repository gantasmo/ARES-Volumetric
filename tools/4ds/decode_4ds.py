#!/usr/bin/env python
"""
decode_4ds.py — Windows decoder host for 4DViews ".4ds" volumetric captures (ARES Task H).

Loads the proprietary BridgeCodec4DS.dll (4DViews' native codec, used under your own
4DViews SDK license; the DLL itself is never committed — copy yours into tools/4ds/bin/) via
ctypes, decodes a .4ds sequence frame-by-frame, and writes a frames-dir that ARES's own
encoder CLI (`packages/encoder/src/cli.ts` `ares encode <frames-dir>`) can bake directly:
one Wavefront OBJ (positions + UVs, shared per-vertex indices) and one atlas PNG per frame,
plus a manifest.json.

DLL contract (as exposed by 4DViews' own Unity bridge, unity4dv.Bridge4DS):
    int   CreateSequence(int key, char* dataPath /*ANSI*/, int rangeBegin, int rangeEnd,
                          int outRangeMode, char* errorMsg /*caller-allocated, 300 bytes*/)
    void  DestroySequence(int key)
    void  Play(int key, bool on)
    void  Stop(int key)
    void  GotoFrame(int key, int frame)
    void  SetSpeed(int key, float speedRatio)
    void  SetChunkBufferMaxSize(int key, int size)
    void  SetMeshBufferMaxSize(int key, int size)
    int   GetSequenceNbFrames(int key)
    float GetSequenceFramerate(int key)
    int   GetSequenceMaxVertices(int key)     // 0 -> default 65535
    int   GetSequenceMaxTriangles(int key)    // 0 -> default 65535
    int   GetTextureSize(int key)             // 0 -> default 1024 (texture is size x size)
    int   GetTextureEncoding(int key)         // 1|100=DXT1, 5|120=ETC_RGB4, 6|130=PVRTC_RGB4,
                                               // 4|131=PVRTC_RGB2, 8|164=ASTC_8x8, else DXT1
    int   GetSequenceCurrentFrame(int key)
    int   UpdateModel(int key, void* vertices, void* uvs, void* triangles, void* texture,
                       void* normals, int lastModelId, int* nbVertices, int* nbTriangles)
                       // returns the decoded frame's model id, or -1 when no new frame is ready

Buffers are caller-allocated and must hold (per the whole-sequence maxima, since the plugin
does not resize per frame): vertices float32 x3 x MaxVertices, uvs float32 x2 x MaxVertices,
triangles int32 x3 x MaxTriangles, normals float32 x3 x MaxVertices, texture bytes =
TextureSize^2 / 2 for DXT1. Output nbVertices/nbTriangles (returned by ref) are the ACTUAL
counts for that frame — the arrays above must be trimmed to them before use.

OUT_RANGE_MODE: Loop=0, Reverse=1, Stop=2, Hide=3 — Stop is used here for batch decode.

Usage:
    python decode_4ds.py <input.4ds> -o <out-dir> [--max-frames N] [--dll <path>] [--info]
    python decode_4ds.py <input.4ds> --info

SEAM WELD (Task J, on by default; --no-weld-seams to disable): the source codec's UV atlas is
REPACKED at irregular "topology reset" frames (runs of 1-24 frames of stable index-buffer content
between resets, measured on the real clip). A sparse set of UV-seam vertices — duplicated by
position so each side of a cut can carry its own UV — drifts apart WITHIN a topology run (up to
~28mm) and snaps back to exactly 0mm at every run's keyframe, producing visible pulsing cracks.
Per run: detect the reset (index-buffer content hash changes vs the previous frame), then at the
keyframe group vertices by EXACT position bits and restrict to groups with >=2 members touching an
OPEN (boundary) edge of the unwelded mesh — this selects true UV-seam duplicates (closing them
collapses ~10,000+ boundary edges to single digits) while excluding vertices that coincide by
chance deep in the (otherwise watertight) interior, which must NOT be welded. The group assignment
is computed once per run and re-applied (vectorized centroid) to EVERY frame in the run, including
the keyframe. UVs are never touched.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from ctypes import c_int, c_float, c_void_p, c_char_p, POINTER, byref, create_string_buffer

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DLL_LOCAL = os.path.join(HERE, "bin", "BridgeCodec4DS.dll")
DEFAULT_DLL_FALLBACK = os.environ.get("FOURDS_DLL", "")

OUT_RANGE_STOP = 2

TEX_ENC_MAP = {
    1: "DXT1", 100: "DXT1",
    5: "ETC_RGB4", 120: "ETC_RGB4",
    6: "PVRTC_RGB4", 130: "PVRTC_RGB4",
    4: "PVRTC_RGB2", 131: "PVRTC_RGB2",
    8: "ASTC_8x8", 164: "ASTC_8x8",
}


def tex_enc_name(code: int) -> str:
    # Unity's DataSource4DS switch defaults to DXT1 for any unrecognized code.
    return TEX_ENC_MAP.get(code, "DXT1")


class Bridge:
    def __init__(self, dll_path: str):
        if not os.path.isfile(dll_path):
            raise FileNotFoundError(f"codec DLL not found: {dll_path}")
        self.dll_path = dll_path
        self.dll = ctypes.WinDLL(dll_path)
        d = self.dll

        d.CreateSequence.argtypes = [c_int, c_char_p, c_int, c_int, c_int, c_void_p]
        d.CreateSequence.restype = c_int

        d.DestroySequence.argtypes = [c_int]
        d.DestroySequence.restype = None

        d.Play.argtypes = [c_int, c_int]
        d.Play.restype = None

        d.Stop.argtypes = [c_int]
        d.Stop.restype = None

        d.GotoFrame.argtypes = [c_int, c_int]
        d.GotoFrame.restype = None

        d.SetSpeed.argtypes = [c_int, c_float]
        d.SetSpeed.restype = None

        d.SetChunkBufferMaxSize.argtypes = [c_int, c_int]
        d.SetChunkBufferMaxSize.restype = None

        d.SetMeshBufferMaxSize.argtypes = [c_int, c_int]
        d.SetMeshBufferMaxSize.restype = None

        d.GetSequenceNbFrames.argtypes = [c_int]
        d.GetSequenceNbFrames.restype = c_int

        d.GetSequenceFramerate.argtypes = [c_int]
        d.GetSequenceFramerate.restype = c_float

        d.GetSequenceMaxVertices.argtypes = [c_int]
        d.GetSequenceMaxVertices.restype = c_int

        d.GetSequenceMaxTriangles.argtypes = [c_int]
        d.GetSequenceMaxTriangles.restype = c_int

        d.GetTextureSize.argtypes = [c_int]
        d.GetTextureSize.restype = c_int

        d.GetTextureEncoding.argtypes = [c_int]
        d.GetTextureEncoding.restype = c_int

        d.GetSequenceCurrentFrame.argtypes = [c_int]
        d.GetSequenceCurrentFrame.restype = c_int

        d.UpdateModel.argtypes = [
            c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, POINTER(c_int), POINTER(c_int),
        ]
        d.UpdateModel.restype = c_int

    def create_sequence(self, path: str, range_begin=0, range_end=-1, out_range_mode=OUT_RANGE_STOP):
        err = create_string_buffer(300)
        path_ansi = path.encode("mbcs")
        handle = self.dll.CreateSequence(0, path_ansi, range_begin, range_end, out_range_mode, err)
        err_text = err.value.decode("mbcs", errors="replace").strip()
        if handle == 0:
            raise RuntimeError(f"CreateSequence failed for {path!r}: {err_text or '(no error message)'}")
        return handle, err_text


def query_sequence_info(bridge: Bridge, handle: int) -> dict:
    nb_frames = bridge.dll.GetSequenceNbFrames(handle)
    fps = bridge.dll.GetSequenceFramerate(handle)
    max_verts = bridge.dll.GetSequenceMaxVertices(handle) or 65535
    max_tris = bridge.dll.GetSequenceMaxTriangles(handle) or 65535
    tex_size = bridge.dll.GetTextureSize(handle) or 1024
    tex_enc_code = bridge.dll.GetTextureEncoding(handle)
    return {
        "nbFrames": nb_frames,
        "framerate": fps,
        "maxVertices": max_verts,
        "maxTriangles": max_tris,
        "textureSize": tex_size,
        "textureEncodingCode": tex_enc_code,
        "textureEncoding": tex_enc_name(tex_enc_code),
    }


def decode_dxt1(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Vectorized BC1/DXT1 -> RGBA8888. `data` is a flat uint8 array of width*height/2 bytes.
    Returns an (height, width, 4) uint8 array in the SAME row order as the input byte stream
    (block 0 = bytes[0:8] = the first 4x4 texel block in the stream, laid out left-to-right
    then top-to-bottom of the BLOCK GRID as stored — orientation relative to "up" is decided
    by the caller, see --flip-tex-v)."""
    assert width % 4 == 0 and height % 4 == 0
    bw, bh = width // 4, height // 4
    nblocks = bw * bh
    raw = data[: nblocks * 8].reshape(nblocks, 8)

    c0 = raw[:, 0].astype(np.uint32) | (raw[:, 1].astype(np.uint32) << 8)
    c1 = raw[:, 2].astype(np.uint32) | (raw[:, 3].astype(np.uint32) << 8)
    idx_bytes = raw[:, 4:8]  # (nblocks,4): byte i = texel row i (LE 32-bit index field)

    def unpack565(c):
        r = (c >> 11) & 0x1F
        g = (c >> 5) & 0x3F
        b = c & 0x1F
        r = (r * 255 + 15) // 31
        g = (g * 255 + 31) // 63
        b = (b * 255 + 15) // 31
        return r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)

    r0, g0, b0 = unpack565(c0)
    r1, g1, b1 = unpack565(c1)
    c0_gt_c1 = c0 > c1

    r2_4, g2_4, b2_4 = (2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3
    r3_4, g3_4, b3_4 = (r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3
    r2_3, g2_3, b2_3 = (r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2

    r2 = np.where(c0_gt_c1, r2_4, r2_3).astype(np.uint8)
    g2 = np.where(c0_gt_c1, g2_4, g2_3).astype(np.uint8)
    b2 = np.where(c0_gt_c1, b2_4, b2_3).astype(np.uint8)
    r3 = np.where(c0_gt_c1, r3_4, 0).astype(np.uint8)
    g3 = np.where(c0_gt_c1, g3_4, 0).astype(np.uint8)
    b3 = np.where(c0_gt_c1, b3_4, 0).astype(np.uint8)
    a3 = np.where(c0_gt_c1, 255, 0).astype(np.uint8)
    a_opaque = np.full(nblocks, 255, dtype=np.uint8)

    pal_r = np.stack([r0.astype(np.uint8), r1.astype(np.uint8), r2, r3], axis=1)  # (nblocks,4)
    pal_g = np.stack([g0.astype(np.uint8), g1.astype(np.uint8), g2, g3], axis=1)
    pal_b = np.stack([b0.astype(np.uint8), b1.astype(np.uint8), b2, b3], axis=1)
    pal_a = np.stack([a_opaque, a_opaque, a_opaque, a3], axis=1)

    codes = np.empty((nblocks, 4, 4), dtype=np.uint8)  # (block, row, col)
    for col in range(4):
        codes[:, :, col] = (idx_bytes >> (2 * col)) & 3

    br = np.arange(nblocks)[:, None, None]
    out = np.empty((nblocks, 4, 4, 4), dtype=np.uint8)
    out[..., 0] = pal_r[br, codes]
    out[..., 1] = pal_g[br, codes]
    out[..., 2] = pal_b[br, codes]
    out[..., 3] = pal_a[br, codes]

    img = np.empty((height, width, 4), dtype=np.uint8)
    block_row = np.arange(nblocks) // bw
    block_col = np.arange(nblocks) % bw
    for by in range(4):
        for bx in range(4):
            img[block_row * 4 + by, block_col * 4 + bx] = out[:, by, bx]
    return img


def hash_indices(idx: np.ndarray) -> bytes:
    """Stable content fingerprint of a trimmed triangle index array — a topology reset is a
    change in this hash vs the previous frame (irregular period; NOT a fixed GOP cadence)."""
    return hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int32).tobytes()).digest()


def compute_boundary_vertices(idx: np.ndarray, n_v: int) -> tuple[np.ndarray, int]:
    """Per-vertex bool: True if the vertex touches >=1 OPEN (boundary) edge of the UNWELDED mesh —
    an edge owned by exactly one triangle. UV-seam duplicate vertices sit exactly on these edges
    (each side of a cut keeps its own vertex index, so the edge along the cut is only ever
    referenced by the triangles on ONE side until the duplicate is welded). Also returns the raw
    boundary-edge count (diagnostic)."""
    if idx.shape[0] == 0:
        return np.zeros(n_v, dtype=bool), 0
    edges = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    is_boundary = np.zeros(n_v, dtype=bool)
    if boundary.size:
        is_boundary[boundary.reshape(-1)] = True
    return is_boundary, int(boundary.shape[0])


def compute_seam_weld_groups(positions: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Per-vertex weld-group id (>=0) or -1 (not welded), computed ONCE per topology run from the
    run's keyframe positions + index buffer (see module docstring for the selection rule). Returns
    (weld_id, boundary_edge_count, seam_group_count) — the latter two are diagnostics only."""
    n_v = positions.shape[0]
    is_boundary, n_boundary_edges = compute_boundary_vertices(idx, n_v)
    if not is_boundary.any():
        return np.full(n_v, -1, dtype=np.int64), n_boundary_edges, 0

    pos_c = np.ascontiguousarray(positions, dtype=np.float32)
    pos_view = pos_c.view([("x", np.float32), ("y", np.float32), ("z", np.float32)]).reshape(n_v)
    _, group_id, group_counts = np.unique(pos_view, return_inverse=True, return_counts=True)
    group_id = np.asarray(group_id).reshape(n_v)

    boundary_count_per_group = np.bincount(group_id[is_boundary], minlength=group_counts.shape[0])
    seam_group = boundary_count_per_group >= 2  # per-group: >=2 boundary members -> real seam duplicate
    weld_id = np.where(is_boundary & seam_group[group_id], group_id, -1).astype(np.int64)
    return weld_id, n_boundary_edges, int(seam_group.sum())


def apply_seam_weld(positions: np.ndarray, weld_id: np.ndarray) -> np.ndarray:
    """Replace each welded vertex's position with its weld-group centroid IN THIS FRAME (vectorized
    bincount mean). No-op (returns positions unchanged) when weld_id has no selected groups."""
    valid = weld_id >= 0
    if not np.any(valid):
        return positions
    ids = weld_id[valid]
    uniq_ids, remapped, counts = np.unique(ids, return_inverse=True, return_counts=True)
    remapped = np.asarray(remapped).reshape(-1)
    k = uniq_ids.shape[0]
    sums = np.zeros((k, 3), dtype=np.float64)
    for ax in range(3):
        sums[:, ax] = np.bincount(remapped, weights=positions[valid, ax].astype(np.float64), minlength=k)
    centroids = (sums / counts[:, None]).astype(positions.dtype)
    positions[valid] = centroids[remapped]
    return positions


def write_obj(path: str, positions: np.ndarray, uvs: np.ndarray, indices: np.ndarray,
              mirror_x: bool, flip_uv_v: bool) -> None:
    nb_v = positions.shape[0]
    px = -positions[:, 0] if mirror_x else positions[:, 0]
    py = positions[:, 1]
    pz = positions[:, 2]
    uv_v = 1.0 - uvs[:, 1] if flip_uv_v else uvs[:, 1]

    lines = []
    lines.append(f"# 4ds decode_4ds.py — {nb_v} verts, {indices.shape[0]} tris")
    for i in range(nb_v):
        lines.append(f"v {px[i]:.6f} {py[i]:.6f} {pz[i]:.6f}")
    for i in range(nb_v):
        lines.append(f"vt {uvs[i, 0]:.6f} {uv_v[i]:.6f}")
    tri = indices + 1  # OBJ is 1-based
    if mirror_x:
        # Negate X flips handedness; reverse winding together to keep front-faces front-facing.
        tri = tri[:, ::-1]
    for a, b, c in tri:
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Decode a 4DViews .4ds volumetric capture into a frames-dir for the ARES encoder.")
    ap.add_argument("input", help="path to the .4ds file")
    ap.add_argument("-o", "--out", help="output frames directory")
    ap.add_argument("--max-frames", type=int, default=None, help="decode at most N frames")
    ap.add_argument("--dll", default=None, help="path to BridgeCodec4DS.dll")
    ap.add_argument("--info", action="store_true", help="print sequence info as JSON and exit (no decode)")
    ap.add_argument("--stall-timeout", type=float, default=10.0, help="seconds with no new frame before giving up")
    ap.add_argument("--flip-tex-v", dest="flip_tex_v", action="store_true", default=True,
                     help="flip decoded texture rows vertically before saving PNG (default: on)")
    ap.add_argument("--no-flip-tex-v", dest="flip_tex_v", action="store_false")
    ap.add_argument("--flip-uv-v", action="store_true", default=False,
                     help="flip UV V in the OBJ instead of flipping the texture (mutually exclusive with --flip-tex-v in intent)")
    ap.add_argument("--mirror-x", action="store_true", default=False,
                     help="negate X and reverse triangle winding (left-handed -> right-handed)")
    ap.add_argument("--weld-seams", dest="weld_seams", action="store_true", default=True,
                     help="weld duplicated UV-seam vertices per topology run (default: on)")
    ap.add_argument("--no-weld-seams", dest="weld_seams", action="store_false",
                     help="disable seam welding (debug / A-B comparison)")
    ap.add_argument("--seam-debug", action="store_true", default=False,
                     help="print boundary-edge/seam-group diagnostics at every topology reset")
    args = ap.parse_args()

    if args.flip_uv_v and args.flip_tex_v:
        # Exactly one V-flip may be active: both at once double-flips (net wrong output).
        # --flip-tex-v defaults ON, so --flip-uv-v must be paired with --no-flip-tex-v.
        ap.error("--flip-uv-v requires --no-flip-tex-v (only one V-flip may be active)")

    dll_path = args.dll or (DEFAULT_DLL_LOCAL if os.path.isfile(DEFAULT_DLL_LOCAL) else DEFAULT_DLL_FALLBACK)
    bridge = Bridge(dll_path)

    input_path = os.path.abspath(args.input)
    handle, _ = bridge.create_sequence(input_path)
    try:
        info = query_sequence_info(bridge, handle)
        if args.info:
            print(json.dumps(info, indent=2))
            return

        if info["textureEncoding"] != "DXT1":
            raise RuntimeError(
                f"unsupported texture encoding for v1: code={info['textureEncodingCode']} "
                f"-> {info['textureEncoding']} (only DXT1 is supported; desktop captures should be DXT1)"
            )

        if not args.out:
            raise SystemExit("error: -o/--out is required for decode (use --info to only inspect the sequence)")
        out_dir = os.path.abspath(args.out)
        os.makedirs(out_dir, exist_ok=True)

        max_verts = info["maxVertices"]
        max_tris = info["maxTriangles"]
        tex_size = info["textureSize"]
        nb_frames_total = info["nbFrames"]
        max_frames = min(args.max_frames, nb_frames_total) if args.max_frames is not None else nb_frames_total

        print(f"[decode_4ds] {input_path}")
        print(f"[decode_4ds] nbFrames={nb_frames_total} fps={info['framerate']:.3f} "
              f"maxVerts={max_verts} maxTris={max_tris} texSize={tex_size} texEnc={info['textureEncoding']} "
              f"-> decoding {max_frames} frame(s)")

        d = bridge.dll
        d.SetChunkBufferMaxSize(handle, 180)
        d.SetMeshBufferMaxSize(handle, 10)
        d.SetSpeed(handle, 1.0)

        vertices = np.zeros(max_verts * 3, dtype=np.float32)
        uvs = np.zeros(max_verts * 2, dtype=np.float32)
        triangles = np.zeros(max_tris * 3, dtype=np.int32)
        normals = np.zeros(max_verts * 3, dtype=np.float32)
        tex_bytes_n = tex_size * tex_size // 2
        texture = np.zeros(tex_bytes_n, dtype=np.uint8)

        v_ptr = vertices.ctypes.data_as(c_void_p)
        uv_ptr = uvs.ctypes.data_as(c_void_p)
        tri_ptr = triangles.ctypes.data_as(c_void_p)
        tex_ptr = texture.ctypes.data_as(c_void_p)
        norm_ptr = normals.ctypes.data_as(c_void_p)

        last_model_id = -1
        frames_written = 0
        t_start = time.time()
        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])

        # Seam-weld run state (see module docstring): a "run" = a span of frames sharing the same
        # index-buffer content. weld_id (vertex -> weld-group id, or -1) is recomputed only when
        # the content hash changes, then re-applied to every frame's own positions in the run.
        prev_idx_hash = None
        run_weld_id: np.ndarray | None = None
        run_count = 0

        # Explicit per-frame seek (Unity's GotoFrame idiom: Play(false) -> GotoFrame -> Play(true)
        # -> poll) rather than one continuous Play() session polled in a loop. Batch decode work
        # per frame (OBJ text + DXT1->PNG) is far slower than real-time playback, and a single
        # continuous Play() session races its background real-time decode thread against slow
        # disk consumption — measured: model ids went non-monotonic and duplicate frame content
        # appeared once consumption fell behind. Seeking explicitly to frame N each iteration
        # makes decode correctness independent of how slow the consumer is.
        while frames_written < max_frames:
            d.Play(handle, 0)
            d.GotoFrame(handle, frames_written)
            d.Play(handle, 1)

            deadline = time.time() + args.stall_timeout
            while True:
                nb_v = c_int(0)
                nb_t = c_int(0)
                model_id = d.UpdateModel(handle, v_ptr, uv_ptr, tri_ptr, tex_ptr, norm_ptr,
                                          last_model_id, byref(nb_v), byref(nb_t))
                if model_id != -1 and model_id != last_model_id:
                    break
                if time.time() > deadline:
                    cur_frame = d.GetSequenceCurrentFrame(handle)
                    raise RuntimeError(
                        f"stalled: no new frame for {args.stall_timeout}s while seeking to frame "
                        f"{frames_written}. last_model_id={last_model_id} current_model_id={model_id} "
                        f"GetSequenceCurrentFrame={cur_frame} nbVertices={nb_v.value} nbTriangles={nb_t.value}"
                    )
                time.sleep(0.003)

            last_model_id = model_id
            n_v, n_t = nb_v.value, nb_t.value

            pos = vertices[: n_v * 3].reshape(n_v, 3)
            uv = uvs[: n_v * 2].reshape(n_v, 2)
            idx = triangles[: n_t * 3].reshape(n_t, 3)

            if args.weld_seams:
                idx_hash = hash_indices(idx)
                if prev_idx_hash is None or idx_hash != prev_idx_hash:
                    run_weld_id, n_boundary, n_seam_groups = compute_seam_weld_groups(pos, idx)
                    run_count += 1
                    if args.seam_debug:
                        print(f"[decode_4ds] seam-weld: topology reset at frame {frames_written} "
                              f"(run #{run_count}) — boundary_edges={n_boundary} seam_groups={n_seam_groups} "
                              f"welded_verts={int((run_weld_id >= 0).sum())}/{n_v}")
                prev_idx_hash = idx_hash
                if run_weld_id is not None:
                    pos = apply_seam_weld(pos, run_weld_id)

            global_min = np.minimum(global_min, pos.min(axis=0))
            global_max = np.maximum(global_max, pos.max(axis=0))

            obj_path = os.path.join(out_dir, f"frame_{frames_written:05d}.obj")
            write_obj(obj_path, pos, uv, idx, mirror_x=args.mirror_x, flip_uv_v=args.flip_uv_v)

            img = decode_dxt1(texture, tex_size, tex_size)
            if args.flip_tex_v:
                img = img[::-1]
            Image.fromarray(img, mode="RGBA").save(os.path.join(out_dir, f"atlas_{frames_written:05d}.png"))

            frames_written += 1
            if frames_written % 30 == 0 or frames_written == max_frames:
                elapsed = time.time() - t_start
                print(f"[decode_4ds] {frames_written}/{max_frames} frames "
                      f"({frames_written / elapsed:.1f} fps decode, model_id={model_id})")

        d.Stop(handle)
        elapsed = time.time() - t_start

        manifest = {
            "source": input_path,
            "nbFramesTotal": nb_frames_total,
            "nbFramesDecoded": frames_written,
            "fps": info["framerate"],
            "textureSize": tex_size,
            "textureEncoding": info["textureEncoding"],
            "maxVertices": max_verts,
            "maxTriangles": max_tris,
            "decodeSeconds": elapsed,
            "decodeFramesPerSec": frames_written / elapsed if elapsed > 0 else None,
            "bounds": {"min": global_min.tolist(), "max": global_max.tolist()},
            "transforms": {"flipTexV": args.flip_tex_v, "flipUvV": args.flip_uv_v, "mirrorX": args.mirror_x},
            "seamWeld": {"enabled": args.weld_seams, "topologyRuns": run_count},
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[decode_4ds] wrote {frames_written} frame(s) to {out_dir} in {elapsed:.2f}s "
              f"({frames_written / elapsed:.2f} fps decode)")
        print(f"[decode_4ds] bounds min={global_min.tolist()} max={global_max.tolist()}")
        if args.weld_seams:
            print(f"[decode_4ds] seam-weld: {run_count} topology run(s) detected and welded "
                  f"(--no-weld-seams to disable)")
    finally:
        bridge.dll.DestroySequence(handle)


if __name__ == "__main__":
    main()
