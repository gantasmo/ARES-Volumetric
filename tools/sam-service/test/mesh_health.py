"""Check a mesh capture folder from avatar_mesh.py for topology and deformation problems.

For each keyframe/deformed group, verifies every deformed frame shares the key's vertex count
and face array, then measures how far the deformation has drifted: seam gap growth (vertices
that started coincident on the key), flipped and degenerate faces, edge length ratio, and new
vertex collisions that were not there on the key. Also reports the enclosed volume of every
mesh against the hull volume recorded in avatar-mesh.json.

    "<repo>/tools/ext/4danyone/venv/Scripts/python.exe" mesh_health.py <frames-dir> [--json out.json]

Runs on CPU only: numpy and scipy, no torch.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

FRAME_RE = re.compile(r"mesh-f(\d+)\.obj$")


def read_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def face_normals(verts, tris):
    """Raw (non-unit) face normals and their lengths (twice the triangle area)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    raw = np.cross(b - a, c - a)
    length = np.linalg.norm(raw, axis=1)
    return raw, length


def mesh_volume_litres(verts, tris):
    """Signed tetrahedron sum against the origin, absolute value, converted to litres."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return abs(signed) * 1000.0


def unique_edges(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def find_frames(frames_dir: Path):
    frames = {}
    for p in frames_dir.glob("mesh-f*.obj"):
        m = FRAME_RE.match(p.name)
        if m:
            frames[int(m.group(1)) - 1] = p
    return dict(sorted(frames.items()))


def build_groups(meta_frames, frame_paths):
    """List of (key_index, [deformed indices in order]) covering every frame in order."""
    groups = []
    current_key = None
    for entry in meta_frames:
        idx = entry["frame"]
        if idx not in frame_paths:
            continue
        if entry["kind"] == "key":
            current_key = idx
            groups.append((current_key, []))
        elif current_key is not None:
            groups[-1][1].append(idx)
    return groups


def seam_pairs_at(verts, tol=1e-5):
    tree = cKDTree(verts)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    return pairs


def frame_metrics(key_verts, key_tris, key_face_raw, key_face_len, key_edges, key_edge_len,
                   seam_pairs, seam_key_gap, seam_pair_set, verts):
    out = {}

    # seam gap on the deformed frame, over the pairs found on the key
    if len(seam_pairs):
        gap = np.linalg.norm(verts[seam_pairs[:, 0]] - verts[seam_pairs[:, 1]], axis=1) * 1000.0
    else:
        gap = np.array([])
    out["seam_gap_mm"] = percentiles(gap)

    # flipped / degenerate faces
    raw, length = face_normals(verts, key_tris)
    degenerate = (length < 1e-12) | (key_face_len < 1e-12)
    valid = ~degenerate
    if valid.any():
        dot = np.einsum("ij,ij->i", raw[valid], key_face_raw[valid])
        unit_dot = dot / (length[valid] * key_face_len[valid])
        flipped_frac = float((unit_dot < 0).mean())
    else:
        flipped_frac = 0.0
    out["flipped"] = flipped_frac
    out["degenerate"] = float(degenerate.mean()) if len(degenerate) else 0.0

    # edge length ratio
    d = verts[key_edges[:, 0]] - verts[key_edges[:, 1]]
    def_len = np.linalg.norm(d, axis=1)
    ok = key_edge_len > 1e-9
    if ok.any():
        ratio = def_len[ok] / key_edge_len[ok]
        out["edge_ratio"] = {
            "p50": float(np.percentile(ratio, 50)),
            "p99": float(np.percentile(ratio, 99)),
            "max": float(ratio.max()),
        }
        out["frac_edges_bad"] = float(((ratio > 2) | (ratio < 0.5)).mean())
    else:
        out["edge_ratio"] = {"p50": 0.0, "p99": 0.0, "max": 0.0}
        out["frac_edges_bad"] = 0.0

    # new duplicate vertices: coincide now (1e-4 m) but not on the key
    tree = cKDTree(verts)
    def_pairs = tree.query_pairs(1e-4, output_type="ndarray")
    if len(def_pairs):
        keys = def_pairs[:, 0].astype(np.int64) * len(verts) + def_pairs[:, 1].astype(np.int64)
        is_new = ~np.isin(keys, seam_pair_set)
        new_pairs = def_pairs[is_new]
        involved = np.unique(new_pairs.reshape(-1)) if len(new_pairs) else np.array([], dtype=np.int64)
        out["dup_new"] = float(len(involved)) / len(verts)
    else:
        out["dup_new"] = 0.0

    out["mesh_litres"] = mesh_volume_litres(verts, key_tris)
    return out


def summarize(rows, key):
    vals = [(r["frame"], r[key]) for r in rows if key in r]
    if not vals:
        return None
    idx = np.array([v[0] for v in vals])
    arr = np.array([v[1] for v in vals], dtype=np.float64)
    worst = int(idx[np.argmax(arr)])
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "worst_frame": worst,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    meta = json.loads((a.frames_dir / "avatar-mesh.json").read_text())
    voxel = meta.get("voxel")
    frame_paths = find_frames(a.frames_dir)
    groups = build_groups(meta["frames"], frame_paths)
    meta_by_frame = {e["frame"]: e for e in meta["frames"]}

    rows = []
    broken_groups = []
    for key_idx, deformed_idxs in groups:
        key_verts, key_tris = read_obj(frame_paths[key_idx])
        key_face_raw, key_face_len = face_normals(key_verts, key_tris)
        key_edges = unique_edges(key_tris)
        key_edge_len = np.linalg.norm(key_verts[key_edges[:, 0]] - key_verts[key_edges[:, 1]], axis=1)
        seam_pairs = seam_pairs_at(key_verts)
        seam_pair_set = (seam_pairs[:, 0].astype(np.int64) * len(key_verts)
                          + seam_pairs[:, 1].astype(np.int64)) if len(seam_pairs) else np.array([], dtype=np.int64)

        entry = meta_by_frame.get(key_idx, {})
        hull_litres = voxel is not None and "voxels" in entry and voxel ** 3 * entry["voxels"] * 1000.0
        rows.append({
            "frame": key_idx,
            "kind": "key",
            "frames_since_key": 0,
            "seam_gap_mm": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "flipped": 0.0,
            "degenerate": 0.0,
            "edge_ratio": {"p50": 1.0, "p99": 1.0, "max": 1.0},
            "frac_edges_bad": 0.0,
            "dup_new": 0.0,
            "mesh_litres": mesh_volume_litres(key_verts, key_tris),
            "hull_litres": hull_litres if hull_litres is not False else None,
        })

        for since, d_idx in enumerate(deformed_idxs, start=1):
            verts, tris = read_obj(frame_paths[d_idx])
            broken = verts.shape[0] != key_verts.shape[0] or tris.shape != key_tris.shape \
                or not np.array_equal(tris, key_tris)
            if broken:
                broken_groups.append((key_idx, d_idx))
                rows.append({
                    "frame": d_idx,
                    "kind": "deformed",
                    "frames_since_key": since,
                    "broken_topology": True,
                })
                continue

            m = frame_metrics(key_verts, key_tris, key_face_raw, key_face_len, key_edges, key_edge_len,
                               seam_pairs, None, seam_pair_set, verts)
            entry = meta_by_frame.get(d_idx, {})
            hull_litres = voxel is not None and "voxels" in entry and voxel ** 3 * entry["voxels"] * 1000.0
            m["frame"] = d_idx
            m["kind"] = "deformed"
            m["frames_since_key"] = since
            m["hull_litres"] = hull_litres if hull_litres is not False else None
            rows.append(m)

    rows.sort(key=lambda r: r["frame"])

    for r in rows:
        if r.get("broken_topology"):
            print(f"frame {r['frame']:5d} {r['kind']:9s} since_key {r['frames_since_key']:3d}  BROKEN TOPOLOGY")
            continue
        sg = r["seam_gap_mm"]
        er = r["edge_ratio"]
        hull = f"{r['hull_litres']:.1f}" if r.get("hull_litres") is not None else "n/a"
        print(f"frame {r['frame']:5d} {r['kind']:9s} since_key {r['frames_since_key']:3d}  "
              f"seam mm p50/p95/max {sg['p50']:6.1f}/{sg['p95']:6.1f}/{sg['max']:7.1f}  "
              f"flipped {r['flipped'] * 100:5.2f}%  degen {r['degenerate'] * 100:5.2f}%  "
              f"edge p99/max {er['p99']:5.2f}/{er['max']:6.2f}  bad_edges {r['frac_edges_bad'] * 100:5.2f}%  "
              f"dup_new {r['dup_new'] * 100:5.2f}%  mesh {r['mesh_litres']:7.1f} L  hull {hull} L")

    deformed_rows = [r for r in rows if r["kind"] == "deformed" and not r.get("broken_topology")]
    key_rows = [r for r in rows if r["kind"] == "key"]
    all_valid_rows = [r for r in rows if not r.get("broken_topology")]

    # flatten nested metrics for summarize()
    for r in deformed_rows:
        r["seam_gap_p50_mm_flat"] = r["seam_gap_mm"]["p50"]
        r["seam_gap_p95_mm_flat"] = r["seam_gap_mm"]["p95"]
        r["seam_gap_max_mm_flat"] = r["seam_gap_mm"]["max"]
        r["edge_p99_flat"] = r["edge_ratio"]["p99"]
        r["edge_max_flat"] = r["edge_ratio"]["max"]

    summary = {
        "keys": len(key_rows),
        "deformed": len([r for r in rows if r["kind"] == "deformed"]),
        "broken_groups": [{"key": k, "frame": d} for k, d in broken_groups],
        "seam_gap_p50_mm": summarize(deformed_rows, "seam_gap_p50_mm_flat"),
        "seam_gap_p95_mm": summarize(deformed_rows, "seam_gap_p95_mm_flat"),
        "seam_gap_max_mm": summarize(deformed_rows, "seam_gap_max_mm_flat"),
        "flipped": summarize(deformed_rows, "flipped"),
        "degenerate": summarize(deformed_rows, "degenerate"),
        "edge_p99": summarize(deformed_rows, "edge_p99_flat"),
        "edge_max": summarize(deformed_rows, "edge_max_flat"),
        "frac_edges_bad": summarize(deformed_rows, "frac_edges_bad"),
        "dup_new": summarize(deformed_rows, "dup_new"),
        "mesh_litres_all_frames": summarize(all_valid_rows, "mesh_litres"),
        "hull_litres_all_frames": summarize(
            [r for r in all_valid_rows if r.get("hull_litres") is not None], "hull_litres"),
    }

    print()
    print(f"keys {summary['keys']}  deformed {summary['deformed']}  broken_groups {len(broken_groups)}")
    for label, key in (
        ("seam_gap p50 mm", "seam_gap_p50_mm"),
        ("seam_gap p95 mm", "seam_gap_p95_mm"),
        ("seam_gap max mm", "seam_gap_max_mm"),
        ("flipped", "flipped"),
        ("degenerate", "degenerate"),
        ("edge p99", "edge_p99"),
        ("edge max", "edge_max"),
        ("frac_edges_bad", "frac_edges_bad"),
        ("dup_new", "dup_new"),
        ("mesh_litres (all frames)", "mesh_litres_all_frames"),
        ("hull_litres (all frames)", "hull_litres_all_frames"),
    ):
        s = summary[key]
        if s is None:
            print(f"  {label:26s} n/a")
        else:
            print(f"  {label:26s} min {s['min']:8.3f}  median {s['median']:8.3f}  "
                  f"p95 {s['p95']:8.3f}  max {s['max']:8.3f}  worst frame {s['worst_frame']}")

    if a.json:
        out = {"frames": rows, "summary": summary}
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")

    return 1 if broken_groups else 0


if __name__ == "__main__":
    sys.exit(main())
