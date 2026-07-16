# SAM 3D Body on a free Colab T4

Owner-facing doc for `ares/tools/sam3d-body-colab.ipynb` — answers "can I have it running in a
Colab instance?" for Meta's **SAM 3D Body** model. Every claim below was verified against the
live GitHub repo / Hugging Face API on 2026-07-10 (URLs inline).

## What it does

Single photo of a person in → rigged, **untextured** 3D human mesh out. Concretely:

- **Model**: [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta Superintelligence
  Labs, checkpoints released 2025-11-19, paper [arXiv 2602.15989](https://arxiv.org/abs/2602.15989)).
- **Mesh**: Momentum Human Rig (MHR) topology — 18,439 vertices, 70 body+hand joints, fixed
  triangle topology (same faces every time, only vertex positions change per photo).
- **Pipeline**: ViTDet-H detector finds the person → SAM 3D Body's DINOv3-H+ (840M param)
  encoder-decoder predicts MHR pose/shape/joints → MoGe2 estimates camera field-of-view for
  scale/depth.
- **Notebook**: `ares/tools/sam3d-body-colab.ipynb`, adapted from the official
  [`notebook/demo_human.ipynb`](https://github.com/facebookresearch/sam-3d-body/blob/main/notebook/demo_human.ipynb)
  in the facebookresearch repo.

### What the output is NOT

- **Not textured.** The mesh has no color/material — it's geometry (vertices + fixed faces) plus
  a skeleton (joint positions + rotations). If you want a colored/textured human, you'd need to
  bake the original photo onto the UVs yourself (not provided here), or use a different model.
- **Not SAM 3D Objects.** [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) is
  Meta's *other* SAM 3D model — it reconstructs full textured 3D shape/geometry for general
  objects and scenes, needs ~32 GB of GPU VRAM, and is effectively cloud-only. It is unrelated
  code from SAM 3D Body despite the shared "SAM 3D" branding and today's task is specifically the
  Body model.
- **Not from AmmarkoV/SAM3DBody-cpp.** An earlier research pass in this project referenced
  "`notebook/demo_human.ipynb` from the AmmarkoV/SAM3DBody-cpp repo" as the plan of record — that
  was incorrect. [AmmarkoV/SAM3DBody-cpp](https://github.com/AmmarkoV/SAM3DBody-cpp) is real (MIT
  licensed, a from-scratch C++/ONNX/ggml runtime for real-time SAM 3D Body inference) but it has
  **no notebook of any kind** in its tree — verified by walking its full git tree. The
  `notebook/demo_human.ipynb` that actually exists lives in the **official
  facebookresearch/sam-3d-body repo**, and that's what this notebook adapts.

## Exact run steps

1. Open `ares/tools/sam3d-body-colab.ipynb` in Colab (upload it, or push it somewhere Colab can
   open it from — File > Upload notebook).
2. `Runtime > Change runtime type > T4 GPU` (free tier — do not need Colab Pro).
3. `Runtime > Run all`.
4. Read the first markdown cell while the early cells run — it has the checkpoint-access links.
5. When the weights cell (Step 2) prompts for a token: paste a Hugging Face access token if
   you've already been granted access to
   [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3), or just
   press Enter to fall back to the ungated mirror automatically.
6. When the upload cell (Step 3) opens a file picker: choose a clear, mostly-frontal, full-body
   photo. Cancel it to use the bundled sample photo instead (useful for a first smoke test).
7. Let inference run (Step 4, ~10-30 sec). The optional preview render (Step 5) may or may not
   work — that's fine either way, see Troubleshooting.
8. The export cell (Step 6) writes `<name>_mesh_000.obj`, `<name>_mesh_000.glb` (best-effort),
   and `<name>_pose_000.json` per detected person, then triggers a browser download for each.

First run end-to-end: **~15-20 minutes**, dominated by building `detectron2` from source
(~5-10 min) and downloading ~6.9 GB of checkpoints. Re-running on a fresh photo in the same
session: **~10-30 seconds**.

## How this could feed the ARES pipeline (later — not wired up today)

- **Pose prior for temporal denoise.** The 70-joint MHR skeleton (`pred_joint_coords`,
  `pred_global_rots` in the exported JSON) is a plausible per-frame human-pose prior that a
  future temporal-consistency pass over the volumetric capture pipeline could condition on —
  e.g. penalizing frame-to-frame joint-angle jumps that aren't physically plausible, or using the
  joint skeleton as a rough correspondence signal between frames of the same performer.
- **Not a texture source.** Because the mesh is untextured, it cannot substitute for or feed the
  ARES texture/atlas pipeline (SD-Forge enhance, video texture, etc.).
  Anywhere ARES needs appearance/color, this model is
  the wrong tool; anywhere it needs geometry/pose *of a person specifically* (not the general
  volumetric scene), it's a candidate.
- **Scale caveat.** Absolute mesh scale/depth depends on the FOV estimate (MoGe2, or the
  fallback heuristic if MoGe2 isn't installed — see Troubleshooting). Before using this as a
  pose prior against ARES's own camera model, re-derive scale from ARES's known capture geometry
  rather than trusting SAM 3D Body's own camera guess.

## Alternatives

| Option | Cost | Setup | Notes |
|---|---|---|---|
| **This notebook** (Colab T4) | Free | ~15-20 min first run | Full control, exports raw mesh/joints, your data stays in your own Colab VM. |
| [fal.ai `fal-ai/sam-3/3d-body`](https://fal.ai/models/fal-ai/sam-3/3d-body) | ~$0.02/generation | None (API call) | Zero setup, pay-per-use, good for one-off conversions or batch jobs without managing a GPU. |
| [Meta playground](https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d) | Free | None (browser) | Official live demo; fastest way to sanity-check a photo before committing to a full pipeline run. No programmatic/batch access. |
| HF Spaces (e.g. `akhaliq/sam-3d-body`) | Free | None (browser) | Community-run; **the one checked on 2026-07-10 was returning a Gradio runtime error** (`theme` kwarg incompatibility) — treat HF Spaces as hit-or-miss, search https://huggingface.co/spaces?search=sam-3d-body for a currently-live one. |

## Troubleshooting

**"No GPU detected" assertion fails in the first cell.**
`Runtime > Change runtime type > Hardware accelerator > GPU`, pick T4, then `Runtime > Run all`
again. Free-tier GPU availability is not guaranteed at all times — if none is offered, wait and
retry, or use a paid tier temporarily.

**`detectron2` build fails (nvcc error / "not compiled with GPU support" / CUDA version mismatch).**
This is the single most likely failure point — detectron2 is built from source against whatever
torch/CUDA Colab currently ships, and Colab's driver/toolkit versions occasionally drift out of
sync (a known recurring Colab issue). First things to try, in order:
1. `Runtime > Disconnect and delete runtime`, then reconnect and `Run all` again (picks up a
   fresh, consistent VM image).
2. Run `!python -m detectron2.utils.collect_env` after a failed install to see which CUDA
   versions (driver / torch build / nvcc) actually disagree.
3. If nvcc is missing entirely, install it via `!apt-get install -y cuda-toolkit-12-x` matching
   `torch.version.cuda` from the GPU-check cell, then re-run the detectron2 cell.

**HF weights cell raises even after supplying a token.**
Means access to `facebook/sam-3d-body-dinov3` genuinely hasn't been approved yet (Meta's approval
can be manual, not instant) — check https://huggingface.co/facebook/sam-3d-body-dinov3 for your
request status. The cell should have already fallen back to the `jetjodh` mirror in this case; if
that also fails, the mirror's gating status may have changed since 2026-07-10 — check
https://huggingface.co/jetjodh/sam-3d-body-dinov3 directly.

**MoGe2 install fails.**
Non-fatal by design — the notebook prints a warning and continues with the model's default FOV
heuristic. Effect: less accurate absolute camera scale/depth for the exported mesh; body pose and
shape recovery are not meaningfully affected (the FOV estimate is a separate input, not a
retrained parameter of the pose/shape decoder).

**Preview render (Step 5) throws an exception.**
Expected sometimes — `pyrender`'s headless EGL backend is finicky on fresh Linux VMs. This does
**not** block the actual deliverable: the export cell (Step 6) only needs the raw NumPy arrays
already sitting in `outputs`, no GL context. If you want the preview working, try setting
`os.environ["PYOPENGL_PLATFORM"] = "osmesa"` in a new cell *before* re-running Step 4 (forces the
software rasterizer instead of EGL).

**Out-of-memory during inference.**
The T4-VRAM estimate in the notebook (~8-11 GB) is based on checkpoint file sizes, not a measured
run. If you OOM: set `MODEL_VARIANT = "vith"` in the weights cell (631M params vs. 840M, smaller
checkpoint) and re-run from there, or skip MoGe2 (frees a further ~1.3 GB of weights).

**"No person detected."**
Try a clearer, more front-facing, better-lit photo with the full body in frame; the ViTDet-H
detector's default confidence threshold is 0.5 and will silently return zero boxes on ambiguous
crops.

## Verification notes

This doc and the notebook were built from live-fetched source (GitHub raw files, GitHub API tree
listings, and the Hugging Face model API — not summaries of summaries) on 2026-07-10.
