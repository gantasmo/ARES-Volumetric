# 2D video to one volumetric clip per person

Status 2026-09-19, in progress. A 2D video of one or more people becomes one `.ares` clip per
person, built from views that were generated around them. The camera saw one side; 4DAnyone
generates the rest as synchronized videos from cameras on a ring, and the per-frame surface is
carved from those views.

State, open failures and next steps: [handoff-4d-people.md](handoff-4d-people.md).

This is a second path beside the relief ones ([depth-2d-to-25d.md](depth-2d-to-25d.md),
[depth-2d-to-volumetric.md](depth-2d-to-volumetric.md)), and it shares their mask pass.

## Pipeline

```
video --> mask (SAM 3 tracker)        --> mask-ids.u8, one id per tracked person
      --> avatar.py refs, tracks      --> per person: ref.png, track.mp4 (fixed 9:16 window)
      --> 4DAnyone                    --> per person: videos/NN.mp4 (N views), cameras.json
      --> avatar_mesh.py              --> per person: mesh-fNNNNN.obj + atlas-fNNNNN.png
      --> ares encode                 --> apps/demo/<name>-p<id>.ares
```

| step | where | what it does |
| --- | --- | --- |
| mask | `tools/sam-service/depth.py` | SAM 3 text tracker over the clip; `mask-ids.u8` holds the tracker's object id on every mask pixel and `mask.objects` lists each id's frames and coverage |
| people | `tools/sam-service/avatar.py` | `refs` cuts each person out of their best frame at source resolution (score: area, edge contact, overlap with others, sharpness; the alpha is the SAM 3 instance that overlaps the id region most). `tracks` writes each person's own 9:16 clip: the window of `frames` consecutive frames where that person is largest with the whole body inside the map |
| generate | `tools/ext/4danyone` | synchronized target-view videos around the person plus an OpenCV camera rig, from the clip alone (motion by GVHMR, conditioning by the MHR70 skeleton). Runs in fp16 with the DiT split across both cards ([tools/4danyone/README.md](../tools/4danyone/README.md)) |
| mesh | `tools/sam-service/avatar_mesh.py` | per frame: BiRefNet masks on every view, a visual hull at `voxel` metres, marching cubes, the pieces above 5 % of the largest, decimation to `faces`, one xatlas unwrap per GOP, and a texture baked from the views that see each texel |
| encode | `packages/encoder` | the frame folder is an ordinary capture folder (`mesh-fNNNNN.obj` + `atlas-fNNNNN.png`) |

## Shared topology

The unwrap costs about 13 s, so it runs once per GOP. Every later frame moves the keyframe's own
vertices onto its hull: the hull's centroid motion first, then each vertex steps along its normal
onto the zero level of the frame's distance field (capped at two voxels a step), with a Taubin pass
between steps and a repair that snaps strays to the nearest fitted vertex. Vertices, triangles and
UVs persist, so the container codes indices and UVs once per GOP and positions as deltas
(`packages/encoder/src/temporal.ts`), and the atlas layout stays put. A frame whose p95 residual
exceeds `--rekey` metres is meshed and unwrapped afresh.

Measured on a synthetic scene with a known answer (an ellipsoid body and a sphere head, ray-traced
through 4DAnyone's own camera ring, 12 views, 40k faces, 1024² atlas):

| figure | keyframe | deformed frame |
| --- | --- | --- |
| time | 15 to 20 s | 3.0 s (carve 0.3, deform 1.9, bake 0.8) |
| distance to the true surface | p50 4.0 mm, p95 9.9 mm | p50 4.3 mm, p95 12.5 mm |
| texels with a colour | 99.1 % | 86 to 95 % |
| atlas colour at the vertices | mean error 10.5/255, wrong stripe on 7.9 % | mean 18.8/255, 15.4 % |

## Dev server and Convert card

`GET /avatar-convert?video=&name=&views=6&pitch=15&frames=61&voxel=0.01&faces=40000&tex=1024`
`&rekey=0.02&people=all&minHeight=0&textureCodec=vp9&texSize=1024&crf=30` streams `start`,
`progress {stage, person}`, `log`, `people`, `person {id, out}`, `done` and `error`. `frames` must
satisfy `(frames - 1) % 4 == 0` and `views` must be a multiple of 6. The chain lives in
[tools/avatar-run.mjs](../tools/avatar-run.mjs); the route owns the mask run, the service, the
history rows and the stream, and drops the run directory unless `keepRun=1`.

In the Convert card the `completion` select takes a third value, `4DAnyone views`. It replaces the
relief Geometry block with a Views block (views, pitch, window, voxel, faces, re-key, people,
minimum height) and disables the relief grid and FOV. One run writes one clip per person, each with
its own Open button.

## Cards with 11 GB

The released configuration wants more than 11 GB and bfloat16 tensor cores. The settings in
[tools/4danyone/README.md](../tools/4danyone/README.md) bring it onto two RTX 2080 Ti: fp16
throughout, the 30 DiT blocks split 15 and 15 across the cards, a 45-frame window instead of 121, a
pose-encoder batch of 3, the DiT's internal chunk budgets at 256 MB instead of 1536 MB, and the
feed-forward run in token chunks under that budget. Measured that way: 652 s for a 45-frame window
at 6 views, peak 10.0 GB, of which the denoise is 4 steps at 40 s.

Each of those was a measured overflow, in order: the pose encoder's batch of 6 full-resolution
skeleton videos, the DiT's activations on a 121-frame window, one card carrying 18 of the 30
blocks, and the feed-forward's 14336-wide hidden tensor at about 2 GB in one piece.

## Open items

1. The generated window is shorter than the released 121 frames; a longer clip needs several
   windows and a rule for joining them, since each run frames its own camera ring.
2. The mask pass tracks at most `DEPTH_SUBJECT_MAX_OBJECTS` (4) people, and a person the tracker
   loses and finds again returns under a new id.
3. The visual hull is the intersection of the view silhouettes, so concavities (between the arms
   and the body) stay filled.
4. The scene itself is not built yet: only the people are.
5. The 4DAnyone install is the manual sequence in its README. Settings shows its three rows
   (`4danyone-code`, `4danyone-weights`, `4danyone-smplx`) as status only, so the tab reports
   what is present without installing it.
