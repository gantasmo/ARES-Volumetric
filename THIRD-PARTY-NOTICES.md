# Third-party notices

The repository's own code is MIT-licensed (see [LICENSE](LICENSE)). The items
below carry their own terms and are **not** covered by the repository license.

## Bundled or linked at runtime

- **meshoptimizer** (MIT, Arseny Kapoulkine) — the vertex/index codecs and the simplifier.
  `npm run bundle` inlines its JavaScript and WASM into `packages/*/dist/bundle/*`; those
  bundles carry a banner naming it, and the MIT text must accompany any redistribution.
- **draco3d** (Apache-2.0, Google) — used by the Phase 0 benchmark only (`bench/`). Not part
  of the runtime or the bundles. If a DRAC intra codec path is ever wired into `@ares/core`, the
  Apache-2.0 license text and NOTICE must ship with it.
- **three**, **react**, **@react-three/fiber** — peer dependencies of the wrapper packages,
  installed by the host application under their own MIT licenses; never bundled here.
- **esbuild** (MIT), **typescript** (Apache-2.0) — build-time only.

## Invoked as external programs

- **ffmpeg** — shelled out to (never linked) by the encoder for VP9/AV1 texture video, Opus
  audio, PNG decode, WebP decode (SOG import) and repack detection. ffmpeg builds are LGPL-2.1+
  or GPL-2.0+ depending on their configuration; redistributing a bundled ffmpeg binary alongside
  this software would bring that build's license with it. This repository does not distribute one.
- **realesrgan-ncnn-vulkan** (BSD-3-Clause) — fetched separately into the git-ignored
  `tools/bin/` directory by the Settings tab; not distributed with this repository.
- **Stable Diffusion WebUI Forge** — an optional local install the enhance tier launches; its
  own license (AGPL-3.0) and the licenses of whatever checkpoints it loads apply to that install.

## Python service (`tools/sam-service`)

Installed into a git-ignored virtual environment from `requirements.txt`:
**torch** / **torchvision** (BSD-3-Clause), **transformers** (Apache-2.0), **fastapi** (MIT),
**uvicorn** (BSD-3-Clause), **pydantic** (MIT), **pillow** (MIT-CMU), **numpy** (BSD-3-Clause),
**segment-anything** (Apache-2.0), **safetensors** / **huggingface_hub** (Apache-2.0),
**spandrel** (MIT), **diffusers** (Apache-2.0), **accelerate** (Apache-2.0).

Model weights the service loads are downloaded by the user and are governed by their own terms:

- **SAM 3 / SAM ViT-H** weights — Meta's SAM License.
- **Real-ESRGAN** weights (`tools/sam-service/models/`, git-ignored) — BSD-3-Clause.
- **Lykon/dreamshaper-8** (default checkpoint of the optional `/detail` endpoint) —
  CreativeML OpenRAIL-M, a use-restricted license: outputs may not be used for the purposes
  it lists. Anyone deploying that endpoint takes on those terms.

## Notebooks and interop tools

- **`tools/sam3d-body-colab.ipynb`** adapts Meta's SAM 3D Body demo notebook
  (`facebookresearch/sam-3d-body`, `notebook/demo_human.ipynb`) and downloads the SAM 3D Body
  model weights at runtime. Both the adapted content and the weights are governed by Meta's SAM
  License, not MIT.
- **BridgeCodec4DS.dll** (4DViews) and **SVFUnityPlugin** (Microsoft) are proprietary components
  the interop tools load from the user's own licensed installs; nothing from either SDK is
  distributed with this repository.

## Format specifications implemented

Niantic **SPZ**, PlayCanvas **SOG**, the antimatter15 **.splat** layout, the INRIA **3D Gaussian
Splatting PLY** layout and Khronos **KHR_gaussian_splatting** (glTF 2.0) are implemented from
their public specifications; no code was copied from their reference implementations.
