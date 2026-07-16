# Third-party notices

The repository's own code is MIT-licensed (see [LICENSE](LICENSE)). The items
below carry their own terms and are **not** covered by the repository license:

- **`tools/sam3d-body-colab.ipynb`** adapts Meta's SAM 3D Body demo notebook
  (`facebookresearch/sam-3d-body`, `notebook/demo_human.ipynb`) and downloads
  the SAM 3D Body model weights at runtime. Both the adapted content and the
  weights are governed by Meta's SAM License, not MIT.
- **realesrgan-ncnn-vulkan** (BSD-3-Clause) is fetched separately into the
  git-ignored `tools/bin/` directory; it is not distributed with this repository.
- **BridgeCodec4DS.dll** (4DViews) and **SVFUnityPlugin** (Microsoft) are
  proprietary components the interop tools load from the user's own licensed
  installs; nothing from either SDK is distributed with this repository.
- Runtime npm dependencies (three, react, meshoptimizer, draco3d, typescript)
  are installed from npm under their own licenses.
