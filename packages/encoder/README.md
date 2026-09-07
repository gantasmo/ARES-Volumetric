# @ares/encoder

The encoder behind [ARES Volumetric](https://github.com/gantasmo/ARES-Volumetric): it turns a folder
of captured frames into one `.ares` file that [@ares/core](https://www.npmjs.com/package/@ares/core)
plays in a browser, and reads that file back out again.

A capture that arrives as thousands of OBJ/PLY meshes plus PNG atlases — or as a per-frame Gaussian
splat sequence — becomes a single GOP-aligned stream: quantized, meshopt-compressed geometry, a
VP9/AV1 video texture encoded through ffmpeg, and an optional Opus audio track.

## Install

```
npm i @ares/encoder      # or: npx @ares/encoder --help
```

Node 22.15 or newer. ffmpeg on `PATH` (or `FFMPEG=/path/to/ffmpeg`) for video textures, the SOG
importer and audio.

## CLI

```
ares synth  [-o out.ares] [--shape object|talk|splat] [--frames 60] [--fps 30]
ares encode <frames-dir> [-o out.ares] [--fps 30] [--gop 30] [--texture-codec vp9|av1] [--crf 32]
                         [--audio track.wav] [--trim-in N] [--trim-out N] [--edits edits.json]
                         splat input: [--sh-degree 0..3] [--splat-temporal auto|index|nn|off]
ares export <file.ares> -o <out> [--frame N]     # .obj/.ply for meshes; .spz/.ply/.glb/.splat for splats
ares info   <file.ares>
```

`ares --help` prints the full flag list.

Imports: OBJ and PLY meshes with PNG atlases; Niantic SPZ v1–v4, 3DGS PLY, `.splat`, glTF/GLB with
`KHR_gaussian_splatting`, and PlayCanvas SOG for splats. Exports: OBJ, PLY, SPZ, GLB, `.splat`.

## API

```js
import { muxClip, synthClip } from "@ares/encoder";
```

The same building blocks the CLI uses — importers, the temporal GOP builder, quantization, the
muxer and the exporters — are exported for programmatic use.

MIT. ffmpeg is shelled out to, never bundled; see `THIRD-PARTY-NOTICES.md` in the repository.
