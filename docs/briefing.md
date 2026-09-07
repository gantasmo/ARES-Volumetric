# ARES in plain language

A short, non-technical companion to [whitepaper.md](whitepaper.md). Written 2026-07-10.

## Overview

Volumetric video is video with a shape. A studio films a person from many cameras at once
and software rebuilds them as a moving 3D surface, so the recording can be watched from
any angle, walked around in VR, or placed on a table through a phone camera. Each frame of
such a recording is a full 3D model: the shape itself (a mesh made of thousands of
triangles) plus a color image that wraps around the shape like a printed skin.

The problem is size and delivery. The test recording in this repository is nine seconds of
one skateboarder. As exported by the capture software, it is 1.58 gigabytes spread over
544 separate files, two per frame. A web page trying to play that has to download
hundreds of files and rebuild a 3D model 30 times every second. It stutters, it takes
forever to start, and it cannot skip around like a normal video player.

ARES is a file format and a player that fix the delivery half of that problem. The same
nine-second recording becomes one file of about 50 megabytes, thirty times smaller,
fetched in a single request, playing at 60 frames per second in an ordinary browser with
nothing installed.

## Compression approach

Two ideas do most of the work.

The color skin is stored as actual video. Instead of 272 separate photographs, the color
data becomes a nine-second video clip, and every phone and laptop has a dedicated chip
that decodes video almost for free. On the
test recording this one change shrinks the color data from 1.11 gigabytes to about 12
megabytes.

The shape data is packed the way the graphics card wants it. Vertex coordinates are stored
as compact whole numbers instead of full-precision decimals, and they stay in that compact
form all the way to the graphics card, which does the final conversion itself during
rendering. The computer's main processor barely touches the data; on the test machines it
spends under one millisecond per frame on shape decoding, which is why playback stays
smooth.

## Current capabilities

A working player, a converter, and an editor, all running locally. On Windows,
double-clicking `ARES.vbs` in the repository opens the player with the test
recording. The same app has a Convert tab that turns a folder of captured frames into an
`.ares` file through a normal folder-picker dialog, a Compare tab that shows two encodes
side by side with a draggable divider, an Inspect tab that reports what is inside a
volumetric file without uploading it anywhere, and an editor that can trim away floors,
stray fragments, or time ranges of the recording without destroying the original.

The size numbers above were measured on this recording, and a script in the repository
regenerates them from the source files. The speed numbers come from the player's own
on-screen meter.

## Known limitations

Google's Draco compressor still packs the shape data alone into fewer bytes than ARES
does. One commercial format (4DViews) goes further, compressing shape data about seven
times smaller than the current ARES encode by reusing information between frames,
something ARES does not yet do for this kind of capture. The same format stores its color
data about ten times larger than ARES, because it skips the video-codec idea. Each format
is strong exactly where the other is weak, and the long-term plan is to combine both
strengths in one format.

Recordings of this type also shimmer slightly, because the capture software rebuilds the
surface from scratch every frame and the small differences between rebuilds are visible
as a faint boiling. Reducing that without smearing fast motion is planned work.

An optional feature that sharpens the color imagery with AI currently produces visible
artifacts on skin and faces, so it is not recommended for people until a better-suited
model is wired in.

## Roadmap

Nearer term: finishing the visual quality evaluation of the current encodes, and extending
the editor's click-to-select segmentation tool so a selection can follow the subject across
time automatically. An optional
setting that trims the shape data further without touching the color mapping is also
planned. Further out: true streaming, where playback starts before the download finishes
and quality adapts to the connection, plus support for a newer capture style called
Gaussian splats and importers for more capture formats. The technical detail and the
measurements behind all of it are in [whitepaper.md](whitepaper.md).
