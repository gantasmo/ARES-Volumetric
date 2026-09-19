/**
 * Cross-language pin on the mask RLE convention.
 *
 * The SAM video tracker's masks arrive from the Python service ALREADY in the edit list's own
 * encoding (docs/sam-propagation-plan.md, Routes: the `mask` event's `rle` is stored verbatim into
 * the sidecar), so nothing between track.py's mask_to_rle and the encoder bake ever re-encodes
 * them. That makes the two encoders one interface with no adapter to catch a disagreement: a
 * Python side that drops the leading 0-run, or forgets to close a trailing 1-run, or encodes per
 * ROW instead of over the flattened bitmap, produces an rle that parses, sums wrong or shifts every
 * pixel, and the only symptom is a mask that lands somewhere else on every frame of a 272-frame
 * track. validateMasks catches the sum error; the other two it cannot see.
 *
 * So the pattern lives in mask-rle-fixture.json as `spans` (data, not a reimplementable
 * algorithm — both sides fill the same pixels or fail here), and `rle` is the encoder output under
 * test. Regenerate that field, and only that field, with track.py's own encoder:
 *
 *   tools/sam-service/env/Scripts/python.exe -c "import json,numpy as np,track; \
 *     f=json.load(open('packages/encoder/test/mask-rle-fixture.json')); \
 *     m=np.zeros(f['width']*f['height'],np.uint8); \
 *     [m.__setitem__(slice(s,s+n),1) for s,n in f['spans']]; \
 *     print(track.mask_to_rle(m.reshape(f['height'],f['width'])))"
 *
 * No GPU, no torch, no service: the fixture is a checked-in file and this test is arithmetic.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { rleEncodeMask, rleDecodeMask } from "@ares/core";

const FIXTURE = fileURLToPath(new URL("./mask-rle-fixture.json", import.meta.url));

// The pattern, duplicated here on purpose: the fixture is the OTHER side's answer sheet, and a test
// that read its question from the same file it grades would pass on any pair of self-consistent
// numbers. Each span is chosen for a convention the two encoders can disagree about:
const W = 64, H = 64;
const SPANS = [
  [0, 3],       // pixel 0 is SET → rleEncodeMask (edits.ts:50) must open with a ZERO-LENGTH 0-run
  [70, 130],    // starts row 1 col 6, ends row 3 col 7 → one run ACROSS rows: the bitmap is flat, not per-row
  [500, 1],     // a lone pixel → a run of length 1, which a "merge short runs" cleanup would eat
  [1000, 64],   // exactly one row's width but offset by 40 → still one run, not two
  [2048, 900],  // a block long enough that a u8/u16 run type would wrap
  [4090, 6],    // reaches the final pixel → the encoder's trailing push (edits.ts:59) must close it
];
const EXPECTED_RLE = [0, 3, 67, 130, 300, 1, 499, 64, 984, 900, 1142, 6];

function bitmapFromSpans(spans) {
  const bits = new Uint8Array(W * H);
  for (const [start, len] of spans) bits.fill(1, start, start + len);
  return bits;
}

/** First differing element plus its neighbours, because a bare deepEqual on 12 numbers still makes
 *  the reader find the index by eye — and on a real 1,520-run mask it prints two walls of digits. */
function firstDiff(actual, expected) {
  const n = Math.max(actual.length, expected.length);
  for (let i = 0; i < n; i++) {
    if (actual[i] !== expected[i]) {
      const win = (a) => JSON.stringify(a.slice(Math.max(0, i - 2), i + 3));
      return `run ${i} (a ${i % 2 ? "1" : "0"}-run) is ${actual[i]}, expected ${expected[i]}`
        + `\n  got      …${win(actual)}…`
        + `\n  expected …${win(expected)}…`;
    }
  }
  return `runs match to ${n} elements but the arrays differ in length: ${actual.length} vs ${expected.length}`;
}

test("rleEncodeMask holds the convention the fixture pattern is designed to break", () => {
  const bits = bitmapFromSpans(SPANS);
  const rle = rleEncodeMask(bits);
  assert.deepEqual(rle, EXPECTED_RLE, "the JS golden vector drifted:\n  " + firstDiff(rle, EXPECTED_RLE));
  // The invariant validateMasks enforces on every stored mask, asserted on the vector itself so a
  // bad golden can never be checked in as the thing everything else is measured against.
  assert.equal(EXPECTED_RLE.reduce((a, b) => a + b, 0), W * H);
  assert.equal(EXPECTED_RLE[0], 0, "a bitmap whose first pixel is set opens with a zero-length 0-run");
  assert.equal(EXPECTED_RLE.length % 2, 0, "a bitmap whose last pixel is set ends on a 1-run");
  assert.deepEqual(rleDecodeMask(rle, W * H), bits);
});

test("the Python mask encoder's fixture round-trips to a byte-identical bitmap", () => {
  const fx = JSON.parse(readFileSync(FIXTURE, "utf8"));

  // The two sides must have encoded the SAME pattern, or the run comparison below is meaningless.
  assert.equal(fx.width, W);
  assert.equal(fx.height, H);
  assert.deepEqual(fx.spans, SPANS, "mask-rle-fixture.json describes a different pattern than this test builds");
  assert.ok(Array.isArray(fx.rle) && fx.rle.length > 0, "mask-rle-fixture.json has no rle array");

  assert.deepEqual(fx.rle, EXPECTED_RLE,
    "the Python mask encoder disagrees with rleEncodeMask (edits.ts:50).\n  " + firstDiff(fx.rle, EXPECTED_RLE)
    + "\n  Its output is written verbatim into the sidecar, so this is a silent per-frame mask shift"
    + "\n  across the whole track. Fix track.py's mask_to_rle — never the fixture.");

  // The end the bake actually depends on: whatever the service sent must decode back to the exact
  // pixels it segmented. Byte-identical, because prepareVolume region-tests these bits directly
  // (edits.ts:374) and an off-by-one run offsets every pixel after it.
  const bits = rleDecodeMask(fx.rle, fx.width * fx.height);
  assert.deepEqual(bits, bitmapFromSpans(SPANS));
  assert.equal(fx.rle.reduce((a, b) => a + b, 0), fx.width * fx.height,
    "the fixture's runs do not cover the bitmap — validateMasks would reject this mask");
});
