// Decode a {"got":N,"png":"data:image/png;base64,..."} result dump into a real PNG. Also
// verifies "got" matches the expected frame index (fails loudly if not — this exists BECAUSE
// a truncated-decimal t= URL param silently landed on the wrong frame earlier in this POC).
import { readFile, writeFile } from "node:fs/promises";

const [inPath, outPath, expectFrame] = process.argv.slice(2);
if (!inPath || !outPath) { console.error("usage: node json-dataurl-to-png.mjs <in.json> <out.png> [expectFrame]"); process.exit(1); }
const j = JSON.parse(await readFile(inPath, "utf8"));
if (expectFrame !== undefined && String(j.got) !== String(expectFrame)) {
  console.error(`MISMATCH: ${inPath} got frame ${j.got}, expected ${expectFrame}`);
  process.exit(1);
}
const b64 = j.png.replace(/^data:image\/png;base64,/, "");
await writeFile(outPath, Buffer.from(b64, "base64"));
console.log(`wrote ${outPath} (frame ${j.got})`);
