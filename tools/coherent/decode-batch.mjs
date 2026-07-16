// Decode a Playwright browser_evaluate array-of-{frame,dataUrl} JSON dump into individual PNGs.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

const [inPath, outDir, prefix] = process.argv.slice(2);
if (!inPath || !outDir) { console.error("usage: node decode-batch.mjs <in.json> <outDir> [prefix]"); process.exit(1); }
await mkdir(outDir, { recursive: true });
const arr = JSON.parse(await readFile(inPath, "utf8"));
for (const item of arr) {
  const b64 = item.dataUrl.replace(/^data:image\/png;base64,/, "");
  const name = `${prefix || ""}f${String(item.frame).padStart(5, "0")}.png`;
  await writeFile(join(outDir, name), Buffer.from(b64, "base64"));
  console.log(`wrote ${name} (requested ${item.frame}, actual ${item.actualIdx})`);
}
