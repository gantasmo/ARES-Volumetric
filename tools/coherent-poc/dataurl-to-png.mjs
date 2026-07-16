// Decode a canvas.toDataURL() text dump (saved by Playwright browser_evaluate) into a real PNG.
import { readFile, writeFile } from "node:fs/promises";

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) { console.error("usage: node dataurl-to-png.mjs <in.txt> <out.png>"); process.exit(1); }
let text = (await readFile(inPath, "utf8")).trim();
if (text.startsWith('"') && text.endsWith('"')) text = JSON.parse(text);
const b64 = text.replace(/^data:image\/png;base64,/, "");
await writeFile(outPath, Buffer.from(b64, "base64"));
console.log(`wrote ${outPath} (${b64.length} b64 chars)`);
