/**
 * Minimal IVF demuxer — splits ffmpeg's IVF output into per-frame coded chunks.
 * IVF: 32-byte header ('DKIF', ver, hdrlen, fourcc, w, h, rate num/den, frame count),
 * then per frame [size u32][timestamp u64][data]. We encode one closed GOP per IVF, so
 * frame 0 is always a keyframe and the rest are deltas.
 */
export interface IvfFrame { data: Uint8Array; timestamp: bigint; }
export interface IvfVideo { fourcc: string; width: number; height: number; frames: IvfFrame[]; }

export function parseIvf(buf: Uint8Array): IvfVideo {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (String.fromCharCode(buf[0]!, buf[1]!, buf[2]!, buf[3]!) !== "DKIF") throw new Error("not an IVF stream");
  const hdrLen = dv.getUint16(6, true);
  const fourcc = String.fromCharCode(buf[8]!, buf[9]!, buf[10]!, buf[11]!);
  const width = dv.getUint16(12, true);
  const height = dv.getUint16(14, true);
  const frames: IvfFrame[] = [];
  let p = hdrLen;
  while (p + 12 <= buf.byteLength) {
    const size = dv.getUint32(p, true);
    const timestamp = dv.getBigUint64(p + 4, true);
    p += 12;
    if (p + size > buf.byteLength) break;
    frames.push({ data: buf.subarray(p, p + size).slice(), timestamp });
    p += size;
  }
  return { fourcc, width, height, frames };
}
