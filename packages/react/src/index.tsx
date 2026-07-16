/** @ares/react — <Ares/> for @react-three/fiber (spec §12.7). */
import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { AresObject } from "@ares/three";

export interface AresProps {
  /** URL of the .ares capture (or a preloaded Uint8Array). */
  src: string | Uint8Array;
  /** The canvas the ARES WebGPU renderer draws into (P1: its own surface). */
  canvas: HTMLCanvasElement;
  loop?: boolean;
  autoOrbit?: boolean;
}

/**
 * Drop-in component. The ARES player renders to `canvas` via WebGPU; this component
 * advances its clock from r3f's frame loop and disposes deterministically (§10.6).
 */
export function Ares({ src, canvas, loop = true, autoOrbit = false }: AresProps): null {
  const ref = useRef<AresObject | null>(null);

  useEffect(() => {
    const obj = new AresObject({ src, canvas, loop, autoOrbit });
    ref.current = obj;
    return () => { obj.dispose(); ref.current = null; };
  }, [src, canvas, loop, autoOrbit]);

  useFrame((_, dt) => ref.current?.tick(dt));
  return null;
}
