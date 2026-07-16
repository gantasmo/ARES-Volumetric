/**
 * @ares/three — THREE.Object3D wrapper around @ares/core's AresPlayer (spec §12.6).
 *
 * P1 note: the ARES WebGPU renderer draws to its OWN canvas/device, decoupled from
 * the render loop (spec §10.1). This wrapper drives that player's clock from the host
 * scene's frame loop via tick(dt) and tears it down deterministically (§10.6). A future
 * revision will present into a shared BufferGeometry when the host is a WebGPURenderer.
 */
import { Object3D } from "three";
import { AresPlayer, type AresPlayerOptions } from "@ares/core";

export interface AresObjectOptions {
  canvas: HTMLCanvasElement;
  src: string | Uint8Array;
  autoOrbit?: boolean;
  loop?: boolean;
}

export class AresObject extends Object3D {
  private player: AresPlayer | null = null;
  private ready: Promise<AresPlayer>;

  constructor(options: AresObjectOptions) {
    super();
    const opts: AresPlayerOptions = {
      canvas: options.canvas,
      src: options.src,
      loop: options.loop ?? true,
      autoOrbit: options.autoOrbit ?? false, // the host scene usually owns the camera
    };
    this.ready = AresPlayer.create(opts).then((p) => {
      this.player = p;
      p.pause(); // the host drives the clock via tick()
      return p;
    });
  }

  whenReady(): Promise<AresPlayer> { return this.ready; }
  get isReady(): boolean { return this.player !== null; }

  /** Advance playback + render one frame; call from the host render loop. */
  tick(dt: number): void { this.player?.tick(dt); }

  dispose(): void {
    this.player?.dispose();
    this.player = null;
  }
}
