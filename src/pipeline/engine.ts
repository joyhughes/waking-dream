import { createGlContext, type GlContext } from '../gpu/gl';
import { Ops } from '../gpu/ops';
import { GpuTensor } from '../gpu/tensor';
import { DreamNet } from '../model/dreamnet';
import { DEFAULT_SHALLOW_PARAMS, ShallowDream, type ShallowDreamParams } from '../model/shallowDream';
import type { FrameSource } from './sources';
import { FrameTimer, type TimingSnapshot } from './stats';

export type ProcessorMode = 'off' | 'shallow' | 'model';

/**
 * How much of the previous output comes back around, and how it is moved before it does.
 *
 * The warp is the difference between a filter and a dream. With `zoom` at exactly 1 and no
 * rotation, hallucinated detail lands on the same pixels every frame and simply saturates. Push it
 * outward slightly each frame and the centre is continuously vacated for new detail to grow into,
 * which is the recursion the original DeepDream zoom videos were made of — and, on a live camera,
 * what keeps the effect attached to the scene instead of baked onto the lens.
 */
export interface FeedbackConfig {
  enabled: boolean;
  /** Weight on the live frame going into the network. */
  source: number;
  /** Weight on the warped previous output. Above about 0.6 the live frame stops being legible. */
  previous: number;
  /** Per-frame magnification. 1.005 is a slow outward crawl; 1.05 is a headlong rush. */
  zoom: number;
  /** Per-frame rotation, in degrees. */
  rotate: number;
  driftX: number;
  driftY: number;
  /**
   * How much of the fed-back frame's contrast survives the trip, the rest decaying toward mid grey.
   * 1 keeps all of it and lets the ascent accumulate without limit; below about 0.97 the recursion
   * settles into a steady state instead of saturating.
   */
  fade: number;
}

export interface DisplayConfig {
  /** Cross-fade between the untouched capture and the network's output. */
  mix: number;
  gain: number;
  saturation: number;
}

export interface EngineConfig {
  processor: ProcessorMode;
  /** Longest side of the tensor the network actually sees. The size/speed dial. */
  captureSize: number;
  shallow: ShallowDreamParams;
  modelControls: number[];
  feedback: FeedbackConfig;
  display: DisplayConfig;
  /** Null follows the source's own default: mirrored for a camera, not for a file. */
  mirror: boolean | null;
}

export const DEFAULT_CONFIG: EngineConfig = {
  processor: 'shallow',
  captureSize: 256,
  shallow: DEFAULT_SHALLOW_PARAMS,
  modelControls: [],
  feedback: {
    enabled: true,
    source: 0.7,
    previous: 0.3,
    zoom: 1.008,
    rotate: 0.05,
    driftX: 0,
    driftY: 0,
    fade: 0.95,
  },
  display: { mix: 1, gain: 1, saturation: 1 },
  mirror: null,
};

export interface BenchmarkRow {
  captureSize: number;
  width: number;
  height: number;
  /** Median wall time for one fully-serialized frame, in milliseconds. */
  msPerFrame: number;
  fps: number;
}

export interface EngineStatus {
  running: boolean;
  captureWidth: number;
  captureHeight: number;
  timing: TimingSnapshot;
  supportsGpuTiming: boolean;
  poolMegabytes: number;
  programCount: number;
  renderer: string;
  modelName: string | null;
  error: string | null;
}

/**
 * The frame loop, and the only thing in the app that owns GPU state.
 *
 * The whole of a frame — uploading the camera texture, mixing in the warped previous output,
 * running the network, compositing to the canvas — is queued without a single readback. That
 * constraint is the reason the feedback loop is affordable at all: the previous frame's output is
 * still a texture, so feeding it back costs a sample, not a round trip through CPU memory.
 */
export class Engine {
  readonly ctx: GlContext;
  private readonly ops: Ops;
  private readonly shallow: ShallowDream;
  private readonly timer: FrameTimer;

  private config: EngineConfig = DEFAULT_CONFIG;
  private source: FrameSource | null = null;
  private model: DreamNet | null = null;

  /** Held outside the pool so it survives the end-of-frame release, which is the entire point of it. */
  private previous: GpuTensor | null = null;
  private previousValid = false;

  private rafHandle: number | null = null;
  private running = false;
  private error: string | null = null;
  private captureWidth = 0;
  private captureHeight = 0;

  onStatus: ((status: EngineStatus) => void) | null = null;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = createGlContext(canvas);
    this.ops = new Ops(this.ctx);
    this.shallow = new ShallowDream(this.ctx, this.ops);
    this.timer = new FrameTimer(this.ctx);
  }

  getConfig(): EngineConfig {
    return this.config;
  }

  setConfig(patch: Partial<EngineConfig>): void {
    const previousSize = this.config.captureSize;
    this.config = { ...this.config, ...patch };
    if (this.config.captureSize !== previousSize) {
      // The persistent feedback buffer is tied to the capture size, so a size change starts the
      // recursion over rather than resampling a buffer of the wrong shape into it.
      this.discardFeedback();
    }
  }

  setSource(source: FrameSource | null): void {
    this.source?.dispose();
    this.source = source;
    this.discardFeedback();
    this.timer.reset();
  }

  get currentSource(): FrameSource | null {
    return this.source;
  }

  async loadModel(url: string): Promise<DreamNet> {
    const model = await DreamNet.load(this.ctx, this.ops, url);
    this.setModel(model);
    return model;
  }

  loadModelFromBuffer(buffer: ArrayBuffer): DreamNet {
    const model = DreamNet.fromBuffer(this.ctx, this.ops, buffer);
    this.setModel(model);
    return model;
  }

  private setModel(model: DreamNet): void {
    this.model?.dispose();
    this.model = model;
    this.config = {
      ...this.config,
      processor: 'model',
      modelControls: model.controls.map((control) => control.default),
    };
    this.discardFeedback();
  }

  get currentModel(): DreamNet | null {
    return this.model;
  }

  clearModel(): void {
    this.model?.dispose();
    this.model = null;
    if (this.config.processor === 'model') this.config = { ...this.config, processor: 'shallow' };
  }

  /** Clears the recursion. Also the escape hatch when feedback has run away into a saturated mess. */
  discardFeedback(): void {
    this.previousValid = false;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    const tick = (now: number) => {
      this.rafHandle = requestAnimationFrame(tick);
      this.renderFrame(now);
    };
    this.rafHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    if (this.rafHandle !== null) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = null;
  }

  private renderFrame(now: number): void {
    const frame = this.source?.frame;
    if (!frame) return;

    const cpuStart = this.timer.beginFrame(now);
    try {
      this.drawOnce(frame);
      this.error = null;
    } catch (error) {
      // A GPU error mid-frame otherwise repeats sixty times a second and buries the console. One
      // report and a stopped loop is more useful than a scrolling wall of the same message.
      this.error = error instanceof Error ? error.message : String(error);
      this.stop();
    } finally {
      this.ops.pool.releaseAll();
      this.timer.endFrame(cpuStart);
      this.onStatus?.(this.status());
    }
  }

  /**
   * One frame, start to finish.
   *
   * Kept separate from `renderFrame` so the benchmark can drive it directly without the timer, the
   * error latch, or `requestAnimationFrame` pacing in the way.
   */
  private drawOnce(frame: TexImageSource): void {
    const source = this.source!;
    const { width, height } = this.captureDimensions(source);
    this.captureWidth = width;
    this.captureHeight = height;
    this.resizeCanvas(width, height);

    const captured = this.ops.pool.acquire({ width, height, channels: 3 });
    this.ops.fromSource(frame, captured, { mirror: this.config.mirror ?? source.defaultMirror });

    const input = this.buildNetworkInput(captured, width, height);
    const output = this.process(input);

    this.retainFeedback(output, width, height);

    this.ops.toCanvas(output, captured, this.canvas.width, this.canvas.height, this.config.display);
  }

  /** Mixes the live frame with the warped previous output, or passes the live frame straight through. */
  private buildNetworkInput(captured: GpuTensor, width: number, height: number): GpuTensor {
    const { feedback } = this.config;
    if (!feedback.enabled || !this.previousValid || !this.previous) return captured;

    const warped = this.ops.pool.acquire({ width, height, channels: 3 });
    this.ops.warp(this.previous, warped, {
      zoom: feedback.zoom,
      rotate: (feedback.rotate * Math.PI) / 180,
      translateX: feedback.driftX,
      translateY: feedback.driftY,
      fade: feedback.fade,
    });

    const mixed = this.ops.pool.acquire({ width, height, channels: 3 });
    this.ops.combine(
      mixed,
      [
        { tensor: captured, weight: feedback.source },
        { tensor: warped, weight: feedback.previous },
      ],
      { clamp01: true },
    );
    this.ops.pool.release(warped);
    return mixed;
  }

  private process(input: GpuTensor): GpuTensor {
    switch (this.config.processor) {
      case 'model': {
        if (!this.model) return input;
        this.model.setControls(this.config.modelControls);
        return this.model.forward(input);
      }
      case 'shallow':
        return this.shallow.run(input, this.config.shallow);
      default:
        return input;
    }
  }

  /** Copies this frame's output into the persistent buffer the next frame will warp and mix in. */
  private retainFeedback(output: GpuTensor, width: number, height: number): void {
    if (!this.config.feedback.enabled) {
      this.previousValid = false;
      return;
    }

    if (!this.previous || this.previous.width !== width || this.previous.height !== height) {
      this.previous?.dispose();
      this.previous = new GpuTensor(this.ctx, width, height, 3);
    }

    // A copy rather than keeping the tensor itself: `output` belongs to the pool and will be handed
    // to some other layer on the next frame, and the network's output size can differ from the
    // capture size by a rounding pixel, which this pass resolves along the way.
    this.ops.combine(this.previous, [{ tensor: output, weight: 1 }], { clamp01: true });
    this.previousValid = true;
  }

  /**
   * The capture size, resolved against the source's aspect ratio.
   *
   * Both sides are rounded to a multiple of eight. The network halves its resolution twice and then
   * doubles it twice, so a size not divisible by four comes back a pixel or two off and every
   * later composite is very slightly stretched; eight leaves room for a third downsample.
   */
  private captureDimensions(source: FrameSource): { width: number; height: number } {
    const longest = Math.max(64, this.config.captureSize);
    const aspect = source.width / source.height;
    const raw = aspect >= 1 ? { width: longest, height: longest / aspect } : { width: longest * aspect, height: longest };
    return {
      width: Math.max(64, Math.round(raw.width / 8) * 8),
      height: Math.max(64, Math.round(raw.height / 8) * 8),
    };
  }

  /**
   * Sizes the canvas to the capture's aspect ratio inside whatever box CSS gave it.
   *
   * Letterboxing here rather than with `object-fit` keeps the drawing buffer exactly the number of
   * pixels being displayed, so the composite pass is never resampling to a mismatched shape.
   */
  private resizeCanvas(captureWidth: number, captureHeight: number): void {
    const container = this.canvas.parentElement;
    if (!container) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const boxWidth = container.clientWidth;
    const boxHeight = container.clientHeight;
    if (boxWidth === 0 || boxHeight === 0) return;

    const aspect = captureWidth / captureHeight;
    const cssWidth = Math.min(boxWidth, boxHeight * aspect);
    const cssHeight = cssWidth / aspect;

    const backingWidth = Math.max(1, Math.round(cssWidth * dpr));
    const backingHeight = Math.max(1, Math.round(cssHeight * dpr));

    if (this.canvas.width !== backingWidth || this.canvas.height !== backingHeight) {
      this.canvas.width = backingWidth;
      this.canvas.height = backingHeight;
    }
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
  }

  /**
   * Measures cost against capture size, which is the tradeoff the whole app is built to expose.
   *
   * Each frame is serialized with `finish()` before the clock is read. That is exactly what you
   * must not do in the real loop — it throws away the overlap between CPU and GPU — but it is the
   * only way to attribute time to one frame rather than to a pipeline several frames deep. The
   * numbers therefore run slightly pessimistic against the live frame rate, and the shape of the
   * curve across sizes, which is what is being asked about, is right.
   */
  async benchmark(sizes: number[], framesPerSize = 12): Promise<BenchmarkRow[]> {
    const frame = this.source?.frame;
    if (!frame) throw new Error('Nothing to benchmark: no source is running.');

    const wasRunning = this.running;
    this.stop();

    const originalSize = this.config.captureSize;
    const rows: BenchmarkRow[] = [];

    try {
      for (const size of sizes) {
        this.config = { ...this.config, captureSize: size };
        this.discardFeedback();

        // Warm-up covers shader compilation for any new variant and the first allocation of every
        // tensor shape at this size. Timing those would measure the compiler, not the pipeline.
        for (let i = 0; i < 3; i++) {
          this.drawOnce(frame);
          this.ops.pool.releaseAll();
        }
        this.ops.finish();

        const samples: number[] = [];
        for (let i = 0; i < framesPerSize; i++) {
          const start = performance.now();
          this.drawOnce(frame);
          this.ops.finish();
          samples.push(performance.now() - start);
          this.ops.pool.releaseAll();
        }

        samples.sort((a, b) => a - b);
        const median = samples[Math.floor(samples.length / 2)];
        rows.push({
          captureSize: size,
          width: this.captureWidth,
          height: this.captureHeight,
          msPerFrame: median,
          fps: median > 0 ? 1000 / median : 0,
        });

        // Let the page breathe between sizes; a long synchronous sweep otherwise freezes the UI and
        // lets the GPU heat-soak into thermal throttling that skews the later rows downward.
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
    } finally {
      this.config = { ...this.config, captureSize: originalSize };
      this.discardFeedback();
      this.timer.reset();
      if (wasRunning) this.start();
    }

    return rows;
  }

  status(): EngineStatus {
    return {
      running: this.running,
      captureWidth: this.captureWidth,
      captureHeight: this.captureHeight,
      timing: this.timer.snapshot(),
      supportsGpuTiming: this.timer.supportsGpuTiming,
      poolMegabytes: this.ops.pool.bytesHeld / (1024 * 1024),
      programCount: this.ops.programCount,
      renderer: this.ctx.caps.rendererName,
      modelName: this.model?.name ?? null,
      error: this.error,
    };
  }

  dispose(): void {
    this.stop();
    this.source?.dispose();
    this.model?.dispose();
    this.previous?.dispose();
    this.shallow.dispose();
    this.timer.dispose();
    this.ops.dispose();
  }
}
