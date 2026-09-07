import { createGlContext, type GlContext } from '../gpu/gl';
import { Ops } from '../gpu/ops';
import { GpuTensor } from '../gpu/tensor';
import { DreamNet } from '../model/dreamnet';
import { DEFAULT_SHALLOW_PARAMS, ShallowDream, type ShallowDreamParams } from '../model/shallowDream';
import type { FrameSource } from './sources';
import { AudioAnalyser, DEFAULT_BAND_ASSIGNMENTS, FREQUENCY_BANDS, mapLevelsToControls } from './audio';
import { getDeviceLimits } from './deviceLimits';
import { applyModulations, type ModulationMap } from './modulation';
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

/**
 * How sound drives the model's controls.
 *
 * The conditioning is already a per-frame CPU calculation, so where its numbers come from is
 * entirely open — sliders or a spectrum analyser cost the same. A model holding several patterns
 * becomes an instrument this way.
 */
export interface AudioConfig {
  enabled: boolean;
  /** Crossfade between the slider value and the band level. 1 is fully sound-driven. */
  amount: number;
  /** Multiplier on the band level before it is clamped, for pushing quiet material into range. */
  gain: number;
  /**
   * How hard the assigned bands compete. At 1 the loudest reads full and the rest are pushed down a
   * power curve, which is what makes the ratio between two bands legible rather than subtle.
   */
  sensitivity: number;
  /** Band index per control. Defaults put bass on the first control and the vocal range on the second. */
  assignments: number[];
}

export interface DisplayConfig {
  /** Cross-fade between the untouched capture and the network's output. */
  mix: number;
  gain: number;
  saturation: number;
}

export interface EngineConfig {
  processor: ProcessorMode;
  /**
   * How strongly the result's hue and saturation are pulled back toward the source frame.
   *
   * 0 leaves the colours the processor chose; 1 keeps the camera's colours exactly and lets only
   * brightness carry what was drawn. Applied before the feedback buffer is written, so the
   * recursion is constrained too rather than the drift merely being hidden at the end.
   */
  colorPreservation: number;
  /** Longest side of the tensor the network actually sees. The size/speed dial. */
  captureSize: number;
  shallow: ShallowDreamParams;
  modelControls: number[];
  /**
   * How many resolutions a trained model is run at, and how far apart they sit.
   *
   * The shallow mode has had octaves from the start because it *generates* structure at run time,
   * so running it at several scales is the natural way to get detail at several sizes. A trained
   * model looked like it did not need them: its pattern scale was decided at training time by how
   * large the style image was read, which is what separates pandas from pandas-big.
   *
   * But the same lever exists at run time. The network draws its motifs at a size fixed in its own
   * input pixels, so feeding it a half-size frame makes everything it draws twice as large relative
   * to the picture. Running it at two or three scales and keeping the coarse structure from one and
   * the fine detail from another gives a single model the range that otherwise took two.
   *
   * 1 is a single pass, which is what it did before.
   */
  modelOctaves: number;
  modelOctaveScale: number;
  /**
   * How much of the coarse pass's structure replaces the full-resolution pass's own.
   *
   * A continuous control rather than an implied all-or-nothing, and the reason is not only taste.
   * Instance normalization makes the model's output level depend on its input resolution, so the
   * coarse pass comes back at a different exposure — taking its low frequencies wholesale shifted
   * the whole frame brighter. Expressed as a swap of one band for another, 0 is exactly the single
   * pass, and how far you go from there is yours to choose.
   */
  modelScaleMix: number;
  feedback: FeedbackConfig;
  display: DisplayConfig;
  audio: AudioConfig;
  /** Sound routings, keyed by modulation target id. See `modulation.ts`. */
  modulations: ModulationMap;
  /** Null follows the source's own default: mirrored for a camera, not for a file. */
  mirror: boolean | null;
  /**
   * Take the capture's aspect ratio from the screen rather than from the source.
   *
   * A phone screen is much taller than any camera's 4:3 or 16:9, so fitting the whole frame leaves
   * bars down the sides — which is not what "full screen" means to anyone. With this on, the
   * capture is shaped like the display and the source is centre-cropped into it, so the picture
   * goes edge to edge. Nothing is stretched: the crop happens on the way in, and the network sees
   * exactly the pixels that end up on screen.
   */
  fillScreen: boolean;
}

export const DEFAULT_CONFIG: EngineConfig = {
  processor: 'shallow',
  colorPreservation: 0,
  captureSize: getDeviceLimits().defaultCaptureSize,
  shallow: DEFAULT_SHALLOW_PARAMS,
  modelControls: [],
  modelOctaves: 1,
  modelOctaveScale: 2,
  modelScaleMix: 0.6,
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
  audio: { enabled: false, amount: 1, gain: 1.2, sensitivity: 0.7, assignments: [...DEFAULT_BAND_ASSIGNMENTS] },
  modulations: {},
  mirror: null,
  fillScreen: false,
};

/** One model's measured cost, for the side-by-side comparison. */
export interface ModelBenchmarkRow {
  label: string;
  detail: string;
  msPerFrame: number;
  fps: number;
}

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
  /** True once the GPU has taken the context away. Nothing will render again without a reload. */
  contextLost: boolean;
  captureWidth: number;
  captureHeight: number;
  timing: TimingSnapshot;
  supportsGpuTiming: boolean;
  poolMegabytes: number;
  programCount: number;
  renderer: string;
  modelName: string | null;
  /** Per-band levels when audio is running, for the meter. Empty when it is not. */
  audioLevels: number[];
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
  private contextLost = false;
  private audio: AudioAnalyser | null = null;
  /** Last read band levels, kept so the meter can be drawn without a second analyser pass. */
  private audioLevels: number[] = new Array(FREQUENCY_BANDS.length).fill(0);
  /**
   * What this frame is actually running: the config with any sound routings applied.
   *
   * Held separately from `config` because `config` stays the user's settings — the sliders must not
   * appear to move on their own, and the values written back out to a saved PNG have to be the ones
   * that were set rather than wherever the music happened to push them at the moment of capture.
   */
  private active: EngineConfig = DEFAULT_CONFIG;

  onStatus: ((status: EngineStatus) => void) | null = null;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = createGlContext(canvas);
    this.ops = new Ops(this.ctx);
    this.shallow = new ShallowDream(this.ctx, this.ops);
    this.timer = new FrameTimer(this.ctx);

    // A lost context does not throw. Every GL call afterwards silently does nothing, so without
    // this the app carries on at full frame rate drawing a black canvas and reporting no error --
    // which is a far worse failure than saying so. Losing it happens for real reasons: the machine
    // sleeping, the GPU being reset under memory pressure, or a driver crash in another tab.
    canvas.addEventListener('webglcontextlost', (event) => {
      // Without preventDefault the browser will not even attempt to give the context back.
      event.preventDefault();
      this.contextLost = true;
      this.error = 'The GPU context was lost. Reload the page to start again.';
      this.stop();
      this.onStatus?.(this.status());
    });

    canvas.addEventListener('webglcontextrestored', () => {
      // Every texture, program and framebuffer this app holds belongs to the dead context and none
      // of them come back. Rebuilding all of it in place would be a second, barely-exercised code
      // path through every module here; a reload does the same thing correctly.
      this.error = 'The GPU context came back, but the pipeline needs a reload to rebuild.';
      this.onStatus?.(this.status());
    });
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
      // Every pooled buffer is now the wrong size. Dropping them immediately matters because this
      // control gets swept rather than set: dragging it end to end asks for over a hundred distinct
      // sizes, and holding a set for each is how the tab gets killed for running out of memory.
      this.ops.trim();
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

  setAudio(analyser: AudioAnalyser | null): void {
    this.audio?.stop();
    this.audio = analyser;
    if (!analyser) this.audioLevels = new Array(FREQUENCY_BANDS.length).fill(0);
  }

  get currentAudio(): AudioAnalyser | null {
    return this.audio;
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
    if (this.running || this.contextLost) return;
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
      this.ops.endFrame();
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

    // Read once per frame. The analyser advances its own attack and release on every call, so
    // asking twice would run the envelopes at double speed.
    const levels =
      this.config.audio.enabled && this.audio ? this.audio.levels() : null;
    if (levels) this.audioLevels = Array.from(levels);

    this.active = levels ? applyModulations(this.config, levels, this.config.modulations) : this.config;

    const { width, height } = this.captureDimensions(source);
    this.captureWidth = width;
    this.captureHeight = height;
    this.resizeCanvas(width, height);

    const captured = this.ops.pool.acquire({ width, height, channels: 3 });
    this.ops.fromSource(frame, captured, { mirror: this.active.mirror ?? source.defaultMirror });

    const input = this.buildNetworkInput(captured, width, height);
    const processed = this.process(input, levels);
    const output = this.applyColorPreservation(processed, captured, width, height);

    this.retainFeedback(output, width, height);

    this.ops.toCanvas(output, captured, this.canvas.width, this.canvas.height, this.active.display);
  }

  /** Mixes the live frame with the warped previous output, or passes the live frame straight through. */
  private buildNetworkInput(captured: GpuTensor, width: number, height: number): GpuTensor {
    const { feedback } = this.active;
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
      { clamp: 'soft' },
    );
    this.ops.pool.release(warped);
    return mixed;
  }

  private process(input: GpuTensor, levels: ArrayLike<number> | null): GpuTensor {
    switch (this.active.processor) {
      case 'model': {
        if (!this.model) return input;
        this.model.setControls(this.resolveControls(levels));
        return this.active.modelOctaves > 1 ? this.runModelOctaves(input) : this.model.forward(input);
      }
      case 'shallow':
        return this.shallow.run(input, this.active.shallow);
      default:
        return input;
    }
  }

  /**
   * Constrains the processor's colours against the captured frame, when asked to.
   *
   * At zero this is skipped entirely rather than run with an amount of zero — it is a full-frame
   * pass, and the common case is not wanting it.
   */
  private applyColorPreservation(
    processed: GpuTensor,
    captured: GpuTensor,
    width: number,
    height: number,
  ): GpuTensor {
    const amount = this.active.colorPreservation;
    if (amount <= 0) return processed;

    const preserved = this.ops.pool.acquire({ width, height, channels: 3 });
    this.ops.preserveColor(processed, captured, preserved, Math.min(1, amount));
    return preserved;
  }

  /**
   * Runs the model at several resolutions and swaps coarse structure into the full-resolution pass.
   *
   * The network draws its motifs at a size fixed in its own input pixels, so a half-size frame makes
   * everything it draws twice as large relative to the picture. That is the same lever `--style-size`
   * pulls during training — the one separating pandas from pandas-big — available here for the cost
   * of a second pass rather than a second model.
   *
   * Written as a band swap rather than a Laplacian sum. Building the frame up from the coarsest pass
   * and adding high-pass bands is the textbook merge, and it has a defect here: instance
   * normalization makes the model's output level depend on its input resolution, so the coarse pass
   * returns at a different exposure and using it as the base shifted the whole frame. Replacing one
   * band of the full-resolution result instead keeps that result's own level, and makes the strength
   * continuous — at a mix of zero this is exactly the single pass it was before.
   *
   * Levels run finest to coarsest, so each swap refines a sub-band of the one before rather than
   * overwriting it.
   */
  private runModelOctaves(input: GpuTensor): GpuTensor {
    const model = this.model!;
    const { pool } = this.ops;
    const octaves = Math.max(1, Math.min(4, Math.round(this.active.modelOctaves)));
    const scale = Math.max(1.2, this.active.modelOctaveScale);
    // Split across the swaps rather than applied in full at each one. Each swap displaces the
    // result a little further from the single pass, and at three or four octaves applying the whole
    // mix every time compounds into a visible exposure drift. This way the control means the total
    // amount of coarse structure, which is what someone moving it is actually asking for.
    const swaps = Math.max(1, octaves - 1);
    const mix = Math.max(0, Math.min(1, this.active.modelScaleMix)) / swaps;

    const full = { width: input.width, height: input.height, channels: 3 };

    // Multiples of eight: the network halves its resolution twice and doubles it back, and an odd
    // size returns a frame a pixel or two off the one it was given.
    const sizeAt = (level: number) => {
      const factor = Math.pow(scale, level);
      return {
        width: Math.max(32, Math.round(input.width / factor / 8) * 8),
        height: Math.max(32, Math.round(input.height / factor / 8) * 8),
      };
    };

    let result = model.forward(input);

    for (let level = 1; level < octaves && mix > 0; level++) {
      const size = sizeAt(level);

      const levelInput = pool.acquire({ ...size, channels: 3 });
      this.ops.resize(input, levelInput, 'linear');
      const dreamed = model.forward(levelInput);

      // This level's contribution, at full size. It carries nothing finer than its own resolution,
      // so upsampling it *is* its low band — no separate low-pass needed on this side.
      const coarse = pool.acquire(full);
      this.ops.resize(dreamed, coarse, 'linear');
      pool.release(levelInput);

      // The band of the current result that is about to be replaced.
      const small = pool.acquire({ ...size, channels: 3 });
      this.ops.resize(result, small, 'linear');
      const lowOfResult = pool.acquire(full);
      this.ops.resize(small, lowOfResult, 'linear');
      pool.release(small);

      const merged = pool.acquire(full);
      this.ops.combine(
        merged,
        [
          { tensor: result, weight: 1 },
          { tensor: coarse, weight: mix },
          { tensor: lowOfResult, weight: -mix },
        ],
        { clamp: 'soft' },
      );

      pool.release(lowOfResult);
      pool.release(coarse);
      result = merged;
    }

    return result;
  }

  /**
   * The control vector for this frame: the sliders, or the spectrum, or a blend.
   *
   * Read here rather than pushed in from the UI because it changes every frame. Routing sixty
   * updates a second through React state would cost far more than the forward pass it is feeding.
   */
  private resolveControls(levels: ArrayLike<number> | null): number[] {
    const base = this.active.modelControls;
    const { enabled, amount, gain, sensitivity, assignments } = this.active.audio;

    if (!enabled || !levels) return base;

    return mapLevelsToControls(base, levels, { assignments, gain, sensitivity, amount });
  }

  /** Copies this frame's output into the persistent buffer the next frame will warp and mix in. */
  private retainFeedback(output: GpuTensor, width: number, height: number): void {
    if (!this.active.feedback.enabled) {
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
    this.ops.combine(this.previous, [{ tensor: output, weight: 1 }], { clamp: 'soft' });
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
    // Clamped rather than trusted. On a phone a capture size that would be merely slow on a desktop
    // is instead the allocation that gets the tab killed, and the failure gives no error to report.
    const longest = Math.max(64, Math.min(getDeviceLimits().maxCaptureSize, this.config.captureSize));

    // With fillScreen the capture is shaped like the display rather than like the camera, and
    // `fromSource` centre-crops into it — so the picture goes edge to edge with nothing stretched.
    const container = this.canvas.parentElement;
    const displayAspect =
      this.config.fillScreen && container && container.clientHeight > 0
        ? container.clientWidth / container.clientHeight
        : 0;
    const aspect = displayAspect > 0 ? displayAspect : source.width / source.height;
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
  /**
   * Median wall time for one fully-serialized frame at the current configuration.
   *
   * Public because the cost of a *model* is as worth measuring as the cost of a capture size, and
   * that comparison has to be driven from outside — it means loading a different model between
   * measurements, which the engine has no business orchestrating.
   */
  measure(frames = 12): number {
    const frame = this.source?.frame;
    if (!frame) throw new Error('Nothing to measure: no source is running.');

    // Warm-up covers shader compilation for any new variant and the first allocation of every
    // tensor shape at this size. Timing those would measure the compiler, not the pipeline.
    for (let i = 0; i < 3; i++) {
      this.drawOnce(frame);
      this.ops.endFrame();
    }
    this.ops.finish();

    const samples: number[] = [];
    for (let i = 0; i < frames; i++) {
      const start = performance.now();
      this.drawOnce(frame);
      this.ops.finish();
      samples.push(performance.now() - start);
      this.ops.endFrame();
    }

    samples.sort((a, b) => a - b);
    return samples[Math.floor(samples.length / 2)];
  }

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
        // Each size in the sweep has its own set of buffers, and eleven sizes' worth held at once
        // is exactly the allocation spike this is meant to be measuring around.
        this.ops.trim();

        const median = this.measure(framesPerSize);
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
      contextLost: this.contextLost,
      captureWidth: this.captureWidth,
      captureHeight: this.captureHeight,
      timing: this.timer.snapshot(),
      supportsGpuTiming: this.timer.supportsGpuTiming,
      poolMegabytes: this.ops.pool.bytesHeld / (1024 * 1024),
      programCount: this.ops.programCount,
      renderer: this.ctx.caps.rendererName,
      modelName: this.model?.name ?? null,
      audioLevels: this.config.audio.enabled && this.audio ? this.audioLevels : [],
      error: this.error,
    };
  }

  dispose(): void {
    this.stop();
    this.source?.dispose();
    this.audio?.stop();
    this.model?.dispose();
    this.previous?.dispose();
    this.shallow.dispose();
    this.timer.dispose();
    this.ops.dispose();
  }
}
