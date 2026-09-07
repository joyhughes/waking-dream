import { groupsFor, type GlContext } from '../gpu/gl';
import { packConv } from './pack';
import { DataTexture, type Ops } from '../gpu/ops';
import type { GpuTensor } from '../gpu/tensor';

/**
 * DeepDream with no trained model at all, running at video rate.
 *
 * Classic DeepDream ascends the gradient of a deep network's activations, and the cost is that the
 * gradient needs a backward pass through that network, dozens of times per image. Take the network
 * down to a single convolution followed by a ReLU and the backward pass stops needing a tape: the
 * derivative of `mean(relu(W * x))` with respect to `x` is exactly a convolution of the rectified
 * activations by the spatially flipped, channel-transposed kernel. Two convolutions and the step is
 * done, both of which the runtime already has.
 *
 * What that buys is not a substitute for the distilled model — one layer of oriented filters
 * hallucinates texture and flow, not eyes and animals, because there is no semantic depth in it to
 * hallucinate from. What it does give is the whole rest of the machine working on day one: real
 * ascent, real octaves, real feedback recursion, at a frame rate the trained model has to match.
 * It also stays useful afterwards as a control — anything the trained model does that this does not
 * is what the teacher's depth actually bought.
 */

export type FeatureBank = 'gabor' | 'blob' | 'random';

export interface ShallowDreamParams {
  bank: FeatureBank;
  /** Ascent steps per octave, per frame. Two or three is usually enough with feedback running. */
  steps: number;
  stepSize: number;
  octaves: number;
  /** Resolution ratio between neighbouring octaves. */
  octaveScale: number;
  /** Filters in the bank, rounded up to a multiple of four. More is richer and proportionally slower. */
  filters: number;
  kernel: number;
  seed: number;
  /**
   * A restoring force pulling each step back toward the frame the octave started from.
   *
   * Ascent has no opinion about colour beyond "more response", and left alone it drives every
   * channel outward until the picture is made of saturated primaries. This is the counterweight:
   * the Yosinski et al. 2015 decay regularizer, aimed at the source frame rather than at grey, so
   * it holds the photograph's colours rather than washing them out. 0 removes it entirely.
   */
  colourHold: number;
  /**
   * Normalize the ascent gradient across R, G and B together rather than per channel.
   *
   * Per-channel normalization gives each colour the same step regardless of what the gradient
   * asked for, so the channels drift independently and the image collapses onto the corners of the
   * RGB cube. Sharing one scale rescales the gradient without rotating it, which keeps hue.
   */
  sharedGradient: boolean;
}

/**
 * Defaults tuned for running inside the feedback loop rather than for a single pass.
 *
 * A step size that looks mild applied once compounds sixty times a second: at 0.08 across three
 * octaves the frame is fully saturated within a couple of seconds. These values are what leaves the
 * recursion in a steady state, growing and shedding structure indefinitely without pinning. Turn
 * the step up when feedback is off, where one pass is all the effect there is.
 */
export const DEFAULT_SHALLOW_PARAMS: ShallowDreamParams = {
  bank: 'gabor',
  steps: 1,
  stepSize: 0.03,
  octaves: 3,
  octaveScale: 1.8,
  filters: 16,
  kernel: 5,
  seed: 7,
  colourHold: 0.08,
  sharedGradient: true,
};

/** Small deterministic PRNG, so a given seed always gives the same bank on any machine. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Three colour directions to hang the spatial filters on.
 *
 * A bank built only on luminance produces grey ridges that the feedback loop then desaturates the
 * whole image toward. Splitting the filters across a luminance axis and two opponent-colour axes —
 * roughly the ones the early visual system uses, and the ones the first layer of a trained network
 * reliably learns — keeps colour in the hallucinated structure.
 */
const COLOR_AXES: [number, number, number][] = [
  [0.577, 0.577, 0.577],
  [0.707, -0.707, 0],
  [-0.408, -0.408, 0.816],
];

/**
 * Builds the filter bank as a `[K, K, 3, F]` kernel.
 *
 * Every filter is made zero-mean and then scaled to unit norm. Zero-mean matters more than it
 * looks: a filter with any DC component responds to flat brightness, and since the ascent adds its
 * response back into the image, that response feeds itself and the frame walks to white in a second
 * or two. With the mean removed, flat regions produce no gradient and only structure grows.
 */
function buildBank(params: ShallowDreamParams): { kernel: Float32Array; filters: number } {
  const filters = Math.max(4, Math.ceil(params.filters / 4) * 4);
  const k = params.kernel;
  const radius = (k - 1) / 2;
  const random = mulberry32(params.seed);
  const kernel = new Float32Array(k * k * 3 * filters);

  const set = (ky: number, kx: number, c: number, f: number, value: number) => {
    kernel[((ky * k + kx) * 3 + c) * filters + f] = value;
  };

  for (let f = 0; f < filters; f++) {
    const axis = COLOR_AXES[f % COLOR_AXES.length];
    const values = new Float32Array(k * k);

    if (params.bank === 'random') {
      for (let i = 0; i < values.length; i++) {
        // Box-Muller, so the bank is Gaussian rather than uniform — uniform weights give filters
        // with a flat spectrum and the result reads as noise amplification rather than structure.
        const u = Math.max(1e-6, random());
        values[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
      }
    } else if (params.bank === 'blob') {
      // Difference of Gaussians: a centre-surround detector, which grows dots, cells, and the
      // clustered blob textures that shallow DeepDream layers are known for.
      const inner = 0.6 + 0.5 * random();
      const outer = inner * (1.6 + 0.8 * random());
      const sign = random() < 0.5 ? -1 : 1;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const dy = ky - radius;
          const dx = kx - radius;
          const r2 = dx * dx + dy * dy;
          const center = Math.exp(-r2 / (2 * inner * inner)) / (inner * inner);
          const surround = Math.exp(-r2 / (2 * outer * outer)) / (outer * outer);
          values[ky * k + kx] = sign * (center - surround);
        }
      }
    } else {
      // Gabor: an oriented grating under a Gaussian envelope. Orientations are spread evenly and
      // frequency and phase are jittered, so the bank covers directions without every filter in a
      // direction being a copy of the others.
      const orientation = (Math.PI * Math.floor(f / COLOR_AXES.length)) / Math.max(1, filters / COLOR_AXES.length);
      const frequency = 0.35 + 0.45 * random();
      const phase = 2 * Math.PI * random();
      const sigma = radius * (0.55 + 0.25 * random());
      const cos = Math.cos(orientation);
      const sin = Math.sin(orientation);
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const dy = ky - radius;
          const dx = kx - radius;
          const along = dx * cos + dy * sin;
          const envelope = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
          values[ky * k + kx] = envelope * Math.cos(2 * Math.PI * frequency * along + phase);
        }
      }
    }

    let mean = 0;
    for (const value of values) mean += value;
    mean /= values.length;

    let norm = 0;
    for (let i = 0; i < values.length; i++) {
      values[i] -= mean;
      norm += values[i] * values[i];
    }
    norm = Math.sqrt(norm) || 1;

    for (let ky = 0; ky < k; ky++) {
      for (let kx = 0; kx < k; kx++) {
        const value = values[ky * k + kx] / norm;
        for (let c = 0; c < 3; c++) set(ky, kx, c, f, value * axis[c]);
      }
    }
  }

  return { kernel, filters };
}

interface CompiledBank {
  texture: DataTexture;
  filters: number;
  kernel: number;
  forwardWeights: number;
  forwardBias: number;
  backwardWeights: number;
  backwardBias: number;
}

export class ShallowDream {
  private compiled: CompiledBank | null = null;
  private compiledKey = '';
  private readonly unitAffine: DataTexture;

  constructor(
    private readonly ctx: GlContext,
    private readonly ops: Ops,
  ) {
    // Gamma of one and beta of zero: the normalization layer is used here purely to rescale the
    // ascent direction, not to learn anything.
    this.unitAffine = new DataTexture(ctx.gl, 2);
    this.unitAffine.upload(new Float32Array([1, 1, 1, 1, 0, 0, 0, 0]));
  }

  private compile(params: ShallowDreamParams): CompiledBank {
    const key = `${params.bank}:${params.filters}:${params.kernel}:${params.seed}`;
    if (this.compiled && this.compiledKey === key) return this.compiled;

    this.compiled?.texture.dispose();

    const { kernel, filters } = buildBank(params);
    const k = params.kernel;

    // The backward kernel is the forward one flipped in both spatial axes with input and output
    // channels swapped. That is the whole of the gradient: d/dx of conv(x, W) is conv(·, Wᵀflipped).
    const backward = new Float32Array(k * k * filters * 3);
    for (let ky = 0; ky < k; ky++) {
      for (let kx = 0; kx < k; kx++) {
        const flipped = (k - 1 - ky) * k + (k - 1 - kx);
        for (let c = 0; c < 3; c++) {
          for (let f = 0; f < filters; f++) {
            backward[(flipped * filters + f) * 3 + c] = kernel[((ky * k + kx) * 3 + c) * filters + f];
          }
        }
      }
    }

    const forward = packConv(kernel, k, 3, filters);
    const reverse = packConv(backward, k, filters, 3);

    const forwardBiasTexels = groupsFor(filters);
    const backwardBiasTexels = groupsFor(3);
    const total = forward.texelCount + forwardBiasTexels + reverse.texelCount + backwardBiasTexels;

    const data = new Float32Array(total * 4);
    let cursor = 0;
    data.set(forward.texels, cursor * 4);
    const forwardWeights = cursor;
    cursor += forward.texelCount;
    const forwardBias = cursor;
    cursor += forwardBiasTexels;
    data.set(reverse.texels, cursor * 4);
    const backwardWeights = cursor;
    cursor += reverse.texelCount;
    const backwardBias = cursor;

    const texture = new DataTexture(this.ctx.gl, total);
    texture.upload(data);

    this.compiled = {
      texture,
      filters,
      kernel: k,
      forwardWeights,
      forwardBias,
      backwardWeights,
      backwardBias,
    };
    this.compiledKey = key;
    return this.compiled;
  }

  /**
   * One ascent step at the current resolution. Returns a new pooled tensor; `image` is untouched.
   *
   * `anchor` is the frame this octave started from, which the step is pulled back toward by
   * `colourHold`. Without something to pull against, the only force acting on a pixel is "make the
   * filter response larger", and that force always points outward.
   */
  private step(
    image: GpuTensor,
    anchor: GpuTensor,
    params: ShallowDreamParams,
    bank: CompiledBank,
  ): GpuTensor {
    const { pool } = this.ops;

    const activations = pool.acquire({ width: image.width, height: image.height, channels: bank.filters });
    this.ops.conv(image, activations, bank.texture, {
      kernel: bank.kernel,
      stride: 1,
      weightOffset: bank.forwardWeights,
      biasOffset: bank.forwardBias,
      activation: 'relu',
    });

    const gradient = pool.acquire({ width: image.width, height: image.height, channels: 3 });
    this.ops.conv(activations, gradient, bank.texture, {
      kernel: bank.kernel,
      stride: 1,
      weightOffset: bank.backwardWeights,
      biasOffset: bank.backwardBias,
      activation: 'none',
    });
    pool.release(activations);

    // Rescaling the gradient to unit deviation per channel is what makes `stepSize` mean the same
    // thing on a dark frame as on a bright one. It is the same idea as the Laplacian gradient
    // normalization the classic DeepDream notebooks use, minus the pyramid — the octave loop below
    // is already providing the multi-scale part.
    const normalized = pool.acquire({ width: image.width, height: image.height, channels: 3 });
    this.ops.instanceNorm(gradient, normalized, this.unitAffine, 0, {
      relu: false,
      sharedRgb: params.sharedGradient,
    });
    pool.release(gradient);

    const hold = Math.max(0, Math.min(0.9, params.colourHold));
    const stepped = pool.acquire({ width: image.width, height: image.height, channels: 3 });
    this.ops.combine(
      stepped,
      [
        { tensor: image, weight: 1 - hold },
        { tensor: anchor, weight: hold },
        { tensor: normalized, weight: params.stepSize },
      ],
      { clamp: 'soft' },
    );
    pool.release(normalized);

    return stepped;
  }

  /**
   * Runs the full octave sweep over one frame.
   *
   * Each octave ascends at its own resolution and contributes only the *detail* it added — the
   * difference between where it started and where it finished — resampled up to full size. Passing
   * the whole coarse result up instead would throw away the fine structure of every octave before
   * it, which is the reason a naive multi-scale dream comes out soft.
   */
  run(input: GpuTensor, params: ShallowDreamParams): GpuTensor {
    const bank = this.compile(params);
    const { pool } = this.ops;

    let current = pool.acquire({ width: input.width, height: input.height, channels: 3 });
    this.ops.combine(current, [{ tensor: input, weight: 1 }]);

    for (let octave = params.octaves - 1; octave >= 0; octave--) {
      const factor = Math.pow(params.octaveScale, octave);
      const width = Math.max(8, Math.round(input.width / factor));
      const height = Math.max(8, Math.round(input.height / factor));

      const base = pool.acquire({ width, height, channels: 3 });
      this.ops.resize(current, base, 'linear');

      let small = pool.acquire({ width, height, channels: 3 });
      this.ops.combine(small, [{ tensor: base, weight: 1 }]);

      for (let step = 0; step < params.steps; step++) {
        const next = this.step(small, base, params, bank);
        pool.release(small);
        small = next;
      }

      // detail = small - base, taken at the octave's own resolution and then resampled up, so the
      // interpolation smooths the added detail rather than the image it is added to.
      const detail = pool.acquire({ width, height, channels: 3 });
      this.ops.combine(detail, [
        { tensor: small, weight: 1 },
        { tensor: base, weight: -1 },
      ]);
      pool.release(small);
      pool.release(base);

      const upsampled = pool.acquire({ width: input.width, height: input.height, channels: 3 });
      this.ops.resize(detail, upsampled, 'linear');
      pool.release(detail);

      const merged = pool.acquire({ width: input.width, height: input.height, channels: 3 });
      this.ops.combine(
        merged,
        [
          { tensor: current, weight: 1 },
          { tensor: upsampled, weight: 1 },
        ],
        { clamp: 'soft' },
      );
      pool.release(upsampled);
      pool.release(current);
      current = merged;
    }

    return current;
  }

  dispose(): void {
    this.compiled?.texture.dispose();
    this.compiled = null;
    this.unitAffine.dispose();
  }
}
