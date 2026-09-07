import type { GlContext } from './gl';
import { ProgramCache, RenderTargets } from './program';
import { GpuTensor, TensorPool } from './tensor';
import {
  REDUCE_BLOCK,
  WEIGHT_TEX_WIDTH,
  combineShader,
  convShader,
  fromSourceShader,
  normShader,
  reduceFrom2DShader,
  reduceFromTensorShader,
  resizeShader,
  toCanvasShader,
  warpShader,
  type ConvActivation,
} from './shaders';

/**
 * How many channel groups one convolution draw writes at once.
 *
 * Four is a deliberate stop short of the eight colour attachments Apple GPUs expose. Every extra
 * target adds four live accumulators to the fragment shader, and past a point the occupancy lost to
 * register pressure costs more than the input bandwidth saved. Four measured fastest on an M3; the
 * value is worth re-checking on other hardware, which is why it is one constant and not scattered.
 */
const DEFAULT_MAX_TARGETS = 4;

/** Guards the variance in instance normalization, matching PyTorch's `InstanceNorm2d` default. */
const NORM_EPSILON = 1e-5;

export interface ConvSpec {
  kernel: number;
  stride: number;
  /** Texel index of this layer's first weight in the packed weight texture. */
  weightOffset: number;
  /** Texel index of this layer's first bias group. */
  biasOffset: number;
  activation: ConvActivation;
}

/** A statistics pair from the reduction: one texel per channel group, sums over the whole image. */
interface Stats {
  sum: WebGLTexture;
  sumSq: WebGLTexture;
  count: number;
}

/**
 * A float texture holding numbers the shaders index by a flat texel number: model weights (written
 * once at load) and the per-channel affine parameters (rewritten every frame from the control MLP).
 */
export class DataTexture {
  readonly texture: WebGLTexture;
  readonly height: number;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    texelCount: number,
  ) {
    this.height = Math.max(1, Math.ceil(texelCount / WEIGHT_TEX_WIDTH));
    const texture = gl.createTexture();
    if (!texture) throw new Error('Could not allocate a data texture.');
    this.texture = texture;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, WEIGHT_TEX_WIDTH, this.height);
    // Only ever read with texelFetch, so filtering is irrelevant — but NEAREST is required for a
    // 32-bit float texture to be complete without OES_texture_float_linear.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** `data` is a flat float array of 4-component texels, padded by the caller to a whole row count. */
  upload(data: Float32Array): void {
    const { gl } = this;
    const needed = WEIGHT_TEX_WIDTH * this.height * 4;
    const padded = data.length === needed ? data : (() => {
      const buffer = new Float32Array(needed);
      buffer.set(data.subarray(0, Math.min(data.length, needed)));
      return buffer;
    })();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WEIGHT_TEX_WIDTH, this.height, gl.RGBA, gl.FLOAT, padded);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  dispose(): void {
    this.gl.deleteTexture(this.texture);
  }
}

/**
 * Every GPU operation the network and the pipeline are built from.
 *
 * The ops are deliberately thin: each one binds a generated program, points the framebuffer at some
 * layers of a tensor, and draws a triangle. Nothing here reads back to the CPU, and nothing here
 * allocates outside the pool, so a whole frame — capture, forward pass, feedback warp, composite —
 * is one uninterrupted stream of GPU work.
 */
export class Ops {
  readonly pool: TensorPool;
  private readonly programs: ProgramCache;
  private readonly targets: RenderTargets;
  private readonly maxTargets: number;
  private readonly scratch2D = new Map<string, WebGLTexture>();
  private sourceTexture: WebGLTexture | null = null;

  constructor(private readonly ctx: GlContext) {
    this.pool = new TensorPool(ctx);
    this.programs = new ProgramCache(ctx);
    this.targets = new RenderTargets(ctx);
    this.maxTargets = Math.max(1, Math.min(DEFAULT_MAX_TARGETS, ctx.caps.maxDrawBuffers));
  }

  get programCount(): number {
    return this.programs.size;
  }

  /**
   * Convolution, issued as one draw per `maxTargets` output channel groups.
   *
   * Input and output are separate tensors even when the shapes match: a fragment shader that both
   * samples and renders to the same texture is undefined behaviour, and the failure mode is not a
   * crash but plausible-looking garbage that only appears on some drivers.
   */
  conv(input: GpuTensor, output: GpuTensor, weights: DataTexture, spec: ConvSpec): void {
    for (let base = 0; base < output.groups; base += this.maxTargets) {
      const count = Math.min(this.maxTargets, output.groups - base);
      const program = this.programs.get(
        convShader({ kernel: spec.kernel, targets: count, stride: spec.stride, activation: spec.activation }),
      );
      this.targets.drawInto(output, base, count, () => {
        program
          .use()
          .tensor('uInput', input)
          .texture2d('uWeights', weights.texture)
          .ivec2('uInSize', input.width, input.height)
          .int('uGIn', input.groups)
          .int('uGOutTotal', output.groups)
          .int('uGOutBase', base)
          .int('uWOffset', spec.weightOffset)
          .int('uBOffset', spec.biasOffset);
      });
    }
  }

  /**
   * Instance normalization: per-image, per-channel mean and variance, then a per-channel affine.
   *
   * `affineOffset` points at this layer's gammas in the affine texture; the betas follow one full
   * set of groups later. `skip` fuses a residual add into the same pass, and `relu` fuses the
   * activation — both are free here and each would otherwise be a full extra read and write of the
   * whole feature map.
   */
  instanceNorm(
    input: GpuTensor,
    output: GpuTensor,
    affine: DataTexture,
    affineOffset: number,
    options: { relu: boolean; skip?: GpuTensor | null },
  ): void {
    const stats = this.reduceStats(input);
    const skip = options.skip ?? null;

    for (let base = 0; base < output.groups; base += this.maxTargets) {
      const count = Math.min(this.maxTargets, output.groups - base);
      const program = this.programs.get(
        normShader({ targets: count, relu: options.relu, residual: skip !== null }),
      );
      this.targets.drawInto(output, base, count, () => {
        program.use().tensor('uInput', input);
        if (skip) program.tensor('uSkip', skip);
        program
          .texture2d('uSum', stats.sum)
          .texture2d('uSumSq', stats.sumSq)
          .texture2d('uAffine', affine.texture)
          .int('uAffineOffset', affineOffset)
          .int('uGTotal', input.groups)
          .int('uGroupBase', base)
          .float('uInvN', 1 / stats.count)
          .float('uEps', NORM_EPSILON);
      });
    }
  }

  /**
   * Folds a feature map down to one texel per channel group holding the sum and the sum of squares.
   *
   * The chain shrinks by a factor of `REDUCE_BLOCK` in each direction per pass, so a 256x256 map is
   * three passes and a 64x64 one is two. Scratch targets are cached by size and reused across
   * layers and frames — there are only a handful of distinct sizes in a given configuration.
   */
  private reduceStats(input: GpuTensor): Stats {
    const groups = input.groups;
    const count = input.width * input.height;

    let tileW = Math.ceil(input.width / REDUCE_BLOCK);
    let tileH = Math.ceil(input.height / REDUCE_BLOCK);

    let sum = this.scratchTexture(tileW * groups, tileH, 'sum');
    let sumSq = this.scratchTexture(tileW * groups, tileH, 'sq');

    const first = this.programs.get(reduceFromTensorShader());
    this.targets.drawInto2D([sum, sumSq], tileW * groups, tileH, () => {
      first.use().tensor('uInput', input).ivec2('uInSize', input.width, input.height).int('uOutTileW', tileW);
    });

    const step = this.programs.get(reduceFrom2DShader());
    while (tileW > 1 || tileH > 1) {
      const nextW = Math.ceil(tileW / REDUCE_BLOCK);
      const nextH = Math.ceil(tileH / REDUCE_BLOCK);
      const nextSum = this.scratchTexture(nextW * groups, nextH, 'sum');
      const nextSumSq = this.scratchTexture(nextW * groups, nextH, 'sq');

      const inTileW = tileW;
      const inHeight = tileH;
      const sourceSum = sum;
      const sourceSumSq = sumSq;
      this.targets.drawInto2D([nextSum, nextSumSq], nextW * groups, nextH, () => {
        step
          .use()
          .texture2d('uSum', sourceSum)
          .texture2d('uSumSq', sourceSumSq)
          .int('uInTileW', inTileW)
          .int('uInHeight', inHeight)
          .int('uOutTileW', nextW);
      });

      sum = nextSum;
      sumSq = nextSumSq;
      tileW = nextW;
      tileH = nextH;
    }

    return { sum, sumSq, count };
  }

  private scratchTexture(width: number, height: number, role: string): WebGLTexture {
    const key = `${role}:${width}x${height}`;
    const existing = this.scratch2D.get(key);
    if (existing) return existing;

    const { gl } = this.ctx;
    const texture = gl.createTexture();
    if (!texture) throw new Error('Could not allocate a reduction scratch texture.');
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.scratch2D.set(key, texture);
    return texture;
  }

  /** Resamples one channel group. The network only ever upsamples RGB-width or feature-width group 0..n. */
  resize(input: GpuTensor, output: GpuTensor, filter: 'nearest' | 'linear' = 'linear'): void {
    const program = this.programs.get(resizeShader(filter));
    for (let group = 0; group < output.groups; group++) {
      this.targets.drawInto(output, group, 1, () => {
        program
          .use()
          .tensor('uInput', input)
          .ivec2('uInSize', input.width, input.height)
          .vec2('uOutSizeInv', 1 / output.width, 1 / output.height)
          .int('uGroupBase', group);
      });
    }
  }

  /** `output = wa*a + wb*b + wc*c + bias`, over the first channel group only. */
  combine(
    output: GpuTensor,
    terms: { tensor: GpuTensor; weight: number }[],
    options: { bias?: number; clamp01?: boolean } = {},
  ): void {
    const count = terms.length as 1 | 2 | 3;
    const program = this.programs.get(combineShader(count, options.clamp01 ?? false));
    this.targets.drawInto(output, 0, 1, () => {
      program.use();
      const names = ['uA', 'uB', 'uC'];
      const weightNames = ['uWa', 'uWb', 'uWc'];
      terms.forEach((term, i) => program.tensor(names[i], term.tensor));
      terms.forEach((term, i) => program.float(weightNames[i], term.weight));
      program.float('uBias', options.bias ?? 0).int('uGroupBase', 0);
    });
  }

  /**
   * Uploads a frame and centre-crops it into a tensor.
   *
   * `texImage2D` from a video element is the one place per frame where data crosses into the GPU,
   * and on every browser worth targeting it is a zero-copy path from the decoder, not a readback.
   */
  fromSource(source: TexImageSource, output: GpuTensor, options: { mirror: boolean }): void {
    const { gl } = this.ctx;

    this.sourceTexture ??= (() => {
      const texture = gl.createTexture();
      if (!texture) throw new Error('Could not allocate the source texture.');
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return texture;
    })();

    gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    const sourceWidth = sourceDimension(source, 'width');
    const sourceHeight = sourceDimension(source, 'height');
    const sourceAspect = sourceWidth / sourceHeight;
    const outputAspect = output.width / output.height;

    // Take the largest centred rectangle of the source that has the output's aspect ratio.
    const scale: [number, number] =
      sourceAspect > outputAspect ? [outputAspect / sourceAspect, 1] : [1, sourceAspect / outputAspect];
    const offset: [number, number] = [(1 - scale[0]) / 2, (1 - scale[1]) / 2];

    const program = this.programs.get(fromSourceShader());
    this.targets.drawInto(output, 0, 1, () => {
      program
        .use()
        .texture2d('uSource', this.sourceTexture!)
        .vec2('uScale', scale[0], scale[1])
        .vec2('uOffset', offset[0], offset[1])
        .vec2('uOutSizeInv', 1 / output.width, 1 / output.height)
        .float('uMirror', options.mirror ? 1 : 0);
    });
  }

  /** The affine warp applied to the previous output before it re-enters the network. */
  warp(
    input: GpuTensor,
    output: GpuTensor,
    params: { zoom: number; rotate: number; translateX: number; translateY: number; fade: number },
  ): void {
    // The matrix maps a destination point back to where it came from, so a zoom greater than one —
    // which should magnify — samples from a *smaller* region, hence the reciprocal.
    const inverseZoom = 1 / Math.max(1e-3, params.zoom);
    const cos = Math.cos(params.rotate) * inverseZoom;
    const sin = Math.sin(params.rotate) * inverseZoom;
    const aspect: [number, number] = output.width >= output.height
      ? [output.width / output.height, 1]
      : [1, output.height / output.width];

    const program = this.programs.get(warpShader());
    const { gl } = this.ctx;
    this.targets.drawInto(output, 0, 1, () => {
      program
        .use()
        .tensor('uInput', input)
        .vec2('uOutSizeInv', 1 / output.width, 1 / output.height)
        .vec2('uAspect', aspect[0], aspect[1])
        .vec2('uTranslate', params.translateX, params.translateY)
        .float('uFade', params.fade)
        .int('uGroupBase', 0);
      const location = gl.getUniformLocation(program.handle, 'uMatrix');
      if (location) gl.uniformMatrix2fv(location, false, [cos, sin, -sin, cos]);
    });
  }

  /** Composites onto the canvas. The only pass that touches the default framebuffer. */
  toCanvas(
    dreamed: GpuTensor,
    original: GpuTensor,
    canvasWidth: number,
    canvasHeight: number,
    options: { mix: number; gain: number; saturation: number },
  ): void {
    const program = this.programs.get(toCanvasShader());
    this.targets.drawToCanvas(canvasWidth, canvasHeight, () => {
      program
        .use()
        .tensor('uOutput', dreamed)
        .tensor('uOriginal', original)
        .vec2('uOutSizeInv', 1 / canvasWidth, 1 / canvasHeight)
        .float('uMix', options.mix)
        .float('uGain', options.gain)
        .float('uSaturation', options.saturation);
    });
  }

  /** Blocks until the GPU has finished everything queued. Used only by the benchmark. */
  finish(): void {
    this.ctx.gl.finish();
  }

  dispose(): void {
    const { gl } = this.ctx;
    for (const texture of this.scratch2D.values()) gl.deleteTexture(texture);
    this.scratch2D.clear();
    if (this.sourceTexture) gl.deleteTexture(this.sourceTexture);
    this.sourceTexture = null;
    this.programs.dispose();
    this.targets.dispose();
    this.pool.dispose();
  }
}

function sourceDimension(source: TexImageSource, axis: 'width' | 'height'): number {
  if (source instanceof HTMLVideoElement) return axis === 'width' ? source.videoWidth : source.videoHeight;
  if (source instanceof HTMLImageElement) return axis === 'width' ? source.naturalWidth : source.naturalHeight;
  const sized = source as { width?: number; height?: number };
  return (axis === 'width' ? sized.width : sized.height) ?? 1;
}
