import type { GlContext } from '../gpu/gl';
import { DataTexture, type Ops } from '../gpu/ops';
import type { GpuTensor } from '../gpu/tensor';
import { AffineEvaluator } from './conditioning';
import { parseDnw, type ModelOp, type ParsedModel } from './format';

/**
 * A trained DreamNet, executing its op list against the GPU ops.
 *
 * The network is the fast-style-transfer shape from Johnson et al. 2016 — a wide first convolution,
 * two stride-2 downsamples, a stack of residual blocks, two upsamples, and a wide convolution back
 * to RGB — because that shape is the reason any of this runs in real time. The residual blocks,
 * which is where nearly all the arithmetic is, run at a quarter of the capture resolution in each
 * direction, so they cost a sixteenth of what they would at full size. Everything at full
 * resolution is one convolution in and one out.
 *
 * What is different here is the training target rather than the architecture: `train/` fits this
 * network to the output of an iterative DeepDream run rather than to a style image, so a single
 * forward pass lands where a few hundred gradient-ascent steps would have.
 */
export class DreamNet {
  readonly parsed: ParsedModel;
  readonly affine: AffineEvaluator;
  private readonly weightTexture: DataTexture;
  private readonly affineTexture: DataTexture;
  /** For each op, the index of the last op that reads its output. Drives tensor recycling. */
  private readonly lastUse: number[];

  private constructor(
    ctx: GlContext,
    private readonly ops: Ops,
    parsed: ParsedModel,
  ) {
    this.parsed = parsed;
    this.affine = new AffineEvaluator(parsed);

    this.weightTexture = new DataTexture(ctx.gl, Math.max(1, parsed.header.gpuTexels));
    this.weightTexture.upload(parsed.gpu);

    this.affineTexture = new DataTexture(ctx.gl, this.affine.texelCount);

    this.lastUse = computeLastUse(parsed.header.ops, parsed.header.outputName);
  }

  static async load(ctx: GlContext, ops: Ops, url: string): Promise<DreamNet> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not fetch model "${url}": ${response.status} ${response.statusText}`);
    }
    return new DreamNet(ctx, ops, parseDnw(await response.arrayBuffer()));
  }

  static fromBuffer(ctx: GlContext, ops: Ops, buffer: ArrayBuffer): DreamNet {
    return new DreamNet(ctx, ops, parseDnw(buffer));
  }

  get name(): string {
    return this.parsed.header.name;
  }

  get description(): string {
    return this.parsed.header.description;
  }

  get controls() {
    return this.affine.controls;
  }

  get trainedAt(): number | undefined {
    return this.parsed.header.trainedAt;
  }

  /** Rewrites the affine texture for a new control vector. A no-op when the controls have not moved. */
  setControls(controls: number[]): void {
    const buffer = this.affine.evaluate(controls);
    if (buffer) this.affineTexture.upload(buffer);
  }

  /**
   * One forward pass. `input` is an RGB tensor and is never written to or released; the returned
   * tensor comes from the pool and is valid until the caller's next `releaseAll`.
   */
  forward(input: GpuTensor): GpuTensor {
    const { header } = this.parsed;
    const tensors = new Map<string, GpuTensor>([[header.inputName, input]]);

    header.ops.forEach((op, index) => {
      const source = tensors.get(op.in);
      if (!source) throw new Error(`Op ${index} reads "${op.in}", which is not live.`);

      let output: GpuTensor;
      switch (op.type) {
        case 'conv': {
          output = this.ops.pool.acquire({
            width: Math.ceil(source.width / op.stride),
            height: Math.ceil(source.height / op.stride),
            channels: op.outChannels,
          });
          this.ops.conv(source, output, this.weightTexture, {
            kernel: op.kernel,
            stride: op.stride,
            weightOffset: op.weightOffset,
            biasOffset: op.biasOffset,
            activation: op.activation,
          });
          break;
        }
        case 'norm': {
          const skip = op.skip ? tensors.get(op.skip) ?? null : null;
          if (op.skip && !skip) throw new Error(`Op ${index} adds skip "${op.skip}", which is not live.`);
          output = this.ops.pool.acquire({ width: source.width, height: source.height, channels: op.channels });
          this.ops.instanceNorm(source, output, this.affineTexture, this.affine.slotOffsets[op.slot], {
            relu: op.relu,
            skip,
          });
          break;
        }
        case 'resize': {
          output = this.ops.pool.acquire({
            width: source.width * op.scale,
            height: source.height * op.scale,
            channels: source.channels,
          });
          this.ops.resize(source, output, 'nearest');
          break;
        }
      }

      tensors.set(op.out, output);

      // Hand back everything whose last reader was this op. Without this a forward pass holds every
      // intermediate at once, which at 512px and 64 channels is a few hundred megabytes of texture
      // for no reason — and the pool would keep growing to that high-water mark for the session.
      header.ops.forEach((earlier, earlierIndex) => {
        if (earlierIndex >= index) return;
        if (this.lastUse[earlierIndex] !== index) return;
        const stale = tensors.get(earlier.out);
        if (stale && stale !== input) {
          this.ops.pool.release(stale);
          tensors.delete(earlier.out);
        }
      });
    });

    const result = tensors.get(header.outputName);
    if (!result) throw new Error(`The model produced no "${header.outputName}".`);
    return result;
  }

  /** Rough parameter count, for the model list. Bias texels are a rounding error and are included. */
  get parameterCount(): number {
    return this.parsed.header.gpuTexels * 4 + this.parsed.cpu.length;
  }

  get byteLength(): number {
    return (this.parsed.gpu.length + this.parsed.cpu.length) * 4;
  }

  dispose(): void {
    this.weightTexture.dispose();
    this.affineTexture.dispose();
  }
}

/**
 * For each op, which later op last reads its output — or the op's own index when nothing does,
 * which for the final op is what marks the network's result.
 */
function computeLastUse(ops: ModelOp[], outputName: string): number[] {
  const lastUse = ops.map((_, index) => index);

  ops.forEach((op, index) => {
    const reads = op.type === 'norm' && op.skip ? [op.in, op.skip] : [op.in];
    for (const name of reads) {
      for (let earlier = index - 1; earlier >= 0; earlier--) {
        if (ops[earlier].out === name) {
          lastUse[earlier] = index;
          break;
        }
      }
    }
  });

  // The declared output has to outlive the loop that produced it, so its producer is pinned.
  for (let index = ops.length - 1; index >= 0; index--) {
    if (ops[index].out === outputName) {
      lastUse[index] = ops.length;
      break;
    }
  }

  return lastUse;
}
