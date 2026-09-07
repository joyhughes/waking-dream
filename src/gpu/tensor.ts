import { groupsFor, type GlContext } from './gl';
import { getDeviceLimits } from '../pipeline/deviceLimits';

/**
 * A feature map living entirely in GPU memory.
 *
 * The layout is NHWC with the channel axis split across the layers of a `TEXTURE_2D_ARRAY`, four
 * channels to a layer. That choice is what lets a convolution read a whole channel group with one
 * `texelFetch` and write up to `maxDrawBuffers` groups per draw, and it is also why nothing in the
 * pipeline ever has to come back to the CPU: a tensor is a render target and a sampler source at
 * the same time, so the camera frame, the network's activations, and the previous output frame all
 * stay in the same address space.
 *
 * Activations are `RGBA16F`. Half precision is ample for a small feed-forward network at inference
 * — the error is far below what the tanh at the end and the eventual 8-bit display would keep —
 * and it halves the bandwidth, which is what actually limits a fragment-shader convolution.
 */
export class GpuTensor {
  readonly texture: WebGLTexture;
  readonly width: number;
  readonly height: number;
  /** Padded to a multiple of four. `channels` is what the model declares; this is what is stored. */
  readonly groups: number;
  /**
   * The channel count the current holder declared. Not readonly, because the pool recycles by
   * padded group count: a buffer allocated for 30 channels serves a 32-channel request byte for
   * byte, and whoever holds it next must see 32 rather than the stale 30 — anything that reads
   * this to size a readback or a copy would otherwise silently work on the wrong number.
   */
  channels: number;

  constructor(
    private readonly ctx: GlContext,
    width: number,
    height: number,
    channels: number,
    internalFormat: number = ctx.gl.RGBA16F,
  ) {
    const { gl, caps } = ctx;
    this.width = width;
    this.height = height;
    this.channels = channels;
    this.groups = groupsFor(channels);

    if (this.groups > caps.maxArrayLayers) {
      throw new Error(`A ${channels}-channel tensor needs ${this.groups} array layers; this GPU allows ${caps.maxArrayLayers}.`);
    }
    if (width > caps.maxTextureSize || height > caps.maxTextureSize) {
      throw new Error(`A ${width}x${height} tensor exceeds this GPU's ${caps.maxTextureSize}px texture limit.`);
    }

    const texture = gl.createTexture();
    if (!texture) throw new Error('Could not allocate a texture for a tensor.');
    this.texture = texture;

    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internalFormat, width, height, this.groups);
    // Clamping is not a default worth relying on: the convolution clamps its own sampling
    // coordinates, but the resize and warp shaders sample with filtering and would wrap without it.
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  }

  get byteLength(): number {
    return this.width * this.height * this.groups * 4 * 2;
  }

  dispose(): void {
    this.ctx.gl.deleteTexture(this.texture);
  }
}

/** The shape half of a tensor, used to ask the pool for one without having it yet. */
export interface TensorShape {
  width: number;
  height: number;
  channels: number;
}

/**
 * Recycles tensors between frames, and — just as importantly — lets go of them again.
 *
 * A single forward pass allocates and drops a dozen intermediates, and at 60fps that is a texture
 * allocation every millisecond or so, which drivers start stalling on. So the pool keys tensors on
 * their padded shape and hands the same buffers back out each frame.
 *
 * The part that has to be got right is eviction. Buckets are keyed by exact pixel size, and the
 * capture-size control moves in steps of eight — so dragging it from 64 to 1024 asks for a hundred
 * and twenty distinct sizes in a couple of seconds. Holding all of them costs gigabytes and the
 * browser responds by killing the renderer, which the user sees as the page resetting itself with
 * no error anywhere. Buckets therefore carry the frame they were last used on, and anything that
 * has gone unused for a moment is deleted; past a byte budget the least recently used go early.
 */
/** How long a shape may sit unused before its buffers are given back. About a second and a half. */
const MAX_IDLE_FRAMES = 90;

/**
 * Ceiling on pooled texture memory, taken from the device rather than fixed.
 *
 * The pool is not the only thing holding GPU memory — the model's weights, the feedback buffer, and
 * the browser's own copy of every video frame all sit outside it — so the budget is well under what
 * a tab is allowed. On a phone that allowance is both much smaller and enforced by the process
 * being killed rather than by an error, which is why `deviceLimits` sets it much lower there.
 */
function defaultBudget(): number {
  return getDeviceLimits().poolBudgetBytes;
}

interface Bucket {
  tensors: GpuTensor[];
  lastUsedFrame: number;
}

export class TensorPool {
  private free = new Map<string, Bucket>();
  private live: GpuTensor[] = [];
  private frame = 0;
  private freeBytes = 0;

  constructor(
    private readonly ctx: GlContext,
    private readonly budgetBytes: number = defaultBudget(),
  ) {}

  private static key(shape: TensorShape): string {
    return `${shape.width}x${shape.height}x${groupsFor(shape.channels)}`;
  }

  acquire(shape: TensorShape): GpuTensor {
    const key = TensorPool.key(shape);
    const bucket = this.free.get(key);

    if (bucket) {
      bucket.lastUsedFrame = this.frame;
      const reused = bucket.tensors.pop();
      if (reused) {
        this.freeBytes -= reused.byteLength;
        // The pool recycles by padded group count, so a buffer allocated for 30 channels serves a
        // 32-channel request byte for byte. Whoever holds it next must see the count they asked
        // for, not the stale one.
        reused.channels = shape.channels;
        this.live.push(reused);
        return reused;
      }
    }

    const tensor = new GpuTensor(this.ctx, shape.width, shape.height, shape.channels);
    this.live.push(tensor);
    return tensor;
  }

  private park(tensor: GpuTensor): void {
    const key = TensorPool.key(tensor);
    let bucket = this.free.get(key);
    if (!bucket) {
      bucket = { tensors: [], lastUsedFrame: this.frame };
      this.free.set(key, bucket);
    }
    bucket.tensors.push(tensor);
    bucket.lastUsedFrame = this.frame;
    this.freeBytes += tensor.byteLength;
  }

  /** Hands one tensor back early, for the long chains where holding every intermediate would be wasteful. */
  release(tensor: GpuTensor): void {
    const index = this.live.indexOf(tensor);
    if (index < 0) return;
    this.live.splice(index, 1);
    this.park(tensor);
  }

  /** Called once per frame. Anything the frame is still using must have been detached first. */
  releaseAll(): void {
    for (const tensor of this.live) this.park(tensor);
    this.live.length = 0;
    this.frame++;
    this.evict();
  }

  /**
   * Drops idle buckets, then oldest-first until the budget is met.
   *
   * Only free tensors are ever deleted. A live one is being read or written by work the GPU has not
   * necessarily finished, and deleting its texture would be a use-after-free with no error to
   * catch — just a frame of garbage, or a driver reset.
   */
  private evict(): void {
    for (const [key, bucket] of this.free) {
      if (this.frame - bucket.lastUsedFrame <= MAX_IDLE_FRAMES) continue;
      for (const tensor of bucket.tensors) {
        this.freeBytes -= tensor.byteLength;
        tensor.dispose();
      }
      this.free.delete(key);
    }

    if (this.freeBytes <= this.budgetBytes) return;

    const oldest = [...this.free.entries()].sort((a, b) => a[1].lastUsedFrame - b[1].lastUsedFrame);
    for (const [key, bucket] of oldest) {
      if (this.freeBytes <= this.budgetBytes) break;
      for (const tensor of bucket.tensors) {
        this.freeBytes -= tensor.byteLength;
        tensor.dispose();
      }
      this.free.delete(key);
    }
  }

  /**
   * Frees every pooled buffer immediately.
   *
   * Used when the capture size changes, where waiting out the idle timer means briefly holding two
   * full sets of buffers — and the whole point of the size control is that people sweep it.
   */
  trim(): void {
    for (const bucket of this.free.values()) {
      for (const tensor of bucket.tensors) tensor.dispose();
    }
    this.free.clear();
    this.freeBytes = 0;
  }

  /** Takes a tensor out of the pool's hands entirely, so it survives `releaseAll`. Caller now owns it. */
  detach(tensor: GpuTensor): GpuTensor {
    const index = this.live.indexOf(tensor);
    if (index >= 0) this.live.splice(index, 1);
    return tensor;
  }

  get bytesHeld(): number {
    let total = this.freeBytes;
    for (const tensor of this.live) total += tensor.byteLength;
    return total;
  }

  dispose(): void {
    this.trim();
    for (const tensor of this.live) tensor.dispose();
    this.live.length = 0;
  }
}
