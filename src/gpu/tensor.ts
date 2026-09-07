import { groupsFor, type GlContext } from './gl';

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
 * Recycles tensors between frames.
 *
 * A single forward pass allocates and drops a dozen intermediates, and at 60fps that is a texture
 * allocation every millisecond or so — enough that drivers start stalling on it. The pool keys on
 * the padded shape rather than the declared channel count, so a 30-channel request happily reuses a
 * 32-channel buffer, and every tensor handed out is returned at the end of the frame by `releaseAll`.
 */
export class TensorPool {
  private free = new Map<string, GpuTensor[]>();
  private live: GpuTensor[] = [];

  constructor(private readonly ctx: GlContext) {}

  private static key(shape: TensorShape): string {
    return `${shape.width}x${shape.height}x${groupsFor(shape.channels)}`;
  }

  acquire(shape: TensorShape): GpuTensor {
    const key = TensorPool.key(shape);
    const bucket = this.free.get(key);
    const reused = bucket?.pop();
    const tensor = reused ?? new GpuTensor(this.ctx, shape.width, shape.height, shape.channels);
    tensor.channels = shape.channels;
    this.live.push(tensor);
    return tensor;
  }

  /** Hands one tensor back early, for the long chains where holding every intermediate would be wasteful. */
  release(tensor: GpuTensor): void {
    const index = this.live.indexOf(tensor);
    if (index < 0) return;
    this.live.splice(index, 1);
    const key = TensorPool.key(tensor);
    const bucket = this.free.get(key);
    if (bucket) bucket.push(tensor);
    else this.free.set(key, [tensor]);
  }

  /** Called once per frame. Anything the frame is still using must have been detached first. */
  releaseAll(): void {
    for (const tensor of this.live) {
      const key = TensorPool.key(tensor);
      const bucket = this.free.get(key);
      if (bucket) bucket.push(tensor);
      else this.free.set(key, [tensor]);
    }
    this.live.length = 0;
  }

  /** Takes a tensor out of the pool's hands entirely, so it survives `releaseAll`. Caller now owns it. */
  detach(tensor: GpuTensor): GpuTensor {
    const index = this.live.indexOf(tensor);
    if (index >= 0) this.live.splice(index, 1);
    return tensor;
  }

  get bytesHeld(): number {
    let total = 0;
    for (const bucket of this.free.values()) for (const t of bucket) total += t.byteLength;
    for (const t of this.live) total += t.byteLength;
    return total;
  }

  dispose(): void {
    for (const bucket of this.free.values()) for (const t of bucket) t.dispose();
    for (const t of this.live) t.dispose();
    this.free.clear();
    this.live.length = 0;
  }
}
