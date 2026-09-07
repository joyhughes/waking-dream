import { tf } from './tfSetup';

/**
 * The same network as `train/model.py` and `src/gpu/ops.ts`, expressed in TFJS so it can be trained
 * in the browser.
 *
 * Three implementations of one architecture now exist, and they have to agree exactly or an
 * exported model draws something different from what was trained. The rules they share:
 *
 *   - padding is replicate, matching the runtime shader's clamped sampling coordinate;
 *   - upsampling is nearest-then-convolve, never a transposed convolution;
 *   - the output is tanh mapped onto [0, 1];
 *   - instance normalization is per-image, per-channel, eps 1e-5, with a conditional affine.
 *
 * Every variable's shape is chosen to be exactly what the `.dnw` format wants, so `exportDnw.ts`
 * copies them out with no transposes to get backwards. The self-test builds one of these with
 * random weights, exports it, and checks the WebGL runtime reproduces it — which is what keeps this
 * claim honest rather than aspirational.
 */

/** Matches PyTorch's `InstanceNorm2d` default, and the runtime's `NORM_EPSILON`. */
const NORM_EPSILON = 1e-5;

export interface DreamNetTfConfig {
  width: number;
  blocks: number;
  filmHidden: number;
  condDims: number;
}

export const DEFAULT_TF_CONFIG: DreamNetTfConfig = {
  // Narrower than the PyTorch default of 16. A browser training run has a fraction of the step
  // budget, and a smaller network reaches something worth looking at inside that budget rather
  // than being a better network that never finishes converging.
  width: 8,
  blocks: 3,
  filmHidden: 16,
  condDims: 1,
};

interface ConvVars {
  kernel: tf.Variable<tf.Rank.R4>;
  bias: tf.Variable<tf.Rank.R1>;
  kernelSize: number;
  stride: number;
  inChannels: number;
  outChannels: number;
}

interface NormVars {
  gamma: tf.Variable<tf.Rank.R1>;
  beta: tf.Variable<tf.Rank.R1>;
  gammaProjection: tf.Variable<tf.Rank.R2>;
  betaProjection: tf.Variable<tf.Rank.R2>;
  channels: number;
}

/**
 * Extends a batch outward by repeating its edge pixels.
 *
 * TFJS has `mirrorPad`, but mirroring is not replication: for a pad of four it reflects four pixels
 * back into the frame, which is a different image from the one the runtime's coordinate clamp sees.
 * The first convolution here is 9x9, so that difference is real and would show up as an edge
 * artifact only after export, which is the worst time to find it.
 */
function padReplicate(x: tf.Tensor4D, pad: number): tf.Tensor4D {
  if (pad <= 0) return x;
  return tf.tidy(() => {
    const [batch, height, width, channels] = x.shape;

    const top = x.slice([0, 0, 0, 0], [batch, 1, width, channels]).tile([1, pad, 1, 1]);
    const bottom = x.slice([0, height - 1, 0, 0], [batch, 1, width, channels]).tile([1, pad, 1, 1]);
    const vertical = tf.concat([top, x, bottom], 1) as tf.Tensor4D;

    const tallHeight = vertical.shape[1];
    const left = vertical.slice([0, 0, 0, 0], [batch, tallHeight, 1, channels]).tile([1, 1, pad, 1]);
    const right = vertical.slice([0, 0, width - 1, 0], [batch, tallHeight, 1, channels]).tile([1, 1, pad, 1]);
    return tf.concat([left, vertical, right], 2) as tf.Tensor4D;
  });
}

/** He initialization, which is the right variance for a ReLU stack and what PyTorch's default approximates. */
function heNormal(kernelSize: number, inChannels: number, outChannels: number): tf.Variable<tf.Rank.R4> {
  const fanIn = kernelSize * kernelSize * inChannels;
  const std = Math.sqrt(2 / fanIn);
  return tf.variable(
    tf.randomNormal([kernelSize, kernelSize, inChannels, outChannels], 0, std) as tf.Tensor4D,
  ) as tf.Variable<tf.Rank.R4>;
}

export class DreamNetTf {
  readonly config: DreamNetTfConfig;
  readonly convs: ConvVars[] = [];
  readonly norms: NormVars[] = [];
  readonly filmWeight: tf.Variable<tf.Rank.R2>;
  readonly filmBias: tf.Variable<tf.Rank.R1>;

  constructor(config: Partial<DreamNetTfConfig> = {}) {
    this.config = { ...DEFAULT_TF_CONFIG, ...config };
    const { width, blocks, filmHidden, condDims } = this.config;
    const w1 = width;
    const w2 = width * 2;
    const w4 = width * 4;

    const addConv = (inChannels: number, outChannels: number, kernelSize: number, stride = 1) => {
      this.convs.push({
        kernel: heNormal(kernelSize, inChannels, outChannels),
        bias: tf.variable(tf.zeros([outChannels]) as tf.Tensor1D) as tf.Variable<tf.Rank.R1>,
        kernelSize,
        stride,
        inChannels,
        outChannels,
      });
    };

    addConv(3, w1, 9);
    addConv(w1, w2, 3, 2);
    addConv(w2, w4, 3, 2);
    for (let i = 0; i < 2 * blocks; i++) addConv(w4, w4, 3);
    addConv(w4, w2, 3);
    addConv(w2, w1, 3);
    addConv(w1, 3, 9);

    const normChannels = [w1, w2, w4, ...Array<number>(2 * blocks).fill(w4), w2, w1];
    for (const channels of normChannels) {
      this.norms.push({
        gamma: tf.variable(tf.ones([channels]) as tf.Tensor1D) as tf.Variable<tf.Rank.R1>,
        beta: tf.variable(tf.zeros([channels]) as tf.Tensor1D) as tf.Variable<tf.Rank.R1>,
        // Zero-initialized, so training starts from an unconditioned network and the conditioning
        // grows out of it. Random projections would push noise through every normalization layer
        // before the control vector means anything, which is a much worse starting point.
        gammaProjection: tf.variable(tf.zeros([filmHidden, channels]) as tf.Tensor2D) as tf.Variable<tf.Rank.R2>,
        betaProjection: tf.variable(tf.zeros([filmHidden, channels]) as tf.Tensor2D) as tf.Variable<tf.Rank.R2>,
        channels,
      });
    }

    this.filmWeight = tf.variable(
      tf.randomNormal([condDims, filmHidden], 0, Math.sqrt(2 / Math.max(1, condDims))) as tf.Tensor2D,
    ) as tf.Variable<tf.Rank.R2>;
    this.filmBias = tf.variable(tf.zeros([filmHidden]) as tf.Tensor1D) as tf.Variable<tf.Rank.R1>;
  }

  get trainableVariables(): tf.Variable[] {
    const variables: tf.Variable[] = [this.filmWeight, this.filmBias];
    for (const conv of this.convs) variables.push(conv.kernel, conv.bias);
    for (const norm of this.norms) {
      variables.push(norm.gamma, norm.beta, norm.gammaProjection, norm.betaProjection);
    }
    return variables;
  }

  get parameterCount(): number {
    return this.trainableVariables.reduce((total, variable) => total + variable.size, 0);
  }

  private conv(x: tf.Tensor4D, index: number): tf.Tensor4D {
    const { kernel, bias, kernelSize, stride } = this.convs[index];
    const padded = padReplicate(x, (kernelSize - 1) / 2);
    return tf.conv2d(padded, kernel as unknown as tf.Tensor4D, stride, 'valid').add(bias) as tf.Tensor4D;
  }

  private norm(x: tf.Tensor4D, index: number, hidden: tf.Tensor2D): tf.Tensor4D {
    const { gamma, beta, gammaProjection, betaProjection } = this.norms[index];

    const { mean, variance } = tf.moments(x, [1, 2], true);
    const normalized = x.sub(mean).mul(tf.rsqrt(variance.add(NORM_EPSILON))) as tf.Tensor4D;

    const scale = gamma.expandDims(0).add(hidden.matMul(gammaProjection as unknown as tf.Tensor2D));
    const shift = beta.expandDims(0).add(hidden.matMul(betaProjection as unknown as tf.Tensor2D));

    return normalized
      .mul(scale.expandDims(1).expandDims(1))
      .add(shift.expandDims(1).expandDims(1)) as tf.Tensor4D;
  }

  /** `images01` is `[B, H, W, 3]` in [0, 1]; `controls` is `[B, condDims]`. */
  forward(images01: tf.Tensor4D, controls: tf.Tensor2D): tf.Tensor4D {
    const { blocks } = this.config;

    const hidden = tf.relu(controls.matMul(this.filmWeight as unknown as tf.Tensor2D).add(this.filmBias)) as tf.Tensor2D;

    let h = tf.relu(this.norm(this.conv(images01, 0), 0, hidden)) as tf.Tensor4D;
    h = tf.relu(this.norm(this.conv(h, 1), 1, hidden)) as tf.Tensor4D;
    h = tf.relu(this.norm(this.conv(h, 2), 2, hidden)) as tf.Tensor4D;

    for (let block = 0; block < blocks; block++) {
      const first = 3 + 2 * block;
      const second = 4 + 2 * block;
      const residual = h;
      h = tf.relu(this.norm(this.conv(h, first), first, hidden)) as tf.Tensor4D;
      // The block's second normalization adds the skip and does not rectify, so the identity path
      // stays linear all the way through the stack.
      h = this.norm(this.conv(h, second), second, hidden).add(residual) as tf.Tensor4D;
    }

    const upsampleFirst = 3 + 2 * blocks;
    h = tf.image.resizeNearestNeighbor(h, [h.shape[1] * 2, h.shape[2] * 2]) as tf.Tensor4D;
    h = tf.relu(this.norm(this.conv(h, upsampleFirst), upsampleFirst, hidden)) as tf.Tensor4D;
    h = tf.image.resizeNearestNeighbor(h, [h.shape[1] * 2, h.shape[2] * 2]) as tf.Tensor4D;
    h = tf.relu(this.norm(this.conv(h, upsampleFirst + 1), upsampleFirst + 1, hidden)) as tf.Tensor4D;

    return tf.tanh(this.conv(h, this.convs.length - 1)).mul(0.5).add(0.5) as tf.Tensor4D;
  }

  dispose(): void {
    for (const variable of this.trainableVariables) variable.dispose();
  }
}
