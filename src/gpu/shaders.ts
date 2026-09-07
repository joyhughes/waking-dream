/**
 * GLSL generators for every pass the runtime makes.
 *
 * Sources are built as strings rather than kept as files because the interesting parameters —
 * kernel size, how many channel groups a single draw writes, whether a normalization fuses a ReLU
 * and a residual add — are things the compiler should see as constants. A loop over four output
 * groups that the driver cannot unroll costs more than compiling four shader variants once at
 * startup, and `ProgramCache` keys on the generated source, so identical variants are shared.
 *
 * All weight and conditioning lookups go through a texture whose width is fixed at 1024, so a
 * linear texel index becomes a shift and a mask instead of an integer divide. That division would
 * otherwise run tens of times per pixel per convolution, which is not where the frame budget should go.
 */

export const WEIGHT_TEX_WIDTH = 1024;

/** Side of the square block each reduction pass folds into one texel. 8x8 gets 256x256 down in three passes. */
export const REDUCE_BLOCK = 8;

const PREAMBLE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DArray;
precision highp sampler2D;

const int TEXW = ${WEIGHT_TEX_WIDTH};
const int TEXSHIFT = 10;
const int TEXMASK = ${WEIGHT_TEX_WIDTH - 1};
`;

function outputDeclarations(targets: number): string {
  const lines: string[] = [];
  for (let t = 0; t < targets; t++) lines.push(`layout(location = ${t}) out vec4 o${t};`);
  return lines.join('\n');
}

export type ConvActivation = 'none' | 'relu' | 'tanh01';

export interface ConvShaderOptions {
  /** Odd kernel side. Compiled in, so 3x3 and 9x9 are separate programs. */
  kernel: number;
  /** Channel groups this draw writes; each is four output channels. */
  targets: number;
  stride: number;
  activation: ConvActivation;
}

/**
 * A convolution as a fragment shader.
 *
 * The loop nest is ordered tap → input group → (unrolled) input channel × output target, which puts
 * the one expensive fetch — the input texel — in the outermost useful position. Each input texel
 * read serves `4 * targets` multiply-adds, so writing four groups per draw cuts input bandwidth by
 * four against the naive one-group-per-draw version. Weight texels are each read exactly once per
 * output pixel either way, and since every fragment reads them in the same order they stay resident
 * in the texture cache.
 *
 * Padding is replicate, not zero: the sampling coordinate is clamped into the image. Zero padding
 * would tell the network there is black just outside every frame, and on a live camera feed that
 * shows up as a dark halo crawling in from the edges. `train/model.py` pads the same way.
 */
export function convShader(options: ConvShaderOptions): string {
  const { kernel, targets, stride, activation } = options;
  const radius = (kernel - 1) / 2;

  const accumulate: string[] = [];
  for (let t = 0; t < targets; t++) {
    const terms = ['x.x', 'x.y', 'x.z', 'x.w']
      .map((component, c) => `wfetch(wb + ${c} * uGOutTotal + ${t}) * ${component}`)
      .join('\n          + ');
    accumulate.push(`      acc${t} += ${terms};`);
  }

  const finish: string[] = [];
  for (let t = 0; t < targets; t++) {
    switch (activation) {
      case 'relu':
        finish.push(`  o${t} = max(acc${t}, vec4(0.0));`);
        break;
      case 'tanh01':
        // The network's last layer emits an unbounded field; tanh maps it onto the display range
        // without the flat clipping that a plain clamp would leave on saturated highlights.
        finish.push(`  o${t} = tanh(acc${t}) * 0.5 + 0.5;`);
        break;
      default:
        finish.push(`  o${t} = acc${t};`);
    }
  }

  return `${PREAMBLE}
const int K = ${kernel};
const int RADIUS = ${radius};

uniform sampler2DArray uInput;
uniform sampler2D uWeights;
uniform ivec2 uInSize;
uniform int uGIn;
uniform int uGOutTotal;
uniform int uGOutBase;
uniform int uWOffset;
uniform int uBOffset;

${outputDeclarations(targets)}

vec4 wfetch(int i) {
  int j = uWOffset + i;
  return texelFetch(uWeights, ivec2(j & TEXMASK, j >> TEXSHIFT), 0);
}

vec4 bfetch(int g) {
  int j = uBOffset + g;
  return texelFetch(uWeights, ivec2(j & TEXMASK, j >> TEXSHIFT), 0);
}

void main() {
  ivec2 dst = ivec2(gl_FragCoord.xy);
  ivec2 origin = dst * ${stride} - ivec2(RADIUS);
  int cin = uGIn * 4;

${Array.from({ length: targets }, (_, t) => `  vec4 acc${t} = bfetch(uGOutBase + ${t});`).join('\n')}

  for (int ky = 0; ky < K; ky++) {
    for (int kx = 0; kx < K; kx++) {
      ivec2 p = clamp(origin + ivec2(kx, ky), ivec2(0), uInSize - ivec2(1));
      int tapBase = (ky * K + kx) * cin * uGOutTotal + uGOutBase;
      for (int gi = 0; gi < uGIn; gi++) {
        vec4 x = texelFetch(uInput, ivec3(p, gi), 0);
        int wb = tapBase + gi * 4 * uGOutTotal;
${accumulate.join('\n')}
      }
    }
  }

${finish.join('\n')}
}
`;
}

/**
 * First stage of the instance-norm statistics reduction: folds `REDUCE_BLOCK` squared regions of the
 * activation tensor into one texel each, emitting the sum and the sum of squares side by side.
 *
 * All channel groups are reduced in a single draw by laying their partial sums out horizontally —
 * output column `g * tileWidth + x` belongs to group `g`. Doing it per group instead would mean a
 * draw call per group per pass, and at three passes and eight groups that is 24 draws for one
 * normalization layer, which the driver's per-draw overhead alone would make the dominant cost.
 *
 * Sums are accumulated rather than means: a mean of means is only the true mean when every block is
 * the same size, and the blocks at the right and bottom edges are not. The totals stay well inside
 * what 32-bit float represents exactly enough for a variance.
 */
export function reduceFromTensorShader(): string {
  return `${PREAMBLE}
const int BLOCK = ${REDUCE_BLOCK};

uniform sampler2DArray uInput;
uniform ivec2 uInSize;
uniform int uOutTileW;

layout(location = 0) out vec4 oSum;
layout(location = 1) out vec4 oSumSq;

void main() {
  ivec2 f = ivec2(gl_FragCoord.xy);
  int g = f.x / uOutTileW;
  int lx = f.x - g * uOutTileW;
  ivec2 origin = ivec2(lx, f.y) * BLOCK;

  vec4 sum = vec4(0.0);
  vec4 sumSq = vec4(0.0);

  for (int dy = 0; dy < BLOCK; dy++) {
    int y = origin.y + dy;
    if (y >= uInSize.y) break;
    for (int dx = 0; dx < BLOCK; dx++) {
      int x = origin.x + dx;
      if (x >= uInSize.x) break;
      vec4 v = texelFetch(uInput, ivec3(x, y, g), 0);
      sum += v;
      sumSq += v * v;
    }
  }

  oSum = sum;
  oSumSq = sumSq;
}
`;
}

/** Later reduction passes, folding the previous pass's partial sums the same way down to one texel per group. */
export function reduceFrom2DShader(): string {
  return `${PREAMBLE}
const int BLOCK = ${REDUCE_BLOCK};

uniform sampler2D uSum;
uniform sampler2D uSumSq;
uniform int uInTileW;
uniform int uInHeight;
uniform int uOutTileW;

layout(location = 0) out vec4 oSum;
layout(location = 1) out vec4 oSumSq;

void main() {
  ivec2 f = ivec2(gl_FragCoord.xy);
  int g = f.x / uOutTileW;
  int lx = f.x - g * uOutTileW;
  ivec2 origin = ivec2(lx, f.y) * BLOCK;

  vec4 sum = vec4(0.0);
  vec4 sumSq = vec4(0.0);

  for (int dy = 0; dy < BLOCK; dy++) {
    int y = origin.y + dy;
    if (y >= uInHeight) break;
    for (int dx = 0; dx < BLOCK; dx++) {
      int x = origin.x + dx;
      if (x >= uInTileW) break;
      ivec2 p = ivec2(g * uInTileW + x, y);
      sum += texelFetch(uSum, p, 0);
      sumSq += texelFetch(uSumSq, p, 0);
    }
  }

  oSum = sum;
  oSumSq = sumSq;
}
`;
}

export interface NormShaderOptions {
  targets: number;
  /** Fuses the ReLU that follows the first normalization in every residual block. */
  relu: boolean;
  /** Fuses the block's skip connection into the second normalization's write. */
  residual: boolean;
}

/**
 * Instance normalization with a per-channel affine, reading its statistics straight out of the
 * reduction's 1x1-per-group result rather than off the CPU.
 *
 * Keeping the readback out is not only a latency question. A `readPixels` here would force a
 * pipeline flush in the middle of the forward pass, once per normalization layer, and stall the
 * GPU on the CPU at exactly the point where both should be running ahead.
 *
 * The affine parameters come from a small texture that is rewritten every frame. When the model is
 * conditioned, `model/conditioning.ts` evaluates its control MLP on the CPU and writes the result
 * there, which is what turns layer/intensity/scale into live controls: the network's weights never
 * change, only the per-channel scale and shift this shader multiplies by.
 */
export function normShader(options: NormShaderOptions): string {
  const { targets, relu, residual } = options;

  const body: string[] = [];
  for (let t = 0; t < targets; t++) {
    body.push(`  {
    int g = uGroupBase + ${t};
    vec4 sum = texelFetch(uSum, ivec2(g, 0), 0);
    vec4 sumSq = texelFetch(uSumSq, ivec2(g, 0), 0);
    vec4 mean = sum * uInvN;
    vec4 variance = max(sumSq * uInvN - mean * mean, vec4(0.0));
    vec4 x = texelFetch(uInput, ivec3(px, g), 0);
    vec4 y = (x - mean) * inversesqrt(variance + uEps) * affine(g) + affine(uGTotal + g);
${residual ? `    y += texelFetch(uSkip, ivec3(px, g), 0);` : ''}
${relu ? `    y = max(y, vec4(0.0));` : ''}
    o${t} = y;
  }`);
  }

  return `${PREAMBLE}
uniform sampler2DArray uInput;
${residual ? 'uniform sampler2DArray uSkip;' : ''}
uniform sampler2D uSum;
uniform sampler2D uSumSq;
uniform sampler2D uAffine;
uniform int uAffineOffset;
uniform int uGTotal;
uniform int uGroupBase;
uniform float uInvN;
uniform float uEps;

${outputDeclarations(targets)}

vec4 affine(int i) {
  int j = uAffineOffset + i;
  return texelFetch(uAffine, ivec2(j & TEXMASK, j >> TEXSHIFT), 0);
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
${body.join('\n')}
}
`;
}

/**
 * Resamples every channel group of a tensor, used both for the network's upsampling stages and for
 * the octave pyramid in the training-free dream mode.
 *
 * Upsampling is nearest-then-convolve rather than a transposed convolution. A transposed
 * convolution whose stride does not divide its kernel lays down a checkerboard of uneven overlap,
 * and on a video feed that checkerboard shimmers as the content moves under it. Odena et al. 2016
 * is the writeup; the fix is this pair of ops, and `train/model.py` upsamples identically so the
 * weights mean the same thing here.
 */
export function resizeShader(filter: 'nearest' | 'linear'): string {
  return `${PREAMBLE}
uniform sampler2DArray uInput;
uniform ivec2 uInSize;
uniform vec2 uOutSizeInv;
uniform int uGroupBase;

layout(location = 0) out vec4 o0;

void main() {
  vec2 uv = gl_FragCoord.xy * uOutSizeInv;
${
  filter === 'nearest'
    ? `  ivec2 p = clamp(ivec2(uv * vec2(uInSize)), ivec2(0), uInSize - ivec2(1));
  o0 = texelFetch(uInput, ivec3(p, uGroupBase), 0);`
    : `  o0 = texture(uInput, vec3(uv, float(uGroupBase)));`
}
}
`;
}

/**
 * Weighted sum of up to three tensors' first channel group.
 *
 * One shader covers the feedback mix (camera plus warped previous output), the ascent step in the
 * training-free mode (image plus scaled gradient), and the octave detail transfer (image plus the
 * difference of two resamplings). They are all `a*A + b*B + c*C + bias`; giving each its own shader
 * would only spread the same three lines across three files.
 */
export function combineShader(terms: 1 | 2 | 3, clamp01: boolean): string {
  const inputs = ['uA', 'uB', 'uC'].slice(0, terms);
  const weights = ['uWa', 'uWb', 'uWc'].slice(0, terms);
  const sum = inputs.map((name, i) => `${weights[i]} * texelFetch(${name}, p, 0)`).join(' + ');

  return `${PREAMBLE}
${inputs.map((name) => `uniform sampler2DArray ${name};`).join('\n')}
${weights.map((name) => `uniform float ${name};`).join('\n')}
uniform float uBias;
uniform int uGroupBase;

layout(location = 0) out vec4 o0;

void main() {
  ivec3 p = ivec3(ivec2(gl_FragCoord.xy), uGroupBase);
  vec4 value = ${sum} + vec4(uBias);
  o0 = ${clamp01 ? 'clamp(value, vec4(0.0), vec4(1.0))' : 'value'};
}
`;
}

/**
 * Pulls a frame from a video element, image, or canvas into a tensor.
 *
 * The source is almost never the same aspect ratio as the capture size, so the transform below does
 * a centre crop — filling the capture square from the middle of the frame — instead of squashing.
 * The network was trained on undistorted photographs, and a stretched input makes it draw stretched
 * features, which reads as the model being worse than it is.
 */
export function fromSourceShader(): string {
  return `${PREAMBLE}
uniform sampler2D uSource;
uniform vec2 uScale;
uniform vec2 uOffset;
uniform vec2 uOutSizeInv;
uniform float uMirror;

layout(location = 0) out vec4 o0;

void main() {
  vec2 uv = gl_FragCoord.xy * uOutSizeInv;
  uv.x = mix(uv.x, 1.0 - uv.x, uMirror);
  // Textures from a video element arrive with their first row at the top, which is the opposite of
  // the framebuffer's origin, so the vertical flip happens here rather than by re-uploading flipped.
  uv.y = 1.0 - uv.y;
  vec2 sourceUv = uv * uScale + uOffset;
  o0 = vec4(texture(uSource, clamp(sourceUv, vec2(0.0), vec2(1.0))).rgb, 0.0);
}
`;
}

/**
 * The final composite onto the canvas: the network's output, optionally mixed back toward the
 * untouched frame, with the result scaled to fit the display.
 *
 * The mix is against the captured tensor rather than the original video element so that the two
 * agree pixel for pixel — at low capture sizes the network's output is a blurry version of a sharp
 * frame, and cross-fading between mismatched sharpness looks like a focus error rather than a dial.
 */
export function toCanvasShader(): string {
  return `${PREAMBLE}
uniform sampler2DArray uOutput;
uniform sampler2DArray uOriginal;
uniform vec2 uOutSizeInv;
uniform float uMix;
uniform float uGain;
uniform float uSaturation;

layout(location = 0) out vec4 oColor;

void main() {
  // No vertical flip here. The capture pass already flipped the source on the way in, which put the
  // tensor in the framebuffer's bottom-up orientation -- the same one the default framebuffer uses.
  // Flipping again would land the picture upside down.
  vec2 uv = gl_FragCoord.xy * uOutSizeInv;
  vec3 dreamed = texture(uOutput, vec3(uv, 0.0)).rgb;
  vec3 original = texture(uOriginal, vec3(uv, 0.0)).rgb;
  vec3 color = mix(original, dreamed, uMix) * uGain;
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  color = mix(vec3(luma), color, uSaturation);
  oColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
`;
}

/**
 * Warps a tensor by an affine transform, used on the previous output before it is fed back in.
 *
 * A slow zoom, rotation, and drift is what turns a per-frame filter into the recursion DeepDream is
 * known for: without it the hallucinated detail is pinned to the pixel grid and simply accumulates
 * in place, and with it the detail is continuously pushed outward while new detail grows behind it.
 * The transform is applied in centred, aspect-corrected coordinates so a rotation stays a rotation
 * on a non-square capture.
 */
export function warpShader(): string {
  return `${PREAMBLE}
uniform sampler2DArray uInput;
uniform vec2 uOutSizeInv;
uniform vec2 uAspect;
uniform mat2 uMatrix;
uniform vec2 uTranslate;
uniform float uFade;
uniform int uGroupBase;

layout(location = 0) out vec4 o0;

void main() {
  vec2 uv = gl_FragCoord.xy * uOutSizeInv;
  vec2 centered = (uv - 0.5) * uAspect;
  vec2 warped = uMatrix * centered + uTranslate;
  vec2 sourceUv = warped / uAspect + 0.5;
  vec4 value = texture(uInput, vec3(clamp(sourceUv, vec2(0.0), vec2(1.0)), float(uGroupBase)));
  // The damper pulls toward mid grey rather than toward black. Scaling toward zero would bleed off
  // the runaway by darkening the frame, which is visible as the picture going out rather than
  // calming down; decaying contrast leaves the exposure alone and only takes back the energy the
  // ascent keeps adding.
  o0 = mix(vec4(0.5), value, uFade);
}
`;
}
