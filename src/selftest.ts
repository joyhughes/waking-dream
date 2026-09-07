/**
 * Numerical self-test for the GPU runtime.
 *
 * The convolution's weight indexing, the padded channel layout, and the two-stage instance-norm
 * reduction are all arithmetic that produces a plausible-looking picture when it is subtly wrong.
 * Every check here runs an op on the GPU and the same op in plain JavaScript on the same numbers,
 * and compares. A mismatch is reported as the worst absolute difference and where it was.
 *
 * Open `/selftest.html` to run it. It is not part of the app bundle.
 */

import { createGlContext, groupsFor } from './gpu/gl';
import { readTensor, writeTensor } from './gpu/debug';
import { DataTexture, Ops } from './gpu/ops';
import { DreamNet } from './model/dreamnet';
import { DNW_FORMAT, DNW_MAGIC, type ModelHeader, type ModelOp } from './model/format';
import { packBias, packConv } from './model/pack';
import { ShallowDream, DEFAULT_SHALLOW_PARAMS } from './model/shallowDream';

export interface CheckResult {
  name: string;
  passed: boolean;
  detail: string;
}

/** Half-float carries about three decimal digits, so this is loose on purpose — it catches
 *  structural errors (a transposed index, a missed pad) without failing on precision. */
const TOLERANCE = 3e-2;

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

function randomArray(length: number, seed: number, scale = 1): Float32Array {
  const random = seeded(seed);
  const array = new Float32Array(length);
  for (let i = 0; i < length; i++) array[i] = random() * 2 * scale;
  return array;
}

function compare(name: string, actual: Float32Array, expected: Float32Array): CheckResult {
  if (actual.length !== expected.length) {
    return { name, passed: false, detail: `length ${actual.length} vs expected ${expected.length}` };
  }
  let worst = 0;
  let worstAt = -1;
  for (let i = 0; i < actual.length; i++) {
    const difference = Math.abs(actual[i] - expected[i]);
    if (difference > worst) {
      worst = difference;
      worstAt = i;
    }
  }
  const passed = worst <= TOLERANCE;
  const detail = passed
    ? `max |Δ| = ${worst.toExponential(2)} over ${actual.length} values`
    : `max |Δ| = ${worst.toExponential(2)} at ${worstAt}: got ${actual[worstAt]?.toFixed(5)}, expected ${expected[worstAt]?.toFixed(5)}`;
  return { name, passed, detail };
}

/** Reference convolution: replicate padding, matching the shader's coordinate clamp. */
function cpuConv(
  input: Float32Array,
  width: number,
  height: number,
  inChannels: number,
  kernel: Float32Array,
  bias: Float32Array,
  kernelSize: number,
  outChannels: number,
  stride: number,
): { data: Float32Array; width: number; height: number } {
  const outWidth = Math.ceil(width / stride);
  const outHeight = Math.ceil(height / stride);
  const radius = (kernelSize - 1) / 2;
  const out = new Float32Array(outWidth * outHeight * outChannels);

  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      for (let co = 0; co < outChannels; co++) {
        let sum = bias[co];
        for (let ky = 0; ky < kernelSize; ky++) {
          const sy = Math.min(height - 1, Math.max(0, y * stride + ky - radius));
          for (let kx = 0; kx < kernelSize; kx++) {
            const sx = Math.min(width - 1, Math.max(0, x * stride + kx - radius));
            const tap = ky * kernelSize + kx;
            for (let ci = 0; ci < inChannels; ci++) {
              sum += kernel[(tap * inChannels + ci) * outChannels + co] * input[(sy * width + sx) * inChannels + ci];
            }
          }
        }
        out[(y * outWidth + x) * outChannels + co] = sum;
      }
    }
  }

  return { data: out, width: outWidth, height: outHeight };
}

function cpuInstanceNorm(
  input: Float32Array,
  width: number,
  height: number,
  channels: number,
  gamma: Float32Array,
  beta: Float32Array,
  options: { relu: boolean; skip?: Float32Array | null } = { relu: false },
): Float32Array {
  const count = width * height;
  const out = new Float32Array(input.length);

  for (let c = 0; c < channels; c++) {
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < count; i++) {
      const value = input[i * channels + c];
      sum += value;
      sumSq += value * value;
    }
    const mean = sum / count;
    const variance = Math.max(0, sumSq / count - mean * mean);
    const inverse = 1 / Math.sqrt(variance + 1e-5);
    for (let i = 0; i < count; i++) {
      let value = (input[i * channels + c] - mean) * inverse * gamma[c] + beta[c];
      if (options.skip) value += options.skip[i * channels + c];
      if (options.relu) value = Math.max(0, value);
      out[i * channels + c] = value;
    }
  }

  return out;
}

/**
 * Encodes a `.dnw` in the browser, so the format itself is exercised by the test rather than only
 * by `train/export.py`. If these two ever disagree the model file is the thing at fault, and this
 * is where that shows up.
 */
function encodeDnw(header: ModelHeader, gpu: Float32Array, cpu: Float32Array): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(header));
  // The payload is float32, so it has to start 4-byte aligned; padding the header with spaces
  // keeps it valid JSON while moving the start.
  const padding = (4 - ((8 + json.length) % 4)) % 4;
  const headerLength = json.length + padding;

  const buffer = new ArrayBuffer(8 + headerLength + (gpu.length + cpu.length) * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set(new TextEncoder().encode(DNW_MAGIC), 0);
  new DataView(buffer).setUint32(4, headerLength, true);
  bytes.set(json, 8);
  bytes.fill(0x20, 8 + json.length, 8 + headerLength);

  const floats = new Float32Array(buffer, 8 + headerLength);
  floats.set(gpu, 0);
  floats.set(cpu, gpu.length);
  return buffer;
}

export async function runSelfTest(): Promise<CheckResult[]> {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = createGlContext(canvas);
  const ops = new Ops(ctx);
  const results: CheckResult[] = [];

  try {
    results.push(checkConv(ctx, ops, { stride: 1, kernelSize: 3, inChannels: 6, outChannels: 12 }));
    results.push(checkConv(ctx, ops, { stride: 2, kernelSize: 3, inChannels: 4, outChannels: 8 }));
    // Nine wide with three input channels is the shape of the network's first layer, and the one
    // where the input padding from three to four channels has to not leak.
    results.push(checkConv(ctx, ops, { stride: 1, kernelSize: 9, inChannels: 3, outChannels: 16 }));
    // More output groups than a single draw can write, so the multi-pass path is covered.
    results.push(checkConv(ctx, ops, { stride: 1, kernelSize: 3, inChannels: 8, outChannels: 32 }));

    results.push(checkNorm(ctx, ops, { width: 24, height: 20, channels: 8, relu: false, skip: false }));
    results.push(checkNorm(ctx, ops, { width: 24, height: 20, channels: 8, relu: true, skip: true }));
    // Larger than one reduction pass can fold, so the multi-pass reduction is covered.
    results.push(checkNorm(ctx, ops, { width: 96, height: 72, channels: 4, relu: false, skip: false }));

    results.push(checkCombine(ctx, ops));
    results.push(checkResize(ctx, ops));
    results.push(checkModel(ctx, ops));
    results.push(checkShallowRuns(ctx, ops));

    results.push(await checkParameterRoundTrip());
    results.push(await checkAudioBands());
    results.push(await checkBrowserTrainerParity(ctx, ops));

    const exported = await checkExportedReference(ctx, ops);
    if (exported) results.push(exported);
  } finally {
    ops.pool.releaseAll();
    ops.dispose();
  }

  return results;
}

function checkConv(
  ctx: ReturnType<typeof createGlContext>,
  ops: Ops,
  spec: { stride: number; kernelSize: number; inChannels: number; outChannels: number },
): CheckResult {
  const { stride, kernelSize, inChannels, outChannels } = spec;
  const width = 20;
  const height = 16;
  const name = `conv ${kernelSize}x${kernelSize} ${inChannels}→${outChannels} stride ${stride}`;

  const input = randomArray(width * height * inChannels, 11 + kernelSize + inChannels);
  const kernel = randomArray(kernelSize * kernelSize * inChannels * outChannels, 23 + outChannels, 0.2);
  const bias = randomArray(outChannels, 31 + stride, 0.1);

  const packed = packConv(kernel, kernelSize, inChannels, outChannels);
  const packedBias = packBias(bias, outChannels);
  const data = new Float32Array((packed.texelCount + packedBias.texelCount) * 4);
  data.set(packed.texels, 0);
  data.set(packedBias.texels, packed.texelCount * 4);

  const weights = new DataTexture(ctx.gl, packed.texelCount + packedBias.texelCount);
  weights.upload(data);

  const source = ops.pool.acquire({ width, height, channels: inChannels });
  writeTensor(ctx, source, input);

  const expected = cpuConv(input, width, height, inChannels, kernel, bias, kernelSize, outChannels, stride);
  const output = ops.pool.acquire({ width: expected.width, height: expected.height, channels: outChannels });
  ops.conv(source, output, weights, {
    kernel: kernelSize,
    stride,
    weightOffset: 0,
    biasOffset: packed.texelCount,
    activation: 'none',
  });

  const actual = readTensor(ctx, output);
  ops.pool.releaseAll();
  weights.dispose();

  return compare(name, actual, expected.data);
}

function checkNorm(
  ctx: ReturnType<typeof createGlContext>,
  ops: Ops,
  spec: { width: number; height: number; channels: number; relu: boolean; skip: boolean },
): CheckResult {
  const { width, height, channels, relu } = spec;
  const name = `instance norm ${width}x${height}x${channels}${relu ? ' +relu' : ''}${spec.skip ? ' +skip' : ''}`;

  const input = randomArray(width * height * channels, 41 + width, 2);
  const skipData = spec.skip ? randomArray(width * height * channels, 53 + channels) : null;
  const gamma = randomArray(channels, 61, 0.5);
  const beta = randomArray(channels, 67, 0.5);
  for (let c = 0; c < channels; c++) gamma[c] += 1;

  const groups = groupsFor(channels);
  const affine = new DataTexture(ctx.gl, 2 * groups);
  const affineData = new Float32Array(2 * groups * 4);
  affineData.set(gamma, 0);
  affineData.set(beta, groups * 4);
  affine.upload(affineData);

  const source = ops.pool.acquire({ width, height, channels });
  writeTensor(ctx, source, input);

  let skipTensor = null;
  if (skipData) {
    skipTensor = ops.pool.acquire({ width, height, channels });
    writeTensor(ctx, skipTensor, skipData);
  }

  const output = ops.pool.acquire({ width, height, channels });
  ops.instanceNorm(source, output, affine, 0, { relu, skip: skipTensor });

  const actual = readTensor(ctx, output);
  const expected = cpuInstanceNorm(input, width, height, channels, gamma, beta, { relu, skip: skipData });

  ops.pool.releaseAll();
  affine.dispose();

  return compare(name, actual, expected);
}

function checkCombine(ctx: ReturnType<typeof createGlContext>, ops: Ops): CheckResult {
  const width = 12;
  const height = 9;
  const a = randomArray(width * height * 3, 71);
  const b = randomArray(width * height * 3, 73);

  const ta = ops.pool.acquire({ width, height, channels: 3 });
  const tb = ops.pool.acquire({ width, height, channels: 3 });
  writeTensor(ctx, ta, a);
  writeTensor(ctx, tb, b);

  const out = ops.pool.acquire({ width, height, channels: 3 });
  ops.combine(out, [
    { tensor: ta, weight: 0.75 },
    { tensor: tb, weight: -0.25 },
  ], { bias: 0.1 });

  const actual = readTensor(ctx, out);
  const expected = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) expected[i] = 0.75 * a[i] - 0.25 * b[i] + 0.1;

  ops.pool.releaseAll();
  return compare('combine 0.75a − 0.25b + 0.1', actual, expected);
}

function checkResize(ctx: ReturnType<typeof createGlContext>, ops: Ops): CheckResult {
  const width = 8;
  const height = 6;
  const input = randomArray(width * height * 3, 79);

  const source = ops.pool.acquire({ width, height, channels: 3 });
  writeTensor(ctx, source, input);

  const output = ops.pool.acquire({ width: width * 2, height: height * 2, channels: 3 });
  ops.resize(source, output, 'nearest');
  const actual = readTensor(ctx, output);

  // Nearest upsampling by two: each source texel becomes a 2x2 block.
  const expected = new Float32Array(width * 2 * height * 2 * 3);
  for (let y = 0; y < height * 2; y++) {
    for (let x = 0; x < width * 2; x++) {
      for (let c = 0; c < 3; c++) {
        expected[(y * width * 2 + x) * 3 + c] = input[((y >> 1) * width + (x >> 1)) * 3 + c];
      }
    }
  }

  ops.pool.releaseAll();
  return compare('nearest upsample ×2', actual, expected);
}

/**
 * End to end: a small but structurally complete network — wide input convolution, a stride-2
 * downsample, a residual block, an upsample, and a wide output convolution with tanh — encoded as a
 * `.dnw`, loaded through the real loader, and compared against the same op list run on the CPU.
 */
function checkModel(ctx: ReturnType<typeof createGlContext>, ops: Ops): CheckResult {
  const width = 16;
  const height = 16;
  const featureChannels = 8;

  interface LayerWeights {
    kernel: Float32Array;
    bias: Float32Array;
    kernelSize: number;
    inChannels: number;
    outChannels: number;
    stride: number;
  }

  const layers: Record<string, LayerWeights> = {
    inConv: { kernel: randomArray(9 * 9 * 3 * featureChannels, 101, 0.1), bias: randomArray(featureChannels, 103, 0.05), kernelSize: 9, inChannels: 3, outChannels: featureChannels, stride: 1 },
    down: { kernel: randomArray(3 * 3 * featureChannels * featureChannels, 107, 0.15), bias: randomArray(featureChannels, 109, 0.05), kernelSize: 3, inChannels: featureChannels, outChannels: featureChannels, stride: 2 },
    res1: { kernel: randomArray(3 * 3 * featureChannels * featureChannels, 113, 0.15), bias: randomArray(featureChannels, 127, 0.05), kernelSize: 3, inChannels: featureChannels, outChannels: featureChannels, stride: 1 },
    res2: { kernel: randomArray(3 * 3 * featureChannels * featureChannels, 131, 0.15), bias: randomArray(featureChannels, 137, 0.05), kernelSize: 3, inChannels: featureChannels, outChannels: featureChannels, stride: 1 },
    outConv: { kernel: randomArray(9 * 9 * featureChannels * 3, 139, 0.1), bias: randomArray(3, 149, 0.05), kernelSize: 9, inChannels: featureChannels, outChannels: 3, stride: 1 },
  };

  const gpuChunks: Float32Array[] = [];
  const offsets: Record<string, { weight: number; bias: number }> = {};
  let cursor = 0;
  for (const [name, layer] of Object.entries(layers)) {
    const packed = packConv(layer.kernel, layer.kernelSize, layer.inChannels, layer.outChannels);
    const packedBias = packBias(layer.bias, layer.outChannels);
    offsets[name] = { weight: cursor, bias: cursor + packed.texelCount };
    gpuChunks.push(packed.texels, packedBias.texels);
    cursor += packed.texelCount + packedBias.texelCount;
  }

  const gpu = new Float32Array(cursor * 4);
  let writeAt = 0;
  for (const chunk of gpuChunks) {
    gpu.set(chunk, writeAt);
    writeAt += chunk.length;
  }

  const normSlots = [featureChannels, featureChannels, featureChannels, featureChannels];
  const cpuValues: number[] = [];
  const cpuTensors: ModelHeader['cpuTensors'] = {};
  const gammas: Float32Array[] = [];
  const betas: Float32Array[] = [];
  normSlots.forEach((channels, slot) => {
    const gamma = randomArray(channels, 151 + slot, 0.3);
    for (let c = 0; c < channels; c++) gamma[c] += 1;
    const beta = randomArray(channels, 163 + slot, 0.2);
    gammas.push(gamma);
    betas.push(beta);
    cpuTensors[`norm${slot}.gamma`] = { offset: cpuValues.length, length: channels, shape: [channels] };
    cpuValues.push(...gamma);
    cpuTensors[`norm${slot}.beta`] = { offset: cpuValues.length, length: channels, shape: [channels] };
    cpuValues.push(...beta);
  });

  const modelOps: ModelOp[] = [
    { type: 'conv', in: 'x', out: 'c0', kernel: 9, stride: 1, inChannels: 3, outChannels: featureChannels, weightOffset: offsets.inConv.weight, biasOffset: offsets.inConv.bias, activation: 'none' },
    { type: 'norm', in: 'c0', out: 'n0', channels: featureChannels, slot: 0, relu: true, skip: null },
    { type: 'conv', in: 'n0', out: 'c1', kernel: 3, stride: 2, inChannels: featureChannels, outChannels: featureChannels, weightOffset: offsets.down.weight, biasOffset: offsets.down.bias, activation: 'none' },
    { type: 'norm', in: 'c1', out: 'n1', channels: featureChannels, slot: 1, relu: true, skip: null },
    { type: 'conv', in: 'n1', out: 'c2', kernel: 3, stride: 1, inChannels: featureChannels, outChannels: featureChannels, weightOffset: offsets.res1.weight, biasOffset: offsets.res1.bias, activation: 'none' },
    { type: 'norm', in: 'c2', out: 'n2', channels: featureChannels, slot: 2, relu: true, skip: null },
    { type: 'conv', in: 'n2', out: 'c3', kernel: 3, stride: 1, inChannels: featureChannels, outChannels: featureChannels, weightOffset: offsets.res2.weight, biasOffset: offsets.res2.bias, activation: 'none' },
    { type: 'norm', in: 'c3', out: 'n3', channels: featureChannels, slot: 3, relu: false, skip: 'n1' },
    { type: 'resize', in: 'n3', out: 'u0', scale: 2 },
    { type: 'conv', in: 'u0', out: 'y', kernel: 9, stride: 1, inChannels: featureChannels, outChannels: 3, weightOffset: offsets.outConv.weight, biasOffset: offsets.outConv.bias, activation: 'tanh01' },
  ];

  const header: ModelHeader = {
    format: DNW_FORMAT,
    name: 'selftest',
    description: 'Synthetic model used by the runtime self-test.',
    inputName: 'x',
    outputName: 'y',
    ops: modelOps,
    normSlots,
    gpuTexels: cursor,
    cpuFloats: cpuValues.length,
    cpuTensors,
    conditioning: null,
  };

  const model = DreamNet.fromBuffer(ctx, ops, encodeDnw(header, gpu, new Float32Array(cpuValues)));
  model.setControls([]);

  const input = randomArray(width * height * 3, 173, 0.5);
  for (let i = 0; i < input.length; i++) input[i] = Math.min(1, Math.max(0, input[i] + 0.5));

  const source = ops.pool.acquire({ width, height, channels: 3 });
  writeTensor(ctx, source, input);
  const output = model.forward(source);
  const actual = readTensor(ctx, output);
  ops.pool.releaseAll();
  model.dispose();

  // The same op list, on the CPU.
  const c0 = cpuConv(input, width, height, 3, layers.inConv.kernel, layers.inConv.bias, 9, featureChannels, 1);
  const n0 = cpuInstanceNorm(c0.data, c0.width, c0.height, featureChannels, gammas[0], betas[0], { relu: true });
  const c1 = cpuConv(n0, c0.width, c0.height, featureChannels, layers.down.kernel, layers.down.bias, 3, featureChannels, 2);
  const n1 = cpuInstanceNorm(c1.data, c1.width, c1.height, featureChannels, gammas[1], betas[1], { relu: true });
  const c2 = cpuConv(n1, c1.width, c1.height, featureChannels, layers.res1.kernel, layers.res1.bias, 3, featureChannels, 1);
  const n2 = cpuInstanceNorm(c2.data, c2.width, c2.height, featureChannels, gammas[2], betas[2], { relu: true });
  const c3 = cpuConv(n2, c2.width, c2.height, featureChannels, layers.res2.kernel, layers.res2.bias, 3, featureChannels, 1);
  const n3 = cpuInstanceNorm(c3.data, c3.width, c3.height, featureChannels, gammas[3], betas[3], { relu: false, skip: n1 });

  const upWidth = c3.width * 2;
  const upHeight = c3.height * 2;
  const u0 = new Float32Array(upWidth * upHeight * featureChannels);
  for (let y = 0; y < upHeight; y++) {
    for (let x = 0; x < upWidth; x++) {
      for (let c = 0; c < featureChannels; c++) {
        u0[(y * upWidth + x) * featureChannels + c] = n3[((y >> 1) * c3.width + (x >> 1)) * featureChannels + c];
      }
    }
  }

  const y = cpuConv(u0, upWidth, upHeight, featureChannels, layers.outConv.kernel, layers.outConv.bias, 9, 3, 1);
  const expected = new Float32Array(y.data.length);
  for (let i = 0; i < y.data.length; i++) expected[i] = Math.tanh(y.data[i]) * 0.5 + 0.5;

  return compare('full DreamNet forward pass vs CPU', actual, expected);
}

/** The training-free path has no CPU reference; this only asserts it runs and stays finite. */
function checkShallowRuns(ctx: ReturnType<typeof createGlContext>, ops: Ops): CheckResult {
  const shallow = new ShallowDream(ctx, ops);
  const width = 64;
  const height = 48;

  const input = new Float32Array(width * height * 3);
  for (let i = 0; i < input.length; i++) input[i] = 0.5 + 0.4 * Math.sin(i * 0.07);

  const source = ops.pool.acquire({ width, height, channels: 3 });
  writeTensor(ctx, source, input);

  let current = new Float32Array(input);
  try {
    // Ten iterations of the feedback loop, which is where a sign error or a runaway gain shows up.
    for (let frame = 0; frame < 10; frame++) {
      writeTensor(ctx, source, current);
      const output = shallow.run(source, { ...DEFAULT_SHALLOW_PARAMS, steps: 2, octaves: 2 });
      current = readTensor(ctx, output);
      ops.pool.releaseAll();
    }
  } catch (error) {
    shallow.dispose();
    return { name: 'shallow ascent runs', passed: false, detail: error instanceof Error ? error.message : String(error) };
  }

  shallow.dispose();

  let min = Infinity;
  let max = -Infinity;
  let moved = 0;
  for (let i = 0; i < current.length; i++) {
    if (!Number.isFinite(current[i])) {
      return { name: 'shallow ascent runs', passed: false, detail: `non-finite value at ${i}` };
    }
    min = Math.min(min, current[i]);
    max = Math.max(max, current[i]);
    moved += Math.abs(current[i] - input[i]);
  }
  moved /= current.length;

  const passed = min >= -1e-3 && max <= 1 + 1e-3 && moved > 1e-3;
  return {
    name: 'shallow ascent runs',
    passed,
    detail: `range [${min.toFixed(3)}, ${max.toFixed(3)}], mean change ${moved.toFixed(4)} over 10 frames`,
  };
}

/**
 * Runs an exported model against the reference PyTorch computed for it.
 *
 * This is the check that actually proves `train/export.py` and this runtime agree. Everything above
 * verifies the runtime against itself; only this one catches a weight packed in the wrong order, a
 * transposed FiLM projection, or an op list that does not match what `model.py` computes -- all of
 * which produce a picture rather than an error.
 *
 * Skipped, not failed, when no reference has been exported. Run:
 *
 *   python export.py --checkpoint runs/dreamnet/checkpoint.pt \\
 *       --out ../public/models/dreamnet.dnw --reference ../public/models/verify.json
 */
async function checkExportedReference(
  ctx: ReturnType<typeof createGlContext>,
  ops: Ops,
): Promise<CheckResult | null> {
  let reference: { model: string; width: number; height: number; controls: number[]; input: number[]; output: number[] };
  try {
    const response = await fetch('/models/verify.json');
    if (!response.ok) return null;
    reference = await response.json();
  } catch {
    return null;
  }

  const name = `exported model "${reference.model}" vs PyTorch`;
  try {
    const model = await DreamNet.load(ctx, ops, `/models/${reference.model}`);
    model.setControls(reference.controls);

    const source = ops.pool.acquire({ width: reference.width, height: reference.height, channels: 3 });
    writeTensor(ctx, source, new Float32Array(reference.input));
    const output = model.forward(source);
    const actual = readTensor(ctx, output);
    ops.pool.releaseAll();
    model.dispose();

    return compare(name, actual, new Float32Array(reference.output));
  } catch (error) {
    return { name, passed: false, detail: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * Checks the in-browser trainer's network against the runtime that will actually run what it trains.
 *
 * `dreamNetTf.ts` is a third implementation of the same architecture, alongside the WebGL runtime
 * and `train/model.py`, and a browser-trained model is only worth anything if all three agree. So:
 * build one with random weights — including conditioning projections large enough that a transposed
 * one would show — run it in TFJS, export it through the real exporter, and run the result here.
 *
 * This is the check that a PyTorch install is not needed to run, which makes it the one that guards
 * the browser training path in CI or on a machine with no Python at all.
 */
async function checkBrowserTrainerParity(
  ctx: ReturnType<typeof createGlContext>,
  ops: Ops,
): Promise<CheckResult> {
  const name = 'browser trainer network vs WebGL runtime';
  try {
    const [{ tf, initializeTraining }, { DreamNetTf }, { exportDnw, styleControls }] = await Promise.all([
      import('./train/tfSetup'),
      import('./train/dreamNetTf'),
      import('./train/exportDnw'),
    ]);
    await initializeTraining();

    const config = { width: 4, blocks: 2, filmHidden: 8, condDims: 2 };
    const model = new DreamNetTf(config);

    // The conditioning projections start at zero, which would let a transposed FiLM weight through
    // unnoticed. Give them real values so every path in the control MLP carries something.
    for (const norm of model.norms) {
      norm.gammaProjection.assign(tf.randomNormal(norm.gammaProjection.shape, 0, 0.3) as never);
      norm.betaProjection.assign(tf.randomNormal(norm.betaProjection.shape, 0, 0.3) as never);
    }
    for (const norm of model.norms) {
      norm.gamma.assign(tf.randomNormal(norm.gamma.shape, 1, 0.2) as never);
      norm.beta.assign(tf.randomNormal(norm.beta.shape, 0, 0.2) as never);
    }

    const size = 32;
    const controls = [0.7, 0.25];
    const inputData = randomArray(size * size * 3, 1009, 0.5);
    for (let i = 0; i < inputData.length; i++) inputData[i] = Math.min(1, Math.max(0, inputData[i] + 0.5));

    const expected = tf.tidy(() => {
      const images = tf.tensor4d(Array.from(inputData), [1, size, size, 3]);
      const controlTensor = tf.tensor2d([controls]);
      return model.forward(images, controlTensor).dataSync() as Float32Array;
    });

    const buffer = exportDnw(model, {
      name: 'tf-parity',
      description: 'Self-test model built by the browser trainer.',
      controls: styleControls(['a', 'b']),
      trainedAt: size,
    });

    const runtimeModel = DreamNet.fromBuffer(ctx, ops, buffer);
    runtimeModel.setControls(controls);

    const source = ops.pool.acquire({ width: size, height: size, channels: 3 });
    writeTensor(ctx, source, inputData);
    const output = runtimeModel.forward(source);
    const actual = readTensor(ctx, output);

    ops.pool.releaseAll();
    runtimeModel.dispose();
    model.dispose();

    return compare(name, actual, new Float32Array(expected));
  } catch (error) {
    return { name, passed: false, detail: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * Checks that settings survive a trip through a PNG, and that the PNG survives the settings.
 *
 * Both halves matter. If the parameters do not come back exactly, a saved frame is not a way back
 * to the look that made it — which is the whole point of embedding them. And if writing the chunk
 * corrupts the file, every saved image is broken everywhere else, which is a far worse trade than
 * simply not storing the settings.
 */
async function checkParameterRoundTrip(): Promise<CheckResult> {
  const name = 'settings survive a round trip through a PNG';
  try {
    const { buildParameters, readParameters } = await import('./pipeline/parameters');
    const { embedParameters } = await import('./pipeline/parameters');
    const { DEFAULT_CONFIG } = await import('./pipeline/engine');

    const config = {
      ...DEFAULT_CONFIG,
      captureSize: 384,
      processor: 'shallow' as const,
      shallow: { ...DEFAULT_CONFIG.shallow, stepSize: 0.0725, bank: 'blob' as const, colourHold: 0.123 },
      feedback: { ...DEFAULT_CONFIG.feedback, zoom: 1.0123, rotate: -0.37 },
      display: { ...DEFAULT_CONFIG.display, saturation: 1.45 },
      modelControls: [0.25, 0.75],
    };
    // A name with a character outside Latin-1, which is exactly what the chunk cannot hold raw and
    // the escaping exists to handle.
    const parameters = buildParameters(config, { source: 'x.dnw', name: 'ノイズ — waves' }, 384, 288);

    const canvas = document.createElement('canvas');
    canvas.width = 24;
    canvas.height = 16;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#3a7bd5';
    context.fillRect(0, 0, 24, 16);
    context.fillStyle = '#d53a7b';
    context.fillRect(4, 4, 8, 8);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) return { name, passed: false, detail: 'the canvas produced no PNG' };

    const original = new Uint8Array(await blob.arrayBuffer());
    const embedded = embedParameters(original, parameters);
    const recovered = readParameters(embedded);

    if (!recovered) return { name, passed: false, detail: 'no parameters could be read back' };

    const differences: string[] = [];
    if (JSON.stringify(recovered.config) !== JSON.stringify(config)) differences.push('config');
    if (recovered.model?.name !== parameters.model?.name) {
      differences.push(`model name (${recovered.model?.name} vs ${parameters.model?.name})`);
    }
    if (recovered.captureWidth !== 384 || recovered.captureHeight !== 288) differences.push('capture size');

    // Re-embedding must replace rather than append, or a frame re-saved a few times grows a chunk
    // each time and the reader starts finding a stale one first.
    const twice = embedParameters(embedded, parameters);
    if (twice.length !== embedded.length) differences.push(`re-embed grew the file by ${twice.length - embedded.length} bytes`);

    // And the result still has to be an image.
    let decoded = false;
    try {
      const bitmap = await createImageBitmap(new Blob([embedded], { type: 'image/png' }));
      decoded = bitmap.width === 24 && bitmap.height === 16;
      bitmap.close();
    } catch (error) {
      differences.push(`the PNG no longer decodes: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (!decoded && differences.length === 0) differences.push('the PNG decoded to the wrong size');

    return {
      name,
      passed: differences.length === 0,
      detail:
        differences.length === 0
          ? `${embedded.length - original.length} bytes added, still decodes, re-embed is idempotent`
          : differences.join('; '),
    };
  } catch (error) {
    return { name, passed: false, detail: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * Drives the band analyser with tones of known frequency.
 *
 * The band edges are the whole point of the sound feature — if bass and the vocal range are not
 * actually separated, two textures assigned to them move together and the instrument does nothing.
 * A microphone cannot be asserted on, so this feeds an oscillator in through the same injected
 * stream path the real input uses.
 */
async function checkAudioBands(): Promise<CheckResult> {
  const name = 'frequency bands separate a 60 Hz tone from a 1 kHz one';
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  try {
    const { AudioAnalyser, FREQUENCY_BANDS } = await import('./pipeline/audio');

    const context = new AudioContext();
    if (context.state === 'suspended') await context.resume();

    const destination = context.createMediaStreamDestination();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = 'sine';
    gain.gain.value = 0.6;
    oscillator.connect(gain).connect(destination);
    oscillator.start();

    const analyser = await AudioAnalyser.open({ stream: destination.stream });

    /** Runs the tone for long enough that the attack/release has settled, then reads the bands. */
    const measure = async (frequency: number): Promise<number[]> => {
      oscillator.frequency.value = frequency;
      for (let i = 0; i < 60; i++) {
        analyser.levels();
        await sleep(8);
      }
      return Array.from(analyser.levels());
    };

    const bass = await measure(60);
    const vocal = await measure(1000);

    analyser.stop();
    oscillator.stop();
    void context.close();

    const loudest = (levels: number[]) => levels.indexOf(Math.max(...levels));
    const bassBand = loudest(bass);
    const vocalBand = loudest(vocal);

    const passed = FREQUENCY_BANDS[bassBand]?.name === 'bass' && FREQUENCY_BANDS[vocalBand]?.name === 'vocal';
    const show = (levels: number[]) => levels.map((value) => value.toFixed(2)).join(' ');

    return {
      name,
      passed,
      detail:
        `60 Hz → ${FREQUENCY_BANDS[bassBand]?.label} [${show(bass)}], ` +
        `1 kHz → ${FREQUENCY_BANDS[vocalBand]?.label} [${show(vocal)}]`,
    };
  } catch (error) {
    return { name, passed: false, detail: error instanceof Error ? error.message : String(error) };
  }
}
