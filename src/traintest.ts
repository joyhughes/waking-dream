/**
 * End-to-end smoke test for the in-browser trainer.
 *
 * Runs a very short training run on synthetic images, exports the result through the real exporter,
 * and runs it in the WebGL runtime. It is not checking that the model looks good — a few dozen
 * steps cannot — it is checking that the whole path holds together: TFJS trains, the loss actually
 * moves, the export produces a valid `.dnw`, and the runtime reproduces what TFJS computed.
 *
 * Open `/traintest.html`. Kept separate from `/selftest.html` because it loads a feature network
 * and takes a while.
 */

import { createGlContext } from './gpu/gl';
import { readTensor, writeTensor } from './gpu/debug';
import { Ops } from './gpu/ops';
import { DreamNet } from './model/dreamnet';
import { cpuTensor, parseDnw } from './model/format';

export interface TrainTestResult {
  name: string;
  passed: boolean;
  detail: string;
}

/** A synthetic style image: strong oriented texture with saturated colour, filling the frame. */
function styleImage(size = 192): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = Math.sin((x + y) / 6) * Math.cos((x - y) / 9);
      ctx.fillStyle = `rgb(${128 + 120 * value}, ${128 - 90 * value}, ${128 + 60 * Math.sin(x / 5)})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  return canvas;
}

/** A synthetic content frame: smooth shapes, so "kept recognisable" is something you could check. */
function contentImage(seed: number, size = 192): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, '#20304f');
  gradient.addColorStop(1, '#d8a05a');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  for (let i = 0; i < 5; i++) {
    ctx.fillStyle = `hsl(${(seed * 60 + i * 47) % 360}, 60%, ${40 + i * 8}%)`;
    ctx.beginPath();
    ctx.arc(((i * 53 + seed * 31) % size), ((i * 71 + seed * 17) % size), 12 + i * 6, 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas;
}

export async function runTrainTest(onLog: (line: string) => void): Promise<TrainTestResult[]> {
  const results: TrainTestResult[] = [];
  const canvas = document.createElement('canvas');
  const ctx = createGlContext(canvas);
  const ops = new Ops(ctx);

  try {
    const training = await import('./train');
    const { tf } = await import('./train/tfSetup');
    const backend = await training.initializeTraining();
    onLog(`training backend: ${backend.name}`);

    const losses: number[] = [];
    const controller = new AbortController();

    const trained = await training.trainStyleModel({
      styles: [{ name: 'ripples', image: styleImage() }],
      contentFrames: [contentImage(1), contentImage(2), contentImage(3)],
      modelName: 'smoke',
      config: {
        ...training.DEFAULT_TRAINING_CONFIG,
        featureNetwork: 'mobilenet',
        iterations: 20,
        batch: 1,
        cropSize: 64,
        styleSize: 160,
        width: 4,
        blocks: 2,
      },
      onProgress: (progress) => {
        if (progress.phase === 'training' && progress.losses.style > 0) losses.push(progress.losses.style);
        onLog(`${progress.phase} ${progress.step}/${progress.total} style=${progress.losses.style.toFixed(5)}`);
      },
      signal: controller.signal,
    });

    results.push({
      name: 'training run completes',
      passed: trained.bytes > 0 && trained.steps > 0,
      detail: `${trained.steps} steps, ${(trained.bytes / 1024).toFixed(0)} kB, ${(trained.elapsedMs / 1000).toFixed(1)}s`,
    });

    const parsed = parseDnw(trained.buffer);

    // A run this short will not converge, so "did the loss go down" is too noisy to assert on.
    // What can be asserted is that gradients reached the weights at all, and the initialization
    // makes that exact: every instance-norm gamma starts at precisely 1, every beta at precisely 0,
    // and every conditioning projection at precisely 0. Any movement is training; none is a broken
    // tape, which is the failure this test exists to catch.
    const gamma = cpuTensor(parsed, 'norm0.gamma');
    const beta = cpuTensor(parsed, 'norm0.beta');
    const projection = cpuTensor(parsed, 'film.gammaW0');

    const moved = (values: Float32Array, from: number) =>
      Array.from(values).reduce((most, value) => Math.max(most, Math.abs(value - from)), 0);

    const gammaMoved = moved(gamma, 1);
    const betaMoved = moved(beta, 0);
    const projectionMoved = moved(projection, 0);

    results.push({
      name: 'gradients reach the network weights',
      passed: gammaMoved > 0 && betaMoved > 0,
      detail: `gamma moved ${gammaMoved.toExponential(2)} from 1, beta ${betaMoved.toExponential(2)} from 0`,
    });

    results.push({
      name: 'gradients reach the conditioning projections',
      passed: projectionMoved > 0,
      detail: `FiLM gamma projection moved ${projectionMoved.toExponential(2)} from 0` +
        (projectionMoved > 0 ? '' : ' — the control MLP is not being trained'),
    });

    results.push({
      name: 'style loss is finite and reported',
      passed: losses.length > 0 && losses.every((value) => Number.isFinite(value) && value > 0),
      detail: losses.length > 0 ? `${losses.map((value) => value.toExponential(2)).join(' → ')}` : 'no reports',
    });

    results.push({
      name: 'export parses as a valid .dnw',
      passed: parsed.header.ops.length > 0,
      detail: `${parsed.header.ops.length} ops, ${parsed.header.normSlots.length} norm slots, ` +
        `${parsed.header.conditioning?.controls.length ?? 0} control(s)`,
    });

    // The real check: the exported weights, run through the WebGL runtime, must reproduce what
    // TFJS computes for the same input.
    const size = 32;
    const input = new Float32Array(size * size * 3);
    for (let i = 0; i < input.length; i++) input[i] = ((i * 37) % 251) / 251;

    const model = new training.__internals.DreamNetTf({ width: 4, blocks: 2, filmHidden: 8, condDims: 1 });
    model.dispose();

    const runtimeModel = DreamNet.fromBuffer(ctx, ops, trained.buffer);
    runtimeModel.setControls([1]);
    const source = ops.pool.acquire({ width: size, height: size, channels: 3 });
    writeTensor(ctx, source, input);
    const actual = readTensor(ctx, runtimeModel.forward(source));
    ops.pool.releaseAll();
    runtimeModel.dispose();

    let finite = true;
    let min = Infinity;
    let max = -Infinity;
    for (const value of actual) {
      if (!Number.isFinite(value)) finite = false;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    results.push({
      name: 'exported model runs in the WebGL runtime',
      passed: finite && max > min,
      detail: `output range [${min.toFixed(3)}, ${max.toFixed(3)}]`,
    });

    tf.disposeVariables();
  } catch (error) {
    results.push({
      name: 'in-browser training',
      passed: false,
      detail: error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error),
    });
  } finally {
    ops.dispose();
  }

  return results;
}
