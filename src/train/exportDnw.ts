import { groupsFor } from '../gpu/gl';
import { DNW_FORMAT, DNW_MAGIC, type ControlSpec, type ModelHeader, type ModelOp } from '../model/format';
import { packBias, packConv } from '../model/pack';
import type { DreamNetTf } from './dreamNetTf';

/**
 * Writes a browser-trained network out as the same `.dnw` file `train/export.py` produces.
 *
 * There is deliberately one format and one runtime. A model trained here is not a lesser kind of
 * model — once exported it is indistinguishable from one trained in PyTorch, runs through the same
 * WebGL kernels at the same speed, and can be handed to someone else, dropped into `public/models/`,
 * and deployed. This function is the Python exporter's `build()` and `write_dnw()`, in TypeScript.
 *
 * The variable shapes in `dreamNetTf.ts` were chosen so nothing needs transposing on the way out.
 * That is not a convenience: a transpose that is wrong in one of the two exporters produces a model
 * that loads, runs, and draws the wrong thing, with no error to notice.
 */

export interface ExportOptions {
  name: string;
  description: string;
  controls: ControlSpec[];
  /** Resolution the run trained at, recorded so the app can say what the model was tuned for. */
  trainedAt: number;
  /** Free-form record of what produced this model. */
  provenance?: Record<string, unknown>;
}

/** Builds the same slider spec `common.py`'s `style_controls` produces, from a list of style names. */
export function styleControls(names: string[]): ControlSpec[] {
  return names.map((name, index) => ({
    name: `style${index}`,
    label: `Style: ${name}`,
    description: 'How strongly this pattern is drawn. Several can be raised at once to blend them.',
    min: 0,
    max: 1,
    default: index === 0 ? 1 : 0,
  }));
}

export function exportDnw(model: DreamNetTf, options: ExportOptions): ArrayBuffer {
  const gpuChunks: Float32Array[] = [];
  const offsets: { weight: number; bias: number }[] = [];
  let texelCursor = 0;

  for (const conv of model.convs) {
    // dataSync on a [kh, kw, cin, cout] variable is already the order packConv indexes.
    const packed = packConv(
      conv.kernel.dataSync() as Float32Array,
      conv.kernelSize,
      conv.inChannels,
      conv.outChannels,
    );
    const packedBias = packBias(conv.bias.dataSync() as Float32Array, conv.outChannels);

    offsets.push({ weight: texelCursor, bias: texelCursor + packed.texelCount });
    gpuChunks.push(packed.texels, packedBias.texels);
    texelCursor += packed.texelCount + packedBias.texelCount;
  }

  const gpu = concat(gpuChunks);

  const cpuChunks: Float32Array[] = [];
  const cpuTensors: ModelHeader['cpuTensors'] = {};
  let cpuCursor = 0;

  const addCpu = (key: string, data: Float32Array, shape: number[]) => {
    cpuTensors[key] = { offset: cpuCursor, length: data.length, shape };
    cpuChunks.push(data);
    cpuCursor += data.length;
  };

  const normSlots = model.norms.map((norm) => norm.channels);
  model.norms.forEach((norm, slot) => {
    addCpu(`norm${slot}.gamma`, norm.gamma.dataSync() as Float32Array, [norm.channels]);
    addCpu(`norm${slot}.beta`, norm.beta.dataSync() as Float32Array, [norm.channels]);
  });

  addCpu('film.w1', model.filmWeight.dataSync() as Float32Array, [model.config.condDims, model.config.filmHidden]);
  addCpu('film.b1', model.filmBias.dataSync() as Float32Array, [model.config.filmHidden]);
  model.norms.forEach((norm, slot) => {
    addCpu(`film.gammaW${slot}`, norm.gammaProjection.dataSync() as Float32Array, [model.config.filmHidden, norm.channels]);
    addCpu(`film.betaW${slot}`, norm.betaProjection.dataSync() as Float32Array, [model.config.filmHidden, norm.channels]);
  });

  const cpu = concat(cpuChunks);
  const ops = buildOps(model, offsets, normSlots);

  const header: ModelHeader = {
    format: DNW_FORMAT,
    name: options.name,
    description: options.description,
    teacher: { kind: 'style', trainedIn: 'browser', ...options.provenance },
    trainedAt: options.trainedAt,
    inputName: 'x',
    outputName: 'y',
    ops,
    normSlots,
    gpuTexels: gpu.length / 4,
    cpuFloats: cpu.length,
    cpuTensors,
    conditioning: {
      dims: model.config.condDims,
      hidden: model.config.filmHidden,
      controls: options.controls,
    },
  };

  return encode(header, gpu, cpu);
}

function buildOps(
  model: DreamNetTf,
  offsets: { weight: number; bias: number }[],
  normSlots: number[],
): ModelOp[] {
  const ops: ModelOp[] = [];

  const conv = (index: number, input: string, output: string, activation: 'none' | 'tanh01' = 'none') => {
    const layer = model.convs[index];
    ops.push({
      type: 'conv',
      in: input,
      out: output,
      kernel: layer.kernelSize,
      stride: layer.stride,
      inChannels: layer.inChannels,
      outChannels: layer.outChannels,
      weightOffset: offsets[index].weight,
      biasOffset: offsets[index].bias,
      activation,
    });
  };

  const norm = (slot: number, input: string, output: string, relu: boolean, skip: string | null = null) => {
    ops.push({ type: 'norm', in: input, out: output, channels: normSlots[slot], slot, relu, skip });
  };

  conv(0, 'x', 'c0');
  norm(0, 'c0', 'h0', true);
  conv(1, 'h0', 'c1');
  norm(1, 'c1', 'h1', true);
  conv(2, 'h1', 'c2');
  norm(2, 'c2', 'h2', true);

  let current = 'h2';
  for (let block = 0; block < model.config.blocks; block++) {
    const first = 3 + 2 * block;
    const second = 4 + 2 * block;
    conv(first, current, `b${block}c0`);
    norm(first, `b${block}c0`, `b${block}h0`, true);
    conv(second, `b${block}h0`, `b${block}c1`);
    // The skip folds into the normalization's write, exactly as the residual add does in forward().
    norm(second, `b${block}c1`, `b${block}out`, false, current);
    current = `b${block}out`;
  }

  const upsampleFirst = 3 + 2 * model.config.blocks;
  ops.push({ type: 'resize', in: current, out: 'u0', scale: 2 });
  conv(upsampleFirst, 'u0', 'uc0');
  norm(upsampleFirst, 'uc0', 'uh0', true);
  ops.push({ type: 'resize', in: 'uh0', out: 'u1', scale: 2 });
  conv(upsampleFirst + 1, 'u1', 'uc1');
  norm(upsampleFirst + 1, 'uc1', 'uh1', true);
  conv(model.convs.length - 1, 'uh1', 'y', 'tanh01');

  return ops;
}

function concat(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(total);
  let cursor = 0;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  return out;
}

function encode(header: ModelHeader, gpu: Float32Array, cpu: Float32Array): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(header));
  // The float payload must start 4-byte aligned; padding the header with spaces moves the start
  // without making it invalid JSON.
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

/** Rounds channel counts the way the runtime pads them, for the size estimate shown before a run. */
export function estimateModelBytes(model: DreamNetTf): number {
  let texels = 0;
  for (const conv of model.convs) {
    const paddedIn = groupsFor(conv.inChannels) * 4;
    const groupsOut = groupsFor(conv.outChannels);
    texels += conv.kernelSize * conv.kernelSize * paddedIn * groupsOut + groupsOut;
  }
  const cpuFloats = model.trainableVariables
    .filter((variable) => variable.rank !== 4)
    .reduce((sum, variable) => sum + variable.size, 0);
  return texels * 16 + cpuFloats * 4;
}
