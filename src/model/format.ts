/**
 * The `.dnw` container: one file holding a JSON header and two blocks of float32.
 *
 * A single file rather than a manifest plus a weights blob, because the browser then makes one
 * request and there is no way for the two halves to drift apart in a cache. `train/export.py`
 * writes exactly this, and the two ends of the format are meant to be read side by side.
 *
 *   bytes 0..3     magic "DNW1"
 *   bytes 4..7     header length, uint32 little-endian
 *   header         UTF-8 JSON, space-padded so the payload starts 16-byte aligned
 *   gpu block      `gpuTexels` * 4 float32 — convolution weights and biases, uploaded verbatim
 *                  into the weight texture and indexed by texel number from the shaders
 *   cpu block      float32 — everything evaluated on the CPU: the instance-norm affines and, when
 *                  the model is conditioned, the control MLP that modulates them
 */

export const DNW_MAGIC = 'DNW1';
export const DNW_FORMAT = 'dreamnet-weights-1';

export type OpActivation = 'none' | 'relu' | 'tanh01';

export interface ConvOp {
  type: 'conv';
  in: string;
  out: string;
  kernel: number;
  stride: number;
  inChannels: number;
  outChannels: number;
  /** Texel index into the GPU block. */
  weightOffset: number;
  biasOffset: number;
  activation: OpActivation;
}

export interface NormOp {
  type: 'norm';
  in: string;
  out: string;
  channels: number;
  /** Index into `normSlots`; selects this layer's region of the affine texture. */
  slot: number;
  relu: boolean;
  /** Name of the tensor added into this layer's output, closing a residual block. */
  skip: string | null;
}

export interface ResizeOp {
  type: 'resize';
  in: string;
  out: string;
  /** Integer upsampling factor. Nearest, to match `nn.Upsample(mode='nearest')` in training. */
  scale: number;
}

export type ModelOp = ConvOp | NormOp | ResizeOp;

export interface ControlSpec {
  name: string;
  label: string;
  description: string;
  min: number;
  max: number;
  default: number;
}

export interface ModelHeader {
  format: string;
  name: string;
  description: string;
  /** Free-form record of the DeepDream configuration this model was distilled from. */
  teacher?: Record<string, unknown>;
  /** Resolution the model was trained at. It still runs at any size; this is what it was tuned for. */
  trainedAt?: number;
  inputName: string;
  outputName: string;
  ops: ModelOp[];
  /** Channel count of each instance-norm layer, in slot order. */
  normSlots: number[];
  gpuTexels: number;
  cpuFloats: number;
  /** Named views into the CPU block. Offsets and lengths are in floats. */
  cpuTensors: Record<string, { offset: number; length: number; shape: number[] }>;
  conditioning: {
    dims: number;
    hidden: number;
    controls: ControlSpec[];
  } | null;
}

export interface ParsedModel {
  header: ModelHeader;
  /** Ready to upload straight into the weight texture. */
  gpu: Float32Array;
  cpu: Float32Array;
}

export function parseDnw(buffer: ArrayBuffer): ParsedModel {
  if (buffer.byteLength < 8) {
    throw new Error('This file is too short to be a DreamNet model.');
  }

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== DNW_MAGIC) {
    throw new Error(`Not a DreamNet model: expected magic "${DNW_MAGIC}", found "${magic}".`);
  }

  const view = new DataView(buffer);
  const headerLength = view.getUint32(4, true);
  const headerStart = 8;
  const payloadStart = headerStart + headerLength;

  if (payloadStart > buffer.byteLength) {
    throw new Error('Model header length runs past the end of the file.');
  }
  if (payloadStart % 4 !== 0) {
    throw new Error('Model payload is not 4-byte aligned; the file was not written by export.py.');
  }

  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, headerStart, headerLength))) as ModelHeader;

  if (header.format !== DNW_FORMAT) {
    throw new Error(`Unsupported model format "${header.format}"; this build reads "${DNW_FORMAT}".`);
  }

  const gpuFloats = header.gpuTexels * 4;
  const expectedBytes = (gpuFloats + header.cpuFloats) * 4;
  if (payloadStart + expectedBytes > buffer.byteLength) {
    throw new Error(
      `Model payload is short: header declares ${expectedBytes} bytes, file has ${buffer.byteLength - payloadStart}.`,
    );
  }

  const gpu = new Float32Array(buffer.slice(payloadStart, payloadStart + gpuFloats * 4));
  const cpu = new Float32Array(buffer.slice(payloadStart + gpuFloats * 4, payloadStart + expectedBytes));

  validate(header);

  return { header, gpu, cpu };
}

/**
 * Catches the mismatches that would otherwise surface as a black canvas rather than an error: an op
 * referring to a tensor nothing produced, a normalization pointing at a slot that does not exist,
 * or a conditioned header missing the MLP the conditioning needs.
 */
function validate(header: ModelHeader): void {
  const produced = new Set<string>([header.inputName]);

  for (const [index, op] of header.ops.entries()) {
    if (!produced.has(op.in)) {
      throw new Error(`Op ${index} (${op.type}) reads "${op.in}", which no earlier op produced.`);
    }
    if (op.type === 'norm') {
      if (op.slot < 0 || op.slot >= header.normSlots.length) {
        throw new Error(`Op ${index} uses norm slot ${op.slot}, but the model declares ${header.normSlots.length}.`);
      }
      if (op.skip !== null && !produced.has(op.skip)) {
        throw new Error(`Op ${index} adds skip "${op.skip}", which no earlier op produced.`);
      }
    }
    produced.add(op.out);
  }

  if (!produced.has(header.outputName)) {
    throw new Error(`The model's declared output "${header.outputName}" is never produced.`);
  }

  for (const slot of header.normSlots.keys()) {
    for (const suffix of ['gamma', 'beta']) {
      const key = `norm${slot}.${suffix}`;
      if (!header.cpuTensors[key]) {
        throw new Error(`Model is missing "${key}" in its CPU block.`);
      }
    }
  }

  if (header.conditioning) {
    for (const key of ['film.w1', 'film.b1']) {
      if (!header.cpuTensors[key]) {
        throw new Error(`Model declares conditioning but is missing "${key}".`);
      }
    }
  }
}

/** Reads one named array out of the CPU block. Throws rather than returning a silently empty view. */
export function cpuTensor(model: ParsedModel, name: string): Float32Array {
  const spec = model.header.cpuTensors[name];
  if (!spec) throw new Error(`Model has no CPU tensor named "${name}".`);
  return model.cpu.subarray(spec.offset, spec.offset + spec.length);
}

export function hasCpuTensor(model: ParsedModel, name: string): boolean {
  return name in model.header.cpuTensors;
}
