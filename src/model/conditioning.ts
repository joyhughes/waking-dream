import { groupsFor } from '../gpu/gl';
import { cpuTensor, hasCpuTensor, type ParsedModel } from './format';

/**
 * The control vector, evaluated on the CPU into the per-channel affine parameters the instance-norm
 * shader reads.
 *
 * This is the mechanism behind live controls on a fixed network. The convolution weights never
 * change — recompiling or re-uploading them per frame would be hopeless — but every normalization
 * layer already multiplies by a per-channel scale and adds a per-channel shift, and those are cheap
 * to recompute. A small MLP maps the control vector to offsets on them, so one trained model covers
 * a continuum of dream configurations instead of one, and moving a slider costs a few thousand
 * multiply-adds on the CPU rather than a second forward pass.
 *
 * The idea is conditional instance normalization (Dumoulin et al. 2017, "A Learned Representation
 * For Artistic Style"), which showed a single network can hold many styles this way; the FiLM
 * framing (Perez et al. 2018) is the same arithmetic with a general conditioner in front.
 * `train/model.py` implements the identical evaluation so the two agree exactly.
 */
export class AffineEvaluator {
  /** Texel offset of each slot's gammas. Betas follow one group-count later. */
  readonly slotOffsets: number[] = [];
  readonly texelCount: number;
  private readonly buffer: Float32Array;
  private readonly hidden: Float32Array;
  private lastControls: number[] | null = null;

  constructor(private readonly model: ParsedModel) {
    const { normSlots, conditioning } = model.header;

    let offset = 0;
    for (const channels of normSlots) {
      this.slotOffsets.push(offset);
      offset += 2 * groupsFor(channels);
    }
    this.texelCount = Math.max(1, offset);
    this.buffer = new Float32Array(this.texelCount * 4);
    this.hidden = new Float32Array(conditioning?.hidden ?? 0);
  }

  get controls(): { name: string; label: string; description: string; min: number; max: number; default: number }[] {
    return this.model.header.conditioning?.controls ?? [];
  }

  get isConditioned(): boolean {
    return this.model.header.conditioning !== null;
  }

  /**
   * Recomputes the affine buffer for a control vector, or returns null when nothing changed.
   *
   * Returning null lets the caller skip the texture upload, which is the only part of this with a
   * real cost. Sliders sit still for most frames of a session.
   */
  evaluate(controls: number[]): Float32Array | null {
    if (this.lastControls && sameControls(this.lastControls, controls)) {
      return null;
    }
    this.lastControls = controls.slice();

    const { normSlots, conditioning } = this.model.header;

    if (conditioning) {
      const w1 = cpuTensor(this.model, 'film.w1');
      const b1 = cpuTensor(this.model, 'film.b1');
      const { dims, hidden } = conditioning;

      // h = relu(control @ W1 + b1), with W1 stored row-major as [dims, hidden].
      for (let j = 0; j < hidden; j++) {
        let sum = b1[j];
        for (let i = 0; i < dims; i++) sum += (controls[i] ?? 0) * w1[i * hidden + j];
        this.hidden[j] = sum > 0 ? sum : 0;
      }
    }

    for (let slot = 0; slot < normSlots.length; slot++) {
      const channels = normSlots[slot];
      const groups = groupsFor(channels);
      const base = this.slotOffsets[slot] * 4;
      const gamma0 = cpuTensor(this.model, `norm${slot}.gamma`);
      const beta0 = cpuTensor(this.model, `norm${slot}.beta`);

      const gammaW = hasCpuTensor(this.model, `film.gammaW${slot}`) ? cpuTensor(this.model, `film.gammaW${slot}`) : null;
      const betaW = hasCpuTensor(this.model, `film.betaW${slot}`) ? cpuTensor(this.model, `film.betaW${slot}`) : null;
      const hiddenSize = this.hidden.length;

      for (let c = 0; c < channels; c++) {
        let gamma = gamma0[c];
        let beta = beta0[c];
        if (gammaW) for (let h = 0; h < hiddenSize; h++) gamma += this.hidden[h] * gammaW[h * channels + c];
        if (betaW) for (let h = 0; h < hiddenSize; h++) beta += this.hidden[h] * betaW[h * channels + c];
        this.buffer[base + c] = gamma;
        this.buffer[base + groups * 4 + c] = beta;
      }

      // Channels beyond the declared count exist only because the layout pads to four. Their gamma
      // is left at zero so any numerical dirt in the padded input cannot leak into a real channel
      // through a later convolution.
      for (let c = channels; c < groups * 4; c++) {
        this.buffer[base + c] = 0;
        this.buffer[base + groups * 4 + c] = 0;
      }
    }

    return this.buffer;
  }
}

function sameControls(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
