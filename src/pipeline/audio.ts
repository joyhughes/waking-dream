/**
 * Microphone input, split into frequency bands, driving the model's controls.
 *
 * The conditioning already exists and is already cheap to move — a control vector goes through a
 * small CPU-side MLP each frame and rewrites every instance-norm affine. So there is nothing
 * stopping those numbers coming from a spectrum analyser instead of from sliders, and a network
 * that holds several patterns at once becomes an instrument: bass pushes one texture forward, the
 * vocal range pushes another, and the mix moves with the track.
 *
 * Bands are the musical ones rather than an even split of the spectrum. An even split would give
 * four of five bands to content above 4 kHz, where almost nothing in music lives — the interesting
 * structure is all crammed into the bottom two octaves, so the edges are placed to match.
 */

export interface FrequencyBand {
  name: string;
  label: string;
  /** Hz. */
  low: number;
  high: number;
}

export const FREQUENCY_BANDS: FrequencyBand[] = [
  { name: 'bass', label: 'Bass', low: 20, high: 160 },
  { name: 'low-mid', label: 'Low mid', low: 160, high: 500 },
  { name: 'vocal', label: 'Vocal range', low: 500, high: 2000 },
  { name: 'presence', label: 'Presence', low: 2000, high: 6000 },
  { name: 'air', label: 'Air', low: 6000, high: 16000 },
];

/**
 * Which band each control gets by default.
 *
 * Bass first and the vocal range second, because with two styles those are the two that move most
 * independently in almost any music — kick and voice rarely peak together, so the two textures
 * actually take turns instead of pumping as one.
 */
export const DEFAULT_BAND_ASSIGNMENTS = [0, 2, 3, 1, 4];

/**
 * Rise almost instantly, fall quickly but not as fast.
 *
 * Deliberately much less smoothing than before. Smoothing is what makes a meter pleasant to look at
 * and what makes a control feel late — every stage of it puts the picture further behind the beat,
 * and three stages (the analyser's own, the attack, the release) had it lagging visibly. What is
 * left is the minimum that stops a control chattering between adjacent frames.
 */
const ATTACK = 0.85;
const RELEASE = 0.25;

/** How quickly the running peak decays, per frame. Sets how fast the gain re-adapts after a loud passage. */
const PEAK_DECAY = 0.999;

/** Below this the band is treated as silence, so room noise does not get normalized up into signal. */
const NOISE_FLOOR = 0.01;

/**
 * How much of a band's level comes from its loudest bin rather than its average.
 *
 * The average alone is stable but badly dilutes narrow content: a sustained 1 kHz tone occupies a
 * handful of bins out of the hundreds in the 500–2000 Hz band, so the mean barely lifts off the
 * noise floor and a control driven by it hardly moves. The peak alone is the opposite — responsive,
 * and jumpy enough that broadband material reads as constant. Weighted toward the peak, because a
 * control that under-responds to a melody line is the worse failure here.
 */
const PEAK_WEIGHT = 0.65;

/** Turns a per-frame change into something on the same scale as a level. */
const ONSET_SCALE = 3;

/** How fast an onset spike falls away once the rise stops. */
const ONSET_DECAY = 0.72;

/** The frame time the onset scale is defined against, so the derivative is per second not per frame. */
const REFERENCE_FRAME_MS = 1000 / 60;

export class AudioAnalyser {
  private readonly context: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly stream: MediaStream;
  private readonly ownsStream: boolean;
  private readonly spectrum: Uint8Array;
  private readonly smoothed: Float32Array;
  private readonly peaks: Float32Array;
  /** Previous frame's normalized level per band, for the derivative. */
  private readonly previous: Float32Array;
  /** Smoothed positive rate of change per band — how hard each band is *arriving*, not how loud it is. */
  private readonly onsets: Float32Array;
  private lastReadAt = 0;
  readonly label: string;

  private constructor(context: AudioContext, analyser: AnalyserNode, stream: MediaStream, ownsStream: boolean, label: string) {
    this.context = context;
    this.analyser = analyser;
    this.stream = stream;
    this.ownsStream = ownsStream;
    this.label = label;
    this.spectrum = new Uint8Array(analyser.frequencyBinCount);
    this.smoothed = new Float32Array(FREQUENCY_BANDS.length);
    this.peaks = new Float32Array(FREQUENCY_BANDS.length).fill(NOISE_FLOOR);
    this.previous = new Float32Array(FREQUENCY_BANDS.length);
    this.onsets = new Float32Array(FREQUENCY_BANDS.length);
  }

  /**
   * Opens the microphone, or wraps a stream supplied by a caller.
   *
   * The injected-stream path is what lets the self-test drive this with an oscillator instead of a
   * microphone, which is the only way to check the band split against a known frequency.
   */
  static async open(options: { deviceId?: string; stream?: MediaStream } = {}): Promise<AudioAnalyser> {
    let stream = options.stream;
    const ownsStream = !stream;

    if (!stream) {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser will not give a page microphone access.');
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: options.deviceId ? { exact: options.deviceId } : undefined,
          // All three of these exist to make speech intelligible and all three destroy music: the
          // gain control flattens exactly the dynamics being measured, and the noise suppressor
          // treats sustained bass as noise.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    }

    const context = new AudioContext();
    // Browsers start an AudioContext suspended until a user gesture; opening the mic is one, but
    // resuming explicitly covers the injected-stream case where no gesture happened.
    if (context.state === 'suspended') await context.resume();

    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    // Low, because everything downstream of it is also smoothing. The transients are the useful
    // part of a drum hit; averaging them into the neighbouring frames is exactly what loses the
    // rhythm the controls are supposed to be following.
    analyser.smoothingTimeConstant = 0.15;

    context.createMediaStreamSource(stream).connect(analyser);

    const label = stream.getAudioTracks()[0]?.label || 'Audio input';
    return new AudioAnalyser(context, analyser, stream, ownsStream, label);
  }

  static async listInputs(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    return (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'audioinput');
  }

  get bandCount(): number {
    return FREQUENCY_BANDS.length;
  }

  /**
   * Current level per band, in [0, 1], smoothed and normalized against a decaying running peak.
   *
   * Normalizing per band matters more than it sounds: raw spectrum energy falls steeply with
   * frequency, so an absolute threshold that bass crosses constantly is one the air band never
   * reaches. Each band is scaled against its own recent maximum, which makes the controls respond
   * to *this* band getting louder rather than to where it sits in the mix.
   */
  levels(): Float32Array {
    this.analyser.getByteFrequencyData(this.spectrum);

    // The derivative is per unit time, not per frame: without this a 30 fps machine would read
    // every onset as twice the size of the same music on a 60 fps one.
    const now = performance.now();
    const elapsed = this.lastReadAt > 0 ? now - this.lastReadAt : REFERENCE_FRAME_MS;
    this.lastReadAt = now;
    const frameScale = Math.min(4, REFERENCE_FRAME_MS / Math.max(1, elapsed));

    const binHz = this.context.sampleRate / this.analyser.fftSize;

    for (let band = 0; band < FREQUENCY_BANDS.length; band++) {
      const { low, high } = FREQUENCY_BANDS[band];
      const first = Math.max(0, Math.floor(low / binHz));
      const last = Math.min(this.spectrum.length - 1, Math.ceil(high / binHz));

      let sum = 0;
      let peak = 0;
      for (let bin = first; bin <= last; bin++) {
        sum += this.spectrum[bin];
        if (this.spectrum[bin] > peak) peak = this.spectrum[bin];
      }
      const mean = last >= first ? sum / (last - first + 1) : 0;
      const raw = (mean * (1 - PEAK_WEIGHT) + peak * PEAK_WEIGHT) / 255;

      this.peaks[band] = Math.max(raw, this.peaks[band] * PEAK_DECAY, NOISE_FLOOR);
      const normalized = raw <= NOISE_FLOOR ? 0 : Math.min(1, raw / this.peaks[band]);

      // The first derivative, taken before smoothing so it is not flattened by it, and only the
      // rising half of it. A band falling away is not an event; a band arriving is. This is what
      // separates a kick from a sustained bass note of the same loudness — the sustained note has
      // level and no onset, the kick has both.
      const rise = Math.max(0, normalized - this.previous[band]) * ONSET_SCALE * frameScale;
      this.previous[band] = normalized;
      // Rises immediately and decays on its own, so a transient reads as a spike rather than a step.
      this.onsets[band] = Math.min(1, rise > this.onsets[band] ? rise : this.onsets[band] * ONSET_DECAY);

      const smoothedPrevious = this.smoothed[band];
      const rate = normalized > smoothedPrevious ? ATTACK : RELEASE;
      this.smoothed[band] = smoothedPrevious + (normalized - smoothedPrevious) * rate;
    }

    return this.smoothed;
  }

  /**
   * Positive rate of change per band, as of the last `levels()` call.
   *
   * Kept separate rather than folded in, because how much of a control's movement should come from
   * onsets and how much from level is a decision that belongs to whoever is playing it — a filter
   * that only reacts to attacks feels percussive, one that only reacts to level feels like a
   * volume pedal, and the interesting settings are in between.
   */
  onsetLevels(): Float32Array {
    return this.onsets;
  }

  stop(): void {
    if (this.ownsStream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    void this.context.close();
  }
}


/**
 * Below this the bands are treated as too quiet to have a winner, so silence does not get scaled up
 * into a control pinned at full.
 */
const CONTRAST_FLOOR = 0.04;

export interface ControlMapping {
  /** Band index per control. */
  assignments: number[];
  /** How far, and which way, an already-leading control is pushed further. */
  leaderBonus: number;
  /**
   * How hard the assigned bands compete with each other.
   *
   * At 0 each band is on its own scale, which is what makes them all sit mid-range: every band is
   * already normalized against its own recent peak, so bass and the vocal range both read "fairly
   * loud for themselves" almost all the time and the ratio between them barely moves.
   *
   * Turning this up does two things at once. The loudest of the assigned bands is scaled to exactly
   * 1, so whichever is winning reads full — and the others are put through a rising power curve, so
   * a band at 80% of the leader drops toward 50% rather than staying alongside it. The first part is
   * what makes the peak reach the top; the second is what makes the gap visible.
   */
  sensitivity: number;
  /** Crossfade between the slider positions and the sound. 1 is fully sound-driven. */
  amount: number;
}

/**
 * What each band is contributing, before anything is mapped to a control.
 *
 * Level plus a share of the onset, then that band's own gain. Per-band gain rather than one master:
 * the bands do not arrive on remotely equal footing in real music — a mix with a loud kick and a
 * quiet hi-hat needs them scaled differently before any of the competition downstream means
 * anything, and a single multiplier can only move all five together.
 */
export function bandActivations(
  levels: ArrayLike<number>,
  onsets: ArrayLike<number>,
  gains: number[],
  onsetBoost: number,
): number[] {
  return FREQUENCY_BANDS.map((_, band) => {
    const level = levels[band] ?? 0;
    const onset = onsets[band] ?? 0;
    const gain = gains[band] ?? 1;
    return Math.min(1, (level + onsetBoost * onset) * gain);
  });
}

/**
 * Turns band activations into a control vector.
 *
 * Pure, and separate from the engine, so the shaping can be checked against known inputs rather
 * than inferred from watching a picture move.
 */
export function mapLevelsToControls(
  base: number[],
  activations: ArrayLike<number>,
  mapping: ControlMapping,
): number[] {
  const { assignments, sensitivity, amount, leaderBonus } = mapping;

  const scaled = base.map((_, index) => {
    const band = assignments[index] ?? DEFAULT_BAND_ASSIGNMENTS[index] ?? index;
    return Math.min(1, activations[band] ?? 0);
  });

  // Whichever control is already ahead is pushed further ahead, by a share of the distance it has
  // left to the top. The competition below reshapes the whole set against its peak; this is the
  // separate thing of rewarding the winner for winning, which is what makes one texture clearly
  // take the frame on a hit rather than the two of them trading small margins.
  if (leaderBonus > 0 && scaled.length > 1) {
    let leader = 0;
    for (let index = 1; index < scaled.length; index++) {
      if (scaled[index] > scaled[leader]) leader = index;
    }
    if (scaled[leader] > CONTRAST_FLOOR) {
      scaled[leader] = scaled[leader] + leaderBonus * (1 - scaled[leader]);
    }
  }

  let shaped = scaled;
  const peak = scaled.reduce((most, value) => Math.max(most, value), 0);

  if (sensitivity > 0 && peak > CONTRAST_FLOOR) {
    // 1 at no sensitivity, 3 at full. A cube is enough to open a 0.8 ratio to about 0.5 without
    // collapsing everything below the leader to zero.
    const exponent = 1 + sensitivity * 2;
    shaped = scaled.map((value) => {
      const competed = Math.pow(value / peak, exponent);
      return value * (1 - sensitivity) + competed * sensitivity;
    });
  }

  return base.map((value, index) => value * (1 - amount) + shaped[index] * amount);
}
