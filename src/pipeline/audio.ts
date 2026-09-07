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

/** Rise fast, fall slowly. A meter that falls as fast as it rises reads as flicker, not as rhythm. */
const ATTACK = 0.55;
const RELEASE = 0.12;

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

export class AudioAnalyser {
  private readonly context: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly stream: MediaStream;
  private readonly ownsStream: boolean;
  private readonly spectrum: Uint8Array;
  private readonly smoothed: Float32Array;
  private readonly peaks: Float32Array;
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
    // Some smoothing in the analyser itself, and the rest in the attack/release below. Doing it all
    // here would make every band equally sluggish, including the transients worth reacting to.
    analyser.smoothingTimeConstant = 0.5;

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

      const previous = this.smoothed[band];
      const rate = normalized > previous ? ATTACK : RELEASE;
      this.smoothed[band] = previous + (normalized - previous) * rate;
    }

    return this.smoothed;
  }

  stop(): void {
    if (this.ownsStream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    void this.context.close();
  }
}
