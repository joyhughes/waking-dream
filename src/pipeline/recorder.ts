import { embedParameters, type EmbeddedParameters } from './parameters';

/**
 * Recording the canvas to a video file, and saving single frames.
 *
 * `captureStream` taps the canvas's existing drawing buffer, so recording adds an encode but no
 * extra render — the frames going to the file are literally the ones on screen. It needs the
 * context to have been created with `preserveDrawingBuffer`, which is why `gl.ts` asks for it.
 */

/**
 * MP4 first, deliberately.
 *
 * WebM is the format browsers have always recorded to, and on a Mac almost nothing opens it —
 * QuickTime, Photos, Final Cut and iMessage all refuse, so a recording arrives as a file the
 * machine cannot play. H.264 in MP4 opens everywhere. Chrome and Safari can both record it now;
 * the WebM entries stay below as the fallback for browsers that cannot.
 */
const CODEC_PREFERENCES = [
  'video/mp4;codecs=avc1.42E01E',
  'video/mp4;codecs=h264',
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  return CODEC_PREFERENCES.find((type) => MediaRecorder.isTypeSupported(type));
}

export class CanvasRecorder {
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  static get supported(): boolean {
    return typeof MediaRecorder !== 'undefined' && pickMimeType() !== undefined;
  }

  get recording(): boolean {
    return this.recorder !== null;
  }

  get elapsedSeconds(): number {
    return this.recorder ? (performance.now() - this.startedAt) / 1000 : 0;
  }

  start(fps = 60, bitsPerSecond = 12_000_000): void {
    if (this.recorder) return;

    const mimeType = pickMimeType();
    if (!mimeType) throw new Error('This browser cannot record canvas video.');

    const stream = this.canvas.captureStream(fps);
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: bitsPerSecond });

    this.chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    // A timeslice keeps chunks arriving during the recording rather than in one blob at the end,
    // which is what stops a long take from being lost if the tab is closed mid-recording.
    recorder.start(1000);

    this.recorder = recorder;
    this.startedAt = performance.now();
  }

  async stop(): Promise<Blob> {
    const recorder = this.recorder;
    if (!recorder) throw new Error('Not recording.');

    const finished = new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
    });
    recorder.stop();
    await finished;

    this.recorder = null;
    for (const track of (recorder.stream as MediaStream).getTracks()) track.stop();

    return new Blob(this.chunks, { type: recorder.mimeType });
  }
}

export function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  // Revoking immediately can cancel the download on some browsers before it has read the blob.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/**
 * Saves the canvas as a PNG, optionally with the settings that made it written into the file.
 *
 * PNG rather than JPEG for two reasons: it is lossless, so a frame is a faithful record rather than
 * a re-encoded approximation of one, and it has somewhere to put the parameters. The metadata rides
 * in a `tEXt` chunk that every other decoder skips, so the file stays an ordinary image.
 */
export function saveCanvasFrame(
  canvas: HTMLCanvasElement,
  filename: string,
  parameters?: EmbeddedParameters,
): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('The canvas produced no image.'));
        return;
      }

      if (!parameters) {
        download(blob, filename);
        resolve();
        return;
      }

      void blob
        .arrayBuffer()
        .then((buffer) => {
          const withParameters = embedParameters(new Uint8Array(buffer), parameters);
          download(new Blob([withParameters], { type: 'image/png' }), filename);
          resolve();
        })
        .catch((error) => {
          // The image matters more than the metadata; if embedding fails, still save the frame.
          console.warn('Could not embed parameters in the saved frame.', error);
          download(blob, filename);
          resolve();
        });
    }, 'image/png');
  });
}

/** The extension matching whatever the recorder actually negotiated. */
export function extensionForMimeType(mimeType: string): string {
  return mimeType.includes('mp4') ? 'mp4' : 'webm';
}

export function timestampedName(prefix: string, extension: string): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `${prefix}-${stamp}.${extension}`;
}
