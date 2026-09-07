/**
 * Recording the canvas to a video file, and saving single frames.
 *
 * `captureStream` taps the canvas's existing drawing buffer, so recording adds an encode but no
 * extra render — the frames going to the file are literally the ones on screen. It needs the
 * context to have been created with `preserveDrawingBuffer`, which is why `gl.ts` asks for it.
 */

const CODEC_PREFERENCES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4',
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

export function saveCanvasFrame(canvas: HTMLCanvasElement, filename: string): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('The canvas produced no image.'));
        return;
      }
      download(blob, filename);
      resolve();
    }, 'image/png');
  });
}

export function timestampedName(prefix: string, extension: string): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `${prefix}-${stamp}.${extension}`;
}
