import type { TrainingImage } from './trainer';

/**
 * Collects the frames a training run repaints.
 *
 * The content images are not what the network learns — that comes entirely from the style images.
 * They only have to be things it must keep recognisable while repainting them, which is why
 * grabbing them from whatever is already on camera works as well as a curated photo set, and is a
 * good deal less trouble. Training on your own footage also means the network sees the lighting and
 * subject matter it will actually run on.
 */

export const DEFAULT_FRAME_COUNT = 12;

/** Snapshots a live source repeatedly, spacing the grabs so the frames are not near-duplicates. */
export async function captureFromSource(
  source: { readonly frame: TexImageSource | null; readonly width: number; readonly height: number },
  count = DEFAULT_FRAME_COUNT,
  spacingMs = 250,
  longestSide = 384,
): Promise<HTMLCanvasElement[]> {
  const frames: HTMLCanvasElement[] = [];

  for (let i = 0; i < count; i++) {
    const frame = source.frame;
    if (frame) frames.push(snapshot(frame, source.width, source.height, longestSide));
    if (i < count - 1) await new Promise((resolve) => setTimeout(resolve, spacingMs));
  }

  if (frames.length === 0) throw new Error('The source produced no frames to capture.');
  return frames;
}

/** Copies one frame into a canvas, scaled down so training crops are cheap to cut. */
export function snapshot(frame: TexImageSource, width: number, height: number, longestSide: number): HTMLCanvasElement {
  const scale = Math.min(1, longestSide / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));

  const context = canvas.getContext('2d');
  if (!context) throw new Error('Could not create a 2D context to capture a frame.');
  context.drawImage(frame as CanvasImageSource, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/** Loads an image file the user picked, as something the trainer can read. */
export function loadImageFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      // Revoked only after decode; releasing earlier can cancel the load on some browsers.
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not decode "${file.name}".`));
    };
    image.src = url;
  });
}

export type { TrainingImage };
