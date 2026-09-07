/**
 * The list of models a deployed build ships with.
 *
 * A page cannot enumerate a directory over HTTP, so the models that were built into `public/models/`
 * have to announce themselves. `train/export.py` writes and updates `index.json` as part of every
 * export, which means a model becomes available in the app the moment it is exported and the same
 * manifest is what gets deployed — there is no separate step to forget.
 */

export interface ModelListing {
  file: string;
  name: string;
  description: string;
  bytes: number;
  trainedAt?: number;
  /** `style` for a pattern trained by style.py, `distilled` for a DeepDream student. */
  kind?: string;
  /** Slider labels, so the list can say what a model offers before it is downloaded. */
  controls?: string[];
  /** Base channel width. Together with `blocks`, this is what actually sets the frame cost. */
  width?: number;
  blocks?: number;
  params?: number;
}

/**
 * A rough cost class, for labelling the model list.
 *
 * Frame time scales with the width squared and linearly with the depth, because the residual stack
 * is where nearly all the arithmetic is. File size tracks that loosely but not reliably — a wide,
 * shallow network and a narrow, deep one can weigh the same and cost very different amounts.
 */
export function costLabel(listing: ModelListing): string {
  const width = listing.width ?? 0;
  const blocks = listing.blocks ?? 0;
  if (width === 0) return '';
  const work = width * width * Math.max(1, blocks);
  if (work <= 2000) return 'fast';
  if (work <= 20000) return 'medium';
  return 'heavy';
}

export const MODELS_BASE = 'models';

/**
 * Reads the manifest, returning an empty list when there is none.
 *
 * A missing `index.json` is the normal state of a fresh clone, not an error — nothing has been
 * trained yet — so it is reported as "no models" rather than surfaced as a failure.
 */
export async function fetchModelListings(): Promise<ModelListing[]> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}${MODELS_BASE}/index.json`, { cache: 'no-cache' });
    if (!response.ok) return [];
    const parsed = (await response.json()) as { models?: ModelListing[] };
    return Array.isArray(parsed.models) ? parsed.models : [];
  } catch {
    return [];
  }
}

export function modelUrl(file: string): string {
  return `${import.meta.env.BASE_URL}${MODELS_BASE}/${file}`;
}

export function formatSize(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} kB`;
}
