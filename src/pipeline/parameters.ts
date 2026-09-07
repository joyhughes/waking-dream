import { DEFAULT_CONFIG, type EngineConfig } from './engine';
import { isPng, readPngText, writePngText } from './pngText';

/**
 * The settings that made a frame, carried inside the frame.
 *
 * A saved image is the only durable record of a look that was found by moving sliders, and without
 * this the record is incomplete in the one way that matters: you can see what it looked like and
 * have no way back to it. Embedding the parameters makes a PNG the unit of exchange — send someone
 * a frame and they can open it and be exactly where you were.
 *
 * The chunk is a `tEXt`, which every decoder skips without complaint, so the file stays an ordinary
 * PNG that Preview, Photos and everything else opens normally.
 */

/** PNG text keyword. Latin-1, 1–79 characters, no leading or trailing spaces. */
const KEYWORD = 'dreamnet';

/** Bumped only when older files would be misread rather than merely missing a field. */
const PARAMETERS_VERSION = 1;

export interface EmbeddedParameters {
  version: number;
  savedAt: string;
  config: EngineConfig;
  /** Which model was running, so the settings can say what they were settings *for*. */
  model: { source: string; name: string } | null;
  captureWidth: number;
  captureHeight: number;
}

/**
 * Escapes anything a PNG `tEXt` chunk cannot hold.
 *
 * The chunk is Latin-1, and a model name or style filename can easily contain a character above
 * U+00FF. JSON's own `\uXXXX` escape is the natural way out: `JSON.parse` understands it already,
 * so the reader needs no matching unescape step and cannot get out of step with this one.
 */
function toLatin1Json(value: unknown): string {
  return JSON.stringify(value).replace(/[Ā-￿]/g, (character) =>
    `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`,
  );
}

export function encodeParameters(parameters: EmbeddedParameters): string {
  return toLatin1Json(parameters);
}

/** Stamps the current settings for embedding. */
export function buildParameters(
  config: EngineConfig,
  model: { source: string; name: string } | null,
  captureWidth: number,
  captureHeight: number,
): EmbeddedParameters {
  return {
    version: PARAMETERS_VERSION,
    savedAt: new Date().toISOString(),
    config,
    model,
    captureWidth,
    captureHeight,
  };
}

/**
 * Reads parameters out of a PNG, returning null for anything that is not one of ours.
 *
 * Every failure here is expected rather than exceptional — a JPEG, a PNG from another program, a
 * file from a future version — so none of them throw. A missing or unreadable chunk simply means
 * this image carries no settings.
 */
export function readParameters(bytes: Uint8Array): EmbeddedParameters | null {
  if (!isPng(bytes)) return null;

  const text = readPngText(bytes, KEYWORD);
  if (!text) return null;

  try {
    const parsed = JSON.parse(text) as Partial<EmbeddedParameters>;
    if (typeof parsed.version !== 'number' || !parsed.config) return null;
    return {
      version: parsed.version,
      savedAt: parsed.savedAt ?? '',
      // Merged over the defaults rather than used raw: a file written by an older build is missing
      // whatever has been added since, and a config with holes in it would break the engine.
      config: mergeConfig(parsed.config),
      model: parsed.model ?? null,
      captureWidth: parsed.captureWidth ?? 0,
      captureHeight: parsed.captureHeight ?? 0,
    };
  } catch {
    return null;
  }
}

/** Fills in anything the stored config is missing, one level into the nested groups. */
function mergeConfig(stored: Partial<EngineConfig>): EngineConfig {
  return {
    ...DEFAULT_CONFIG,
    ...stored,
    shallow: { ...DEFAULT_CONFIG.shallow, ...(stored.shallow ?? {}) },
    feedback: { ...DEFAULT_CONFIG.feedback, ...(stored.feedback ?? {}) },
    display: { ...DEFAULT_CONFIG.display, ...(stored.display ?? {}) },
    // Files written before the gain became per-band carry a single number; spreading it across the
    // bands reproduces what that file actually sounded like rather than silently reverting to the
    // defaults.
    audio: (() => {
      const audio = { ...DEFAULT_CONFIG.audio, ...(stored.audio ?? {}) };
      const legacy = (stored.audio as { gain?: number } | undefined)?.gain;
      if (typeof legacy === 'number' && !Array.isArray((stored.audio as { gains?: number[] })?.gains)) {
        audio.gains = DEFAULT_CONFIG.audio.gains.map(() => legacy);
      }
      return audio;
    })(),
    modelControls: stored.modelControls ?? [],
  };
}

export function embedParameters(bytes: Uint8Array, parameters: EmbeddedParameters): Uint8Array {
  return writePngText(bytes, KEYWORD, encodeParameters(parameters));
}

/** Whether a file is worth reading parameters out of at all. */
export function couldCarryParameters(file: File): boolean {
  return file.type === 'image/png' || /\.png$/i.test(file.name);
}

/** A one-line summary of what a set of embedded parameters describes, for the UI to show. */
export function describeParameters(parameters: EmbeddedParameters): string {
  const { config } = parameters;
  const processor =
    config.processor === 'model'
      ? `model${parameters.model ? ` "${parameters.model.name}"` : ''}`
      : config.processor === 'shallow'
        ? `shallow ${config.shallow.bank}`
        : 'passthrough';
  const size = parameters.captureWidth > 0 ? `${parameters.captureWidth}×${parameters.captureHeight}` : `${config.captureSize}px`;
  const feedback = config.feedback.enabled ? `feedback ${config.feedback.previous.toFixed(2)}` : 'no feedback';
  return `${processor} · ${size} · ${feedback}`;
}
