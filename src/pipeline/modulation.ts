import type { EngineConfig } from './engine';

/**
 * Routing sound to any parameter, not just to a model's own controls.
 *
 * A model control was the obvious thing to drive from audio because the conditioning is already
 * recomputed every frame. But most of what makes a look is elsewhere — how much of the previous
 * frame comes back around, how fast the feedback zooms, how far the colour is held to the source —
 * and all of those are read fresh each frame too, so there is nothing stopping a band driving them
 * as well.
 *
 * What is *not* here matters as much as what is. Anything that reallocates or recompiles is left
 * out: capture size resizes every pooled buffer, and the shallow bank's filter count and kernel
 * size rebuild the weight texture. Modulating those sixty times a second would spend the whole
 * frame budget on allocation, so they stay manual.
 */

export interface Modulation {
  /** Index into `FREQUENCY_BANDS`. */
  band: number;
  /**
   * How far, and in which direction, the band pushes the parameter.
   *
   * The slider stays the resting value and sound pushes away from it: at depth 1 a full band reaches
   * the parameter's maximum, at −1 it reaches the minimum, and at silence the parameter sits exactly
   * where the slider was left. That way routing a band never takes the control away from you — it
   * decides which direction the sound travels, and the slider still decides where it starts.
   */
  depth: number;
}

export type ModulationMap = Record<string, Modulation>;

export interface ModulationTarget {
  id: string;
  label: string;
  min: number;
  max: number;
  read(config: EngineConfig): number;
  write(config: EngineConfig, value: number): EngineConfig;
}

export const MODULATION_TARGETS: ModulationTarget[] = [
  {
    id: 'colorPreservation',
    label: 'Colour preservation',
    min: 0,
    max: 1,
    read: (config) => config.colorPreservation,
    write: (config, value) => ({ ...config, colorPreservation: value }),
  },
  {
    id: 'feedback.previous',
    label: 'Previous output',
    min: 0,
    max: 1,
    read: (config) => config.feedback.previous,
    write: (config, value) => ({ ...config, feedback: { ...config.feedback, previous: value } }),
  },
  {
    id: 'feedback.zoom',
    label: 'Zoom per frame',
    min: 0.97,
    max: 1.05,
    read: (config) => config.feedback.zoom,
    write: (config, value) => ({ ...config, feedback: { ...config.feedback, zoom: value } }),
  },
  {
    id: 'feedback.rotate',
    label: 'Rotate per frame',
    min: -1,
    max: 1,
    read: (config) => config.feedback.rotate,
    write: (config, value) => ({ ...config, feedback: { ...config.feedback, rotate: value } }),
  },
  {
    id: 'feedback.fade',
    label: 'Fade',
    min: 0.9,
    max: 1,
    read: (config) => config.feedback.fade,
    write: (config, value) => ({ ...config, feedback: { ...config.feedback, fade: value } }),
  },
  {
    id: 'display.mix',
    label: 'Effect amount',
    min: 0,
    max: 1,
    read: (config) => config.display.mix,
    write: (config, value) => ({ ...config, display: { ...config.display, mix: value } }),
  },
  {
    id: 'display.gain',
    label: 'Gain',
    min: 0.4,
    max: 2,
    read: (config) => config.display.gain,
    write: (config, value) => ({ ...config, display: { ...config.display, gain: value } }),
  },
  {
    id: 'display.saturation',
    label: 'Saturation',
    min: 0,
    max: 2,
    read: (config) => config.display.saturation,
    write: (config, value) => ({ ...config, display: { ...config.display, saturation: value } }),
  },
  {
    id: 'shallow.stepSize',
    label: 'Step size',
    min: 0,
    max: 0.4,
    read: (config) => config.shallow.stepSize,
    write: (config, value) => ({ ...config, shallow: { ...config.shallow, stepSize: value } }),
  },
  {
    id: 'shallow.octaveScale',
    label: 'Octave scale',
    min: 1.2,
    max: 2.5,
    read: (config) => config.shallow.octaveScale,
    write: (config, value) => ({ ...config, shallow: { ...config.shallow, octaveScale: value } }),
  },
];

const BY_ID = new Map(MODULATION_TARGETS.map((target) => [target.id, target]));

export function modulationTarget(id: string): ModulationTarget | undefined {
  return BY_ID.get(id);
}

/**
 * Applies every routing to a config, returning what this frame should actually run with.
 *
 * Returns the config unchanged when nothing is routed, so the common case allocates nothing — this
 * runs once per frame and the great majority of frames have no routing at all.
 */
export function applyModulations(
  config: EngineConfig,
  levels: ArrayLike<number>,
  modulations: ModulationMap,
): EngineConfig {
  let result = config;

  for (const [id, modulation] of Object.entries(modulations)) {
    if (!modulation || modulation.depth === 0) continue;

    const target = BY_ID.get(id);
    if (!target) continue;

    const level = Math.min(1, Math.max(0, levels[modulation.band] ?? 0));
    if (level === 0) continue;

    const base = target.read(result);
    // The reachable span depends on which way the depth points, so a slider near one end still has
    // its full travel available in the other direction rather than a squashed version of it.
    const span = modulation.depth >= 0 ? target.max - base : base - target.min;
    const value = base + modulation.depth * level * span;

    result = target.write(result, Math.min(target.max, Math.max(target.min, value)));
  }

  return result;
}
