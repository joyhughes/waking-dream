import { useState, type ReactNode } from 'react';
import { FREQUENCY_BANDS } from '../pipeline/audio';
import type { Modulation, ModulationMap } from '../pipeline/modulation';

/**
 * The control-panel primitives.
 *
 * Every slider shows its value, because the point of this app is reading a number off a change you
 * just made — "that got better" is not usable without "at 0.42".
 */

/**
 * An explanation that stays out of the way until asked for.
 *
 * These used to sit under their controls as paragraphs, which made the panel long enough that
 * reaching a control meant scrolling past the reasons for the ones above it — and the reasons are
 * worth reading once, not on every pass. Opens on hover and on focus, so it is reachable by tap and
 * by keyboard rather than by pointer alone.
 */
export function Hint({ text }: { text: string }) {
  return (
    <span className="hint" tabIndex={0} role="note" aria-label={text}>
      <span aria-hidden="true">i</span>
      <span className="hint-bubble">{text}</span>
    </span>
  );
}

export function Section({ title, hint, info, children, defaultOpen = true, lazy = false }: {
  title: string;
  hint?: string;
  /** Explanation of what the whole section is for, shown on the ⓘ next to its title. */
  info?: string;
  children: ReactNode;
  defaultOpen?: boolean;
  /**
   * Hold the children back until the section is first opened.
   *
   * A `<details>` renders its contents whether or not it is open, so without this the training
   * panel would mount — and start fetching its several megabytes of TensorFlow — on page load for
   * everyone, including the majority who never open it.
   */
  lazy?: boolean;
}) {
  const [opened, setOpened] = useState(defaultOpen);

  return (
    <details className="section" open={defaultOpen} onToggle={(event) => {
      if ((event.currentTarget as HTMLDetailsElement).open) setOpened(true);
    }}>
      <summary>
        <span className="section-title">
          {title}
          {info ? <Hint text={info} /> : null}
        </span>
        {hint ? <span className="section-hint">{hint}</span> : null}
      </summary>
      <div className="section-body">{!lazy || opened ? children : null}</div>
    </details>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, title, info }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  title?: string;
  /** Longer explanation, on a ⓘ beside the label. `title` stays for one-line hover text. */
  info?: string;
}) {
  return (
    <label className="control" title={title}>
      <span className="control-label">
        <span>
          {label}
          {info ? <Hint text={info} /> : null}
        </span>
        <span className="control-value">{format ? format(value) : value.toFixed(stepDecimals(step))}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function stepDecimals(step: number): number {
  if (Number.isInteger(step)) return 0;
  return Math.min(4, String(step).split('.')[1]?.length ?? 2);
}

export function Toggle({ label, checked, onChange, title }: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  title?: string;
}) {
  return (
    <label className="control control-inline" title={title}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function Choice<T extends string>({ label, value, options, onChange, info }: {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (value: T) => void;
  info?: string;
}) {
  return (
    <label className="control">
      <span className="control-label">
        <span>
          {label}
          {info ? <Hint text={info} /> : null}
        </span>
      </span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ButtonRow({ children }: { children: ReactNode }) {
  return <div className="button-row">{children}</div>;
}

export function FileButton({ label, accept, onFile, onFiles, multiple }: {
  label: string;
  accept: string;
  /** Called with the first file picked. Use `onFiles` when a selection of several is meaningful. */
  onFile?: (file: File) => void;
  onFiles?: (files: File[]) => void;
  multiple?: boolean;
}) {
  return (
    <label className="button file-button">
      {label}
      <input
        type="file"
        accept={accept}
        multiple={multiple ?? Boolean(onFiles)}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length > 0) {
            onFiles?.(files);
            onFile?.(files[0]);
          }
          // Cleared so picking the same file twice in a row still fires a change event.
          event.target.value = '';
        }}
      />
    </label>
  );
}


/**
 * A slider that can hand itself over to a frequency band.
 *
 * The routing sits under the slider rather than in a separate matrix panel, because the question
 * "what is driving this control" is one you ask while looking at the control. The slider keeps
 * meaning what it meant — it is the resting value, and the band pushes away from it toward whichever
 * end the depth points at — so routing a parameter never takes it out of your hands.
 */
export function ModulatedSlider({
  targetId,
  modulations,
  onModulationChange,
  audioEnabled,
  ...slider
}: Parameters<typeof Slider>[0] & {
  targetId: string;
  modulations: ModulationMap;
  onModulationChange: (id: string, modulation: Modulation | null) => void;
  audioEnabled: boolean;
}) {
  const routing = modulations[targetId];

  return (
    <div className="modulated">
      <Slider {...slider} />

      {routing ? (
        <div className="routing">
          <select
            value={routing.band}
            onChange={(event) => onModulationChange(targetId, { ...routing, band: Number(event.target.value) })}
          >
            {FREQUENCY_BANDS.map((band, index) => (
              <option key={band.name} value={index}>{band.label}</option>
            ))}
          </select>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.05}
            value={routing.depth}
            onChange={(event) => onModulationChange(targetId, { ...routing, depth: Number(event.target.value) })}
            title="How far, and which way, the band pushes this. Negative drives it toward the low end instead."
          />
          <span className="routing-depth">{routing.depth > 0 ? '+' : ''}{routing.depth.toFixed(2)}</span>
          <button
            className="routing-clear"
            onClick={() => onModulationChange(targetId, null)}
            title="Stop driving this from sound"
          >
            ×
          </button>
        </div>
      ) : (
        <button
          className="routing-add"
          onClick={() => onModulationChange(targetId, { band: 0, depth: 1 })}
          title={audioEnabled ? 'Drive this from a frequency band' : 'Drive this from a frequency band — turn Sound on to hear it'}
        >
          ♪ route to a band
        </button>
      )}
    </div>
  );
}
