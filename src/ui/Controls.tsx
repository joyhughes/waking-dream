import { useState, type ReactNode } from 'react';

/**
 * The control-panel primitives.
 *
 * Every slider shows its value, because the point of this app is reading a number off a change you
 * just made — "that got better" is not usable without "at 0.42".
 */

export function Section({ title, hint, children, defaultOpen = true, lazy = false }: {
  title: string;
  hint?: string;
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
        <span className="section-title">{title}</span>
        {hint ? <span className="section-hint">{hint}</span> : null}
      </summary>
      <div className="section-body">{!lazy || opened ? children : null}</div>
    </details>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, title }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  title?: string;
}) {
  return (
    <label className="control" title={title}>
      <span className="control-label">
        {label}
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

export function Choice<T extends string>({ label, value, options, onChange }: {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (value: T) => void;
}) {
  return (
    <label className="control">
      <span className="control-label">{label}</span>
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
