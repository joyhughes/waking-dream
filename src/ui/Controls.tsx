import type { ReactNode } from 'react';

/**
 * The control-panel primitives.
 *
 * Every slider shows its value, because the point of this app is reading a number off a change you
 * just made — "that got better" is not usable without "at 0.42".
 */

export function Section({ title, hint, children, defaultOpen = true }: {
  title: string;
  hint?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="section" open={defaultOpen}>
      <summary>
        <span className="section-title">{title}</span>
        {hint ? <span className="section-hint">{hint}</span> : null}
      </summary>
      <div className="section-body">{children}</div>
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

export function FileButton({ label, accept, onFile }: {
  label: string;
  accept: string;
  onFile: (file: File) => void;
}) {
  return (
    <label className="button file-button">
      {label}
      <input
        type="file"
        accept={accept}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          // Cleared so picking the same file twice in a row still fires a change event.
          event.target.value = '';
        }}
      />
    </label>
  );
}
