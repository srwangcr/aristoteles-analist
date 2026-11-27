import type { ReactNode } from 'react';

interface NumberFieldProps {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  hint?: string;
  suffix?: string;
}

export function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  hint,
  suffix,
}: NumberFieldProps) {
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type="number"
          className={`field-input ${suffix ? 'pr-10' : ''}`}
          value={Number.isFinite(value) ? value : ''}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(event) => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) onChange(next);
          }}
        />
        {suffix && (
          <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 font-mono text-[10px] text-slate-500">
            {suffix}
          </span>
        )}
      </div>
      {hint && <p className="mt-1 text-[10px] text-slate-500">{hint}</p>}
    </div>
  );
}

interface SelectFieldProps<T extends string> {
  id: string;
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}

export function SelectField<T extends string>({
  id,
  label,
  value,
  options,
  onChange,
  disabled,
}: SelectFieldProps<T>) {
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className="field-input"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}

export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: SegmentedControlProps<T>) {
  return (
    <fieldset disabled={disabled}>
      <legend className="field-label">{label}</legend>
      <div className="flex gap-1 rounded-md border border-slate-700/80 bg-obsidian-950/70 p-1">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`btn flex-1 border-transparent px-2 py-1 ${
              value === option.value
                ? 'bg-cyan-500/20 text-cyan-200 shadow-sm'
                : 'text-slate-400 hover:bg-slate-800/70 hover:text-slate-200'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

interface ToggleProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  hint?: string;
}

export function Toggle({ id, label, checked, onChange, disabled, hint }: ToggleProps) {
  return (
    <label htmlFor={id} className={`flex items-center gap-2.5 ${disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-4 w-8 shrink-0 rounded-full border transition-colors ${
          checked ? 'border-cyan-400/70 bg-cyan-500/50' : 'border-slate-600 bg-slate-800'
        }`}
      >
        <span
          className={`absolute top-0.5 h-2.5 w-2.5 rounded-full bg-slate-100 transition-all ${
            checked ? 'left-4' : 'left-0.5'
          }`}
        />
      </button>
      <span className="text-xs text-slate-300">
        {label}
        {hint && <span className="ml-1.5 text-[10px] text-slate-500">{hint}</span>}
      </span>
    </label>
  );
}

interface RangeNumberProps {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  suffix?: string;
  hint?: ReactNode;
}

export function RangeNumber({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  suffix,
  hint,
}: RangeNumberProps) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label className="field-label mb-0" htmlFor={id}>
          {label}
        </label>
        <span className="font-mono text-[11px] text-cyan-200">
          {value.toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0)}
          {suffix && <span className="ml-0.5 text-slate-500">{suffix}</span>}
        </span>
      </div>
      <input
        id={id}
        type="range"
        className="w-full accent-cyan-400"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint && <div className="mt-1 text-[10px] text-slate-500">{hint}</div>}
    </div>
  );
}
