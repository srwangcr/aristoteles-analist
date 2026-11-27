import type { ReactNode } from 'react';

export function Spinner({ className = '' }: { className?: string }) {
  return (
    <span
      className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-500 border-t-cyan-300 ${className}`}
      aria-hidden="true"
    />
  );
}

export function ErrorBox({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-200"
    >
      <span aria-hidden="true" className="mt-px font-bold">
        !
      </span>
      <span className="flex-1 break-words">{message}</span>
      {onDismiss && (
        <button type="button" className="text-rose-300 hover:text-rose-100" onClick={onDismiss} aria-label="Dismiss error">
          ×
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: ReactNode }) {
  return (
    <div className="flex min-h-[180px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-slate-700/70 bg-obsidian-950/40 px-5 py-8 text-center">
      <div className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-700 bg-slate-800/50 text-slate-400" aria-hidden="true">
        ◎
      </div>
      <p className="text-xs font-medium text-slate-300">{title}</p>
      {detail && <p className="max-w-sm text-[11px] leading-relaxed text-slate-500">{detail}</p>}
    </div>
  );
}

export function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex min-h-[180px] flex-col items-center justify-center gap-3 text-slate-400" role="status">
      <Spinner className="h-5 w-5" />
      <span className="text-[11px]">{label}</span>
    </div>
  );
}

export function StatusDot({ tone, label }: { tone: 'idle' | 'active' | 'success' | 'error'; label: string }) {
  const color = {
    idle: 'bg-slate-500',
    active: 'bg-amber-400',
    success: 'bg-emerald-400',
    error: 'bg-rose-400',
  }[tone];
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-400">
      <span className={`h-1.5 w-1.5 rounded-full ${color}`} aria-hidden="true" />
      {label}
    </span>
  );
}
