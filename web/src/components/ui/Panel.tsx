import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
  bodyClassName = 'p-4',
}: PanelProps) {
  return (
    <section className={`panel ${className}`} aria-label={title}>
      <header className="panel-header">
        <div>
          <h2 className="panel-title">{title}</h2>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </header>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}
