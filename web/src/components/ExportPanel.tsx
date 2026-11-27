import { useState } from 'react';
import { errorMessage, getExport } from '../api';
import type { ExportFormat } from '../types';
import { Panel } from './ui/Panel';
import { ErrorBox, Spinner } from './ui/StateViews';

const FORMATS: { format: ExportFormat; label: string; detail: string }[] = [
  { format: 'json', label: 'JSON', detail: 'full records' },
  { format: 'mpc', label: 'MPC', detail: 'observations' },
  { format: 'parquet', label: 'Parquet', detail: 'columnar' },
];

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ExportPanel({ jobId }: { jobId: string | null }) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const downloadCatalog = async (format: ExportFormat) => {
    if (!jobId) return;
    setBusy(format);
    setError(null);
    try {
      const { blob, filename } = await getExport(jobId, format);
      download(blob, filename);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(null);
    }
  };

  return (
    <Panel title="6 · Export" subtitle="Download the detection catalog from the API.">
      {!jobId ? (
        <p className="text-[11px] text-slate-500">A detection job is required before exporting.</p>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            {FORMATS.map(({ format, label, detail }) => (
              <button
                key={format}
                type="button"
                className="btn-ghost flex-col gap-0.5 py-2"
                disabled={busy !== null}
                onClick={() => void downloadCatalog(format)}
              >
                {busy === format ? <Spinner /> : <span className="text-xs font-semibold text-cyan-200">{label}</span>}
                <span className="text-[9px] text-slate-500">{detail}</span>
              </button>
            ))}
          </div>
          {error && <div className="mt-3"><ErrorBox message={error} /></div>}
        </>
      )}
    </Panel>
  );
}
