import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import { errorMessage, getStack } from '../api';
import type { Candidate, JobStatus, StackMap } from '../types';
import { canvasCoordinates, drawHeatmap, magmaGradientCss, stats } from '../utils/image';
import { NumberField, RangeNumber } from './ui/Controls';
import { Panel } from './ui/Panel';
import { EmptyState, ErrorBox, Spinner } from './ui/StateViews';

interface StackInspectorProps {
  jobId: string | null;
  jobStatus: JobStatus;
  velocityMax: number;
  velocityStep: number;
  selectedCandidate: Candidate | null;
}

export function StackInspector({
  jobId,
  jobStatus,
  velocityMax,
  velocityStep,
  selectedCandidate,
}: StackInspectorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef(0);
  const loadedForJob = useRef<string | null>(null);
  const [vx, setVx] = useState(0);
  const [vy, setVy] = useState(0);
  const [stack, setStack] = useState<StackMap | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clicked, setClicked] = useState<{ x: number; y: number; value: number } | null>(null);

  const load = useCallback(
    async (nextVx: number, nextVy: number) => {
      if (!jobId) return;
      const requestId = ++requestRef.current;
      setLoading(true);
      setError(null);
      setClicked(null);
      try {
        const response = await getStack(jobId, nextVx, nextVy);
        if (requestId !== requestRef.current) return;
        const summary = stats(response.data);
        setStack({
          ...response,
          vx: nextVx,
          vy: nextVy,
          min: summary.min,
          max: summary.max,
          peak: summary.max,
        });
      } catch (caught) {
        if (requestId === requestRef.current) setError(errorMessage(caught));
      } finally {
        if (requestId === requestRef.current) setLoading(false);
      }
    },
    [jobId],
  );

  useEffect(() => {
    loadedForJob.current = null;
    setStack(null);
    setError(null);
    setClicked(null);
    setVx(0);
    setVy(0);
  }, [jobId]);

  useEffect(() => {
    if (!jobId || jobStatus !== 'done' || loadedForJob.current === jobId) return;
    loadedForJob.current = jobId;
    void load(0, 0);
  }, [jobId, jobStatus, load]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !stack) return;
    drawHeatmap(canvas, stack.data, stack.width, stack.height, stack.min, stack.max);
    const context = canvas.getContext('2d');
    if (!context) return;
    if (clicked) {
      const radius = Math.max(3, canvas.width / 120);
      context.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      context.lineWidth = Math.max(1, canvas.width / 400);
      context.beginPath();
      context.arc(clicked.x + 0.5, clicked.y + 0.5, radius, 0, Math.PI * 2);
      context.stroke();
    }
  }, [stack, clicked]);

  const onCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!stack) return;
    const point = canvasCoordinates(event, event.currentTarget);
    if (!point) return;
    setClicked({ ...point, value: stack.data[point.y * stack.width + point.x] });
  };

  const range = Math.max(0.1, velocityMax);
  const step = Math.max(0.01, velocityStep);

  return (
    <Panel
      title="4 · Stack map inspector"
      subtitle="Shift-and-stack response at a trial velocity (vx, vy)."
      actions={
        selectedCandidate && (
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              setVx(selectedCandidate.vx);
              setVy(selectedCandidate.vy);
              void load(selectedCandidate.vx, selectedCandidate.vy);
            }}
          >
            Use candidate velocity
          </button>
        )
      }
    >
      {!jobId ? (
        <EmptyState title="No detection job" detail="Run a detection to inspect its shift-stack velocity maps." />
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="space-y-2">
              <RangeNumber id="stack-vx" label="vₓ" value={vx} onChange={setVx} min={-range} max={range} step={step} suffix="px/f" />
              <NumberField id="stack-vx-number" label="Exact vₓ" value={vx} onChange={setVx} min={-20} max={20} step={step} />
            </div>
            <div className="space-y-2">
              <RangeNumber id="stack-vy" label="vᵧ" value={vy} onChange={setVy} min={-range} max={range} step={step} suffix="px/f" />
              <NumberField id="stack-vy-number" label="Exact vᵧ" value={vy} onChange={setVy} min={-20} max={20} step={step} />
            </div>
          </div>
          <button type="button" className="btn-primary w-full" disabled={loading} onClick={() => void load(vx, vy)}>
            {loading ? <Spinner /> : null} Load stack at ({vx.toFixed(2)}, {vy.toFixed(2)})
          </button>

          {error && <ErrorBox message={error} />}

          {loading && !stack ? (
            <div className="flex h-[240px] items-center justify-center gap-2 text-[11px] text-slate-400" role="status">
              <Spinner /> Fetching float32 stack…
            </div>
          ) : !stack ? (
            <EmptyState title="No stack loaded" detail="Choose a velocity hypothesis and load the corresponding map." />
          ) : (
            <div className="flex flex-wrap gap-3 sm:flex-nowrap">
              <div className="min-w-0 flex-1">
                <canvas
                  ref={canvasRef}
                  onClick={onCanvasClick}
                  className="block h-[260px] w-full cursor-crosshair rounded-md border border-slate-700 bg-black object-contain"
                  role="img"
                  aria-label={`Shift-stack heatmap at vx ${stack.vx}, vy ${stack.vy}`}
                />
                <p className="mt-1 text-[10px] text-slate-500">Click a pixel to inspect its position and value.</p>
              </div>
              <div className="flex w-full shrink-0 flex-col items-center gap-1 sm:w-10" aria-hidden="true">
                <span className="font-mono text-[9px] text-slate-400">{stack.max.toFixed(2)}</span>
                <div className="h-[220px] w-3 rounded-sm border border-slate-700" style={{ background: magmaGradientCss() }} />
                <span className="font-mono text-[9px] text-slate-400">{stack.min.toFixed(2)}</span>
                <span className="text-[9px] uppercase tracking-wider text-slate-500">SNR</span>
              </div>
            </div>
          )}

          {stack && (
            <div className="grid grid-cols-2 gap-2 rounded-lg border border-slate-800 bg-obsidian-950/50 p-3 sm:grid-cols-4">
              <div><p className="stat-label">Peak SNR</p><p className="stat-value text-cyan-200">{stack.peak.toFixed(3)}</p></div>
              <div><p className="stat-label">Shape</p><p className="stat-value">{stack.height}×{stack.width}</p></div>
              <div><p className="stat-label">Mean</p><p className="stat-value">{stats(stack.data).mean.toFixed(3)}</p></div>
              <div>
                <p className="stat-label">Clicked pixel</p>
                <p className="stat-value">{clicked ? `(${clicked.x}, ${clicked.y}) ${clicked.value.toFixed(2)}` : '—'}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
