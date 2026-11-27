import type { Backend, DetectionParams, JobViewState } from '../types';
import { NumberField, SegmentedControl } from './ui/Controls';
import { Panel } from './ui/Panel';
import { Spinner, StatusDot } from './ui/StateViews';

interface ParameterPanelProps {
  cubeId: string | null;
  params: DetectionParams;
  onParamsChange: (params: DetectionParams) => void;
  onRun: () => void;
  running: boolean;
  jobId: string | null;
  job: JobViewState;
}

const BACKENDS: { value: Backend; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'cpu', label: 'CPU' },
  { value: 'gpu', label: 'GPU' },
];

export function ParameterPanel({
  cubeId,
  params,
  onParamsChange,
  onRun,
  running,
  jobId,
  job,
}: ParameterPanelProps) {
  const patch = (change: Partial<DetectionParams>) => onParamsChange({ ...params, ...change });
  const percent = Math.round(job.progress * (job.progress <= 1 ? 100 : 1));

  return (
    <Panel title="2 · Detection" subtitle="Tune the velocity sweep and launch a job.">
      <div className="space-y-3">
        <SegmentedControl
          label="Backend"
          value={params.backend}
          options={BACKENDS}
          onChange={(backend) => patch({ backend })}
          disabled={running}
        />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <NumberField
            id="velocity-max"
            label="v max"
            value={params.velocityMax}
            onChange={(velocityMax) => patch({ velocityMax })}
            min={0}
            max={20}
            step={0.05}
            suffix="px/f"
            disabled={running}
          />
          <NumberField
            id="velocity-step"
            label="Δv step"
            value={params.velocityStep}
            onChange={(velocityStep) => patch({ velocityStep })}
            min={0.01}
            max={5}
            step={0.01}
            suffix="px/f"
            disabled={running}
          />
        </div>
        <NumberField
          id="detection-sigma"
          label="Detection σ"
          value={params.detectionSigma}
          onChange={(detectionSigma) => patch({ detectionSigma })}
          min={0.5}
          max={50}
          step={0.1}
          hint="Threshold applied to shift-stack peaks."
          disabled={running}
        />
        <button
          type="button"
          className="btn-primary w-full py-2"
          disabled={!cubeId || running || params.velocityStep <= 0 || params.velocityMax <= 0}
          onClick={onRun}
        >
          {running ? <Spinner /> : null}
          {running ? 'Detection running…' : 'Run detection'}
        </button>
        {!cubeId && <p className="text-[10px] text-slate-500">Charge a cube first to enable detection.</p>}
      </div>

      <div className="mt-4 border-t border-slate-800 pt-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="stat-label">Job progress</span>
          <div className="flex items-center gap-2">
            <StatusDot
              tone={job.status === 'error' ? 'error' : job.status === 'done' ? 'success' : job.status === 'idle' ? 'idle' : 'active'}
              label={job.socketConnected ? 'WS live' : job.status}
            />
            <span className="font-mono text-[11px] text-cyan-200">{percent}%</span>
          </div>
        </div>
        <div
          className="h-2 overflow-hidden rounded-full bg-slate-800"
          role="progressbar"
          aria-label="Detection progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
        >
          <div
            className={`h-full rounded-full transition-[width] duration-300 ${job.status === 'error' ? 'bg-rose-400' : 'bg-gradient-to-r from-cyan-500 to-teal-300'}`}
            style={{ width: `${percent}%` }}
          />
        </div>
        <p className="mt-2 min-h-[1.1rem] text-[11px] text-slate-400" aria-live="polite">
          {job.message || (jobId ? `Job ${jobId}` : 'No job started yet.')}
        </p>
        <div className="mt-2 flex flex-wrap justify-between gap-2 font-mono text-[10px] text-slate-500">
          <span>{jobId ? `job: ${jobId}` : 'job: —'}</span>
          {job.elapsedS !== undefined && <span>elapsed: {job.elapsedS.toFixed(2)} s</span>}
        </div>
      </div>
    </Panel>
  );
}
