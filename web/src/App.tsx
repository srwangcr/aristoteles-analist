import { useCallback, useState } from 'react';
import { API_BASE, errorMessage, getCandidates, startDetection } from './api';
import { BlinkComparator } from './components/BlinkComparator';
import { ChargePanel } from './components/ChargePanel';
import { CandidateTable } from './components/CandidateTable';
import { ExportPanel } from './components/ExportPanel';
import { ParameterPanel } from './components/ParameterPanel';
import { StackInspector } from './components/StackInspector';
import { ErrorBox } from './components/ui/StateViews';
import { useJobWebSocket } from './hooks/useJobWebSocket';
import type {
  Candidate,
  CandidateFilters,
  CubeMeta,
  DetectionParams,
  PreviewCube,
} from './types';

const INITIAL_PARAMS: DetectionParams = {
  backend: 'auto',
  velocityMax: 1.5,
  velocityStep: 0.25,
  detectionSigma: 5,
};

const INITIAL_FILTERS: CandidateFilters = {
  minSnr: 0,
  maxTrackRms: 1,
  maxCosmicFraction: 1,
};

export default function App() {
  const [cube, setCube] = useState<CubeMeta | null>(null);
  const [preview, setPreview] = useState<PreviewCube | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [params, setParams] = useState<DetectionParams>(INITIAL_PARAMS);
  const [jobId, setJobId] = useState<string | null>(null);
  const [filters, setFilters] = useState<CandidateFilters>(INITIAL_FILTERS);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [candidatesLoading, setCandidatesLoading] = useState(false);
  const [candidateError, setCandidateError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const loadCandidates = useCallback(
    async (activeFilters: CandidateFilters) => {
      if (!jobId) return;
      setCandidatesLoading(true);
      setCandidateError(null);
      try {
        const response = await getCandidates(jobId, {
          minSnr: activeFilters.minSnr,
          maxTrackRms: activeFilters.maxTrackRms,
        });
        const filtered = response.candidates.filter(
          (candidate) => candidate.cosmic_fraction <= activeFilters.maxCosmicFraction,
        );
        setCandidates(filtered);
        setSelectedIndex(null);
      } catch (caught) {
        setCandidateError(errorMessage(caught));
        setCandidates([]);
      } finally {
        setCandidatesLoading(false);
      }
    },
    [jobId],
  );

  const job = useJobWebSocket(jobId, {
    onDone: () => {
      void loadCandidates(filters);
    },
    onError: (message) => {
      setCandidateError(message);
    },
  });

  const handleCubeReady = (nextCube: CubeMeta, nextPreview: PreviewCube | null, cubeNotice?: string) => {
    setCube(nextCube);
    setPreview(nextPreview);
    setNotice(cubeNotice ?? null);
    setJobId(null);
    setCandidates([]);
    setCandidateError(null);
    setSelectedIndex(null);
    setRunError(null);
    setFilters(INITIAL_FILTERS);
  };

  const handleRun = async () => {
    if (!cube) {
      setRunError('Charge a cube before starting detection.');
      return;
    }
    setRunError(null);
    setCandidateError(null);
    setCandidates([]);
    setSelectedIndex(null);
    setJobId(null);
    try {
      const response = await startDetection({
        cube_id: cube.cubeId,
        backend: params.backend,
        velocity_max: params.velocityMax,
        velocity_step: params.velocityStep,
        detection_sigma: params.detectionSigma,
      });
      setJobId(response.job_id);
    } catch (caught) {
      setRunError(errorMessage(caught));
    }
  };

  const applyFilters = (next: CandidateFilters) => {
    setFilters(next);
    void loadCandidates(next);
  };

  const selectedCandidate = selectedIndex !== null ? candidates[selectedIndex] ?? null : null;

  return (
    <div className="mx-auto max-w-[1800px] px-3 py-5 sm:px-5 sm:py-7">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.3em] text-cyan-400/80">
            Moving-object detection engine
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 sm:text-3xl">
            Observatory <span className="text-cyan-300">Workbench</span>
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Charge, detect, blink, stack and export — a complete shift-and-stack investigation loop.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 font-mono text-[10px]">
          <span className="badge border-slate-700 bg-slate-800/50 text-slate-300">
            API {API_BASE}
          </span>
          <span className="badge border-slate-700 bg-slate-800/50 text-slate-300">
            cube {cube ? cube.cubeId.slice(0, 12) : '—'}
          </span>
          <span
            className={`badge ${
              job.status === 'done'
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : job.status === 'error'
                  ? 'border-rose-500/40 bg-rose-500/10 text-rose-300'
                  : 'border-slate-700 bg-slate-800/50 text-slate-300'
            }`}
          >
            job {job.status}
          </span>
        </div>
      </header>

      {runError && <div className="mb-4"><ErrorBox message={runError} onDismiss={() => setRunError(null)} /></div>}

      <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="space-y-4">
          <ChargePanel cube={cube} onCubeReady={handleCubeReady} />
          <ParameterPanel
            cubeId={cube?.cubeId ?? null}
            params={params}
            onParamsChange={setParams}
            onRun={() => void handleRun()}
            running={job.status === 'queued' || job.status === 'running'}
            jobId={jobId}
            job={job}
          />
          <ExportPanel jobId={jobId} />
        </aside>

        <div className="min-w-0 space-y-4">
          <div className="grid gap-4 2xl:grid-cols-2">
            <BlinkComparator
              cube={cube}
              preview={preview}
              notice={notice}
              selectedCandidate={selectedCandidate}
            />
            <StackInspector
              jobId={jobId}
              jobStatus={job.status}
              velocityMax={params.velocityMax}
              velocityStep={params.velocityStep}
              selectedCandidate={selectedCandidate}
            />
          </div>
          <CandidateTable
            candidates={candidates}
            loading={candidatesLoading}
            error={candidateError}
            filters={filters}
            onFiltersChange={applyFilters}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
        </div>
      </div>

      <footer className="mt-6 border-t border-slate-800/70 pt-3 font-mono text-[10px] text-slate-600">
        Observatory Workbench · React 18 + TypeScript · configure VITE_API_URL to point at the engine.
      </footer>
    </div>
  );
}
