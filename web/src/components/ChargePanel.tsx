import { useRef, useState, type DragEvent } from 'react';
import { errorMessage, simulateCube, uploadFits } from '../api';
import type { CubeMeta, PreviewCube, SimulateRequest } from '../types';
import { parseFitsPreview, synthesizePreview } from '../utils/fits';
import { NumberField } from './ui/Controls';
import { Panel } from './ui/Panel';
import { ErrorBox, LoadingState, Spinner } from './ui/StateViews';

type SourceTab = 'fits' | 'simulate';

interface ChargePanelProps {
  cube: CubeMeta | null;
  onCubeReady: (cube: CubeMeta, preview: PreviewCube | null, notice?: string) => void;
}

export function ChargePanel({ cube, onCubeReady }: ChargePanelProps) {
  const [tab, setTab] = useState<SourceTab>('simulate');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [snrFrame, setSnrFrame] = useState(2);
  const [seed, setSeed] = useState(1234);
  const [nFrames, setNFrames] = useState(12);
  const [height, setHeight] = useState(96);
  const [width, setWidth] = useState(96);
  const [vx, setVx] = useState(0.37);
  const [vy, setVy] = useState(-0.21);
  const fileInput = useRef<HTMLInputElement>(null);

  const submitSimulation = async () => {
    setBusy(true);
    setError(null);
    try {
      if (snrFrame <= 0 || nFrames < 1 || height < 8 || width < 8) {
        throw new Error('SNR must be positive, frames ≥ 1 and shape axes ≥ 8.');
      }
      const payload: SimulateRequest = {
        snr_frame: snrFrame,
        seed,
        n_frames: nFrames,
        shape: [height, width],
        vx,
        vy,
      };
      const response = await simulateCube(payload);
      const preview = synthesizePreview(width, height, nFrames, snrFrame, seed, response.truth);
      onCubeReady(
        {
          cubeId: response.cube_id,
          frames: nFrames,
          height,
          width,
          truth: response.truth,
          origin: 'simulation',
          sourceName: `sim seed ${seed}`,
        },
        preview,
        'Blink preview is synthesised locally from the returned truth table; detection uses the server cube.',
      );
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const submitFits = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const response = await uploadFits(file);
      let preview: PreviewCube | null = null;
      let notice: string | undefined;
      try {
        preview = parseFitsPreview(await file.arrayBuffer());
      } catch (parseError) {
        notice = `Uploaded successfully; local preview unavailable (${errorMessage(parseError)}).`;
      }
      onCubeReady(
        {
          cubeId: response.cube_id,
          frames: response.frames,
          height: response.height,
          width: response.width,
          truth: [],
          origin: 'fits',
          sourceName: file.name,
        },
        preview,
        notice,
      );
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void submitFits(file);
  };

  return (
    <Panel
      title="1 · Charge a cube"
      subtitle="Upload a FITS cube or ask the engine to simulate a labelled one."
    >
      <div className="mb-3 flex gap-1 rounded-md border border-slate-700/80 bg-obsidian-950/70 p-1" role="tablist" aria-label="Cube source">
        {(
          [
            ['fits', 'Load FITS'],
            ['simulate', 'Simulate'],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={tab === value}
            disabled={busy}
            onClick={() => {
              setTab(value);
              setError(null);
            }}
            className={`btn flex-1 border-transparent ${tab === value ? 'bg-cyan-500/20 text-cyan-200' : 'text-slate-400 hover:text-slate-200'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {error && <div className="mb-3"><ErrorBox message={error} onDismiss={() => setError(null)} /></div>}
      {busy && <div className="mb-3"><LoadingState label={tab === 'fits' ? 'Uploading FITS bytes…' : 'Simulating cube…'} /></div>}

      {tab === 'fits' ? (
        <div role="tabpanel" aria-label="Load FITS">
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`flex flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-7 text-center transition-colors ${
              dragging ? 'border-cyan-400 bg-cyan-500/10' : 'border-slate-700 bg-obsidian-950/40'
            }`}
          >
            <span className="font-mono text-2xl text-slate-500" aria-hidden="true">⇩</span>
            <p className="text-xs text-slate-300">Drop a <code className="font-mono text-cyan-200">.fits</code> cube here</p>
            <p className="text-[10px] text-slate-500">Raw bytes are POSTed to <code className="font-mono">/api/fits</code></p>
            <input
              ref={fileInput}
              type="file"
              accept=".fits,.fit,.fts,application/octet-stream"
              className="sr-only"
              aria-label="Choose FITS cube file"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void submitFits(file);
                event.target.value = '';
              }}
            />
            <button type="button" className="btn-ghost" disabled={busy} onClick={() => fileInput.current?.click()}>
              {busy ? <Spinner /> : null} Choose file
            </button>
          </div>
        </div>
      ) : (
        <div role="tabpanel" aria-label="Simulate" className="grid grid-cols-2 gap-3">
          <NumberField id="sim-snr" label="SNR / frame" value={snrFrame} onChange={setSnrFrame} min={0.1} max={100} step={0.1} />
          <NumberField id="sim-seed" label="Seed" value={seed} onChange={setSeed} min={0} step={1} />
          <NumberField id="sim-frames" label="Frames" value={nFrames} onChange={setNFrames} min={1} max={500} step={1} suffix="N" />
          <NumberField id="sim-vx" label="vₓ (px/frame)" value={vx} onChange={setVx} step={0.01} />
          <NumberField id="sim-height" label="Shape · height" value={height} onChange={setHeight} min={8} max={2048} step={1} suffix="px" />
          <NumberField id="sim-width" label="Shape · width" value={width} onChange={setWidth} min={8} max={2048} step={1} suffix="px" />
          <div className="col-span-2">
            <NumberField id="sim-vy" label="vᵧ (px/frame)" value={vy} onChange={setVy} step={0.01} />
          </div>
          <button type="button" className="btn-primary col-span-2 mt-1" disabled={busy} onClick={() => void submitSimulation()}>
            {busy ? <Spinner /> : null} Generate &amp; charge cube
          </button>
        </div>
      )}

      {cube && !busy && (
        <dl className="mt-4 space-y-1.5 rounded-lg border border-slate-800 bg-obsidian-950/50 p-3">
          <div className="flex justify-between gap-3">
            <dt className="stat-label">Cube ID</dt>
            <dd className="stat-value truncate" title={cube.cubeId}>{cube.cubeId}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="stat-label">Geometry</dt>
            <dd className="stat-value">{cube.frames} × {cube.height} × {cube.width}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="stat-label">Source</dt>
            <dd className="stat-value truncate" title={cube.sourceName}>{cube.origin === 'fits' ? 'FITS' : 'Simulated'}{cube.sourceName ? ` · ${cube.sourceName}` : ''}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="stat-label">Truth points</dt>
            <dd className="stat-value">{cube.truth.length}</dd>
          </div>
        </dl>
      )}
    </Panel>
  );
}
