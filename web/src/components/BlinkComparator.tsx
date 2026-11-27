import { useEffect, useMemo, useRef, useState } from 'react';
import type { Candidate, CubeMeta, PreviewCube } from '../types';
import { drawGrayscale, drawMarker } from '../utils/image';
import { NumberField, Toggle } from './ui/Controls';
import { Panel } from './ui/Panel';
import { EmptyState } from './ui/StateViews';

interface BlinkComparatorProps {
  cube: CubeMeta | null;
  preview: PreviewCube | null;
  notice?: string | null;
  selectedCandidate: Candidate | null;
}

export function BlinkComparator({ cube, preview, notice, selectedCandidate }: BlinkComparatorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [abMode, setAbMode] = useState(false);
  const [frameA, setFrameA] = useState(0);
  const [frameB, setFrameB] = useState(1);
  const [truthVisible, setTruthVisible] = useState(true);
  const [loPercentile, setLoPercentile] = useState(2);
  const [hiPercentile, setHiPercentile] = useState(98);
  const [stretch, setStretch] = useState<{ lo: number; hi: number } | null>(null);

  const frameCount = preview?.frames.length ?? cube?.frames ?? 0;
  const shownFrame = Math.min(frame, Math.max(0, frameCount - 1));
  const currentFrame = preview?.frames[shownFrame] ?? null;
  const hasTruth = (cube?.truth.length ?? 0) > 0;

  useEffect(() => {
    setFrame(0);
    setPlaying(false);
    setFrameA(0);
    setFrameB(1);
  }, [cube?.cubeId]);

  useEffect(() => {
    if (!playing || frameCount < 2) return;
    const timer = window.setInterval(() => {
      setFrame((current) => {
        if (abMode) {
          const a = Math.min(frameA, frameCount - 1);
          const b = Math.min(frameB, frameCount - 1);
          return current === a ? b : a;
        }
        return (current + 1) % frameCount;
      });
    }, 1000 / Math.max(0.5, speed));
    return () => window.clearInterval(timer);
  }, [playing, speed, frameCount, abMode, frameA, frameB]);

  const truthForFrame = useMemo(
    () => (truthVisible ? cube?.truth.filter((point) => Math.round(point.t) === shownFrame) ?? [] : []),
    [cube, truthVisible, shownFrame],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentFrame || !preview) return;
    const range = drawGrayscale(
      canvas,
      currentFrame,
      preview.width,
      preview.height,
      Math.min(loPercentile, hiPercentile - 0.1),
      Math.max(hiPercentile, loPercentile + 0.1),
    );
    setStretch(range);
    const context = canvas.getContext('2d');
    if (!context) return;
    for (const point of truthForFrame) {
      drawMarker(context, point.x, point.y, 'rgba(103, 232, 249, 0.95)', point.object_id);
    }
    if (selectedCandidate && !abMode) {
      drawMarker(context, selectedCandidate.x, selectedCandidate.y, 'rgba(251, 191, 36, 0.95)', 'C');
    }
  }, [currentFrame, preview, loPercentile, hiPercentile, truthForFrame, selectedCandidate, abMode, shownFrame]);

  const abFrame = frame === frameA ? frameA : frameB;

  return (
    <Panel
      title="3 · Blink comparator"
      subtitle={cube ? `${cube.frames} frames · ${cube.width}×${cube.height} px` : 'Awaiting a cube'}
      actions={
        cube && (
          <Toggle
            id="truth-overlay"
            label="Truth overlay"
            checked={truthVisible && hasTruth}
            disabled={!hasTruth}
            hint={hasTruth ? `${cube?.truth.length ?? 0} points` : 'simulation only'}
            onChange={setTruthVisible}
          />
        )
      }
    >
      {!preview || !cube ? (
        <EmptyState
          title="No frames to blink"
          detail="Charge a FITS cube or run a simulation to populate the comparator."
        />
      ) : (
        <div className="space-y-3">
          <div className="relative overflow-hidden rounded-lg border border-slate-700/80 bg-black">
            <canvas
              ref={canvasRef}
              className="block h-[280px] w-full object-contain sm:h-[360px]"
              aria-label={`Frame ${abMode ? abFrame : shownFrame} grayscale image`}
              role="img"
            />
            <div className="pointer-events-none absolute left-2 top-2 rounded bg-slate-950/75 px-2 py-1 font-mono text-[10px] text-cyan-200">
              {abMode ? `A/B · frame ${abFrame}` : `frame ${shownFrame}`}
            </div>
            {truthForFrame.length > 0 && (
              <div className="pointer-events-none absolute bottom-2 left-2 rounded bg-slate-950/75 px-2 py-1 font-mono text-[10px] text-cyan-200">
                {truthForFrame.length} truth source{truthForFrame.length === 1 ? '' : 's'}
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              className={playing ? 'btn-primary' : 'btn-ghost'}
              onClick={() => setPlaying((value) => !value)}
              disabled={frameCount < 2}
              aria-label={playing ? 'Pause playback' : 'Play frames'}
            >
              {playing ? '❚❚ Pause' : '▶ Play'}
            </button>
            <div className="min-w-[180px] flex-1">
              <label className="field-label" htmlFor="frame-slider">
                Frame {shownFrame} / {Math.max(0, frameCount - 1)}
              </label>
              <input
                id="frame-slider"
                type="range"
                className="w-full accent-cyan-400"
                min={0}
                max={Math.max(0, frameCount - 1)}
                value={shownFrame}
                disabled={abMode}
                onChange={(event) => setFrame(Number(event.target.value))}
              />
            </div>
            <div className="w-20">
              <label className="field-label" htmlFor="blink-speed">fps</label>
              <input
                id="blink-speed"
                type="number"
                className="field-input"
                min={0.5}
                max={30}
                step={0.5}
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value) || 1)}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-800 bg-obsidian-950/50 p-3">
            <Toggle id="ab-mode" label="A/B blink" checked={abMode} onChange={setAbMode} />
            <span className="font-mono text-[11px] text-slate-400">
              A: {Math.min(frameA, frameCount - 1)} · B: {Math.min(frameB, frameCount - 1)}
            </span>
            <button type="button" className="btn-ghost" disabled={abMode} onClick={() => setFrameA(shownFrame)}>
              Set A = current
            </button>
            <button type="button" className="btn-ghost" disabled={abMode} onClick={() => setFrameB(shownFrame)}>
              Set B = current
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <NumberField id="stretch-lo" label="Stretch low %" value={loPercentile} onChange={setLoPercentile} min={0} max={99} step={0.5} />
            <NumberField id="stretch-hi" label="Stretch high %" value={hiPercentile} onChange={setHiPercentile} min={1} max={100} step={0.5} />
            <div>
              <span className="stat-label">Displayed min</span>
              <p className="stat-value">{stretch ? stretch.lo.toFixed(2) : '—'}</p>
            </div>
            <div>
              <span className="stat-label">Displayed max</span>
              <p className="stat-value">{stretch ? stretch.hi.toFixed(2) : '—'}</p>
            </div>
          </div>

          {notice && <p className="text-[10px] leading-relaxed text-amber-300/80">{notice}</p>}
        </div>
      )}
    </Panel>
  );
}
