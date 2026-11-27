import { useEffect, useMemo, useState } from 'react';
import type { Candidate, CandidateFilters } from '../types';
import { NumberField } from './ui/Controls';
import { Panel } from './ui/Panel';
import { EmptyState, ErrorBox, Spinner } from './ui/StateViews';

type SortKey = keyof Candidate;

interface Column {
  key: SortKey;
  label: string;
  title: string;
  digits: number;
}

const COLUMNS: Column[] = [
  { key: 'x', label: 'x', title: 'Pixel x', digits: 2 },
  { key: 'y', label: 'y', title: 'Pixel y', digits: 2 },
  { key: 'vx', label: 'vₓ', title: 'Velocity x (px/frame)', digits: 3 },
  { key: 'vy', label: 'vᵧ', title: 'Velocity y (px/frame)', digits: 3 },
  { key: 'snr', label: 'SNR', title: 'Stacked signal-to-noise ratio', digits: 2 },
  { key: 'stacked_flux', label: 'Flux', title: 'Stacked flux', digits: 1 },
  { key: 'peak', label: 'Peak', title: 'Peak pixel value', digits: 2 },
  { key: 'track_rms', label: 'RMS', title: 'RANSAC track RMS residual', digits: 3 },
  { key: 'photometric_cv', label: 'CV', title: 'Photometric coefficient of variation', digits: 3 },
  { key: 'cosmic_fraction', label: 'Cosmic %', title: 'Fraction of samples affected by cosmic rays', digits: 3 },
];

const DEFAULT_FILTERS: CandidateFilters = { minSnr: 0, maxTrackRms: 1, maxCosmicFraction: 1 };

interface CandidateTableProps {
  candidates: Candidate[];
  loading: boolean;
  error: string | null;
  filters: CandidateFilters;
  onFiltersChange: (filters: CandidateFilters) => void;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export function CandidateTable({
  candidates,
  loading,
  error,
  filters,
  onFiltersChange,
  selectedIndex,
  onSelect,
}: CandidateTableProps) {
  const [draft, setDraft] = useState<CandidateFilters>(filters);
  const [sortKey, setSortKey] = useState<SortKey>('snr');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => setDraft(filters), [filters]);

  const sorted = useMemo(() => {
    const copy = candidates.map((candidate, index) => ({ candidate, index }));
    copy.sort((a, b) => {
      const difference = a.candidate[sortKey] - b.candidate[sortKey];
      return sortDirection === 'asc' ? difference : -difference;
    });
    return copy;
  }, [candidates, sortKey, sortDirection]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDirection((direction) => (direction === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  return (
    <Panel
      title="5 · Candidate catalog"
      subtitle="RANSAC track and photometric diagnostics · click a row to mark it on the frame."
      actions={
        <span className="badge border-slate-700 bg-slate-800/60 text-slate-300">
          {candidates.length} candidate{candidates.length === 1 ? '' : 's'}
        </span>
      }
      bodyClassName="p-0"
    >
      <div className="flex flex-wrap items-end gap-3 border-b border-slate-800 px-4 py-3">
        <div className="w-24">
          <NumberField id="filter-snr" label="Min SNR" value={draft.minSnr} onChange={(minSnr) => setDraft({ ...draft, minSnr })} min={0} step={0.1} />
        </div>
        <div className="w-28">
          <NumberField id="filter-rms" label="Max RMS" value={draft.maxTrackRms} onChange={(maxTrackRms) => setDraft({ ...draft, maxTrackRms })} min={0} step={0.01} />
        </div>
        <div className="w-32">
          <NumberField id="filter-cosmic" label="Max cosmic" value={draft.maxCosmicFraction} onChange={(maxCosmicFraction) => setDraft({ ...draft, maxCosmicFraction })} min={0} max={1} step={0.01} />
        </div>
        <button type="button" className="btn-primary" onClick={() => onFiltersChange(draft)} disabled={loading}>
          Apply filters
        </button>
        <button type="button" className="btn-ghost" onClick={() => { setDraft(DEFAULT_FILTERS); onFiltersChange(DEFAULT_FILTERS); }} disabled={loading}>
          Reset
        </button>
      </div>

      {error && <div className="p-4"><ErrorBox message={error} /></div>}

      {loading ? (
        <div className="flex h-40 items-center justify-center gap-2 text-[11px] text-slate-400" role="status">
          <Spinner /> Loading candidates…
        </div>
      ) : candidates.length === 0 ? (
        <div className="p-4">
          <EmptyState
            title="No candidates match"
            detail="Run a detection job, then adjust the SNR, track RMS and cosmic-ray filters."
          />
        </div>
      ) : (
        <div className="max-h-[430px] overflow-auto">
          <table className="w-full min-w-[900px] border-collapse text-left font-mono text-[11px]">
            <thead className="sticky top-0 z-10 bg-obsidian-850 text-[10px] uppercase tracking-wider text-slate-400">
              <tr>
                <th scope="col" className="px-2 py-2 font-medium">#</th>
                {COLUMNS.map((column) => {
                  const active = sortKey === column.key;
                  return (
                    <th
                      key={column.key}
                      scope="col"
                      aria-sort={active ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                      className="whitespace-nowrap px-2 py-2 font-medium"
                    >
                      <button
                        type="button"
                        className={`hover:text-cyan-200 ${active ? 'text-cyan-300' : ''}`}
                        title={column.title}
                        onClick={() => toggleSort(column.key)}
                      >
                        {column.label} {active ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                      </button>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {sorted.map(({ candidate, index }) => (
                <tr
                  key={`${index}-${candidate.x}-${candidate.y}`}
                  tabIndex={0}
                  aria-selected={selectedIndex === index}
                  onClick={() => onSelect(index)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      onSelect(index);
                    }
                  }}
                  className={`cursor-pointer border-t border-slate-800/70 transition-colors ${
                    selectedIndex === index ? 'bg-cyan-500/15 text-cyan-100' : 'text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <td className="px-2 py-1.5 text-slate-500">{index + 1}</td>
                  {COLUMNS.map((column) => (
                    <td key={column.key} className="whitespace-nowrap px-2 py-1.5">
                      <span className={column.key === 'snr' && candidate.snr >= 5 ? 'text-emerald-300' : ''}>
                        {candidate[column.key].toFixed(column.digits)}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
