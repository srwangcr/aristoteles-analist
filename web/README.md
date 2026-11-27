# Observatory Workbench

React 18 + TypeScript + Vite + Tailwind CSS 3 frontend for the moving-object detection engine.

## Run

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend talks to the API at `http://localhost:8080` by default. Point it elsewhere with a `.env` file:

```bash
VITE_API_URL=http://localhost:8080
```

Build the production bundle with `npm run build` (output: `web/dist`).

## What it does

1. **Charge** a cube: drag/drop or select a FITS file (raw bytes POSTed to `/api/fits`), or call `/api/simulate` with SNR, seed, frame count, shape and velocity.
2. **Detect**: choose Auto/CPU/GPU, set `v_max`, `Δv` and detection σ, then run. Progress comes from `/api/jobs/{job_id}/ws` with automatic reconnection and REST polling fallback.
3. **Blink** through frames with play/pause, frame slider, speed control, A/B comparison, percentile stretch and a truth overlay for simulated cubes. For FITS uploads the browser parses the primary HDU locally for the preview; for simulated cubes a deterministic noise+PSF preview is synthesized locally from the returned truth table (the server cube itself is what detection uses).
4. **Inspect** shift-stack maps: choose `(vx, vy)` with sliders or exact number inputs, fetch the little-endian float32 stack from `/api/stack/{job_id}`, view it as a heatmap with colorbar and peak SNR, and click a pixel to read its position/value. Selecting a catalog row also offers its velocity as a stack hypothesis.
5. **Filter/sort** candidates by SNR, RANSAC track RMS and cosmic-ray fraction; click a row to mark it on the current blink frame.
6. **Export** JSON, MPC and Parquet catalogs from `/api/export/{job_id}?format=...`.

## Scripts

| Script | Purpose |
| --- | --- |
| `npm run dev` | Vite dev server with HMR |
| `npm run build` | Typecheck (`tsc -b`) + production build |
| `npm run preview` | Preview the built bundle |
| `npm run typecheck` | TypeScript only |
