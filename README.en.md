# Project 2 — Detection of wanderers in noisy astronomical images

[Spanish version: README.md](README.md)

A modular, reproducible engine for detecting objects with linear or quasi-linear motion hidden in the noise of FITS image time series. The pipeline covers calibrated simulation with ground truth, astrometric preprocessing, multi-hypothesis shift-and-stack, track validation and scientific evaluation.

## Installation

Requires Python 3.12+ (tested with 3.14).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/pytest -q
```

Core dependencies: NumPy, SciPy, Astropy (FITS/WCS) and Polars (catalogs and metrics). The stacking kernel is pure NumPy; if profiling justifies it, `ShiftStackEngine.stack` is the single hot spot to port to Rust/PyO3.

## Structure

```
src/errantes/
  types.py        # configs, Wanderer, Candidate, GroundTruth, SimulatedCube
  physics.py      # Gaussian/Moffat PSF, Poisson+Gaussian noise, flux↔SNR↔magnitude
  simulator.py    # synthetic cube, ground-truth injection, FITS I/O
  preprocess.py   # MAD mesh, phase correlation, master image and residuals
  detection.py    # ShiftStackEngine, kσ thresholding, components, centroids
  validation.py   # linear RANSAC, photometric consistency, cosmic-ray rejection
  pipeline.py     # end-to-end orchestration
  metrics.py      # Hungarian matching, Precision/Recall/F1, ROC/PR, benchmark
benchmarks/
  run_benchmark.py  # sensitivity and scaling CLI
  results/          # empirical Parquet/JSON results
tests/              # 24 unit and integration tests
```

## Physical model

For frame $i$, the expected number of electrons per pixel is

$$
\mu_i(x,y) = \underbrace{b_0 + g_x(x-c_x) + g_y(y-c_y) + \delta_i}_{\text{sky with gradient and drift}}
+ \sum_s F_s\,P_i(x-x_s-\Delta x_i,\,y-y_s-\Delta y_i)
+ F_{\rm obj}\,P_i(x-x_i,\,y-y_i) + C_i(x,y),
$$

where $\delta_i\sim\mathcal{N}(0,\sigma_{\rm drift}^2)$, $\Delta x_i,\Delta y_i\sim\mathcal{N}(0,\sigma_{\rm jitter}^2)$ represent guide jitter of the stars, $C_i$ are cosmic rays (single-frame pixels) and defective columns (constant additive bias per column). The digitized image is

$$
D_i(x,y) = \frac{\mathrm{Poisson}\!\left(\mu_i(x,y)\right) + \mathcal{N}(0,\sigma_{\rm read}^2)}{g},
$$

with $g$ the gain in e⁻/ADU. The local electron noise is $\sigma_{\rm loc}=\sqrt{\mu_{\rm sky}+\sigma_{\rm read}^2}$.

Gaussian PSF: $P=\exp[-r^2/(2\sigma^2)]$, normalized to unit sum, with peak $P(0)=1/(2\pi\sigma^2)$. Moffat PSF: $P=(1+r^2/\alpha^2)^{-\beta}$, with $\alpha=2\sigma\sqrt{2^{1/\beta}-1}$ to match the Gaussian FWHM. The flux injected for a target SNR per frame follows from

$$
\mathrm{SNR}_{\rm frame}=\frac{F_{\rm obj}P(0)}{\sqrt{\mu_{\rm sky}+\sigma_{\rm read}^2}}
\quad\Longrightarrow\quad
F_{\rm obj}=\frac{\mathrm{SNR}_{\rm frame}\,\sigma_{\rm loc}}{P(0)}.
$$

All magnitudes are instrumental: $m = m_{\rm zp}-2.5\log_{10}(F/g)$.

## Algorithm

1. **Background and stars.** A 2D mesh of medians with iterative sigma-clipping (median + $1.4826\,\mathrm{MAD}$) and bilinear interpolation. Alignment uses phase correlation with zero-padding and a parabolic fit of the peak (sub-pixel precision). The master image is the temporal median of aligned frames, normalized by the flux of bright stars. It is subtracted **only over the support of stationary sources** (a dilated mask of the master image): subtracting the median over the whole field would self-subtract slow wanderers that remain on the same pixel for several frames. When per-frame seeing values are known, the reference is smoothed with $\sigma_{\rm extra}=\sqrt{\sigma_i^2-\sigma_{\rm ref}^2}$ before subtraction.

2. **Shift-and-stack (Track-Before-Detect).** For each hypothesis $\vec v=(v_x,v_y)$, the residuals are shifted to the reference frame $t_r$ and summed:

$$
S_{\vec v}(x,y)=\sum_{i=0}^{N-1} R_i\!\left(x+v_x(t_i-t_r),\,y+v_y(t_i-t_r)\right).
$$

In Fourier space, with $k_x,k_y$ as frequency indices and $s_x=-v_x(t_i-t_r)$, $s_y=-v_y(t_i-t_r)$:

$$
\hat R_i^{\,s}(k_x,k_y)=\hat R_i(k_x,k_y)\,
\exp\!\left[-2\pi i\left(\frac{k_xs_x}{W}+\frac{k_ys_y}{H}\right)\right].
$$

The $N$ forward FFTs are computed once; each hypothesis costs $N$ inverse transforms. A Gaussian matched filter ($\sigma=0.8$ px) is applied to the stacked map before thresholding at $k\sigma$ (MAD). Connected 3×3 components are labeled and sub-pixel weighted centroids are computed.

3. **Validation.** Aperture flux and centroid are measured in every frame along the predicted track; $(x(t),y(t))$ is fitted with linear RANSAC. Candidates are rejected if RMS > 0.9 px, single-frame energy fraction > 0.55, photometric CV > 0.65, or fewer than 70 % of frames lie inside the field. Cosmic rays concentrate all their energy in one frame and are removed by this filter.

4. **Metrics.** Optimal candidate↔truth assignment via the Hungarian algorithm (tolerances 2.5 px in position and 0.45 px/frame in velocity), Precision, Recall, F1, and ROC/PR curves sweeping candidate SNR.

### Complexity

| Stage | Cost |
|---|---|
| Background + subtraction | $O(NHW)$ |
| Alignment | $O(NHW\log(HW))$ |
| Residual FFTs (once) | $O(NHW\log(HW))$ |
| Sweep over $M$ velocities | $O(M\,N\,HW\log(HW))$ |
| Memory | $O(NHW + HW)$ plus the $N$ complex spectra |

With $M=\left(2v_{\max}/\Delta v+1\right)^2$: $v_{\max}=1.5$, $\Delta v=0.25$ give $M=169$; $\Delta v=0.125$ give $M=625$.

## Quick start

```python
from errantes import simulate_default_cube, run_pipeline
from errantes.metrics import match_detections

cube = simulate_default_cube(snr_frame=2.0, seed=1234)
result = run_pipeline(cube)

for candidate in result.candidates:
    print(candidate.x, candidate.y, candidate.vx, candidate.vy, candidate.snr)

matched = match_detections(cube.truth.positions, result.candidates, ref_frame=5)
print(matched.as_dict())
```

Ground truth is exported with `cube.truth.write("truth/")` to `objects.json`, `objects.parquet`, `positions.json` and `positions.parquet`. `write_fits_cube(cube, "cube.fits")` writes the cube with a linear WCS and a `TRUTH` extension containing $(t,x,y,v_x,v_y,\mathrm{SNR},F)$ per object and frame.

## Empirical results

Benchmark over 10 seeds per SNR, $12\times96\times96$ px cubes, $v=(0.37,-0.21)$ px/frame, with stars, varying seeing, jitter, defective columns and ~3 cosmic rays/frame. AMD Ryzen 5 3400G CPU, NumPy 2.5.5 / SciPy 1.18.1, single thread. Full data in `benchmarks/results/`.

| SNR/frame | Precision | Recall | F1 | Mean FP | Time/cube (s) |
|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.60 | 0.60 | 0.60 | 0.0 | 1.25 |
| 2.0 | 0.85 | 0.90 | 0.87 | 0.1 | 1.35 |
| 3.0 | 0.90 | 0.90 | 0.90 | 0.0 | 1.29 |
| 5.0 | 1.00 | 1.00 | 1.00 | 0.0 | 1.31 |

Sweep scaling (same cube, mean of 3 seeds):

| $\Delta v$ (px/frame) | $M$ | Time (s) |
|---:|---:|---:|
| 0.5 | 49 | 0.70 |
| 0.25 | 169 | 1.32 |
| 0.125 | 625 | 3.20 |

Runtime grows approximately linearly with $M$, as predicted by $O(M N H W\log(HW))$. Stacking integrates the signal by $\sqrt{N}\approx3.46$; the matched filter and partial star subtraction reduce the effective observed gain to a stacked SNR of ~5–9 for $\mathrm{SNR}_{\rm frame}=2$.

Reproduce with:

```bash
.venv/bin/python benchmarks/run_benchmark.py --scaling
.venv/bin/pytest -q
.venv/bin/mypy src/errantes
```

Included verification: recovery of wanderers at $\mathrm{SNR}_{\rm frame}=2.0$ with position error < 1.5 px and velocity error < 0.5 px/frame; **100 % cosmic-ray rejection** in 4 cubes with a rate of 12 cosmic rays/frame (~48 injections), plus a dedicated unit test for the single-frame spike.

## Limitations and future work

- Star subtraction masks the cores of fixed sources: a wanderer crossing the core of a bright star may lose signal in those frames.
- Fourier shifts assume periodic edges; candidates within 2 px of the border are discarded during validation.
- The velocity grid is regular; a hierarchical search (coarse → fine) would reduce cost for large $v_{\max}$.
- Current profile: ~98 % of the time is spent in the $M\times N$ inverse FFTs. The natural next step is a Rust/PyO3 kernel with planned FFTs and in-place accumulation, keeping the `ShiftStackEngine` API.

## Technology Readiness Roadmap (TRL 4 -> TRL 9)

Current status: **TRL 4** (modular Python/NumPy/Astropy/Polars engine with calibrated synthetic cube generator, star subtraction, Fourier shift-and-stack, and RANSAC track validation verified in lab benchmarks at $	ext{F1} = 0.87$ for $	ext{SNR}_{	ext{frame}} = 2.0$ and 100% cosmic-ray rejection).

- [x] **TRL 4 — Component validation in laboratory:** Physical simulator with Gaussian/Moffat PSF and Poisson+Gaussian noise, MAD background + phase-correlation alignment, Fourier-domain shift-and-stack, RANSAC/photometric filtering, and reproducible benchmarks (`pytest` + strict `mypy`).
- [ ] **TRL 5 — Core acceleration and observational FITS support:** Port `ShiftStackEngine.stack` to Rust (`PyO3` + `rustfft` + `rayon`) for multi-core parallel execution, add hierarchical coarse-to-fine velocity search, and ingest real multi-extension FITS cubes with irregular timestamps (`MJD-OBS`), bad-pixel masks, and WCS astrometric rates (`arcsec/hr`).
- [ ] **TRL 6 — Validation on real survey datasets:** Benchmark the pipeline on archival survey imagery (ZTF, TESS, or Pan-STARRS), recovering known Minor Planet Center (MPC) moving objects and measuring empirical completeness and false-positive rates.
- [ ] **TRL 7 — Operational pilot in nightly observing pipelines:** Deploy on a Linux server processing nightly batches from robotic telescopes or survey feeds, emitting standard MPC astrometric reports (ADES / 80-column) and low-latency transient alerts.
- [ ] **TRL 8 — Flight/survey-qualified production system:** Support accelerated/curved trajectories for very close NEOs (topocentric parallax and angular acceleration), full telemetry, reference-catalog regression suites, and reproducible container builds.
- [ ] **TRL 9 — Nominal scientific operations:** Continuous production operation in discovery, follow-up, and precision astrometry campaigns for Near-Earth Objects (NEOs) and faint transients.
