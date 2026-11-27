"""Tests de fondo robusto, alineamiento y sustraccion de estrellas."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from errantes.preprocess import align_frames, estimate_background, phase_correlation, subtract_reference
from errantes.simulator import simulate
from errantes.types import PipelineConfig, SimulatorConfig


def test_estimate_background_recupera_gradiente() -> None:
    yy, xx = np.mgrid[0:80, 0:80]
    truth = 100.0 + 0.4 * xx - 0.25 * yy
    image = truth + np.random.default_rng(1).normal(0, 3, size=(80, 80))
    image[40, 40] = 5000.0  # estrella que no debe sesgar la mediana
    background = estimate_background(image, mesh_size=16)
    assert np.mean(np.abs(background - truth)) < 1.0


def test_phase_correlation_subpixel() -> None:
    rng = np.random.default_rng(4)
    reference = np.zeros((80, 80))
    for _ in range(12):
        x, y = rng.integers(10, 70, size=2)
        reference[y - 2 : y + 3, x - 2 : x + 3] += rng.uniform(50, 200)
    shifted = ndimage.shift(reference, shift=(2.3, -1.7), order=3, mode="constant")
    dy, dx = phase_correlation(reference, shifted, upsample=8)
    assert abs(dy - 2.3) < 0.2
    assert abs(dx + 1.7) < 0.2


def test_align_frames_aplica_desplazamiento_opuesto() -> None:
    rng = np.random.default_rng(8)
    yy, xx = np.mgrid[0:64, 0:64]
    reference = np.zeros((64, 64))
    for _ in range(10):
        x, y = rng.integers(10, 54, size=2)
        reference += 100.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))
    shifted = ndimage.shift(reference, shift=(1.5, 2.5), order=3, mode="nearest")
    aligned, shifts = align_frames(np.stack([reference, shifted]), ref_index=0)
    assert abs(shifts[1, 0] - 1.5) < 0.2
    assert abs(shifts[1, 1] - 2.5) < 0.2
    interior = np.s_[6:-6, 6:-6]
    assert np.std(aligned[1][interior] - aligned[0][interior]) < 2.0


def test_sustraccion_de_estrellas_no_auto_sustrae_errante() -> None:
    config = SimulatorConfig(n_stars=3, cosmic_rate=0.0, n_defective_columns=0, rng_seed=10)
    cube = simulate(config)
    star_peak = cube.frames[5][round(cube.star_positions[0][1]), round(cube.star_positions[0][0])]
    aligned = cube.frames - np.median(cube.frames)
    residuals, _ = subtract_reference(aligned, cube.frame_psf_sigma)
    residual_peak = residuals[5][round(cube.star_positions[0][1]), round(cube.star_positions[0][0])]
    assert abs(residual_peak) < 0.5 * abs(star_peak)
    wanderer = cube.wanderers[0]
    x, y = wanderer.position(5)
    assert residuals[5][round(y), round(x)] > 0.5 * residuals[5].std()
    assert len(PipelineConfig().velocities) > 0
