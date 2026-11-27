"""Tests del kernel de shift-and-stack y extraccion de candidatos."""

from __future__ import annotations

import numpy as np

from errantes.detection import ShiftStackEngine, extract_candidates, robust_sigma
from errantes.physics import poisson_gaussian_noise, render_sources
from errantes.types import PipelineConfig, PSFModel


def _moving_cube(snr: float = 8.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    model = PSFModel(sigma=1.3)
    cube = np.empty((9, 48, 48), dtype=np.float64)
    for t in range(9):
        x, y = 20.0 + 0.5 * t, 24.0 - 0.25 * t
        sources = render_sources((48, 48), np.array([[x, y]]), np.array([300.0]), model)
        cube[t] = poisson_gaussian_noise(60.0 + sources, 4.0, 1.0, rng)
    return cube - np.median(cube)


def test_shift_and_stack_integra_senal() -> None:
    cube = _moving_cube()
    engine = ShiftStackEngine(cube, ref_frame=4)
    correct = engine.stack(0.5, -0.25)
    wrong = engine.stack(-0.5, 0.25)
    x, y = 20.0 + 0.5 * 4, 24.0 - 0.25 * 4
    assert correct[round(y), round(x)] > 3.0 * wrong[round(y), round(x)]
    assert correct[round(y), round(x)] / robust_sigma(correct) > 8.0


def test_extract_candidates_centroide_subpixel() -> None:
    yy, xx = np.mgrid[0:48, 0:48]
    image = np.exp(-((xx - 23.4) ** 2 + (yy - 17.6) ** 2) / (2 * 1.3**2))
    image += np.random.default_rng(2).normal(0, 0.02, size=image.shape)
    candidates = extract_candidates(image, 0.5, -0.25, PipelineConfig(detection_sigma=4.0))
    assert len(candidates) == 1
    assert abs(candidates[0].x - 23.4) < 0.2
    assert abs(candidates[0].y - 17.6) < 0.2


def test_grilla_de_velocidades() -> None:
    config = PipelineConfig(velocity_max=1.0, velocity_step=0.5)
    assert config.n_velocities == 25
    velocities = config.velocities
    assert np.isclose(velocities[:, 0].min(), -1.0)
    assert np.isclose(velocities[:, 1].max(), 1.0)
