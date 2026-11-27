"""Tests de extremo a extremo: recuperacion a SNR=2 y rechazo de rayos cosmicos."""

from __future__ import annotations

import numpy as np

from errantes.metrics import match_detections
from errantes.pipeline import run_pipeline
from errantes.simulator import simulate, simulate_default_cube
from errantes.types import Candidate, PipelineConfig, SimulatorConfig
from errantes.validation import validate_candidates


def test_recupera_errante_snr_frame_2() -> None:
    for seed in (1234, 11, 22, 33):
        cube = simulate_default_cube(snr_frame=2.0, seed=seed)
        result = run_pipeline(cube)
        matched = match_detections(cube.truth.positions, result.candidates, ref_frame=5)
        assert matched.tp == 1, f"fallo en seed {seed}"
        assert matched.fp == 0, f"falso positivo en seed {seed}"
        candidate = matched.pairs[0][1]
        x_truth, y_truth = cube.wanderers[0].position(5)
        assert np.hypot(candidate.x - x_truth, candidate.y - y_truth) < 1.5
        assert np.hypot(candidate.vx - 0.37, candidate.vy + 0.21) < 0.5
        assert candidate.cosmic_fraction < 0.55


def test_rechazo_del_100_por_ciento_de_rayos_cosmicos() -> None:
    config = SimulatorConfig(cosmic_rate=12.0, rng_seed=3)
    for seed in (3, 11, 22, 33):
        cube = simulate(SimulatorConfig(**{**config.__dict__, "rng_seed": seed}), ())
        result = run_pipeline(cube)
        assert result.n_candidates == 0, f"cosmico aceptado en seed {seed}"


def test_validador_rechaza_pico_de_un_solo_frame() -> None:
    rng = np.random.default_rng(0)
    residual = rng.normal(0, 5.0, size=(9, 48, 48))
    residual[2, 24, 24] += 200.0  # rayo cosmico de un solo frame
    candidate = Candidate(
        x=24.0,
        y=24.0,
        vx=0.0,
        vy=0.0,
        snr=30.0,
        stacked_flux=200.0,
        peak=200.0,
        n_pixels=1,
        ref_frame=4,
    )
    assert validate_candidates(residual, [candidate], PipelineConfig()) == []


def test_pipeline_acepta_cubo_np_sin_metadatos() -> None:
    cube = simulate_default_cube(5.0, seed=6)
    result = run_pipeline(cube.frames, PipelineConfig())
    assert result.n_candidates >= 1
    assert result.runtime_s > 0
