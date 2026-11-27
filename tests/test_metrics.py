"""Tests de asociacion, curvas ROC/PR y benchmark."""

from __future__ import annotations

import polars as pl

from errantes.metrics import (
    area_under_curve,
    benchmark_snr_sweep,
    match_detections,
    roc_pr_curve,
    summarize_benchmark,
    write_benchmark,
)
from errantes.simulator import simulate_default_cube
from errantes.types import Candidate


def _candidate(x: float, y: float, vx: float, vy: float, snr: float = 6.0) -> Candidate:
    return Candidate(
        x=x, y=y, vx=vx, vy=vy, snr=snr, stacked_flux=100.0, peak=20.0,
        n_pixels=5, ref_frame=5,
    )


def test_match_detections_hungaro() -> None:
    truth = pl.DataFrame(
        {
            "object_id": ["A", "B"],
            "t": [5, 5],
            "x": [30.0, 60.0],
            "y": [40.0, 20.0],
            "vx": [0.4, -0.2],
            "vy": [-0.1, 0.3],
        }
    )
    matched = match_detections(
        truth,
        [_candidate(60.5, 20.2, -0.2, 0.3), _candidate(10.0, 10.0, 1.2, 0.0)],
        ref_frame=5,
    )
    assert (matched.tp, matched.fp, matched.fn) == (1, 1, 1)
    assert np_isclose(matched.precision, 0.5)
    assert np_isclose(matched.recall, 0.5)
    assert np_isclose(matched.f1, 0.5)


def np_isclose(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def test_curvas_roc_pr() -> None:
    curve = roc_pr_curve([0.9, 0.7, 0.1], [True, False, False])
    assert curve.height == 4
    assert curve["tpr"][0] == 0.0
    assert curve["tpr"][-1] == 1.0
    assert curve["precision"][-1] < 1.0
    assert area_under_curve(curve, "recall", "precision") > 0.0


def test_benchmark_y_escritura(tmp_path: object) -> None:
    results = benchmark_snr_sweep(snrs=(2.0,), seeds=(1234,))
    assert results.height == 1
    assert results["f1"][0] == 1.0
    assert results["runtime_s"][0] > 0
    summary = summarize_benchmark(results)
    assert summary["snr_frame"][0] == 2.0
    parquet, json_path = write_benchmark(results, tmp_path)  # type: ignore[arg-type]
    assert parquet.exists() and json_path.exists()
    assert pl.read_parquet(parquet).height == 1


def test_match_sin_candidatos() -> None:
    truth = pl.DataFrame(
        {
            "object_id": ["A"],
            "t": [5],
            "x": [30.0],
            "y": [40.0],
            "vx": [0.4],
            "vy": [-0.1],
        }
    )
    matched = match_detections(truth, [], ref_frame=5)
    assert (matched.tp, matched.fp, matched.fn) == (0, 0, 1)
    cube = simulate_default_cube(5.0, seed=1)
    assert len(cube.wanderers) == 1
