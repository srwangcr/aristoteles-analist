"""Tests del simulador calibrado y de la verdad terrestre."""

from __future__ import annotations

import numpy as np
import polars as pl

from errantes.simulator import (
    WandererSpec,
    read_fits_cube,
    simulate,
    simulate_default_cube,
    write_fits_cube,
)
from errantes.types import GroundTruth, SimulatorConfig


def test_cubo_y_trayectoria_exacta(tmp_path: object) -> None:
    cube = simulate_default_cube(snr_frame=2.0, seed=1234)
    assert cube.frames.shape == (12, 96, 96)
    assert cube.frames.dtype == np.float32
    assert len(cube.wanderers) == 1
    wanderer = cube.wanderers[0]
    assert np.isclose(wanderer.x0, 36.0)
    assert np.isclose(wanderer.y0, 41.0)
    assert np.isclose(wanderer.position(5)[0], 37.85)
    assert np.isclose(wanderer.position(5)[1], 39.95)
    rows = cube.truth.positions.filter(pl.col("object_id") == "W01")
    assert rows.height == 12
    assert np.isclose(rows["vx"][0], 0.37)
    assert np.allclose(rows["vx"].to_numpy(), 0.37)

    paths = cube.truth.write(tmp_path)  # type: ignore[arg-type]
    assert all(path.exists() for path in paths)
    loaded = GroundTruth.read(tmp_path)  # type: ignore[arg-type]
    assert loaded.positions["x"].to_list() == cube.truth.positions["x"].to_list()


def test_fits_roundtrip(tmp_path: object) -> None:
    cube = simulate_default_cube(3.0, seed=9)
    path = write_fits_cube(cube, tmp_path / "cube.fits")  # type: ignore[operator]
    frames, truth, header = read_fits_cube(path)
    assert frames.shape == cube.frames.shape
    assert np.allclose(frames, cube.frames, atol=1e-6)
    assert truth.height == cube.n_frames * len(cube.wanderers)
    assert header["GAIN"] == cube.config.gain


def test_estadisticas_de_fondo_sin_fuentes() -> None:
    config = SimulatorConfig(
        n_stars=0,
        cosmic_rate=0.0,
        n_defective_columns=0,
        background=100.0,
        gradient=(0.0, 0.0),
        read_noise=4.0,
        background_drift=0.0,
        rng_seed=5,
    )
    cube = simulate(config, ())
    assert abs(float(cube.frames.mean()) - 100.0) < 1.0
    assert abs(float(cube.frames.std()) - np.sqrt(116.0)) < 3.0


def test_snr_inyectado_es_calibrado() -> None:
    config = SimulatorConfig(
        n_stars=0,
        cosmic_rate=0.0,
        n_defective_columns=0,
        gradient=(0.0, 0.0),
        background_drift=0.0,
        rng_seed=2,
    )
    for snr in (2.0, 5.0):
        cube = simulate(config, (WandererSpec("W", 40, 40, 0.0, 0.0, snr, ref_frame=0),))
        mean_peak = float(
            np.mean([cube.frames[t][40, 40] - config.background for t in range(config.n_frames)])
        )
        expected = snr * np.sqrt(config.background + config.read_noise**2)
        assert abs(mean_peak - expected) < 0.25 * expected
