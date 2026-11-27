"""Regresión del pipeline de detección compilado en Rust (`errantes_py`)."""

from __future__ import annotations

import numpy as np
import pytest

errantes_py = pytest.importorskip("errantes_py")


def _cube_with_star_and_wanderer() -> np.ndarray:
    rng = np.random.default_rng(7)
    n_frames, height, width = 8, 160, 160
    cube = rng.normal(160.0, 4.0, size=(n_frames, height, width))
    yy, xx = np.mgrid[0:height, 0:width]
    for t in range(n_frames):
        # Estrella estacionaria muy brillante: antes generaba cientos de duplicados.
        cube[t] += 50_000.0 * np.exp(-((xx - 28) ** 2 + (yy - 140) ** 2) / (2 * 1.4**2))
        x, y = 80.0 - 0.03 * t, 80.0 + 0.20 * t
        cube[t] += 120.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4**2))
    return cube


def test_rust_deteccion_top_candidato_y_metricas_reales() -> None:
    result = errantes_py.detect_candidates(
        _cube_with_star_and_wanderer(), backend="cpu"
    )
    candidates = result["candidates"]
    assert candidates, "no se detectó la fuente móvil"

    top = candidates[0]
    assert np.hypot(top["x"] - 80.0, top["y"] - 80.0) < 2.0
    assert 0.1 < top["vy"] < 0.5
    assert top["track_rms"] > 0.0
    assert top["photometric_cv"] > 0.0
    assert 0.0 <= top["cosmic_fraction"] <= 1.0

    # Ni la estrella ni hipótesis estacionarias deben sobrevivir al filtro/NMS.
    assert all(np.hypot(c["x"] - 28.0, c["y"] - 140.0) > 5.0 for c in candidates)
    assert all(np.hypot(c["vx"], c["vy"]) >= 0.08 for c in candidates)
