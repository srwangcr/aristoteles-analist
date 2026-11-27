"""Tests del modelo de PSF y del ruido Poisson-Gaussiano."""

from __future__ import annotations

import numpy as np

from errantes.physics import (
    flux_for_snr,
    flux_to_magnitude,
    poisson_gaussian_noise,
    psf_peak,
    render_sources,
)
from errantes.types import PSFModel


def test_render_sources_conserva_flujo() -> None:
    model = PSFModel(sigma=1.35)
    image = render_sources(
        (64, 64), np.array([[31.4, 42.7]]), np.array([1000.0]), model
    )
    assert np.isclose(image.sum(), 1000.0, rtol=1e-6)
    # El maximo cae en el pixel entero mas cercano al centro sub-pixel.
    assert np.isclose(image.max(), 1000.0 * psf_peak(model), rtol=0.1)


def test_moffat_alpha_y_peak() -> None:
    model = PSFModel(kind="moffat", sigma=1.35, beta=3.5)
    assert 0.5 * model.sigma < model.alpha < 2.0 * model.sigma
    assert 0 < psf_peak(model) < 1.0
    image = render_sources((48, 48), np.array([[24.0, 24.0]]), np.array([500.0]), model)
    assert np.isclose(image.sum(), 500.0, rtol=1e-6)


def test_flux_for_snr_definicion() -> None:
    model = PSFModel(sigma=1.4)
    sigma_local = 12.0
    flux = flux_for_snr(2.0, sigma_local, model)
    assert np.isclose(flux * psf_peak(model) / sigma_local, 2.0)


def test_ruido_poisson_gaussiano_media_y_varianza() -> None:
    rng = np.random.default_rng(7)
    expectation = np.full((200, 200), 100.0)
    frame = poisson_gaussian_noise(expectation, read_noise=5.0, gain=1.0, rng=rng)
    assert abs(frame.mean() - 100.0) < 1.0
    assert abs(frame.std() - np.sqrt(125.0)) < 3.0


def test_magnitud_instrumental() -> None:
    assert np.isclose(flux_to_magnitude(100.0, 25.0), 20.0)
