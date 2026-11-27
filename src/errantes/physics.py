"""Modelo fisico: PSF, ruido Poisson-Gaussiano y conversion de flujo a magnitud."""

from __future__ import annotations

import numpy as np

from errantes.types import PSFModel


def psf_values(dy: np.ndarray, dx: np.ndarray, model: PSFModel) -> np.ndarray:
    """PSF 2D no normalizada evaluada en desplazamientos (dy, dx)."""
    r2 = dx * dx + dy * dy
    if model.kind == "gaussian":
        return np.asarray(np.exp(-0.5 * r2 / (model.sigma * model.sigma)), dtype=np.float64)
    return np.asarray(
        np.power(1.0 + r2 / (model.alpha * model.alpha), -model.beta), dtype=np.float64
    )


def psf_peak(model: PSFModel) -> float:
    """Fraccion de flujo total que cae en el centro de la PSF ideal."""
    if model.kind == "gaussian":
        return float(1.0 / (2.0 * np.pi * model.sigma**2))
    return float((model.beta - 1.0) / (np.pi * model.alpha**2))


def flux_for_snr(snr: float, sigma_local: float, model: PSFModel) -> float:
    """Flujo total (electrones) necesario para alcanzar SNR en el piculo central.

    Se usa sigma_local = sqrt(mu_cielo + sigma_read^2), de modo que
    SNR = (flujo * peak_psf) / sigma_local.
    """
    return float(snr * sigma_local / psf_peak(model))


def render_sources(
    shape: tuple[int, int],
    positions: np.ndarray,
    fluxes: np.ndarray,
    model: PSFModel,
) -> np.ndarray:
    """Renderiza fuentes sub-pixel en un recorte local para evitar FFTs costosas.

    Parameters
    ----------
    shape: (altura, anchura) de la imagen.
    positions: array (M, 2) con coordenadas (x, y) en pixeles.
    fluxes: array (M,) con flujo total por fuente en electrones.
    model: PSF comun a todas las fuentes de esta imagen.
    """
    height, width = shape
    image = np.zeros(shape, dtype=np.float64)
    radius = int(np.ceil(5.0 * (model.sigma if model.kind == "gaussian" else model.alpha)))
    radius = min(radius, max(height, width))
    for (x, y), flux in zip(positions, fluxes):
        x0, x1 = max(0, int(np.floor(x)) - radius), min(width, int(np.floor(x)) + radius + 2)
        y0, y1 = max(0, int(np.floor(y)) - radius), min(height, int(np.floor(y)) + radius + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=np.float64)[None, :]
        ys = np.arange(y0, y1, dtype=np.float64)[:, None]
        psf = psf_values(ys - y, xs - x, model)
        image[y0:y1, x0:x1] += flux * psf / psf.sum()
    return image


def poisson_gaussian_noise(
    expectation: np.ndarray, read_noise: float, gain: float, rng: np.random.Generator
) -> np.ndarray:
    """Ruido de fotones (Poisson) + lectura (Gaussiano), devuelto en ADU."""
    electrons = rng.poisson(np.maximum(expectation, 0.0)).astype(np.float64)
    electrons += rng.normal(0.0, read_noise, size=expectation.shape)
    return np.asarray(electrons / float(gain), dtype=np.float64)


def flux_to_magnitude(flux: float, zeropoint: float) -> float:
    """Magnitud instrumental: m = zp - 2.5 log10(F)."""
    return float(zeropoint - 2.5 * np.log10(max(float(flux), 1e-9)))
