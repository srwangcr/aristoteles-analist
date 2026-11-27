"""Deteccion multi-frame: shift-and-stack, umbralado y centroides sub-pixel."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from errantes.types import Candidate, PipelineConfig


class ShiftStackEngine:
    """Apilamiento sintetico por hipotesis de velocidad en el dominio de Fourier.

    Los FFT de los N frames se calculan una sola vez. Cada hipotesis cuesta
    N transformadas inversas: O(M * N * H * W * log(H * W)), con M el numero
    de velocidades probadas.
    """

    def __init__(self, residuals: np.ndarray, ref_frame: int | None = None) -> None:
        cube = np.asarray(residuals, dtype=np.float64)
        if cube.ndim != 3:
            raise ValueError("residuals debe ser un cubo (N, H, W)")
        self.residuals = cube
        self.n_frames, self.height, self.width = cube.shape
        self.ref_frame = (self.n_frames - 1) // 2 if ref_frame is None else ref_frame
        self.fft = np.fft.rfft2(cube, axes=(-2, -1))
        self.ky = (np.fft.fftfreq(self.height) * self.height)[:, None]
        self.kx = (np.fft.rfftfreq(self.width) * self.width)[None, :]

    def stack(self, vx: float, vy: float) -> np.ndarray:
        """Suma los frames desplazados para que un errante con (vx, vy) quede fijo."""
        stacked = np.zeros((self.height, self.width), dtype=np.float64)
        for t in range(self.n_frames):
            dt = float(t - self.ref_frame)
            shift_x, shift_y = -vx * dt, -vy * dt
            if abs(shift_x) < 1e-12 and abs(shift_y) < 1e-12:
                stacked += np.fft.irfft2(self.fft[t], s=(self.height, self.width))
                continue
            phase = np.exp(
                -2j
                * np.pi
                * (self.kx * shift_x / self.width + self.ky * shift_y / self.height)
            )
            stacked += np.fft.irfft2(self.fft[t] * phase, s=(self.height, self.width))
        return stacked


def robust_sigma(image: np.ndarray) -> float:
    """Estimacion MAD robusta de la dispersion del mapa apilado."""
    finite = np.asarray(image, dtype=np.float64)
    median = float(np.median(finite))
    return float(max(1.4826 * np.median(np.abs(finite - median)), 1e-6))


def extract_candidates(
    stacked: np.ndarray,
    vx: float,
    vy: float,
    config: PipelineConfig,
    velocity_index: int = -1,
) -> list[Candidate]:
    """Umbraliza el mapa apilado, etiqueta componentes y ajusta centroides."""
    sigma = robust_sigma(stacked)
    threshold = config.detection_sigma * sigma
    mask = stacked > threshold
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    candidates: list[Candidate] = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < config.min_component_pixels:
            continue
        weights = stacked[ys, xs] - threshold
        total = float(weights.sum())
        if total <= 0:
            continue
        x = float(np.average(xs, weights=weights))
        y = float(np.average(ys, weights=weights))
        peak = float(stacked[ys, xs].max())
        candidates.append(
            Candidate(
                x=x,
                y=y,
                vx=float(vx),
                vy=float(vy),
                snr=peak / sigma,
                stacked_flux=total,
                peak=peak,
                n_pixels=len(xs),
                ref_frame=0,  # se fija en detect_candidates
                velocity_index=velocity_index,
            )
        )
    return candidates


def deduplicate(candidates: list[Candidate], radius: float) -> list[Candidate]:
    """Conserva el candidato de mayor SNR dentro de un radio dado."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.snr, reverse=True):
        if any(
            np.hypot(candidate.x - other.x, candidate.y - other.y) <= radius for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def detect_candidates(
    residuals: np.ndarray,
    config: PipelineConfig,
    ref_frame: int | None = None,
) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    """Barrido completo sobre la grilla de velocidades.

    Devuelve (candidatos deduplicados, mejor mapa apilado, SNR maximo por velocidad).
    """
    engine = ShiftStackEngine(residuals, ref_frame=ref_frame)
    velocities = config.velocities
    best_map = np.zeros((engine.height, engine.width), dtype=np.float64)
    best_peak = -np.inf
    snr_by_velocity = np.zeros(len(velocities), dtype=np.float64)
    all_candidates: list[Candidate] = []
    for index, (vx, vy) in enumerate(velocities):
        stacked = engine.stack(float(vx), float(vy))
        if config.match_filter_sigma > 0:
            stacked = ndimage.gaussian_filter(stacked, sigma=config.match_filter_sigma)
        peak = float(stacked.max())
        snr_by_velocity[index] = peak / robust_sigma(stacked)
        if peak > best_peak:
            best_peak, best_map = peak, stacked
        for candidate in extract_candidates(stacked, float(vx), float(vy), config, index):
            all_candidates.append(
                Candidate(**{**candidate.__dict__, "ref_frame": engine.ref_frame})
            )
    detected = deduplicate(all_candidates, config.dedup_radius)
    return detected, best_map, snr_by_velocity
