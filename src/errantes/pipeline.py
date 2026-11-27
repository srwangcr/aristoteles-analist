"""Orquestacion de extremo a extremo del pipeline de deteccion."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from errantes.detection import detect_candidates
from errantes.preprocess import PreprocessedCube, preprocess_cube
from errantes.types import Candidate, PipelineConfig, SimulatedCube
from errantes.validation import validate_candidates


@dataclass(frozen=True)
class DetectionResult:
    """Salida completa del pipeline, incluyendo diagnostics de cada etapa."""

    candidates: tuple[Candidate, ...]
    preprocessed: PreprocessedCube
    best_stack: np.ndarray
    snr_by_velocity: np.ndarray
    runtime_s: float
    config: PipelineConfig

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def run_pipeline(
    cube: SimulatedCube | np.ndarray,
    config: PipelineConfig | None = None,
    frame_psf_sigma: tuple[float, ...] | None = None,
    validate: bool = True,
) -> DetectionResult:
    """Ejecuta preprocesamiento, shift-and-stack y filtrado de candidatos."""
    start = perf_counter()
    cfg = config or PipelineConfig()
    sigmas: tuple[float, ...] | None
    if isinstance(cube, SimulatedCube):
        frames = cube.frames
        sigmas = cube.frame_psf_sigma if frame_psf_sigma is None else frame_psf_sigma
    else:
        frames = cube
        sigmas = frame_psf_sigma
    preprocessed = preprocess_cube(frames, cfg, frame_psf_sigma=sigmas)
    raw_candidates, best_stack, snr_by_velocity = detect_candidates(
        preprocessed.residuals, cfg, ref_frame=(len(frames) - 1) // 2
    )
    if validate:
        candidates = validate_candidates(
            preprocessed.residuals, raw_candidates, cfg, ref_frame=(len(frames) - 1) // 2
        )
    else:
        candidates = [
            Candidate(**{**candidate.__dict__, "ref_frame": (len(frames) - 1) // 2})
            for candidate in raw_candidates
        ]
    return DetectionResult(
        candidates=tuple(candidates),
        preprocessed=preprocessed,
        best_stack=best_stack,
        snr_by_velocity=snr_by_velocity,
        runtime_s=perf_counter() - start,
        config=cfg,
    )
