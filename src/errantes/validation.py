"""Filtros de falsos positivos: rayos cosmicos, ajuste de traza y consistencia fotometrica."""

from __future__ import annotations

import numpy as np

from errantes.types import Candidate, PipelineConfig


def aperture_flux(
    image: np.ndarray, x: float, y: float, radius: float, annulus_width: float = 3.0
) -> float:
    """Flujo en una apertura circular con sustraccion de fondo anular local."""
    height, width = image.shape
    x0, x1 = max(0, int(np.floor(x - radius - annulus_width))), min(width, int(np.ceil(x + radius + annulus_width)) + 1)
    y0, y1 = max(0, int(np.floor(y - radius - annulus_width))), min(height, int(np.ceil(y + radius + annulus_width)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    xs = np.arange(x0, x1)[None, :] - x
    ys = np.arange(y0, y1)[:, None] - y
    r2 = xs * xs + ys * ys
    patch = image[y0:y1, x0:x1]
    aperture = r2 <= radius**2
    annulus = (r2 > (radius + 1.0) ** 2) & (r2 <= (radius + annulus_width) ** 2)
    background = float(np.median(patch[annulus])) if annulus.sum() >= 4 else 0.0
    return float(np.sum(patch[aperture] - background))


def refined_centroid(
    image: np.ndarray, x: float, y: float, radius: float
) -> tuple[float, float]:
    """Centroide ponderado sub-pixel dentro de la apertura, tras restar el fondo."""
    height, width = image.shape
    x0, x1 = max(0, int(np.floor(x - radius))), min(width, int(np.ceil(x + radius)) + 1)
    y0, y1 = max(0, int(np.floor(y - radius))), min(height, int(np.ceil(y + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return x, y
    ys, xs = np.mgrid[y0:y1, x0:x1]
    patch = image[y0:y1, x0:x1]
    r2 = (xs - x) ** 2 + (ys - y) ** 2
    aperture = r2 <= radius**2
    if aperture.sum() < 3:
        return x, y
    weights = np.clip(patch[aperture] - np.median(patch[aperture]), 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return x, y
    return float(np.average(xs[aperture], weights=weights)), float(
        np.average(ys[aperture], weights=weights)
    )


def fit_linear_ransac(
    t: np.ndarray, values: np.ndarray, tolerance: float = 0.8
) -> tuple[float, float, np.ndarray]:
    """Ajuste lineal robusto por RANSAC; devuelve (pendiente, intercepto, inliers)."""
    if len(t) < 2:
        return 0.0, float(values[0]) if len(values) else 0.0, np.ones(len(t), dtype=bool)
    best_inliers = np.ones(len(t), dtype=bool)
    best_count = -1
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            denominator = t[j] - t[i]
            if abs(denominator) < 1e-9:
                continue
            slope = (values[j] - values[i]) / denominator
            intercept = values[i] - slope * t[i]
            inliers = np.abs(values - (intercept + slope * t)) <= tolerance
            count = int(inliers.sum())
            if count > best_count:
                best_count, best_inliers = count, inliers
    if best_count < 2:
        best_inliers = np.ones(len(t), dtype=bool)
    slope, intercept = np.polyfit(t[best_inliers], values[best_inliers], 1)
    return float(slope), float(intercept), best_inliers


def validate_candidates(
    residuals: np.ndarray,
    candidates: list[Candidate],
    config: PipelineConfig,
    ref_frame: int | None = None,
) -> list[Candidate]:
    """Refina la traza y descarta candidatos inconsistentes o de un solo frame."""
    if not candidates:
        return []
    n_frames = len(residuals)
    reference = (n_frames - 1) // 2 if ref_frame is None else ref_frame
    t = np.arange(n_frames, dtype=np.float64)
    validated: list[Candidate] = []
    for candidate in candidates:
        height, width = residuals[0].shape
        if not (2 <= candidate.x < width - 2 and 2 <= candidate.y < height - 2):
            continue
        fluxes: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        for frame_index, frame in enumerate(residuals):
            dt = frame_index - reference
            px, py = candidate.x + candidate.vx * dt, candidate.y + candidate.vy * dt
            if not (1 <= px < frame.shape[1] - 1 and 1 <= py < frame.shape[0] - 1):
                fluxes.append(0.0)
                xs.append(float("nan"))
                ys.append(float("nan"))
                continue
            fluxes.append(aperture_flux(frame, px, py, config.aperture_radius))
            rx, ry = refined_centroid(frame, px, py, config.aperture_radius)
            xs.append(rx)
            ys.append(ry)
        flux_array = np.asarray(fluxes, dtype=np.float64)
        valid_points = np.isfinite(xs) & np.isfinite(ys)
        required_points = max(config.min_detection_frames, int(np.ceil(0.7 * n_frames)))
        if valid_points.sum() < required_points:
            continue
        tv, xv, yv = t[valid_points], np.asarray(xs)[valid_points], np.asarray(ys)[valid_points]
        vx, x0, inliers_x = fit_linear_ransac(tv, xv, tolerance=config.max_track_rms)
        vy, y0, inliers_y = fit_linear_ransac(tv, yv, tolerance=config.max_track_rms)
        good = inliers_x & inliers_y
        if good.sum() < config.min_detection_frames:
            continue
        rms_x = float(np.sqrt(np.mean((xv[good] - (x0 + vx * tv[good])) ** 2)))
        rms_y = float(np.sqrt(np.mean((yv[good] - (y0 + vy * tv[good])) ** 2)))
        positive = np.clip(flux_array[valid_points], 0.0, None)
        total_flux = float(positive.sum())
        if total_flux <= 0:
            continue
        cosmic_fraction = float(positive.max() / total_flux)
        mean_flux = float(positive.mean())
        photometric_cv = float(np.std(positive) / mean_flux) if mean_flux > 0 else np.inf
        n_detected = int((positive > 0.15 * positive.max()).sum())
        if rms_x > config.max_track_rms or rms_y > config.max_track_rms:
            continue
        if cosmic_fraction > config.max_cosmic_fraction:
            continue
        if photometric_cv > config.max_photometric_cv:
            continue
        if n_detected < config.min_detection_frames:
            continue
        validated.append(
            Candidate(
                x=float(x0 + vx * reference),
                y=float(y0 + vy * reference),
                vx=float(np.clip(vx, -config.velocity_max, config.velocity_max)),
                vy=float(np.clip(vy, -config.velocity_max, config.velocity_max)),
                snr=candidate.snr,
                stacked_flux=candidate.stacked_flux,
                peak=candidate.peak,
                n_pixels=candidate.n_pixels,
                ref_frame=reference,
                track_fluxes=tuple(float(f) for f in flux_array),
                track_residual_rms=float(np.sqrt((rms_x**2 + rms_y**2) / 2.0)),
                track_rms_x=rms_x,
                track_rms_y=rms_y,
                photometric_cv=photometric_cv,
                cosmic_fraction=cosmic_fraction,
                velocity_index=candidate.velocity_index,
            )
        )
    return validated
