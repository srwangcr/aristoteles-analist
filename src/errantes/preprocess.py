"""Fondo robusto, alineamiento sub-pixel y sustraccion de fuentes estacionarias."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from errantes.types import PipelineConfig


@dataclass(frozen=True)
class PreprocessedCube:
    """Resultado intermediates del preprocesamiento."""

    backgrounds: np.ndarray  # (N, H, W)
    aligned: np.ndarray  # (N, H, W), fondo eliminado y alineado
    reference: np.ndarray  # (H, W), mediana temporal
    residuals: np.ndarray  # (N, H, W)
    shifts: np.ndarray  # (N, 2), (dy, dx) aplicados a cada frame


def _robust_level(values: np.ndarray, nsigma: float, max_iter: int) -> tuple[float, float]:
    """Mediana + MAD con sigma-clipping iterativo."""
    if values.size == 0:
        return 0.0, 1.0
    centre = float(np.median(values))
    scatter = float(1.4826 * np.median(np.abs(values - centre)))
    for _ in range(max_iter):
        if scatter <= 0:
            break
        keep = np.abs(values - centre) <= nsigma * scatter
        if keep.sum() < max(4, values.size // 10):
            break
        new_centre = float(np.median(values[keep]))
        new_scatter = float(1.4826 * np.median(np.abs(values[keep] - new_centre)))
        if abs(new_centre - centre) < 1e-4 and abs(new_scatter - scatter) < 1e-4:
            centre, scatter = new_centre, new_scatter
            break
        centre, scatter = new_centre, new_scatter
    return centre, max(scatter, 1e-6)


def estimate_background(
    image: np.ndarray,
    mesh_size: int = 16,
    nsigma: float = 3.0,
    max_iter: int = 4,
) -> np.ndarray:
    """Malla 2D de mediana/MAD recortada e interpolada bilinealmente."""
    height, width = image.shape
    ny, nx = int(np.ceil(height / mesh_size)), int(np.ceil(width / mesh_size))
    grid = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        y0, y1 = j * mesh_size, min(height, (j + 1) * mesh_size)
        for i in range(nx):
            x0, x1 = i * mesh_size, min(width, (i + 1) * mesh_size)
            grid[j, i], _ = _robust_level(image[y0:y1, x0:x1].ravel(), nsigma, max_iter)
    gy = np.clip((np.arange(height) + 0.5) / mesh_size - 0.5, 0, ny - 1)
    gx = np.clip((np.arange(width) + 0.5) / mesh_size - 0.5, 0, nx - 1)
    coords_y = gy[:, None] * np.ones((1, width))
    coords_x = np.ones((height, 1)) * gx[None, :]
    return np.asarray(
        ndimage.map_coordinates(grid, [coords_y, coords_x], order=1, mode="nearest"),
        dtype=np.float64,
    )


def _parabolic_offset(a: float, b: float, c: float) -> float:
    """Desplazamiento sub-pixel del maximo de una parabola (-1, 1)."""
    denominator = a - 2.0 * b + c
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denominator, -1.0, 1.0))


def phase_correlation(
    reference: np.ndarray, target: np.ndarray, upsample: int = 8
) -> tuple[float, float]:
    """Desplazamiento (dy, dx) de ``target`` respecto a ``reference``.

    Se usa correlacion de fase con zero-padding y ajuste parabolico del pico,
    por lo que la precision es mejor que un pixel incluso con SNR bajo.
    """
    if reference.shape != target.shape:
        raise ValueError("las imagenes deben tener el mismo tamano")
    height, width = reference.shape
    u = max(1, int(upsample))
    ph, pw = height * u, width * u
    window = np.outer(np.hanning(height), np.hanning(width))
    ref_pad = np.zeros((ph, pw), dtype=np.float64)
    tar_pad = np.zeros((ph, pw), dtype=np.float64)
    ref_pad[:height, :width] = reference * window
    tar_pad[:height, :width] = target * window
    cross = np.fft.irfft2(
        np.fft.rfft2(tar_pad) * np.conj(np.fft.rfft2(ref_pad)), s=(ph, pw)
    )
    power = np.abs(cross)
    py, px = np.unravel_index(int(np.argmax(power)), power.shape)
    # El zero-padding interpola la funcion de correlacion, pero no escala las
    # coordenadas: un desplazamiento de d pixeles aparece en el indice d.
    y_raw = py if py <= ph // 2 else py - ph
    x_raw = px if px <= pw // 2 else px - pw
    dy = float(y_raw)
    dx = float(x_raw)
    dy += _parabolic_offset(
        power[(py - 1) % ph, px], power[py, px], power[(py + 1) % ph, px]
    )
    dx += _parabolic_offset(
        power[py, (px - 1) % pw], power[py, px], power[py, (px + 1) % pw]
    )
    return float(dy), float(dx)


def align_frames(
    images: np.ndarray, ref_index: int = 0, upsample: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Alinea todos los frames al de referencia mediante correlacion de fase."""
    n_frames = len(images)
    reference = images[ref_index] - np.median(images[ref_index])
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    aligned = np.empty_like(images, dtype=np.float64)
    aligned[ref_index] = images[ref_index] - np.median(images[ref_index])
    for t in range(n_frames):
        if t == ref_index:
            continue
        image = images[t] - np.median(images[t])
        dy, dx = phase_correlation(reference, image, upsample=upsample)
        shifts[t] = (dy, dx)
        aligned[t] = ndimage.shift(
            image, shift=(-dy, -dx), order=3, mode="constant", cval=0.0, prefilter=True
        )
    return aligned, shifts


def subtract_reference(
    aligned: np.ndarray,
    frame_psf_sigma: tuple[float, ...] | None = None,
    reference_sigma: float | None = None,
    normalize: bool = True,
    match_seeing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la imagen maestra y resta cada frame (normalizado por flujo y PSF)."""
    n_frames = len(aligned)
    working = aligned.copy()
    if normalize:
        master = np.median(working, axis=0)
        bright = master > np.quantile(master, 0.985)
        for t in range(n_frames):
            if bright.sum() < 4:
                break
            ratio = float(np.median(working[t][bright] / np.maximum(master[bright], 1e-6)))
            if np.isfinite(ratio) and ratio > 0:
                working[t] /= float(np.clip(ratio, 0.8, 1.25))
    reference = np.median(working, axis=0)
    # La mediana temporal contiene tambien el rastro de los errantes. Restarla
    # en todo el campo produce auto-sustraccion severa para velocidades
    # sub-pixel, por lo que solo se resta donde hay fuentes estacionarias
    # detectadas en la imagen maestra (con un borde suave alrededor).
    ref_sigma = float(1.4826 * np.median(np.abs(reference - np.median(reference))))
    star_kernel = np.ones((3, 3), dtype=bool)
    star_mask = ndimage.binary_dilation(
        reference > max(4.0 * ref_sigma, 3.0), structure=star_kernel, iterations=2
    )
    star_weight = ndimage.gaussian_filter(star_mask.astype(np.float64), sigma=0.8)
    residuals = np.empty_like(working, dtype=np.float64)
    can_match = match_seeing and frame_psf_sigma is not None
    base_sigma = reference_sigma
    if can_match and base_sigma is None:
        base_sigma = float(np.median(frame_psf_sigma))  # type: ignore[arg-type]
    for t in range(n_frames):
        ref_t = reference
        if can_match:
            assert frame_psf_sigma is not None
            extra = float(np.sqrt(max(frame_psf_sigma[t] ** 2 - base_sigma**2, 0.0)))  # type: ignore[operator]
            if extra > 0.05:
                ref_t = ndimage.gaussian_filter(reference, sigma=extra, mode="nearest")
        residuals[t] = working[t] - ref_t * star_weight
    return residuals, reference


def preprocess_cube(
    frames: np.ndarray,
    config: PipelineConfig,
    frame_psf_sigma: tuple[float, ...] | None = None,
) -> PreprocessedCube:
    """Ejecuta fondo robusto, alineamiento y sustraccion de estrellas."""
    cube = np.asarray(frames, dtype=np.float64)
    backgrounds = np.stack(
        [
            estimate_background(
                frame,
                mesh_size=config.mesh_size,
                nsigma=config.background_sigma,
                max_iter=config.background_iterations,
            )
            for frame in cube
        ]
    )
    cleaned = cube - backgrounds
    ref_index = (len(cleaned) - 1) // 2
    if config.align:
        aligned, shifts = align_frames(cleaned, ref_index=ref_index, upsample=config.alignment_upsample)
    else:
        aligned, shifts = cleaned.copy(), np.zeros((len(cleaned), 2))
    residuals, reference = subtract_reference(
        aligned,
        frame_psf_sigma=frame_psf_sigma,
        normalize=config.normalize_flux,
        match_seeing=config.match_seeing,
    )
    return PreprocessedCube(
        backgrounds=backgrounds,
        aligned=aligned,
        reference=reference,
        residuals=residuals,
        shifts=shifts,
    )
