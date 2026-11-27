"""Generador sintetico calibrado con verdad terrestre exacta."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl
from astropy.io import fits

from errantes.physics import (
    flux_for_snr,
    flux_to_magnitude,
    poisson_gaussian_noise,
    render_sources,
)
from errantes.types import (
    GroundTruth,
    PSFModel,
    SimulatedCube,
    SimulatorConfig,
    Wanderer,
)


@dataclass(frozen=True)
class WandererSpec:
    """Especificacion de entrada de un errante; el flujo se deriva del SNR deseado."""

    object_id: str
    x0: float
    y0: float
    vx: float
    vy: float
    snr_frame: float
    ref_frame: int = 0

    def __post_init__(self) -> None:
        if self.snr_frame <= 0:
            raise ValueError("snr_frame debe ser positivo")
        if self.ref_frame < 0:
            raise ValueError("ref_frame debe ser no negativo")


DEFAULT_WANDERER = WandererSpec(
    object_id="W01", x0=36.0, y0=41.0, vx=0.37, vy=-0.21, snr_frame=2.0, ref_frame=5
)


def simulate(
    config: SimulatorConfig | None = None,
    wanderers: Sequence[WandererSpec] | None = None,
    rng: np.random.Generator | None = None,
) -> SimulatedCube:
    """Simula un cubo de imagenes con ruido, estrellas, artefactos y errantes.

    El flujo de cada errante se calcula para que el SNR del pixel central,
    definido como F * PSF_peak / sqrt(mu_cielo + sigma_read^2), sea exactamente
    ``snr_frame`` en el frame de referencia.
    """
    cfg = config or SimulatorConfig()
    specs = (DEFAULT_WANDERER,) if wanderers is None else tuple(wanderers)
    generator = np.random.default_rng(cfg.rng_seed) if rng is None else rng
    height, width = cfg.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0

    n_stars = cfg.n_stars
    margin = 8
    # Evitamos estrellas superpositionadas con la traza inyectada: en ese caso
    # el residuo de la estrella domina la senal de bajo SNR y el benchmark no
    # evaluaria la capacidad de deteccion sino un caso patologico de confusion.
    track_points = np.array(
        [
            [spec.x0 + spec.vx * t, spec.y0 + spec.vy * t]
            for spec in specs
            for t in range(cfg.n_frames)
        ],
        dtype=np.float64,
    )
    kept_x: list[float] = []
    kept_y: list[float] = []
    attempts = 0
    while len(kept_x) < n_stars and attempts < 200:
        batch_x = generator.uniform(margin, width - margin, size=2 * n_stars)
        batch_y = generator.uniform(margin, height - margin, size=2 * n_stars)
        for x, y in zip(batch_x, batch_y):
            if len(kept_x) >= n_stars:
                break
            if track_points.size and np.min(np.hypot(track_points[:, 0] - x, track_points[:, 1] - y)) < 6.0:
                continue
            kept_x.append(float(x))
            kept_y.append(float(y))
        attempts += 1
    sx = np.array(kept_x, dtype=np.float64)
    sy = np.array(kept_y, dtype=np.float64)
    sflux = np.exp(generator.normal(np.log(cfg.star_flux_median), cfg.star_flux_sigma, size=n_stars))

    defect_columns: tuple[int, ...] = ()
    if cfg.n_defective_columns:
        chosen = generator.choice(np.arange(4, width - 4), size=cfg.n_defective_columns, replace=False)
        defect_columns = tuple(int(c) for c in chosen)

    frames = np.empty((cfg.n_frames, height, width), dtype=np.float32)
    frame_psf_sigma: list[float] = []
    object_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    injected: list[Wanderer] = []

    for spec in specs:
        t_ref = min(spec.ref_frame, cfg.n_frames - 1)
        px, py = spec.x0 + spec.vx * t_ref, spec.y0 + spec.vy * t_ref
        bg_ref = cfg.background + cfg.gradient[0] * (px - cx) + cfg.gradient[1] * (py - cy)
        sigma_ref = float(np.sqrt(bg_ref + cfg.read_noise**2))
        flux = flux_for_snr(spec.snr_frame, sigma_ref, cfg.psf)
        mag = flux_to_magnitude(flux / cfg.gain, cfg.magnitude_zeropoint)
        wanderer = Wanderer(
            object_id=spec.object_id,
            x0=spec.x0,
            y0=spec.y0,
            vx=spec.vx,
            vy=spec.vy,
            snr_frame=spec.snr_frame,
            flux=flux,
            magnitude=mag,
            ref_frame=t_ref,
        )
        injected.append(wanderer)
        object_rows.append(
            {
                "object_id": wanderer.object_id,
                "x0": wanderer.x0,
                "y0": wanderer.y0,
                "vx": wanderer.vx,
                "vy": wanderer.vy,
                "snr_frame": wanderer.snr_frame,
                "flux_e": wanderer.flux,
                "magnitude": wanderer.magnitude,
                "ref_frame": wanderer.ref_frame,
            }
        )

    for t in range(cfg.n_frames):
        sigma_frame = float(
            np.clip(
                cfg.psf.sigma * (1.0 + generator.normal(0.0, cfg.seeing_jitter)),
                0.5 * cfg.psf.sigma,
                1.8 * cfg.psf.sigma,
            )
        )
        frame_psf_sigma.append(sigma_frame)
        psf_t = PSFModel(kind=cfg.psf.kind, sigma=sigma_frame, beta=cfg.psf.beta)
        dx_jitter, dy_jitter = generator.normal(0.0, cfg.guide_jitter, size=2)
        star_xy = np.column_stack((sx + dx_jitter, sy + dy_jitter))
        offset = generator.normal(0.0, cfg.background_drift)
        background = cfg.background + cfg.gradient[0] * (xx - cx) + cfg.gradient[1] * (yy - cy) + offset
        expectation = background + render_sources(cfg.shape, star_xy, sflux, psf_t)

        for col in defect_columns:
            local_sigma = np.sqrt(cfg.background + cfg.read_noise**2)
            bias = generator.uniform(-3.0, 3.0) * local_sigma
            expectation[:, col] += bias * generator.uniform(0.98, 1.02)

        for wanderer in injected:
            wx, wy = wanderer.position(t)
            if 1.0 <= wx < width - 1.0 and 1.0 <= wy < height - 1.0:
                expectation += render_sources(
                    cfg.shape,
                    np.array([[wx, wy]], dtype=np.float64),
                    np.array([wanderer.flux]),
                    psf_t,
                )
            position_rows.append(
                {
                    "object_id": wanderer.object_id,
                    "t": t,
                    "x": wx,
                    "y": wy,
                    "vx": wanderer.vx,
                    "vy": wanderer.vy,
                    "flux_e": wanderer.flux,
                    "magnitude": wanderer.magnitude,
                    "snr_frame": wanderer.snr_frame,
                    "psf_sigma": sigma_frame,
                }
            )

        n_cosmic = int(generator.poisson(cfg.cosmic_rate))
        if n_cosmic:
            cosmic_x = generator.integers(2, width - 2, size=n_cosmic)
            cosmic_y = generator.integers(2, height - 2, size=n_cosmic)
            local_sigma = np.sqrt(
                cfg.background
                + cfg.gradient[0] * (cosmic_x - cx)
                + cfg.gradient[1] * (cosmic_y - cy)
                + cfg.read_noise**2
            )
            amplitudes = generator.uniform(*cfg.cosmic_amplitude) * local_sigma
            expectation[cosmic_y, cosmic_x] += amplitudes

        frames[t] = poisson_gaussian_noise(expectation, cfg.read_noise, cfg.gain, generator).astype(
            np.float32
        )

    truth = GroundTruth(
        objects=pl.DataFrame(object_rows, infer_schema_length=None),
        positions=pl.DataFrame(position_rows, infer_schema_length=None),
    )
    return SimulatedCube(
        frames=frames,
        truth=truth,
        wanderers=tuple(injected),
        config=cfg,
        star_positions=np.column_stack((sx, sy)),
        star_fluxes=sflux,
        frame_psf_sigma=tuple(frame_psf_sigma),
        defect_columns=defect_columns,
    )


def simulate_default_cube(snr_frame: float = 2.0, seed: int = 1234) -> SimulatedCube:
    """Cubo de prueba rapido con un unico errante de SNR configurable."""
    config = SimulatorConfig(rng_seed=seed)
    spec = WandererSpec(
        object_id="W01", x0=36.0, y0=41.0, vx=0.37, vy=-0.21, snr_frame=snr_frame, ref_frame=5
    )
    return simulate(config, (spec,))


def write_fits_cube(cube: SimulatedCube, path: str | Path) -> Path:
    """Escribe el cubo en FITS con WCS lineal y tabla de verdad terrestre."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["BUNIT"] = "ADU"
    header["GAIN"] = cube.config.gain
    header["NFRAMES"] = cube.n_frames
    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    primary = fits.PrimaryHDU(data=cube.frames, header=header)
    truth = cube.truth.positions
    columns = [
        fits.Column(name="OBJECT_ID", format="8A", array=np.array(truth["object_id"].to_list(), dtype="S8")),
        fits.Column(name="T", format="J", array=truth["t"].to_numpy()),
        fits.Column(name="X", format="D", array=truth["x"].to_numpy()),
        fits.Column(name="Y", format="D", array=truth["y"].to_numpy()),
        fits.Column(name="VX", format="D", array=truth["vx"].to_numpy()),
        fits.Column(name="VY", format="D", array=truth["vy"].to_numpy()),
        fits.Column(name="SNR", format="D", array=truth["snr_frame"].to_numpy()),
        fits.Column(name="FLUX_E", format="D", array=truth["flux_e"].to_numpy()),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="TRUTH")
    fits.HDUList([primary, table]).writeto(output, overwrite=True)
    return output


def read_fits_cube(path: str | Path) -> tuple[np.ndarray, pl.DataFrame, fits.Header]:
    """Lee un cubo FITS escrito por :func:`write_fits_cube`."""
    with fits.open(path) as hdul:
        frames = np.array(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
        table = pl.DataFrame(
            {
                "object_id": np.array(hdul["TRUTH"].data["OBJECT_ID"]).astype(str),
                "t": hdul["TRUTH"].data["T"],
                "x": hdul["TRUTH"].data["X"],
                "y": hdul["TRUTH"].data["Y"],
                "vx": hdul["TRUTH"].data["VX"],
                "vy": hdul["TRUTH"].data["VY"],
                "snr_frame": hdul["TRUTH"].data["SNR"],
                "flux_e": hdul["TRUTH"].data["FLUX_E"],
            }
        )
    return frames, table, header
