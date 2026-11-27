"""Estructuras de datos inmutables y configuraciones tipadas del pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PSFModel:
    """Modelo de funcion de dispersion de punto (PSF).

    Para Moffat, ``alpha`` se deriva de ``sigma`` preservando la misma FWHM que
    una Gaussiana con desviacion ``sigma``.
    """

    kind: Literal["gaussian", "moffat"] = "gaussian"
    sigma: float = 1.35
    beta: float = 3.5

    def __post_init__(self) -> None:
        if self.kind not in ("gaussian", "moffat"):
            raise ValueError(f"PSF desconocida: {self.kind}")
        if not 0.4 <= self.sigma <= 10.0:
            raise ValueError("sigma debe estar en [0.4, 10]")
        if self.beta <= 1.0:
            raise ValueError("beta debe ser > 1")

    @property
    def alpha(self) -> float:
        """Parametro de escala de Moffat con FWHM equivalente a la Gaussiana."""
        return float(self.sigma * 2.0 * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0))


@dataclass(frozen=True)
class SimulatorConfig:
    """Parametros fisicos del simulador. Todas las intensidades estan en electrones."""

    n_frames: int = 12
    shape: tuple[int, int] = (96, 96)  # (altura, anchura)
    background: float = 120.0
    gradient: tuple[float, float] = (0.35, -0.20)  # electrones / pixel
    background_drift: float = 0.6  # sigma del offset de cielo por frame
    read_noise: float = 5.0  # sigma del ruido de lectura, electrones
    gain: float = 1.0  # electrones / ADU (invertido al leer)
    psf: PSFModel = field(default_factory=PSFModel)
    seeing_jitter: float = 0.10  # dispersion relativa de sigma por frame
    guide_jitter: float = 0.12  # dispersion del jitter de guiado en pixeles
    n_stars: int = 30
    star_flux_median: float = 1600.0
    star_flux_sigma: float = 0.65  # sigma del log de flujo
    cosmic_rate: float = 3.0  # rayos cosmicos esperados por frame
    cosmic_amplitude: tuple[float, float] = (12.0, 35.0)  # multiples de sigma local
    n_defective_columns: int = 2
    magnitude_zeropoint: float = 25.0
    rng_seed: int = 1234

    def __post_init__(self) -> None:
        if self.n_frames < 3:
            raise ValueError("se requieren al menos 3 frames")
        if min(self.shape) < 32:
            raise ValueError("la imagen debe ser al menos 32x32")
        if self.background <= 1 or self.read_noise < 0:
            raise ValueError("fondo y ruido de lectura deben ser positivos")


@dataclass(frozen=True)
class Wanderer:
    """Errante con trayectoria lineal y flujo constante por frame."""

    object_id: str
    x0: float
    y0: float
    vx: float
    vy: float
    snr_frame: float
    flux: float  # electrones totales por frame
    magnitude: float
    ref_frame: int = 0

    def position(self, t: int) -> tuple[float, float]:
        """Posicion (x, y) en el frame ``t``."""
        return self.x0 + self.vx * t, self.y0 + self.vy * t


@dataclass(frozen=True)
class GroundTruth:
    """Catalogo de verdad terrestre: objetos y posiciones por frame."""

    objects: pl.DataFrame
    positions: pl.DataFrame

    def write(self, directory: str | Path) -> tuple[Path, Path, Path, Path]:
        """Escribe JSON y Parquet de objetos y posiciones; devuelve las rutas."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        paths = (
            root / "objects.json",
            root / "objects.parquet",
            root / "positions.json",
            root / "positions.parquet",
        )
        self.objects.write_json(paths[0])
        self.objects.write_parquet(paths[1])
        self.positions.write_json(paths[2])
        self.positions.write_parquet(paths[3])
        return paths

    @classmethod
    def read(cls, directory: str | Path) -> GroundTruth:
        """Lee un catalogo previamente escrito por :meth:`write`."""
        root = Path(directory)
        return cls(
            objects=pl.read_json(root / "objects.json"),
            positions=pl.read_json(root / "positions.json"),
        )


@dataclass(frozen=True)
class SimulatedCube:
    """Cubo sintetico mas su verdad terrestre y metadatos de calibracion."""

    frames: np.ndarray  # (N, H, W), float32
    truth: GroundTruth
    wanderers: tuple[Wanderer, ...]
    config: SimulatorConfig
    star_positions: np.ndarray  # (M, 2), posiciones (x, y) de estrellas
    star_fluxes: np.ndarray  # (M,), electrones por frame
    frame_psf_sigma: tuple[float, ...]
    defect_columns: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True)
class Candidate:
    """Candidato a errante, con coordenadas referidas al frame central."""

    x: float
    y: float
    vx: float
    vy: float
    snr: float
    stacked_flux: float
    peak: float
    n_pixels: int
    ref_frame: int
    track_fluxes: tuple[float, ...] = ()
    track_residual_rms: float = float("nan")
    track_rms_x: float = float("nan")
    track_rms_y: float = float("nan")
    photometric_cv: float = float("nan")
    cosmic_fraction: float = float("nan")
    velocity_index: int = -1


@dataclass(frozen=True)
class PipelineConfig:
    """Hiperparametros del preprocesamiento, barrido y filtrado."""

    mesh_size: int = 16
    background_sigma: float = 3.0
    background_iterations: int = 4
    align: bool = True
    alignment_upsample: int = 8
    normalize_flux: bool = True
    match_seeing: bool = True
    velocity_max: float = 1.5
    velocity_step: float = 0.25
    detection_sigma: float = 5.0
    match_filter_sigma: float = 0.8  # filtromatched Gaussiano sobre el mapa apilado
    min_component_pixels: int = 2
    dedup_radius: float = 3.0
    aperture_radius: float = 2.2
    max_track_rms: float = 0.9
    max_cosmic_fraction: float = 0.55
    max_photometric_cv: float = 0.65
    min_detection_frames: int = 3
    backend: str = "auto"  # auto | cpu | gpu
    v_min: float = 0.08  # px/frame; descarta hipótesis estacionarias
    r_nms: float = 5.0  # radio de supresión no-máxima conjunta (x, y, vx, vy)
    star_sigma: float = 5.0  # MAD por encima del cual enmascarar estrellas
    star_mask_radius: int = 3  # dilatación del halo estelar
    star_core_sigma: float = 100.0  # contraste del núcleo (protege fuentes lentas)
    temporal_median: bool = True  # restar mediana temporal antes del stack
    diagnostic_radius: float = 2.5  # 5x5

    @property
    def velocities(self) -> np.ndarray:
        """Grilla de hipotesis (vx, vy) de forma (M, 2)."""
        axis = np.arange(
            -self.velocity_max, self.velocity_max + self.velocity_step / 2, self.velocity_step
        )
        gx, gy = np.meshgrid(axis, axis)
        return np.column_stack((gx.ravel(), gy.ravel()))

    @property
    def valid_velocities(self) -> np.ndarray:
        """Grilla de velocidades sin hipótesis estacionarias (`v_min`)."""
        grid = self.velocities
        return grid[np.hypot(grid[:, 0], grid[:, 1]) >= self.v_min]

    @property
    def n_velocities(self) -> int:
        return int(len(self.velocities))
