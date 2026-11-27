"""Pipeline modular de deteccion de objetos moviles en cubos FITS de bajo SNR."""

from errantes.types import (
    Candidate,
    GroundTruth,
    PipelineConfig,
    PSFModel,
    SimulatedCube,
    SimulatorConfig,
    Wanderer,
)
from errantes.simulator import simulate, simulate_default_cube, write_fits_cube, read_fits_cube
from errantes.pipeline import run_pipeline

__all__ = [
    "Candidate",
    "GroundTruth",
    "PipelineConfig",
    "PSFModel",
    "SimulatedCube",
    "SimulatorConfig",
    "Wanderer",
    "simulate",
    "simulate_default_cube",
    "write_fits_cube",
    "read_fits_cube",
    "run_pipeline",
]
