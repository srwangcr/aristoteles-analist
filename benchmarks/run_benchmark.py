"""Benchmark reproducible de sensibilidad y coste computacional.

Uso:
    python benchmarks/run_benchmark.py --seeds 11 12 13 --output benchmarks/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from errantes.metrics import benchmark_snr_sweep, summarize_benchmark, write_benchmark
from errantes.pipeline import run_pipeline
from errantes.simulator import simulate_default_cube
from errantes.types import PipelineConfig


def measure_velocity_scaling(
    seeds: tuple[int, ...] = (11, 12, 13),
    steps: tuple[float, ...] = (0.5, 0.25, 0.125),
) -> pl.DataFrame:
    """Mide el tiempo total en funcion del numero de hipotesis de velocidad."""
    rows: list[dict[str, float | int]] = []
    for step in steps:
        config = PipelineConfig(velocity_step=step)
        for seed in seeds:
            cube = simulate_default_cube(3.0, seed)
            result = run_pipeline(cube, config)
            rows.append(
                {
                    "velocity_step": step,
                    "n_velocities": config.n_velocities,
                    "runtime_s": result.runtime_s,
                }
            )
    return (
        pl.DataFrame(rows)
        .group_by("velocity_step", "n_velocities")
        .agg(pl.col("runtime_s").mean().alias("runtime_s"))
        .sort("n_velocities")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snrs", type=float, nargs="+", default=[1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--scaling", action="store_true", help="incluye escalado con la grilla")
    args = parser.parse_args()

    results = benchmark_snr_sweep(snrs=args.snrs, seeds=tuple(args.seeds))
    summary = summarize_benchmark(results)
    parquet, json_path = write_benchmark(results, args.output)
    summary.write_parquet(args.output / "summary.parquet")
    summary.write_json(args.output / "summary.json")
    print(summary)
    print(f"\nGuardado: {parquet} / {json_path}")
    if args.scaling:
        scaling = measure_velocity_scaling(tuple(args.seeds[:3]))
        scaling.write_parquet(args.output / "scaling.parquet")
        print("\nEscalado con el numero de velocidades:")
        print(scaling)


if __name__ == "__main__":
    main()
