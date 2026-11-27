"""Asociacion verdad-deteccion, metricas cientificas y benchmarks reproducibles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from errantes.pipeline import run_pipeline
from errantes.simulator import WandererSpec, simulate
from errantes.types import Candidate, PipelineConfig, SimulatorConfig


@dataclass(frozen=True)
class MatchResult:
    """Resultado de la asociacion entre errantes reales y candidatos."""

    tp: int
    fp: int
    fn: int
    pairs: tuple[tuple[str, Candidate], ...]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def match_detections(
    truth_positions: pl.DataFrame,
    candidates: Sequence[Candidate],
    ref_frame: int,
    position_tolerance: float = 2.5,
    velocity_tolerance: float = 0.45,
) -> MatchResult:
    """Asocia candidatos a errantes usando el algoritmo hungaro (coste minimo)."""
    rows = truth_positions.filter(pl.col("t") == ref_frame)
    if rows.height == 0:
        rows = truth_positions.with_columns(
            (pl.col("x") - pl.col("vx") * (pl.col("t") - ref_frame)).alias("x"),
            (pl.col("y") - pl.col("vy") * (pl.col("t") - ref_frame)).alias("y"),
        )
        rows = rows.filter(pl.col("t") == truth_positions["t"].min())
    truth_ids = rows["object_id"].to_list()
    if not candidates or not truth_ids:
        return MatchResult(tp=0, fp=len(candidates), fn=len(truth_ids), pairs=())
    tx = rows["x"].to_numpy()
    ty = rows["y"].to_numpy()
    tvx = rows["vx"].to_numpy()
    tvy = rows["vy"].to_numpy()
    cx = np.array([c.x for c in candidates])
    cy = np.array([c.y for c in candidates])
    cvx = np.array([c.vx for c in candidates])
    cvy = np.array([c.vy for c in candidates])
    position_cost = np.hypot(tx[:, None] - cx[None, :], ty[:, None] - cy[None, :])
    velocity_cost = np.hypot(tvx[:, None] - cvx[None, :], tvy[:, None] - cvy[None, :])
    cost = position_cost / position_tolerance + velocity_cost / velocity_tolerance
    cost[cost > 2.0] = 1e6
    truth_indices, candidate_indices = linear_sum_assignment(cost)
    pairs: list[tuple[str, Candidate]] = []
    used_candidates: set[int] = set()
    for i, j in zip(truth_indices, candidate_indices):
        if cost[i, j] < 1e5:
            pairs.append((str(truth_ids[i]), candidates[int(j)]))
            used_candidates.add(int(j))
    fp = len(candidates) - len(used_candidates)
    return MatchResult(
        tp=len(pairs), fp=fp, fn=len(truth_ids) - len(pairs), pairs=tuple(pairs)
    )


def roc_pr_curve(
    scores: Sequence[float], labels: Sequence[bool], n_negatives: int | None = None
) -> pl.DataFrame:
    """Curvas ROC y precision-recall al barrer el umbral de score.

    ``n_negatives`` permite expresar FPR como falsos positivos por celda de
    busqueda cuando se conoce el tamano del espacio de hipotesis.
    """
    if len(scores) != len(labels):
        raise ValueError("scores y labels deben tener la misma longitud")
    positives = sum(labels)
    negatives = len(labels) - positives if n_negatives is None else n_negatives
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    tp = fp = 0
    points: list[dict[str, float]] = [
        {"threshold": float("inf"), "tpr": 0.0, "fpr": 0.0, "precision": 1.0, "recall": 0.0}
    ]
    for index in order:
        if labels[index]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        points.append(
            {
                "threshold": float(scores[index]),
                "tpr": recall,
                "fpr": fp / negatives if negatives > 0 else 0.0,
                "precision": precision,
                "recall": recall,
            }
        )
    return pl.DataFrame(points)


def area_under_curve(curve: pl.DataFrame, x: str, y: str) -> float:
    """Area bajo una curva muestreada (trapezoides)."""
    xs = curve[x].to_numpy()
    ys = curve[y].to_numpy()
    return float(np.trapezoid(ys, xs))


def benchmark_snr_sweep(
    snrs: Sequence[float] = (1.5, 2.0, 3.0, 5.0),
    seeds: Sequence[int] = (11, 22, 33),
    simulator_config: SimulatorConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> pl.DataFrame:
    """Mide precision/recall/F1 y tiempo para cada SNR por frame."""
    sim_cfg = simulator_config or SimulatorConfig()
    pipe_cfg = pipeline_config or PipelineConfig()
    rows: list[dict[str, float | int]] = []
    for snr in snrs:
        for seed in seeds:
            config = SimulatorConfig(**{**sim_cfg.__dict__, "rng_seed": int(seed)})
            spec = WandererSpec(
                object_id="W01", x0=36.0, y0=41.0, vx=0.37, vy=-0.21, snr_frame=float(snr), ref_frame=5
            )
            cube = simulate(config, (spec,))
            result = run_pipeline(cube, pipe_cfg)
            matched = match_detections(cube.truth.positions, result.candidates, ref_frame=5)
            row: dict[str, float | int] = {
                "snr_frame": float(snr),
                "seed": int(seed),
                "runtime_s": result.runtime_s,
                "n_velocity_hypotheses": pipe_cfg.n_velocities,
                **matched.as_dict(),
            }
            rows.append(row)
    return pl.DataFrame(rows)


def summarize_benchmark(results: pl.DataFrame) -> pl.DataFrame:
    """Promedia metricas por regimen de SNR."""
    return results.group_by("snr_frame").agg(
        pl.col("precision").mean().alias("precision"),
        pl.col("recall").mean().alias("recall"),
        pl.col("f1").mean().alias("f1"),
        pl.col("runtime_s").mean().alias("runtime_s"),
        pl.col("fp").mean().alias("fp_mean"),
    ).sort("snr_frame")


def write_benchmark(results: pl.DataFrame, directory: str | Path) -> tuple[Path, Path]:
    """Guarda el benchmark en Parquet y JSON."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    parquet, json_path = root / "benchmark.parquet", root / "benchmark.json"
    results.write_parquet(parquet)
    results.write_json(json_path)
    return parquet, json_path
