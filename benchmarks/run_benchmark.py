"""Frozen-schema benchmark runner.

The runner produces one CSV row per (cell, method) pair. Rows from a
single cell are written incrementally so partial output is durable. The
schema is fixed by ``docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md``
section 8 and section 7.3.

Usage:

    uv run python benchmarks/run_benchmark.py --split training \\
        --output data/bench_training.csv

    uv run python benchmarks/run_benchmark.py --split holdout \\
        --output data/bench_holdout.csv --reps 3 \\
        --methods weighted_sparse,adaptive_threshold,sparcc_closed_form

The runner does NOT regenerate figures or claim that any estimator
clears the acceptance gates. Those decisions happen elsewhere, after
this CSV has been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

# Allow `python benchmarks/run_benchmark.py` from the project root by
# putting the repo on sys.path before the benchmarks/* imports below.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from benchmarks.baselines import AVAILABLE_PYTHON_BASELINES, BaselineResult
from benchmarks.r_adapters import R_BASELINES, RAdapterResult, RAdapterSkip, run_r_baseline
from falcon import infer_network
from falcon.sim import (
    auroc_score,
    average_precision_score,
    fdr_at_target,
    generate_scenario,
    holdout_grid,
    precision_at_k,
    recall_at_k,
    training_grid,
)

SCHEMA_FIELDS = (
    "scenario",
    "split",
    "seed",
    "n",
    "p",
    "density",
    "zero_fraction",
    "distribution",
    "method",
    "estimand_family",
    "auroc",
    "average_precision",
    "recall_at_k",
    "precision_at_k",
    "fdr_at_target",
    "wallclock_seconds",
    "peak_bytes",
    "converged",
    "lambda_value",
    "iterations",
    "min_eigenvalue",
    "skip_reason",
)


def _selected_mask_from_correlation(corr: np.ndarray, target: float = 0.05) -> np.ndarray:
    """Stand-in 'selected' mask used to compute fdr_at_target until real
    calibration ships. Selects the top ``target`` fraction of off-diagonal
    entries by absolute correlation. Real calibration replaces this with
    a thresholded q-value mask."""
    p = corr.shape[0]
    iu, ju = np.triu_indices(p, k=1)
    s = np.abs(corr[iu, ju])
    if s.size == 0:
        return np.zeros_like(corr, dtype=bool)
    k = max(1, int(round(target * s.size)))
    cutoff = np.partition(-s, k - 1)[k - 1]
    cutoff = -cutoff
    mask = np.zeros_like(corr, dtype=bool)
    flag = np.abs(corr) >= cutoff
    np.fill_diagonal(flag, False)
    mask = flag
    return mask


def _row(scenario, method, estimand_family, correlation, support, wall, peak, converged, lambda_value, iterations, min_eig, skip_reason=""):
    if correlation is None:
        auc = ap = rk = pk = fdr = float("nan")
    else:
        k_top = max(1, int(round(0.05 * support.shape[0] * (support.shape[0] - 1) / 2)))
        auc = auroc_score(correlation, support)
        ap = average_precision_score(correlation, support)
        rk = recall_at_k(correlation, support, k_top)
        pk = precision_at_k(correlation, support, k_top)
        sel = _selected_mask_from_correlation(correlation, target=0.05)
        fdr = fdr_at_target(correlation, support, sel)
    md = scenario.metadata
    return {
        "scenario": md["scenario"],
        "split": md.get("split", ""),
        "seed": md["seed"],
        "n": md["n"],
        "p": md["p"],
        "density": md["density"],
        "zero_fraction": md["zero_fraction"],
        "distribution": md.get("distribution", "lognormal"),
        "method": method,
        "estimand_family": estimand_family,
        "auroc": auc,
        "average_precision": ap,
        "recall_at_k": rk,
        "precision_at_k": pk,
        "fdr_at_target": fdr,
        "wallclock_seconds": wall,
        "peak_bytes": peak,
        "converged": converged,
        "lambda_value": lambda_value,
        "iterations": iterations,
        "min_eigenvalue": min_eig,
        "skip_reason": skip_reason,
    }


def _time_and_run(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, wall, peak


def run_falcon_estimator(scenario, estimator: str, *, n_resamples: int = 0):
    md = scenario.metadata
    md["split"] = md.get("split", "training")

    def go():
        return infer_network(
            scenario.counts,
            estimator=estimator,
            zero_policy="multiplicative",
            selection="stability" if n_resamples > 0 else "none",
            n_resamples=max(n_resamples, 1),
            seed=int(scenario.metadata["seed"]),
        )

    result, wall, peak = _time_and_run(go)
    return _row(
        scenario,
        method=f"falcon_{estimator}",
        estimand_family="latent_log_abundance_correlation",
        correlation=result.correlation,
        support=scenario.support,
        wall=wall,
        peak=peak,
        converged=result.diagnostics.converged,
        lambda_value=result.diagnostics.lambda_value,
        iterations=result.diagnostics.iterations,
        min_eig=result.diagnostics.min_eigenvalue,
    )


def run_python_baseline(scenario, name: str):
    fn = AVAILABLE_PYTHON_BASELINES[name]
    out, wall, peak = _time_and_run(lambda: fn(scenario.counts))
    return _row(
        scenario,
        method=name,
        estimand_family=out.estimand_family,
        correlation=out.correlation,
        support=scenario.support,
        wall=wall,
        peak=peak,
        converged=out.converged,
        lambda_value=float("nan"),
        iterations=out.iterations,
        min_eig=float("nan"),
    )


def run_r_baseline_row(scenario, name: str):
    out, wall, peak = _time_and_run(lambda: run_r_baseline(name, scenario.counts))
    if isinstance(out, RAdapterSkip):
        return _row(
            scenario,
            method=name,
            estimand_family="latent_log_abundance_correlation",
            correlation=None,
            support=scenario.support,
            wall=wall,
            peak=peak,
            converged=False,
            lambda_value=float("nan"),
            iterations=0,
            min_eig=float("nan"),
            skip_reason=out.reason,
        )
    return _row(
        scenario,
        method=name,
        estimand_family=out.estimand_family,
        correlation=out.correlation,
        support=scenario.support,
        wall=wall,
        peak=peak,
        converged=out.converged,
        lambda_value=float("nan"),
        iterations=out.iterations,
        min_eig=float("nan"),
    )


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("training", "holdout"), default="training")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--methods",
        default="falcon_weighted_sparse,falcon_adaptive_threshold,falcon_pd_sparse,sparcc_closed_form,pearson_clr",
    )
    p.add_argument("--reps", type=int, default=1, help="how many seed reps per cell")
    p.add_argument("--n-resamples", type=int, default=0)
    return p.parse_args(argv)


def _open_writer(path: str):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=SCHEMA_FIELDS)
    if new:
        writer.writeheader()
    return fh, writer


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cells = training_grid() if args.split == "training" else holdout_grid()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fh, writer = _open_writer(args.output)
    try:
        for cell in cells:
            for rep in range(args.reps):
                seed_offset = rep * 1009
                scenario = generate_scenario(
                    cell.scenario,
                    n=cell.n,
                    p=cell.p,
                    seed=cell.seed + seed_offset,
                    density=cell.density,
                    edge_strength=cell.edge_strength,
                    depth=cell.depth,
                )
                scenario.metadata["split"] = cell.split
                for method in methods:
                    if method.startswith("falcon_"):
                        row = run_falcon_estimator(
                            scenario,
                            method.removeprefix("falcon_"),
                            n_resamples=args.n_resamples,
                        )
                    elif method in AVAILABLE_PYTHON_BASELINES:
                        row = run_python_baseline(scenario, method)
                    elif method in R_BASELINES:
                        row = run_r_baseline_row(scenario, method)
                    else:
                        raise SystemExit(f"unknown method {method!r}")
                    writer.writerow(row)
                    fh.flush()
    finally:
        fh.close()


if __name__ == "__main__":
    run(sys.argv[1:])
