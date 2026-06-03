"""Differential test: weighted_sparse must not regress AUROC/AP per-cell
by more than 0.005 (or mean by more than 0.001) against the pinned
baseline. Runs on the 39 training cells at n_resamples=30.

The baseline at tests/baselines/weighted_sparse_baseline.csv was
captured by tests/baselines/generate_weighted_sparse_baseline.py before
any Line A optimization. Re-pinning the baseline is an explicit
decision-log event, not a quiet test update.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from falcon import infer_network
from falcon.sim import (
    auroc_score,
    average_precision_score,
    generate_scenario,
    training_grid,
)


PER_CELL_TOLERANCE = 0.005
MEAN_TOLERANCE = 0.001
N_RESAMPLES = 30


def _baseline_index() -> dict:
    path = Path(__file__).resolve().parent / "baselines" / "weighted_sparse_baseline.csv"
    if not path.exists():
        pytest.skip(f"baseline missing at {path}; run generator before this test")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {
            (r["scenario"], int(r["n"]), int(r["p"]), int(r["seed"])): {
                "auroc": float(r["auroc"]),
                "ap": float(r["ap"]),
            }
            for r in reader
        }


@pytest.mark.diff_baseline
def test_weighted_sparse_does_not_regress_against_baseline():
    baseline = _baseline_index()
    auroc_deltas = []
    ap_deltas = []
    failing_cells = []
    for cell in training_grid():
        md = cell.metadata()
        key = (md["scenario"], md["n"], md["p"], md["seed"])
        if key not in baseline:
            pytest.skip(f"baseline missing cell {key}; re-pin required")
        scenario = generate_scenario(
            md["scenario"], n=md["n"], p=md["p"], seed=md["seed"],
            density=md["density"], edge_strength=md["edge_strength"],
            depth=md["depth"],
        )
        result = infer_network(
            scenario.counts,
            estimator="weighted_sparse",
            selection="stability",
            n_resamples=N_RESAMPLES,
            seed=md["seed"],
        )
        a = auroc_score(result.correlation, scenario.support)
        p = average_precision_score(result.correlation, scenario.support)
        d_a = abs(a - baseline[key]["auroc"])
        d_p = abs(p - baseline[key]["ap"])
        auroc_deltas.append(d_a)
        ap_deltas.append(d_p)
        if d_a > PER_CELL_TOLERANCE or d_p > PER_CELL_TOLERANCE:
            failing_cells.append(
                (key, d_a, d_p, baseline[key]["auroc"], a, baseline[key]["ap"], p)
            )
    assert not failing_cells, (
        f"{len(failing_cells)} cell(s) exceed per-cell tolerance ({PER_CELL_TOLERANCE}):\n"
        + "\n".join(
            f"  {k} ΔAUROC={da:.4f} ΔAP={dp:.4f} (baseline {ba:.4f}->{a:.4f}, {bp:.4f}->{p:.4f})"
            for (k, da, dp, ba, a, bp, p) in failing_cells
        )
    )
    mean_da = sum(auroc_deltas) / len(auroc_deltas)
    mean_dp = sum(ap_deltas) / len(ap_deltas)
    assert mean_da <= MEAN_TOLERANCE, (
        f"mean ΔAUROC {mean_da:.5f} exceeds {MEAN_TOLERANCE}"
    )
    assert mean_dp <= MEAN_TOLERANCE, (
        f"mean ΔAP {mean_dp:.5f} exceeds {MEAN_TOLERANCE}"
    )
