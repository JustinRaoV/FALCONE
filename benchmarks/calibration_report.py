"""Run Line B's calibration evaluation on training and holdout.

Procedure (spec v2 §6):
1. Fit IsotonicCalibrator(mode="per_scenario") on the 39 training
   cells.
2. Apply to 54 holdout cells.
3. Emit a per-cell row + aggregate summary.

Output:
    data/calibration_holdout_v2.csv (per-cell rows)
    data/calibration_summary_v2.json (aggregate ECE / Brier)

LIMITATION (spec self-review feas-5): infer_network's EdgeTable only
carries sel_prob for selected edges. This runner sets sel_prob=0 for
the unselected off-diagonal pairs — a conservative approximation. A
follow-up task (post Line A) extends NetworkResult with the full
sel_prob matrix; until then this benchmark is a smoke-tier evaluation.

Usage:
    uv run python benchmarks/calibration_report.py --smoke
    uv run python benchmarks/calibration_report.py            # full run (slow)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from falcon import infer_network
from falcon.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    pfer_bound,
    reliability_diagram,
)
from falcon.sim import generate_scenario, holdout_grid, training_grid


N_RESAMPLES_TRAIN = 30
N_RESAMPLES_HOLDOUT = 100


def _gather(cells, n_resamples):
    """For each cell, return (meta, sel_prob_per_off_diag_pair, truth_per_pair)."""
    out = []
    for cell in cells:
        md = cell.metadata()
        scenario = generate_scenario(
            md["scenario"], n=md["n"], p=md["p"], seed=md["seed"],
            density=md["density"], edge_strength=md["edge_strength"],
            depth=md["depth"],
        )
        result = infer_network(
            scenario.counts,
            estimator="weighted_sparse",
            selection="stability",
            n_resamples=n_resamples,
            seed=md["seed"],
        )
        p = scenario.counts.shape[1]
        triu_i, triu_j = np.triu_indices(p, k=1)
        sel_full = np.zeros(triu_i.size, dtype=np.float64)
        if result.edges.selection_probability is not None and len(result.edges.pairs) > 0:
            edge_i = result.edges.pairs[:, 0]
            edge_j = result.edges.pairs[:, 1]
            # Build a (i, j) → triu index lookup once, reuse for each edge.
            triu_index_map = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(triu_i, triu_j))}
            for k_edge, (i, j) in enumerate(zip(edge_i, edge_j)):
                k_triu = triu_index_map[(int(i), int(j))]
                sel_full[k_triu] = float(result.edges.selection_probability[k_edge])
        truth = scenario.support[triu_i, triu_j].astype(np.float64)
        out.append((md, sel_full, truth))
        print(
            f"  gathered {md['scenario']} n={md['n']} p={md['p']} seed={md['seed']}: "
            f"{int(truth.sum())} true edges, {int((sel_full > 0).sum())} non-zero sel_prob",
            flush=True,
        )
    return out


def run(smoke: bool) -> int:
    train_cells = list(training_grid())
    holdout_cells = list(holdout_grid())
    if smoke:
        train_cells = train_cells[:3]
        holdout_cells = holdout_cells[:1]
        print(f"[calibration] SMOKE: {len(train_cells)} train + {len(holdout_cells)} holdout", flush=True)

    print(f"[calibration] gathering training cells (n_resamples={N_RESAMPLES_TRAIN}) ...", flush=True)
    train = _gather(train_cells, N_RESAMPLES_TRAIN)
    print(f"[calibration] training: {len(train)} cells", flush=True)

    cal_per_scenario = IsotonicCalibrator(mode="per_scenario")
    cal_pooled = IsotonicCalibrator(mode="pooled")
    pooled_sel, pooled_truth = [], []
    for md, sel_full, truth in train:
        cal_per_scenario.fit(sel_full, truth, scenario=md["scenario"])
        pooled_sel.append(sel_full)
        pooled_truth.append(truth)
    cal_pooled.fit(np.concatenate(pooled_sel), np.concatenate(pooled_truth), scenario="*")
    pi_train = float(np.concatenate(pooled_truth).mean())
    print(f"[calibration] pi_train = {pi_train:.4f}", flush=True)
    print(f"[calibration] scenarios fitted: {cal_per_scenario.scenarios}", flush=True)

    print(f"[calibration] applying to holdout (n_resamples={N_RESAMPLES_HOLDOUT}) ...", flush=True)
    rows = []
    out_csv = _REPO / "data" / ("calibration_smoke_v2.csv" if smoke else "calibration_holdout_v2.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "cell_id", "scenario", "n", "p", "seed", "density",
        "n_off_diagonal_pairs", "ece_per_cell",
        "brier_score_per_cell", "calibration_method", "pi_train",
        "q_avg_at_0_8", "pfer_bound_at_0_8",
    )
    holdout = _gather(holdout_cells, N_RESAMPLES_HOLDOUT)
    for md, sel_full, truth in holdout:
        scen = md["scenario"]
        if scen in cal_per_scenario.scenarios:
            post = cal_per_scenario.predict(sel_full, scenario=scen)
            method = "empirical_isotonic_per_scenario"
        else:
            post = cal_pooled.predict(sel_full, scenario="*")
            method = "empirical_isotonic_pooled"
        mids, obs, counts = reliability_diagram(post, truth, n_bins=10)
        ece = expected_calibration_error(mids, obs, counts)
        brier = brier_score(post, truth)
        q_avg = int((sel_full >= 0.8).sum())
        try:
            pfer = pfer_bound(q_avg=q_avg, pi_thr=0.8, p_off=int(truth.size))
        except ValueError:
            pfer = float("nan")
        rows.append(dict(
            cell_id=f"{scen}_n{md['n']}_p{md['p']}_seed{md['seed']}",
            scenario=scen, n=md["n"], p=md["p"], seed=md["seed"],
            density=md["density"], n_off_diagonal_pairs=int(truth.size),
            ece_per_cell=ece, brier_score_per_cell=brier,
            calibration_method=method, pi_train=pi_train,
            q_avg_at_0_8=q_avg, pfer_bound_at_0_8=pfer,
        ))

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[calibration] wrote {out_csv} ({len(rows)} rows)", flush=True)

    aggregate_ece = float(np.mean([r["ece_per_cell"] for r in rows])) if rows else 0.0
    aggregate_brier = float(np.mean([r["brier_score_per_cell"] for r in rows])) if rows else 0.0
    summary = dict(
        aggregate_ece=aggregate_ece,
        aggregate_brier=aggregate_brier,
        pi_train=pi_train,
        cells=len(rows),
        per_scenario_ece={
            s: float(np.mean([r["ece_per_cell"] for r in rows if r["scenario"] == s]))
            for s in sorted({r["scenario"] for r in rows})
        },
        smoke=smoke,
    )
    out_json = _REPO / "data" / ("calibration_smoke_summary_v2.json" if smoke else "calibration_summary_v2.json")
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[calibration] wrote {out_json}", flush=True)
    print(f"[calibration] aggregate ECE = {aggregate_ece:.4f}, Brier = {aggregate_brier:.4f}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Run on a tiny subset for sanity (default: full)")
    args = p.parse_args()
    return run(smoke=args.smoke)


if __name__ == "__main__":
    sys.exit(main())
