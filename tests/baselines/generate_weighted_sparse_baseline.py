"""Generate the pinned AUROC/AP baseline for the differential test.

Run once before any Line A optimization commit. Output is committed
under tests/baselines/. Re-running after any algorithmic change to
weighted_sparse would invalidate the test's purpose — so run only when
intentionally resetting the baseline (record a decision-log entry).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from falcon import infer_network
from falcon.sim import (
    auroc_score,
    average_precision_score,
    generate_scenario,
    training_grid,
)


N_RESAMPLES = 30  # fits 10-minute CI; spec §5.5 step 1


def main() -> int:
    out_path = _REPO / "tests" / "baselines" / "weighted_sparse_baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("scenario", "n", "p", "seed", "density", "auroc", "ap",
              "n_edges", "converged", "iterations", "wallclock_s")
    rows = []
    for cell in training_grid():
        md = cell.metadata()
        scenario = generate_scenario(
            md["scenario"], n=md["n"], p=md["p"], seed=md["seed"],
            density=md["density"], edge_strength=md["edge_strength"],
            depth=md["depth"],
        )
        t0 = time.perf_counter()
        result = infer_network(
            scenario.counts,
            estimator="weighted_sparse",
            selection="stability",
            n_resamples=N_RESAMPLES,
            seed=md["seed"],
        )
        wall = time.perf_counter() - t0
        rows.append(dict(
            scenario=md["scenario"],
            n=md["n"], p=md["p"], seed=md["seed"], density=md["density"],
            auroc=auroc_score(result.correlation, scenario.support),
            ap=average_precision_score(result.correlation, scenario.support),
            n_edges=int(len(result.edges.pairs)),
            converged=bool(result.diagnostics.converged),
            iterations=int(result.diagnostics.iterations),
            wallclock_s=wall,
        ))
        print(
            f"  {md['scenario']:>22} n={md['n']:>3} p={md['p']:>4} seed={md['seed']} "
            f"AUROC={rows[-1]['auroc']:.4f} AP={rows[-1]['ap']:.4f} wall={wall:.2f}s",
            flush=True,
        )
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
