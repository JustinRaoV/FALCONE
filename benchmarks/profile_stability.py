"""Profile weighted_sparse + stability_selection on a single p=1000 cell.

Measurement-only script written 2026-06-03 for the method-optimization
spec v2 revision. Does NOT modify any code under src/falcon/. Generates
a fresh holdout-equivalent cell with seed=200 so the frozen-grid
holdout cells (seeds 10/11/12) remain untouched.

Output:
    data/profile_weighted_sparse_p1000_n100.txt — cProfile pstats text dump
    data/profile_weighted_sparse_p1000_n100.summary.json — per-function % wallclock

Usage:
    uv run python benchmarks/profile_stability.py
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from falcon import infer_network  # noqa: E402
from falcon.sim import generate_scenario  # noqa: E402


# Holdout equivalent: sparse_random, n=500, p=1000, density=0.002.
# Seed=200 so we are NOT touching the frozen holdout cells (seeds 10/11/12).
PROFILE_CELL = dict(
    scenario="sparse_random",
    n=500,
    p=1000,
    seed=200,
    density=0.002,
    edge_strength=0.6,
    depth=5000,
)
N_RESAMPLES = 100  # production default, NOT the 30 used in the 2026-06-03 holdout


def main() -> int:
    print(f"[profile] generating scenario {PROFILE_CELL}", flush=True)
    call_kwargs = dict(PROFILE_CELL)
    call_kwargs["name"] = call_kwargs.pop("scenario")
    scenario = generate_scenario(**call_kwargs)
    print(
        f"[profile] counts shape={scenario.counts.shape}, "
        f"true edges={int(scenario.support.sum() // 2)}",
        flush=True,
    )

    print(f"[profile] running infer_network with n_resamples={N_RESAMPLES} under cProfile", flush=True)
    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    result = infer_network(
        scenario.counts,
        estimator="weighted_sparse",
        zero_policy="multiplicative",
        selection="stability",
        n_resamples=N_RESAMPLES,
        seed=PROFILE_CELL["seed"],
    )
    profiler.disable()
    wall_total = time.perf_counter() - wall_start
    print(f"[profile] wallclock_total_s={wall_total:.3f}", flush=True)
    print(
        f"[profile] converged={result.diagnostics.converged} "
        f"iterations={result.diagnostics.iterations} "
        f"min_eigenvalue={result.diagnostics.min_eigenvalue:.3e}",
        flush=True,
    )
    print(f"[profile] edges_selected={len(result.edges.pairs)}", flush=True)

    output_txt = _REPO / "data" / "profile_weighted_sparse_p1000_n100.txt"
    output_json = _REPO / "data" / "profile_weighted_sparse_p1000_n100.summary.json"
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # Dump full pstats output.
    with output_txt.open("w") as fh:
        fh.write(f"# Profile of weighted_sparse + stability, n_resamples={N_RESAMPLES}\n")
        fh.write(f"# Cell: {PROFILE_CELL}\n")
        fh.write(f"# Wallclock total: {wall_total:.3f}s\n")
        fh.write(f"# Edges selected: {len(result.edges.pairs)}\n")
        fh.write(f"# Converged: {result.diagnostics.converged}\n")
        fh.write("\n## Top 40 by cumulative time\n")
        ps = pstats.Stats(profiler, stream=fh).sort_stats("cumulative")
        ps.print_stats(40)
        fh.write("\n## Top 40 by total (self) time\n")
        ps = pstats.Stats(profiler, stream=fh).sort_stats("tottime")
        ps.print_stats(40)

    # Extract structured percentages for the spec.
    sio = io.StringIO()
    ps = pstats.Stats(profiler, stream=sio).sort_stats("cumulative")
    ps.print_stats(80)
    raw = sio.getvalue()

    hot_keywords = {
        "eigvalsh": "np.linalg.eigvalsh and friends",
        "eigh": "eigendecomposition (LAPACK syevd/syevr/syevr)",
        "estimate_weighted_sparse": "weighted_sparse alternating loop entry",
        "_default_weights": "GEMM-based theta_ij estimate",
        "_correlation_from_covariance": "correlation extraction",
        "select_by_stability": "stability selection driver",
        "estimator_fn": "support_fn closure invocation",
        "default_rng": "RNG construction",
        "choice": "subsample index draw",
    }
    summary = {
        "wallclock_total_s": wall_total,
        "n_resamples": N_RESAMPLES,
        "cell": PROFILE_CELL,
        "edges_selected": int(len(result.edges.pairs)),
        "converged": bool(result.diagnostics.converged),
        "iterations": int(result.diagnostics.iterations),
        "hot_function_cumulative_seconds": {},
        "hot_function_percent_wallclock": {},
        "notes": "",
    }
    for kw, label in hot_keywords.items():
        for line in raw.splitlines():
            if kw in line:
                tokens = line.split()
                # pstats line: ncalls tottime percall cumtime percall filename:lineno(name)
                if len(tokens) >= 6:
                    try:
                        cumtime = float(tokens[3])
                    except ValueError:
                        continue
                    pct = (cumtime / wall_total) * 100.0
                    key = f"{kw} ({label})"
                    prior = summary["hot_function_cumulative_seconds"].get(key, 0.0)
                    if cumtime > prior:
                        summary["hot_function_cumulative_seconds"][key] = cumtime
                        summary["hot_function_percent_wallclock"][key] = pct
                break

    with output_json.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[profile] hot function summary (percent of wallclock):")
    for k, v in summary["hot_function_percent_wallclock"].items():
        cum = summary["hot_function_cumulative_seconds"][k]
        print(f"  {v:6.2f}%  ({cum:7.3f}s)  {k}")

    print(f"\n[profile] wrote {output_txt}")
    print(f"[profile] wrote {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
