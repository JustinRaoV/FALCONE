"""Measure true wallclock + memory gap at n_resamples=100 (production default).

Measurement-only script written 2026-06-03 for the method-optimization
spec v2 revision. The 2026-06-03 holdout was run with --n-resamples 30,
which understates the production-default gap to sparcc_closed_form by
a factor of ~3.3x. This script remeasures the gap on a small set of
holdout-equivalent cells, using fresh seeds (200+) so the actual
holdout cells (seeds 10/11/12) remain untouched.

Also records both tracemalloc-based peak memory (the metric the existing
benchmark uses) AND psutil RSS-based peak memory (which captures LAPACK
workspaces that tracemalloc misses). This validates whether the spec's
gate-3 memory metric is measurable as defined.

Output:
    data/bench_gap_n100.csv

Usage:
    uv run python benchmarks/bench_gap_n100.py
"""

from __future__ import annotations

import csv
import sys
import threading
import time
import tracemalloc
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

try:
    import psutil  # noqa: E402

    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

from benchmarks.baselines import pearson_clr, sparcc_closed_form  # noqa: E402
from falcon import infer_network  # noqa: E402
from falcon.sim import generate_scenario  # noqa: E402


CELLS = [
    # Match holdout (500, 500) and (500, 1000) sizes / densities, but with
    # seeds 200/201/202 so we are NOT touching the frozen holdout cells
    # (seeds 10/11/12).
    dict(scenario="sparse_random", n=500, p=500, seed=200, density=0.005),
    dict(scenario="sparse_random", n=500, p=1000, seed=200, density=0.002),
    dict(scenario="hub", n=500, p=500, seed=201, density=0.005),
    dict(scenario="hub", n=500, p=1000, seed=201, density=0.002),
    dict(scenario="heavy_tailed", n=500, p=500, seed=202, density=0.005),
    dict(scenario="block", n=500, p=1000, seed=202, density=0.002),
]
N_RESAMPLES_FALCON = 100  # production default
EDGE_STRENGTH = 0.6
DEPTH = 5000


def _measure_rss_during(fn, poll_interval_s: float = 0.05):
    """Run fn() and concurrently sample RSS at poll_interval_s.
    Returns (result, wallclock_s, peak_rss_bytes, baseline_rss_bytes)."""
    if not _PSUTIL_OK:
        # Fallback: tracemalloc only.
        tracemalloc.start()
        t0 = time.perf_counter()
        out = fn()
        wall = time.perf_counter() - t0
        _, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return out, wall, peak_trace, 0

    proc = psutil.Process()
    baseline_rss = proc.memory_info().rss
    peak_rss = baseline_rss
    stop_flag = threading.Event()

    def poller():
        nonlocal peak_rss
        while not stop_flag.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
            stop_flag.wait(poll_interval_s)

    th = threading.Thread(target=poller, daemon=True)
    th.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    stop_flag.set()
    th.join(timeout=1.0)
    return out, wall, peak_rss, baseline_rss


def _measure_tracemalloc(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, wall, peak


def run_cell(cell):
    print(f"[gap] {cell}", flush=True)
    call_kwargs = dict(cell)
    call_kwargs["name"] = call_kwargs.pop("scenario")
    scenario = generate_scenario(
        **call_kwargs, edge_strength=EDGE_STRENGTH, depth=DEPTH
    )
    rows = []

    # weighted_sparse at n_resamples=100 (the heavy one)
    print("  [gap] weighted_sparse n_resamples=100 ...", flush=True)

    def run_ws():
        return infer_network(
            scenario.counts,
            estimator="weighted_sparse",
            selection="stability",
            n_resamples=N_RESAMPLES_FALCON,
            seed=cell["seed"],
        )

    out_ws, wall_ws, peak_rss_ws, baseline_rss = _measure_rss_during(run_ws)
    _, _, peak_trace_ws = _measure_tracemalloc(run_ws)
    rows.append(
        dict(
            scenario=cell["scenario"],
            n=cell["n"],
            p=cell["p"],
            seed=cell["seed"],
            density=cell["density"],
            method="weighted_sparse",
            n_resamples=N_RESAMPLES_FALCON,
            wallclock_s=wall_ws,
            peak_rss_bytes=peak_rss_ws,
            baseline_rss_bytes=baseline_rss,
            peak_tracemalloc_bytes=peak_trace_ws,
            converged=bool(out_ws.diagnostics.converged),
            iterations=int(out_ws.diagnostics.iterations),
            n_edges=int(len(out_ws.edges.pairs)),
        )
    )
    print(
        f"    wall={wall_ws:.2f}s rss_peak={peak_rss_ws/(1024**2):.1f}MB "
        f"tracemalloc_peak={peak_trace_ws/(1024**2):.1f}MB",
        flush=True,
    )

    # sparcc_closed_form — closed form, no stability loop
    print("  [gap] sparcc_closed_form ...", flush=True)

    def run_sparcc():
        return sparcc_closed_form(scenario.counts)

    out_sp, wall_sp, peak_rss_sp, _ = _measure_rss_during(run_sparcc)
    _, _, peak_trace_sp = _measure_tracemalloc(run_sparcc)
    rows.append(
        dict(
            scenario=cell["scenario"],
            n=cell["n"],
            p=cell["p"],
            seed=cell["seed"],
            density=cell["density"],
            method="sparcc_closed_form",
            n_resamples=1,
            wallclock_s=wall_sp,
            peak_rss_bytes=peak_rss_sp,
            baseline_rss_bytes=baseline_rss,
            peak_tracemalloc_bytes=peak_trace_sp,
            converged=True,
            iterations=1,
            n_edges=int((np.abs(out_sp.correlation) > 0).sum() // 2),
        )
    )
    print(f"    wall={wall_sp:.4f}s rss_peak={peak_rss_sp/(1024**2):.1f}MB", flush=True)

    # pearson_clr — closed form, no stability loop
    print("  [gap] pearson_clr ...", flush=True)

    def run_pearson():
        return pearson_clr(scenario.counts)

    out_pe, wall_pe, peak_rss_pe, _ = _measure_rss_during(run_pearson)
    _, _, peak_trace_pe = _measure_tracemalloc(run_pearson)
    rows.append(
        dict(
            scenario=cell["scenario"],
            n=cell["n"],
            p=cell["p"],
            seed=cell["seed"],
            density=cell["density"],
            method="pearson_clr",
            n_resamples=1,
            wallclock_s=wall_pe,
            peak_rss_bytes=peak_rss_pe,
            baseline_rss_bytes=baseline_rss,
            peak_tracemalloc_bytes=peak_trace_pe,
            converged=True,
            iterations=1,
            n_edges=int((np.abs(out_pe.correlation) > 0).sum() // 2),
        )
    )
    print(f"    wall={wall_pe:.4f}s rss_peak={peak_rss_pe/(1024**2):.1f}MB", flush=True)

    gap_to_sparcc = wall_ws / max(wall_sp, 1e-9)
    print(f"  [gap] weighted_sparse / sparcc gap: {gap_to_sparcc:.1f}x", flush=True)
    return rows


def main() -> int:
    if not _PSUTIL_OK:
        print("[gap] WARNING: psutil not installed; RSS column will be zero.", flush=True)
        print("[gap] Install with: uv add psutil", flush=True)

    out_csv = _REPO / "data" / "bench_gap_n100.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario", "n", "p", "seed", "density", "method", "n_resamples",
        "wallclock_s", "peak_rss_bytes", "baseline_rss_bytes",
        "peak_tracemalloc_bytes", "converged", "iterations", "n_edges",
    )
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        all_rows = []
        for cell in CELLS:
            rows = run_cell(cell)
            for r in rows:
                writer.writerow(r)
                fh.flush()
                all_rows.append(r)

    print(f"\n[gap] wrote {out_csv} ({len(all_rows)} rows)")

    # Print summary.
    print("\n[gap] === GAP SUMMARY at n_resamples=100 (weighted_sparse) ===")
    print(f"{'scenario':>20} {'p':>5} {'ws_wall':>10} {'sparcc':>10} {'gap_x':>8} {'rss_ratio':>10} {'trace_ratio':>12}")
    by_cell = {}
    for r in all_rows:
        key = (r["scenario"], r["p"])
        by_cell.setdefault(key, {})[r["method"]] = r
    for (sc, p), methods in sorted(by_cell.items()):
        ws = methods.get("weighted_sparse")
        sp = methods.get("sparcc_closed_form")
        if ws and sp:
            gap = ws["wallclock_s"] / max(sp["wallclock_s"], 1e-9)
            rss_r = (ws["peak_rss_bytes"] - ws["baseline_rss_bytes"]) / max(
                sp["peak_rss_bytes"] - sp["baseline_rss_bytes"], 1
            )
            trace_r = ws["peak_tracemalloc_bytes"] / max(sp["peak_tracemalloc_bytes"], 1)
            print(
                f"{sc:>20} {p:>5} {ws['wallclock_s']:>10.2f} {sp['wallclock_s']:>10.4f} "
                f"{gap:>8.0f} {rss_r:>10.2f} {trace_r:>12.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
