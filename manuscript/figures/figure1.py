"""Generate Figure 1 (single-domain feasibility) from
``data/falcon_sr_single_feasibility.csv``.

Run after the benchmark has produced rows:

    uv run --extra figures python manuscript/figures/figure1.py

Writes ``manuscript/figures/figure1.svg`` and ``figure1.pdf``. The figure
is a 2x2 grid following spec §17 Figure Contract:

    panel a: schematic of the screen-refine workflow (text label only;
             a richer schematic is produced by the artist downstream).
    panel b: candidate recall vs top_k budget per method.
    panel c: edge overlap and sign accuracy across (n, p) cells.
    panel d: wall-clock vs feature count p, log-log.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "falcon_sr_single_feasibility.csv"
OUT_DIR = Path(__file__).resolve().parent


def _load_rows():
    rows = []
    if not DATA.exists():
        return rows
    with DATA.open() as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def main():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(
            "matplotlib is required. Install with `uv sync --extra figures`."
        )

    rows = _load_rows()
    if not rows:
        sys.exit(
            f"No feasibility data at {DATA}. Run benchmarks/falcon_sr_single.py."
        )

    # Aggregate to cell-method level
    grouped: dict[tuple, list[float]] = defaultdict(list)
    grouped_time: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["method"], int(row["n"]), int(row["p"]), int(row["top_k"]))
        grouped[key + ("overlap",)].append(row["edge_overlap_vs_sparcc"])
        grouped[key + ("recall",)].append(row["candidate_recall"])
        grouped[key + ("sign",)].append(row["sign_accuracy_vs_truth"])
        grouped_time[key].append(row["wallclock_seconds"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax_workflow, ax_recall), (ax_overlap, ax_time) = axes

    ax_workflow.set_title("a  screen-refine workflow")
    ax_workflow.text(0.5, 0.5,
                     "counts -> base -> top-k -> sparse refine -> calibrate",
                     ha="center", va="center", wrap=True)
    ax_workflow.axis("off")

    # Panel b: candidate recall vs top_k for Falcon-SR fast
    method = "falcon_sr_fast"
    points = defaultdict(list)
    for key, vals in grouped.items():
        m, n, p, k, label = key
        if m != method or label != "recall":
            continue
        points[(n, p)].append((k, sum(vals) / len(vals)))
    for (n, p), pts in sorted(points.items()):
        pts.sort()
        ks = [t[0] for t in pts]
        ys = [t[1] for t in pts]
        ax_recall.plot(ks, ys, marker="o", label=f"n={n}, p={p}")
    ax_recall.set_xlabel("top_k")
    ax_recall.set_ylabel("candidate recall (vs planted truth)")
    ax_recall.set_title("b  candidate recall vs budget")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.legend(fontsize=8)

    # Panel c: edge overlap vs SparCC per (n, p) for falcon_sr_fast
    points = defaultdict(list)
    for key, vals in grouped.items():
        m, n, p, k, label = key
        if m != method or label != "overlap":
            continue
        points[(n, p)].append((k, sum(vals) / len(vals)))
    for (n, p), pts in sorted(points.items()):
        pts.sort()
        ks = [t[0] for t in pts]
        ys = [t[1] for t in pts]
        ax_overlap.plot(ks, ys, marker="s", label=f"n={n}, p={p}")
    ax_overlap.set_xlabel("top_k")
    ax_overlap.set_ylabel("edge overlap vs SparCC")
    ax_overlap.set_title("c  edge overlap vs SparCC")
    ax_overlap.set_ylim(0, 1.05)
    ax_overlap.legend(fontsize=8)

    # Panel d: wall-clock vs p, log-log, per method
    methods = sorted({key[0] for key in grouped_time})
    for m in methods:
        ps, ts = [], []
        for (mm, n, p, k), seconds in grouped_time.items():
            if mm != m:
                continue
            ps.append(p)
            ts.append(sum(seconds) / len(seconds))
        if not ps:
            continue
        order = sorted(zip(ps, ts))
        ax_time.plot([x for x, _ in order], [y for _, y in order],
                     marker="o", label=m)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("p (features)")
    ax_time.set_ylabel("wall-clock seconds")
    ax_time.set_title("d  wall-clock vs p")
    ax_time.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure1.svg")
    fig.savefig(OUT_DIR / "figure1.pdf")
    print(f"Wrote {OUT_DIR / 'figure1.svg'} and figure1.pdf")


if __name__ == "__main__":
    main()
