"""Generate Figure 2 (cross-domain feasibility) from
``data/falcon_sr_cross_feasibility.csv``.

Run after the benchmark has produced rows:

    uv run --extra figures python manuscript/figures/figure2.py

Writes ``manuscript/figures/figure2.svg`` and ``figure2.pdf``. The figure
is a 2x2 grid following spec §17 Figure Contract:

    panel a: schematic of the cross-domain screen-refine workflow (text
             label only; richer schematic produced downstream).
    panel b: edge overlap and sign accuracy against SparXCC iter.
    panel c: wall-clock vs (p, q).
    panel d: prior ablation: candidate recall with vs without prior.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "falcon_sr_cross_feasibility.csv"
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
            f"No feasibility data at {DATA}. Run benchmarks/falcon_sr_cross.py."
        )

    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["method"], int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
        grouped[key]["overlap"].append(r["edge_overlap_vs_sparxcc_iter"])
        grouped[key]["sign"].append(r["sign_accuracy_vs_truth"])
        grouped[key]["recall"].append(r["candidate_recall"])
        grouped[key]["time"].append(r["wallclock_seconds"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax_workflow, ax_overlap), (ax_time, ax_prior) = axes

    ax_workflow.set_title("a  cross-domain screen-refine workflow")
    ax_workflow.text(
        0.5, 0.5,
        "(X, Y) -> base (SparXCC Case-C) -> bidirectional top-k "
        "-> sparse refine (row/col pruning) -> optional prior shrinkage "
        "-> calibrate",
        ha="center", va="center", wrap=True,
    )
    ax_workflow.axis("off")

    methods = sorted({key[0] for key in grouped})
    for m in methods:
        xs, ys_overlap, ys_sign = [], [], []
        for (mm, n, p, q, k), vals in grouped.items():
            if mm != m:
                continue
            xs.append((n, p, q, k))
            ys_overlap.append(sum(vals["overlap"]) / len(vals["overlap"]))
            ys_sign.append(sum(vals["sign"]) / len(vals["sign"]))
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        labels = [f"n={xs[i][0]}\np={xs[i][1]}\nq={xs[i][2]}\nk={xs[i][3]}"
                  for i in order]
        x_idx = list(range(len(order)))
        ax_overlap.plot(x_idx, [ys_overlap[i] for i in order],
                        marker="o", label=m)
    ax_overlap.set_xlabel("cell")
    ax_overlap.set_ylabel("edge overlap vs SparXCC iter")
    ax_overlap.set_title("b  edge overlap vs SparXCC iter")
    ax_overlap.set_ylim(0, 1.05)
    ax_overlap.legend(fontsize=8)

    for m in methods:
        ps, ts = [], []
        for (mm, n, p, q, k), vals in grouped.items():
            if mm != m:
                continue
            ps.append(p * q)
            ts.append(sum(vals["time"]) / len(vals["time"]))
        if not ps:
            continue
        order = sorted(zip(ps, ts))
        ax_time.plot([x for x, _ in order], [y for _, y in order],
                     marker="o", label=m)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("p * q")
    ax_time.set_ylabel("wall-clock seconds")
    ax_time.set_title("c  wall-clock vs (p, q)")
    ax_time.legend(fontsize=8)

    # Prior ablation: cross_fast vs cross_prior on the same cells
    cells = sorted({(n, p, q, k) for (m, n, p, q, k) in grouped})
    fast_recall = []
    prior_recall = []
    labels = []
    for cell in cells:
        n, p, q, k = cell
        fast_key = ("falcon_sr_cross_fast", n, p, q, k)
        prior_key = ("falcon_sr_cross_prior", n, p, q, k)
        if fast_key not in grouped or prior_key not in grouped:
            continue
        fast_recall.append(sum(grouped[fast_key]["recall"]) /
                           len(grouped[fast_key]["recall"]))
        prior_recall.append(sum(grouped[prior_key]["recall"]) /
                            len(grouped[prior_key]["recall"]))
        labels.append(f"n={n}\np={p}\nq={q}\nk={k}")
    width = 0.4
    x = list(range(len(labels)))
    ax_prior.bar([i - width / 2 for i in x], fast_recall, width=width,
                  label="cross_fast")
    ax_prior.bar([i + width / 2 for i in x], prior_recall, width=width,
                  label="cross_prior")
    ax_prior.set_xticks(x)
    ax_prior.set_xticklabels(labels, fontsize=6)
    ax_prior.set_ylabel("candidate recall (vs planted truth)")
    ax_prior.set_title("d  prior ablation")
    ax_prior.set_ylim(0, 1.05)
    ax_prior.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2.svg")
    fig.savefig(OUT_DIR / "figure2.pdf")
    print(f"Wrote {OUT_DIR / 'figure2.svg'} and figure2.pdf")


if __name__ == "__main__":
    main()
