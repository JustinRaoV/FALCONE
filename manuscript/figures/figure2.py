"""Figure 2: Falcon-SR cross-domain feasibility.

Schematic-led 2x2 composite:
    a) cross-domain screen-refine workflow with optional prior branch.
    b) edge overlap against SparXCC iter per (n, p, q, top-k) cell.
    c) wall-clock vs p * q, log-log, all methods on the same simulator.
    d) prior ablation: candidate recall and AUROC, fast vs prior, per cell.

Run after the cross feasibility benchmark has produced rows:

    uv run --extra figures python manuscript/figures/figure2.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "falcon_sr_cross_feasibility.csv"
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

from _style import (  # noqa: E402
    MM,
    METHOD_COLOR,
    METHOD_LABEL,
    METHOD_MARKER,
    PALETTE,
    install_pub_style,
    panel_letter,
    save_pub,
)


def _load_rows():
    if not DATA.exists():
        sys.exit(f"No data at {DATA}; run benchmarks/falcon_sr_cross.py.")
    rows = []
    with DATA.open() as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if v in ("", None):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _cell_label(n, p, q, k):
    return f"n{n}, p=q={p}, k={k}"


def _draw_schematic(ax) -> None:
    import matplotlib.patches as patches

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    box_specs = [
        ("counts X\n(n × p)",           3, 19, 13, 9),
        ("counts Y\n(n × q)",           3,  4, 13, 9),
        ("SparXCC\nbase score",        22, 13, 14, 11),
        ("bidirectional\ntop-k union", 41, 13, 14, 11),
        ("sparse refine\n(row / col pruning)", 60, 13, 17, 11),
        ("calibrate\n(permutation)",   82, 13, 15, 11),
    ]
    for label, x, y, w, h in box_specs:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4",
            linewidth=0.7,
            edgecolor=PALETTE["schematic_edge"],
            facecolor=PALETTE["schematic_box"],
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=6.4,
            color="#1F3A56",
        )

    # X, Y → base
    ax.annotate("", xy=(22, 21), xytext=(16, 23),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    ax.annotate("", xy=(22, 17), xytext=(16, 8),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    # base → top-k → refine → calibrate (horizontal at y=18)
    for x_from, x_to in [(36, 41), (55, 60), (77, 82)]:
        ax.annotate("", xy=(x_to, 18), xytext=(x_from, 18),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                    color=PALETTE["schematic_edge"]))

    # Prior box centered above refine/top-k area
    prior_box = patches.FancyBboxPatch(
        (50, 26), 18, 4,
        boxstyle="round,pad=0.4",
        linewidth=0.7,
        edgecolor=PALETTE["schematic_prior_edge"],
        facecolor=PALETTE["schematic_prior_box"],
    )
    ax.add_patch(prior_box)
    ax.text(59, 28, "signed prior (PriorEdge)",
            ha="center", va="center", fontsize=6.2,
            color="#28503A")
    # Prior → top-k (inject)
    ax.annotate(
        "",
        xy=(48, 24), xytext=(55, 26),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.5,
                        color=PALETTE["schematic_prior_edge"],
                        connectionstyle="arc3,rad=-0.2"),
    )
    ax.text(46, 25, "inject", ha="right", va="center",
            fontsize=5.8, color="#28503A", style="italic")
    # Prior → calibrate (shrinkage)
    ax.annotate(
        "",
        xy=(89, 24), xytext=(66, 26),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.5,
                        color=PALETTE["schematic_prior_edge"],
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.text(78, 28, "soft shrinkage",
            ha="center", va="bottom",
            fontsize=5.8, color="#28503A", style="italic")

    # Output below calibrate
    ax.annotate("", xy=(90, 8), xytext=(90, 13),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    ax.text(90, 5, "cross edge table\n+ q-values",
            ha="center", va="top", fontsize=6.2,
            color="#1F3A56")


def _panel_overlap(ax, rows) -> None:
    cells = set()
    by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell = (int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
        cells.add(cell)
        by_method[r["method"]][cell].append(
            r["edge_overlap_vs_sparxcc_iter"]
        )
    cells = sorted(cells)
    labels = [_cell_label(*c) for c in cells]
    x = list(range(len(cells)))

    method_order = [
        "falcon_sr_cross_fast",
        "falcon_sr_cross_fast_calibrated",
        "falcon_sr_cross_prior",
        "sparxcc_base",
    ]
    width = 0.18
    for i, m in enumerate(method_order):
        if m not in by_method:
            continue
        ys = [_avg(by_method[m][c]) for c in cells]
        offset = (i - (len(method_order) - 1) / 2) * width
        ax.bar(
            [xi + offset for xi in x], ys, width=width,
            color=METHOD_COLOR[m], label=METHOD_LABEL[m],
            edgecolor="white", linewidth=0.3,
        )
    ax.axhline(0.95, linestyle="--", linewidth=0.5,
               color=PALETTE["neutral"], alpha=0.7)
    ax.text(x[0] - 0.5, 0.965, "spec gate 0.95",
            fontsize=5.5, color=PALETTE["neutral"],
            ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.6, rotation=60, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("edge overlap\nvs SparXCC iter")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45),
              ncol=2, columnspacing=0.6, handlelength=0.9,
              fontsize=5.5)


def _panel_time(ax, rows) -> None:
    methods_to_plot = [
        "sparxcc_iter", "sparxcc_base",
        "falcon_sr_cross_fast",
        "falcon_sr_cross_fast_calibrated",
        "falcon_sr_cross_prior",
    ]
    by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in methods_to_plot:
            continue
        key = int(r["p"]) * int(r["q"])
        by_method[r["method"]][key].append(r["wallclock_seconds"])
    for m in methods_to_plot:
        if m not in by_method:
            continue
        keys = sorted(by_method[m])
        ys = [_avg(by_method[m][k]) for k in keys]
        ax.plot(
            keys, ys,
            color=METHOD_COLOR[m],
            marker=METHOD_MARKER[m],
            label=METHOD_LABEL[m],
            linewidth=1.1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p × q (bipartite pairs)")
    ax.set_ylabel("wall-clock (s)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.02),
              ncol=1, handlelength=1.0, fontsize=5.4)


def _panel_prior(ax, rows) -> None:
    cells = set()
    fast = defaultdict(list)
    prior = defaultdict(list)
    for r in rows:
        cell = (int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
        cells.add(cell)
        if r["method"] == "falcon_sr_cross_fast":
            fast[cell].append((r["candidate_recall"], r["auroc_vs_truth"]))
        elif r["method"] == "falcon_sr_cross_prior":
            prior[cell].append((r["candidate_recall"], r["auroc_vs_truth"]))
    # Order: prior-relevant cells first (n=100, p=500)
    cells = sorted(cells, key=lambda c: (
        0 if (c[0] == 100 and c[1] == 500) else 1, c
    ))
    labels = [_cell_label(*c) for c in cells]
    x = list(range(len(cells)))
    width = 0.32
    fast_recall = [_avg([t[0] for t in fast[c]]) for c in cells]
    prior_recall = [_avg([t[0] for t in prior[c]]) for c in cells]
    fast_auroc = [_avg([t[1] for t in fast[c]]) for c in cells]
    prior_auroc = [_avg([t[1] for t in prior[c]]) for c in cells]

    ax.bar(
        [xi - width / 2 for xi in x], fast_recall, width=width,
        color=METHOD_COLOR["falcon_sr_cross_fast"],
        label="fast (no prior)",
        edgecolor="white", linewidth=0.3,
    )
    ax.bar(
        [xi + width / 2 for xi in x], prior_recall, width=width,
        color=METHOD_COLOR["falcon_sr_cross_prior"],
        label="fast + prior",
        edgecolor="white", linewidth=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.6, rotation=60, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("candidate recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45),
              ncol=2, handlelength=1.0, fontsize=5.5)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0.4, 1.05)
    ax2.set_ylabel("AUROC", color=PALETTE["neutral"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["neutral"], labelsize=6)
    ax2.plot(
        [xi - width / 2 for xi in x], fast_auroc,
        linestyle="", marker="o", color="#1F3A56", markersize=3,
    )
    ax2.plot(
        [xi + width / 2 for xi in x], prior_auroc,
        linestyle="", marker="D", color="#2E6A40", markersize=3,
    )


def main():
    install_pub_style()
    rows = _load_rows()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[0.65, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=1.05, wspace=0.55,
        left=0.06, right=0.95, top=0.97, bottom=0.20,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    _draw_schematic(ax_a)
    _panel_overlap(ax_b, rows)
    _panel_time(ax_c, rows)
    _panel_prior(ax_d, rows)

    panel_letter(ax_a, "a", dx=-0.005, dy=0.95)
    panel_letter(ax_b, "b")
    panel_letter(ax_c, "c")
    panel_letter(ax_d, "d")

    save_pub(fig, OUT_DIR / "figure2")
    print("Wrote figure2.{svg,pdf,tiff}")


if __name__ == "__main__":
    main()
