"""Figure 2: Falcon-SR cross-domain feasibility.

Layout (schematic-led 2x3 composite):
    a) cross-domain screen-refine workflow with optional prior branch.
    b) AUROC heatmap, methods x (n, p, q, top-k) cells, with family
       grouping. Edge overlap vs SparXCC iter is annotated below the
       diagonal of each cell as a secondary metric.
    c) wall-clock vs p * q, log-log, every benchmarked method.
    d) prior ablation focused on the cells where the prior actually
       matters: candidate recall and AUROC for Falcon-SR fast vs
       Falcon-SR fast + prior on the n = 100 / p = q = 500 cells.

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
    METHOD_FAMILY,
    METHOD_LABEL,
    METHOD_MARKER,
    PALETTE,
    install_pub_style,
    panel_letter,
    save_pub,
)

CROSS_METHOD_ORDER = [
    # Falcon-SR family first
    "falcon_sr_cross_fast",
    "falcon_sr_cross_fast_calibrated",
    "falcon_sr_cross_prior",
    # SparXCC (same estimand)
    "sparxcc_iter",
    "sparxcc_base",
    # SPIEC-EASI cross (adjacent estimand)
    "spieceasi_cross_glasso",
]


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
    xs = [x for x in xs if x is not None and not _isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _isnan(v) -> bool:
    return v is None or v != v


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

    ax.annotate("", xy=(22, 21), xytext=(16, 23),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    ax.annotate("", xy=(22, 17), xytext=(16, 8),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    for x_from, x_to in [(36, 41), (55, 60), (77, 82)]:
        ax.annotate("", xy=(x_to, 18), xytext=(x_from, 18),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                    color=PALETTE["schematic_edge"]))

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
    ax.annotate(
        "", xy=(48, 24), xytext=(55, 26),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.5,
                        color=PALETTE["schematic_prior_edge"],
                        connectionstyle="arc3,rad=-0.2"),
    )
    ax.text(46, 25, "inject", ha="right", va="center",
            fontsize=5.8, color="#28503A", style="italic")
    ax.annotate(
        "", xy=(89, 24), xytext=(66, 26),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.5,
                        color=PALETTE["schematic_prior_edge"],
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.text(78, 28, "soft shrinkage",
            ha="center", va="bottom",
            fontsize=5.8, color="#28503A", style="italic")

    ax.annotate("", xy=(90, 8), xytext=(90, 13),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                                color=PALETTE["schematic_edge"]))
    ax.text(90, 5, "cross edge table\n+ q-values",
            ha="center", va="top", fontsize=6.2,
            color="#1F3A56")


def _panel_overlap_bars(ax, rows) -> None:
    """Edge overlap against the SparXCC iter reference, per method,
    averaged across cells. Mirrors the single-domain headline panel:
    Falcon-SR variants match SparXCC iter; SPIEC-EASI cross-glasso
    drops to ~0.74 averaged (and to ~0.006 on the hardest cell)
    because it targets the partial cross-correlation on stacked
    compositions, not the SparXCC Case-C estimand.
    """
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r["edge_overlap_vs_sparxcc_iter"])
    method_order = [m for m in CROSS_METHOD_ORDER if m in by_method]
    means = [_avg(by_method[m]) for m in method_order]

    x = list(range(len(method_order)))
    ax.barh(x, means,
            color=[METHOD_COLOR[m] for m in method_order],
            edgecolor="white", linewidth=0.4)
    for xi, mean in zip(x, means):
        ax.text(min(mean + 0.015, 1.02), xi, f"{mean:.2f}",
                ha="left", va="center", fontsize=5.6, color="black")

    ax.axvline(0.95, linestyle="--", linewidth=0.6,
               color="#888888", alpha=0.9, zorder=0)
    ax.text(0.94, len(method_order) - 0.5, "0.95",
            fontsize=5.0, color="#555555",
            ha="right", va="top")

    for i in range(1, len(method_order)):
        if METHOD_FAMILY[method_order[i]] != METHOD_FAMILY[method_order[i - 1]]:
            ax.axhline(i - 0.5, color="#CCCCCC", linewidth=0.4, zorder=0)

    ax.set_yticks(x)
    ax.set_yticklabels([METHOD_LABEL[m] for m in method_order], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("edge overlap\nvs SparXCC iter")


def _heatmap_panel(ax, rows, metric_key, title) -> None:
    """Generic methods x cells heatmap for the cross benchmark."""
    import numpy as np

    methods = [m for m in CROSS_METHOD_ORDER
               if any(r["method"] == m for r in rows)]
    cells = sorted({(int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
                    for r in rows})
    grid = np.full((len(methods), len(cells)), np.nan)
    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        if r["method"] not in methods:
            continue
        m_idx = methods.index(r["method"])
        c_idx = cells.index(
            (int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
        )
        bucket[(m_idx, c_idx)].append(r[metric_key])
    for (m_idx, c_idx), values in bucket.items():
        values = [v for v in values if not _isnan(v)]
        if not values:
            continue
        grid[m_idx, c_idx] = sum(values) / len(values)

    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="RdYlBu",
        vmin=0.4, vmax=1.0,
        interpolation="nearest",
    )
    # Compact one-line cell labels: n / p (=q) / k. Rotated 35° so they
    # stay legible inside the narrow heatmap column.
    cell_labels = [f"n{n}, p={p}, k={k}" for (n, p, q, k) in cells]
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cell_labels, fontsize=5.4, rotation=35,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods], fontsize=5.8)

    for i in range(len(methods)):
        for j in range(len(cells)):
            v = grid[i, j]
            if _isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=4.6,
                    color="black" if v > 0.6 else "white")

    for i in range(1, len(methods)):
        if METHOD_FAMILY[methods[i]] != METHOD_FAMILY[methods[i - 1]]:
            ax.axhline(i - 0.5, color="white", linewidth=1.0)

    ax.set_title(title, fontsize=6.8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)


def _panel_time(ax, rows) -> None:
    methods_to_plot = CROSS_METHOD_ORDER
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
            linewidth=1.0,
            markersize=3.0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p × q (bipartite pairs)")
    ax.set_ylabel("wall-clock (s)")
    ax.legend(loc="lower right",
              ncol=1, handlelength=1.0, fontsize=4.8)


def _panel_prior(ax, rows) -> None:
    """Prior ablation, focused on the cells where the prior actually
    moves the needle. On the easy cells Falcon-SR fast already saturates
    at AUROC >= 0.998, so the prior produces no measurable change; we
    flag that ceiling rather than burying it in a wall of identical
    bars.
    """
    import numpy as np

    fast = defaultdict(list)
    prior = defaultdict(list)
    for r in rows:
        cell = (int(r["n"]), int(r["p"]), int(r["q"]), int(r["top_k"]))
        if r["method"] == "falcon_sr_cross_fast":
            fast[cell].append((r["candidate_recall"], r["auroc_vs_truth"]))
        elif r["method"] == "falcon_sr_cross_prior":
            prior[cell].append((r["candidate_recall"], r["auroc_vs_truth"]))

    # Only show cells where Falcon-SR fast did not already ceiling out.
    target_cells = sorted(
        c for c in fast
        if fast[c] and prior[c]
        and _avg([t[0] for t in fast[c]]) < 0.99
    )
    if not target_cells:
        ax.text(0.5, 0.5, "prior ablation pending data",
                ha="center", va="center", transform=ax.transAxes)
        return

    n_cells = len(target_cells)
    # Two paired groups per cell: recall (left), AUROC (right). Each group
    # has two bars (fast vs prior). Cells are separated by a vertical gap.
    group_width = 0.85
    pair_width = group_width / 2.0
    bar_width = pair_width * 0.42
    cell_centres = np.arange(n_cells) * 1.4
    recall_centres = cell_centres - pair_width / 2
    auroc_centres = cell_centres + pair_width / 2

    fast_recall = np.array(
        [_avg([t[0] for t in fast[c]]) for c in target_cells]
    )
    prior_recall = np.array(
        [_avg([t[0] for t in prior[c]]) for c in target_cells]
    )
    fast_auroc = np.array(
        [_avg([t[1] for t in fast[c]]) for c in target_cells]
    )
    prior_auroc = np.array(
        [_avg([t[1] for t in prior[c]]) for c in target_cells]
    )

    ax.bar(recall_centres - bar_width / 2, fast_recall, width=bar_width,
           color=METHOD_COLOR["falcon_sr_cross_fast"],
           edgecolor="white", linewidth=0.4,
           label="fast")
    ax.bar(recall_centres + bar_width / 2, prior_recall, width=bar_width,
           color=METHOD_COLOR["falcon_sr_cross_prior"],
           edgecolor="white", linewidth=0.4,
           label="fast + prior")
    ax.bar(auroc_centres - bar_width / 2, fast_auroc, width=bar_width,
           color=METHOD_COLOR["falcon_sr_cross_fast"],
           edgecolor="white", linewidth=0.4)
    ax.bar(auroc_centres + bar_width / 2, prior_auroc, width=bar_width,
           color=METHOD_COLOR["falcon_sr_cross_prior"],
           edgecolor="white", linewidth=0.4)

    # Sub-group labels under each pair (cell metric)
    for centre in recall_centres:
        ax.text(centre, -0.06, "recall",
                ha="center", va="top", fontsize=5.6,
                color=PALETTE["neutral"])
    for centre in auroc_centres:
        ax.text(centre, -0.06, "AUROC",
                ha="center", va="top", fontsize=5.6,
                color=PALETTE["neutral"])

    # Delta labels above the prior bars
    for centre, before, after in zip(recall_centres, fast_recall, prior_recall):
        delta = after - before
        if abs(delta) < 0.005:
            continue
        ax.annotate(
            f"+{delta:.2f}",
            xy=(centre + bar_width / 2, after),
            xytext=(0, 1.5), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.5, color=PALETTE["falcon_prior"], fontweight="bold",
        )
    for centre, before, after in zip(auroc_centres, fast_auroc, prior_auroc):
        delta = after - before
        if abs(delta) < 0.005:
            continue
        ax.annotate(
            f"+{delta:.2f}",
            xy=(centre + bar_width / 2, after),
            xytext=(0, 1.5), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=5.5, color=PALETTE["falcon_prior"], fontweight="bold",
        )

    # Cell labels below sub-group labels
    cell_labels = [f"n={n}\np=q={p}, k={k}" for (n, p, q, k) in target_cells]
    ax.set_xticks(cell_centres)
    ax.set_xticklabels(cell_labels, fontsize=6.0)
    ax.tick_params(axis="x", pad=12)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("score (recall or AUROC)")
    ax.set_title("prior gain on under-determined cells",
                 fontsize=6.8)
    ax.legend(loc="lower right", ncol=2, fontsize=5.5, handlelength=1.0,
              columnspacing=0.8)


def main():
    install_pub_style()
    rows = _load_rows()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(195 * MM, 135 * MM))
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[0.5, 1.0],
        width_ratios=[0.95, 1.25, 0.8, 0.95],
        hspace=0.65, wspace=1.05,
        left=0.06, right=0.98, top=0.96, bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[1, 3])

    _draw_schematic(ax_a)
    _panel_overlap_bars(ax_b, rows)
    _heatmap_panel(ax_c, rows, "auroc_vs_truth",
                   "AUROC vs planted truth")
    _panel_time(ax_d, rows)
    _panel_prior(ax_e, rows)

    panel_letter(ax_a, "a", dx=-0.005, dy=0.95)
    panel_letter(ax_b, "b")
    panel_letter(ax_c, "c", dx=-0.05)
    panel_letter(ax_d, "d")
    panel_letter(ax_e, "e")

    save_pub(fig, OUT_DIR / "figure2")
    print("Wrote figure2.{svg,pdf,tiff}")


if __name__ == "__main__":
    main()
