"""Figure 1: Falcon-SR single-domain feasibility.

Schematic-led 2x2 composite:
    a) screen-refine workflow diagram (top row, full width).
    b) candidate recall vs top-k budget per (n, p) cell for Falcon-SR fast.
    c) edge overlap against the SparCC reference per cell.
    d) wall-clock vs p, log-log, comparing every benchmarked method.

Run after the feasibility benchmark has produced rows:

    uv run --extra figures python manuscript/figures/figure1.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "falcon_sr_single_feasibility.csv"
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


def _load_rows() -> list[dict]:
    if not DATA.exists():
        sys.exit(
            f"No data at {DATA}; run benchmarks/falcon_sr_single.py first."
        )
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


def _draw_schematic(ax) -> None:
    import matplotlib.patches as patches

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    # 5 boxes evenly spaced across [3, 97]
    box_specs = [
        ("counts\n(n × p)",          3, 17, 14, 11),
        ("dense base\nscore (GEMM)", 22, 17, 14, 11),
        ("top-k\ncandidate\nunion", 41, 14, 14, 14),
        ("sparse\nexclusion\nrefine", 60, 14, 14, 14),
        ("calibrate\n(permutation)", 79, 17, 18, 11),
    ]
    centers_y = []
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
        centers_y.append(y + h / 2)

    # Horizontal arrows between boxes (use mid-line y=22)
    arrow_y = 22
    for i in range(4):
        x_from = box_specs[i][1] + box_specs[i][3]
        x_to = box_specs[i + 1][1]
        ax.annotate(
            "",
            xy=(x_to, arrow_y), xytext=(x_from, arrow_y),
            arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                            color=PALETTE["schematic_edge"]),
        )

    # Adaptive growth back-loop: top-k → refine, return to top-k if unstable
    ax.annotate(
        "",
        xy=(48, 12), xytext=(67, 12),
        arrowprops=dict(
            arrowstyle="-|>", linewidth=0.6,
            color=PALETTE["neutral"],
            connectionstyle="arc3,rad=0.3",
        ),
    )
    ax.text(
        57, 6, "grow top-k if unstable",
        ha="center", va="center",
        fontsize=5.8, color=PALETTE["neutral"], style="italic",
    )

    # Output annotation below calibrate box
    ax.annotate(
        "",
        xy=(88, 12), xytext=(88, 17),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.7,
                        color=PALETTE["schematic_edge"]),
    )
    ax.text(
        88, 9, "edge table + q-values",
        ha="center", va="top",
        fontsize=6.2, color="#1F3A56",
    )


def _panel_recall(ax, rows) -> None:
    fast_by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] != "falcon_sr_fast":
            continue
        cell = (int(r["n"]), int(r["p"]))
        fast_by_cell[cell][int(r["top_k"])].append(r["candidate_recall"])

    p_color = {100: "#88AED0", 500: "#3C6997", 1000: "#1E3F5F"}
    n_dash = {100: ":", 500: "-"}
    for (n, p), k_to_recalls in sorted(fast_by_cell.items()):
        ks = sorted(k_to_recalls)
        means = [_avg(k_to_recalls[k]) for k in ks]
        ax.plot(
            ks, means,
            color=p_color.get(p, PALETTE["neutral"]),
            linestyle=n_dash.get(n, "-"),
            marker="o",
            label=f"n={n}, p={p}",
        )
    ax.set_xlabel("top-$k$ candidate budget")
    ax.set_ylabel("candidate recall\n(vs planted truth)")
    ax.set_xticks([10, 25, 50])
    ax.set_ylim(-0.02, 1.08)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=2, columnspacing=0.8, handlelength=1.2,
        fontsize=5.6,
    )


def _panel_overlap(ax, rows) -> None:
    fast_by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] != "falcon_sr_fast":
            continue
        cell = (int(r["n"]), int(r["p"]))
        fast_by_cell[cell][int(r["top_k"])].append(
            r["edge_overlap_vs_sparcc"]
        )
    p_color = {100: "#88AED0", 500: "#3C6997", 1000: "#1E3F5F"}
    n_dash = {100: ":", 500: "-"}
    for (n, p), k_to_vals in sorted(fast_by_cell.items()):
        ks = sorted(k_to_vals)
        means = [_avg(k_to_vals[k]) for k in ks]
        ax.plot(
            ks, means,
            color=p_color.get(p, PALETTE["neutral"]),
            linestyle=n_dash.get(n, "-"),
            marker="s",
            label=f"n={n}, p={p}",
        )
    ax.axhline(0.95, linestyle="--", linewidth=0.5,
               color=PALETTE["neutral"], alpha=0.7)
    ax.text(10, 0.96, "spec gate 0.95",
            fontsize=5.5, color=PALETTE["neutral"],
            ha="left", va="bottom")
    ax.set_xlabel("top-$k$ candidate budget")
    ax.set_ylabel("edge overlap\nvs SparCC reference")
    ax.set_xticks([10, 25, 50])
    ax.set_ylim(-0.02, 1.08)


def _panel_time(ax, rows) -> None:
    methods_to_plot = [
        "sparcc_py", "pearson_clr",
        "falcon_sr_fast", "falcon_sr_strict",
        "falcon_sr_fast_calibrated",
    ]
    by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in methods_to_plot:
            continue
        # Average over n and top_k for each p (representative cost vs p)
        by_method[r["method"]][int(r["p"])].append(r["wallclock_seconds"])
    for m in methods_to_plot:
        if m not in by_method:
            continue
        ps = sorted(by_method[m])
        ys = [_avg(by_method[m][p]) for p in ps]
        ax.plot(
            ps, ys,
            color=METHOD_COLOR[m],
            marker=METHOD_MARKER[m],
            label=METHOD_LABEL[m],
            linewidth=1.1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (features per domain)")
    ax.set_ylabel("wall-clock (s)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1, columnspacing=0.6, handlelength=1.2,
        fontsize=5.5,
    )


def main():
    install_pub_style()
    rows = _load_rows()

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(183 * MM, 100 * MM))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[0.7, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.45,
        left=0.06, right=0.98, top=0.96, bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    _draw_schematic(ax_a)
    _panel_recall(ax_b, rows)
    _panel_overlap(ax_c, rows)
    _panel_time(ax_d, rows)

    panel_letter(ax_a, "a", dx=-0.005, dy=0.95)
    panel_letter(ax_b, "b")
    panel_letter(ax_c, "c")
    panel_letter(ax_d, "d")

    save_pub(fig, OUT_DIR / "figure1")
    print("Wrote figure1.{svg,pdf,tiff}")


if __name__ == "__main__":
    main()
