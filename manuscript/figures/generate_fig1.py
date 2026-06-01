"""
Fig 1 — Single-domain inference (FastProp): pipeline + accuracy + scaling.

4 panels, 2x2 grid, single-column width (~170 mm):
  a. Pipeline schematic (FastProp algorithm flow)
  b. AUROC vs p, four methods (head-to-head with SparCC + Pearson baselines)
  c. Recall@K vs p, four methods (where FastProp's shrinkage wins)
  d. Wall-clock scalability (FastProp + RandProp + SparCC extrapolation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import C, FIG_DIR, apply_style, load_csv, save_fig  # noqa: E402

apply_style()

METHOD_INFO = {
    "fastprop":         ("FastProp",         C["red"],         "o"),
    "sparcc_py":        ("SparCC",           C["blue"],        "s"),
    "pearson_clr":      ("Pearson (CLR)",    C["blue_mid"],    "^"),
    "pearson_raw":      ("Pearson (raw)",    C["gray"],        "v"),
    "spieceasi_glasso": ("SPIEC-EASI-glasso", C["green"],      "D"),
    "spieceasi_mb":     ("SPIEC-EASI-MB",    "#d4a017",        "P"),
}

COMPARISON = load_csv("method_comparison")
SCALABILITY = load_csv("scalability")


# ---------------------------------------------------------------------------
# Panel a — FastProp pipeline schematic
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, *, fc, lw=0.7):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                       facecolor=fc, edgecolor="#2f2f2f", linewidth=lw,
                       transform=ax.transAxes)
    ax.add_patch(p)


def _arrow(ax, start, end, *, lw=1.0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#2f2f2f", lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                xycoords="axes fraction", textcoords="axes fraction")


def panel_a_pipeline(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "a    FastProp pipeline", ha="left", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)

    steps = [
        (0.02, "Counts\n" + r"$X \in \mathbb{N}^{n \times p}$", C["cream"], "black"),
        (0.21, "Zero replace\n+ CLR", C["blue_light"], "black"),
        (0.40, "Ledoit-Wolf\nshrinkage", C["blue_mid"], "white"),
        (0.59, "Proportionality\n" + r"$\rho_p = 2\hat\Sigma_{ij} / (\hat\Sigma_{ii}+\hat\Sigma_{jj})$",
         C["red"], "white"),
        (0.81, "Fisher-$z$\n+ BH-FDR", C["green"], "white"),
    ]
    w, h = 0.17, 0.45
    centers = []
    for x, text, color, tc in steps:
        _box(ax, x, 0.25, w, h, fc=color)
        ax.text(x + w / 2, 0.25 + h / 2, text, ha="center", va="center",
                fontsize=7.5, color=tc, transform=ax.transAxes,
                linespacing=1.25)
        centers.append((x + w / 2, 0.25 + h / 2))
    for i in range(len(steps) - 1):
        xL = steps[i][0] + w
        xR = steps[i + 1][0]
        y = 0.25 + h / 2
        _arrow(ax, (xL + 0.002, y), (xR - 0.002, y))

    ax.text(0.5, 0.10,
            r"Single BLAS GEMM   ·   $O(np^{2})$   ·   "
            r"no bootstrap, no iterative exclusion",
            ha="center", va="center", fontsize=8.5, color=C["gray"],
            style="italic", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Panel b & c helpers
# ---------------------------------------------------------------------------


def _series_by_p(metric, n_fixed, effect_fixed):
    series = {}
    for m in METHOD_INFO:
        rows = [r for r in COMPARISON
                if r["method"] == m
                and abs(r["n"] - n_fixed) < 0.5
                and abs(r["effect"] - effect_fixed) < 1e-6
                and r[metric] is not None
                and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["p"])
        ps = np.array([r["p"] for r in rows])
        ys = np.array([r[metric] for r in rows])
        stds = np.array([r.get(metric.replace("_mean", "_std"), 0) or 0
                         for r in rows])
        series[m] = (ps, ys, stds)
    return series


def panel_b_auroc(ax) -> None:
    series = _series_by_p("auroc_mean", n_fixed=1000, effect_fixed=0.7)
    if not series:
        ax.text(0.5, 0.5, "(no comparison data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"], style="italic")
        ax.set_axis_off()
        return
    for m, (ps, ys, stds) in series.items():
        label, color, marker = METHOD_INFO[m]
        ax.errorbar(ps, ys, yerr=stds, marker=marker, color=color,
                    label=label, capsize=2.5, markersize=5,
                    linewidth=1.3, capthick=0.6, elinewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Feature dimension  $p$")
    ax.set_ylabel("AUROC")
    ax.set_title(r"b    AUROC ($n = 1{,}000$, $\rho = 0.7$)",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":")
    ax.grid(True, which="both", alpha=0.15, linewidth=0.4)
    ax.legend(loc="lower left", fontsize=7.5, handlelength=1.4,
              labelspacing=0.3, frameon=True, fancybox=False,
              edgecolor="#bbbbbb").get_frame().set_linewidth(0.4)


def panel_c_recall(ax) -> None:
    series = _series_by_p("recall_at_K_mean", n_fixed=1000, effect_fixed=0.7)
    if not series:
        ax.text(0.5, 0.5, "(no comparison data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"], style="italic")
        ax.set_axis_off()
        return
    for m, (ps, ys, stds) in series.items():
        label, color, marker = METHOD_INFO[m]
        ax.errorbar(ps, ys, yerr=stds, marker=marker, color=color,
                    label=label, capsize=2.5, markersize=5,
                    linewidth=1.3, capthick=0.6, elinewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Feature dimension  $p$")
    ax.set_ylabel(r"Recall@$K$")
    ax.set_title(r"c    Recall@$K$ ($n = 1{,}000$, $\rho = 0.7$)",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":")
    ax.grid(True, which="both", alpha=0.15, linewidth=0.4)


# ---------------------------------------------------------------------------
# Panel d — Wall-clock scalability
# ---------------------------------------------------------------------------


def panel_d_scalability(ax) -> None:
    if not SCALABILITY:
        ax.text(0.5, 0.5, "(no scalability data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"], style="italic")
        ax.set_axis_off()
        return

    groups = {}
    for r in SCALABILITY:
        n = int(r["n"])
        if n < 500:
            continue
        groups.setdefault(n, {"p": [], "fp": [], "rp": []})
        groups[n]["p"].append(int(r["p"]))
        groups[n]["fp"].append(r["fastprop_sec"])
        groups[n]["rp"].append(r["randprop_sec"])

    palette = {500: C["blue_light"], 1000: C["blue_mid"],
               2000: C["blue"], 5000: C["blue_deep"]}
    markers = {500: "^", 1000: "D", 2000: "v", 5000: "P"}

    for n in sorted(groups):
        g = groups[n]
        order = np.argsort(g["p"])
        ps = np.array(g["p"])[order]
        fp = np.array([v if v is not None else np.nan for v in g["fp"]])[order]
        mask = ~np.isnan(fp)
        if mask.sum() < 2:
            continue
        ax.loglog(ps[mask], fp[mask],
                  marker=markers.get(n, "o"), markersize=4.8,
                  color=palette.get(n, C["blue"]), linewidth=1.4,
                  label=f"FastProp  $n={n}$")

    # RandProp at n=500
    if 500 in groups:
        g = groups[500]
        order = np.argsort(g["p"])
        ps = np.array(g["p"])[order]
        rp = np.array([v if v is not None else np.nan for v in g["rp"]])[order]
        mask = ~np.isnan(rp)
        if mask.sum() >= 2:
            ax.loglog(ps[mask], rp[mask], marker="x", markersize=5.5,
                      markeredgewidth=1.0, color=C["red"], linewidth=1.4,
                      linestyle="--", label=r"RandProp  $n=500$")

    # Reference lines: SparCC (B*I times FastProp at n=500) and FastSpar.
    # FastSpar paper [Watts 2019] reports mean 32x single-thread speedup
    # over SparCC and 221x with 16 threads, so:
    #   FastSpar (1 thread)  ≈ SparCC / 32
    #   FastSpar (16 threads) ≈ SparCC / 221
    if 500 in groups:
        g = groups[500]
        order = np.argsort(g["p"])
        ps = np.array(g["p"])[order]
        fp = np.array([v if v is not None else np.nan for v in g["fp"]])[order]
        mask = ~np.isnan(fp)
        if mask.sum() >= 2:
            sparcc_full = 100 * 10 * fp[mask]
            ax.loglog(ps[mask], sparcc_full, marker="v", markersize=4.5,
                      color=C["gray"], linewidth=1.4, linestyle=":",
                      label=r"SparCC est. ($B$=100, $I$=10)")
            # FastSpar single-thread = SparCC / 32 (Watts et al. 2019)
            ax.loglog(ps[mask], sparcc_full / 32.0, marker="*",
                      markersize=6.5, color="#d4a017", linewidth=1.4,
                      linestyle="-.", label=r"FastSpar 1-thr (Watts 2019)")
            # FastSpar 16-thread = SparCC / 221
            ax.loglog(ps[mask], sparcc_full / 221.0, marker="*",
                      markersize=6.5, color="#8b6914", linewidth=1.4,
                      linestyle="--", label=r"FastSpar 16-thr (Watts 2019)")

    ax.set_xlabel(r"Feature dimension  $p$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("d    Computational scalability",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, which="both", alpha=0.15, linewidth=0.4)
    # Two-column legend to keep panel d tidy with 7 lines.
    leg = ax.legend(loc="upper left", ncol=2, fontsize=6.8,
                    handlelength=1.6, labelspacing=0.25, columnspacing=0.8,
                    handletextpad=0.4, frameon=True, fancybox=False,
                    edgecolor="#bbbbbb", facecolor="white", framealpha=0.95)
    leg.get_frame().set_linewidth(0.4)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def create() -> plt.Figure:
    fig = plt.figure(figsize=(170 / 25.4, 165 / 25.4))
    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.50, wspace=0.32,
                  left=0.09, right=0.96, top=0.95, bottom=0.08,
                  height_ratios=[0.85, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    # Move panel d to second row, but we already put b/c in second row
    # → use 2x2 layout: a spans top, b/c side by side. Then add d below
    # Actually re-do as 3x2 for cleanliness.
    plt.close(fig)

    fig = plt.figure(figsize=(170 / 25.4, 195 / 25.4))
    gs = GridSpec(3, 2, figure=fig,
                  hspace=0.65, wspace=0.32,
                  left=0.10, right=0.96, top=0.96, bottom=0.06,
                  height_ratios=[0.75, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    panel_a_pipeline(ax_a)
    panel_b_auroc(ax_b)
    panel_c_recall(ax_c)
    panel_d_scalability(ax_d)
    return fig


if __name__ == "__main__":
    fig = create()
    save_fig(fig, "fig1_falcon_single_domain")
    print("Saved fig1_falcon_single_domain.{pdf,svg,png} to", FIG_DIR)
    plt.close(fig)
