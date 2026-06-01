"""
Fig 2 — Cross-domain inference (CrossNet): pipeline + accuracy + FPR.

3 panels, single-column width (~170 mm):
  a. CrossNet pipeline schematic (centering identity + FISTA + biological prior)
  b. Cross-domain accuracy: 5 methods x 4 metrics (mean +/- s.d.)
  c. False-positive rate close-up: where CrossNet differentiates
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

METHOD_ORDER = ["naive_clr", "sparxcc_base", "sparxcc_iter",
                "spieceasi_cross", "bias_corrected"]
METHOD_LABEL = {"naive_clr":       "Naive CLR",
                "sparxcc_like":    "SparXCC-like",
                "sparxcc_base":    "SparXCC (base)",
                "sparxcc_iter":    "SparXCC (iter)",
                "spieceasi_cross": "SPIEC-EASI-cross",
                "bias_corrected":  "CrossNet"}
METHOD_COLOR = {"naive_clr":       C["gray"],
                "sparxcc_base":    C["blue_mid"],
                "sparxcc_iter":    C["blue"],
                "spieceasi_cross": C["green"],
                "bias_corrected":  C["red"]}

CROSS = load_csv("cross_domain")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, *, fc, lw=0.7):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                       facecolor=fc, edgecolor="#2f2f2f", linewidth=lw,
                       transform=ax.transAxes)
    ax.add_patch(p)


def _txt(ax, x, y, text, *, fs=8, fw="normal", tc="black"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, transform=ax.transAxes,
            linespacing=1.25)


def _arrow(ax, start, end, *, lw=1.0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#2f2f2f", lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                xycoords="axes fraction", textcoords="axes fraction")


# ---------------------------------------------------------------------------
# Panel a — CrossNet pipeline schematic
# ---------------------------------------------------------------------------


def panel_a_pipeline(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.97, "a    CrossNet pipeline", ha="left", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)

    # Top row: two domain inputs (taller boxes so two-line text fits cleanly)
    _box(ax, 0.04, 0.66, 0.26, 0.20, fc=C["blue"])
    _txt(ax, 0.17, 0.78, r"Phage  $X \in \Delta^{p-1}$",
         fs=8.5, fw="bold", tc="white")
    _txt(ax, 0.17, 0.71, r"$\sum_i x_i = 1$", fs=7.5, tc="white")

    _box(ax, 0.70, 0.66, 0.26, 0.20, fc=C["red"])
    _txt(ax, 0.83, 0.78, r"Bacteria  $Y \in \Delta^{q-1}$",
         fs=8.5, fw="bold", tc="white")
    _txt(ax, 0.83, 0.71, r"$\sum_j y_j = 1$", fs=7.5, tc="white")

    # Middle: arrows from each domain into the centering identity
    _arrow(ax, (0.17, 0.66), (0.30, 0.56))
    _arrow(ax, (0.83, 0.66), (0.70, 0.56))

    # Centering identity box
    _box(ax, 0.10, 0.43, 0.80, 0.13, fc=C["yellow"])
    _txt(ax, 0.50, 0.49,
         r"$T = H_p\,\Omega\,H_q^{\!\top}$,  "
         r"$H_p = I_p - \frac{1}{p}\mathbf{1}\mathbf{1}^{\!\top}$",
         fs=11, fw="bold")

    # FISTA box (left) + Prior box (right)
    _arrow(ax, (0.36, 0.43), (0.30, 0.30))
    _arrow(ax, (0.64, 0.43), (0.70, 0.30))

    _box(ax, 0.04, 0.11, 0.52, 0.18, fc=C["red"])
    _txt(ax, 0.30, 0.225,
         r"$\min_{\Omega}\; \frac{1}{2}\|T - H_p\Omega H_q^{\!\top}\|_F^2"
         r" + \lambda_1\|\Omega\|_1$",
         fs=8.5, fw="bold", tc="white")
    _txt(ax, 0.30, 0.15,
         r"FISTA  ·  $O(1/k^{2})$  ·  $K \leq 20$",
         fs=8, tc="white")

    _box(ax, 0.60, 0.11, 0.36, 0.18, fc=C["highlight"])
    _txt(ax, 0.78, 0.225, r"Biological prior  $\Omega_{\mathrm{prior}}$",
         fs=8.5, fw="bold")
    _txt(ax, 0.78, 0.15, "CRISPR-spacer / iPHoP\n(lytic = negative entry)",
         fs=7.5)

    # Side arrow showing prior feeds into FISTA
    _arrow(ax, (0.60, 0.20), (0.56, 0.20))


# ---------------------------------------------------------------------------
# Panel b — Cross-domain accuracy bars
# ---------------------------------------------------------------------------


def panel_b_accuracy(ax) -> None:
    if not CROSS:
        ax.text(0.5, 0.5, "(no cross-domain data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"], style="italic")
        ax.set_axis_off()
        return

    metrics = ["corr", "sign_acc", "sensitivity", "specificity"]
    metric_labels = [r"Corr$(\hat\Omega, \Omega)$",
                     "Sign acc.", "Sens.", "Spec."]
    x = np.arange(len(metrics))
    w = 0.21
    for i, m in enumerate(METHOD_ORDER):
        rows = [r for r in CROSS if r["method"] == m]
        if not rows:
            continue
        means = []
        stds = []
        for key in metrics:
            vals = [r[key] for r in rows if r[key] is not None]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=stds, capsize=2,
               color=METHOD_COLOR[m], edgecolor="none", alpha=0.92,
               label=METHOD_LABEL[m],
               error_kw={"linewidth": 0.7, "capthick": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=8.5)
    ax.tick_params(axis="x", pad=3)
    ax.set_ylabel(r"Score (mean $\pm$ s.d.)")
    ax.set_title(r"b    Cross-domain accuracy",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":")
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=4, fontsize=8, handlelength=1.2,
                    columnspacing=1.0, handletextpad=0.4,
                    frameon=False)


# ---------------------------------------------------------------------------
# Panel c — Cross-domain FPR
# ---------------------------------------------------------------------------


def panel_c_fpr(ax) -> None:
    if not CROSS:
        ax.text(0.5, 0.5, "(no cross-domain data)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"], style="italic")
        ax.set_axis_off()
        return

    means = []
    stds = []
    for m in METHOD_ORDER:
        rows = [r for r in CROSS if r["method"] == m
                and r.get("specificity") is not None]
        if not rows:
            means.append(np.nan)
            stds.append(0)
            continue
        spec = np.array([r["specificity"] for r in rows])
        fpr = 1.0 - spec
        means.append(float(fpr.mean()))
        stds.append(float(fpr.std()))
    x = np.arange(len(METHOD_ORDER))
    ax.bar(x, means, 0.55, yerr=stds, capsize=2.5,
           color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           edgecolor="none", alpha=0.92,
           error_kw={"linewidth": 0.7, "capthick": 0.7})
    for xi, v in zip(x, means):
        if not np.isnan(v):
            ax.text(xi, v + max(means) * 0.04,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER],
                       fontsize=8, rotation=12, ha="right")
    ax.set_ylabel(r"False-positive rate  ($1 - $specificity)")
    ax.set_title(r"c    CrossNet reduces FPR by $\sim 25\%$",
                 loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(m for m in means if not np.isnan(m)) * 1.35)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)

    # Relative-improvement annotation: CrossNet vs SparXCC iterative (most relevant baseline)
    last = len(METHOD_ORDER) - 1
    if not np.isnan(means[2]) and not np.isnan(means[last]):
        rel = (means[2] - means[last]) / means[2] * 100
        ax.annotate(f"$-{rel:.0f}\\%$ rel.",
                    xy=(last, means[last] + 0.012),
                    xytext=(last - 0.9, means[2] * 0.5),
                    fontsize=10, color=C["red"], ha="center",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0,
                                    connectionstyle="arc3,rad=-0.3"))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def create() -> plt.Figure:
    fig = plt.figure(figsize=(170 / 25.4, 165 / 25.4))
    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.80, wspace=0.32,
                  left=0.10, right=0.96, top=0.95, bottom=0.13,
                  height_ratios=[1.05, 0.95])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_a_pipeline(ax_a)
    panel_b_accuracy(ax_b)
    panel_c_fpr(ax_c)
    return fig


if __name__ == "__main__":
    fig = create()
    save_fig(fig, "fig2_falcon_cross_domain")
    print("Saved fig2_falcon_cross_domain.{pdf,svg,png} to", FIG_DIR)
    plt.close(fig)
