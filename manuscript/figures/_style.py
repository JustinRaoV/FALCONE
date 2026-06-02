"""Shared publication style for Falcon-SR figures.

Targets a double-column journal panel (~183 mm wide) with 7 pt sans-serif
labels, editable text in SVG and PDF, 600-DPI TIFF, restrained palette
(SparCC-family blue, Falcon-SR-family orange, prior-accent green, neutral
grey), white background, hairline spines. Panel letters are 8 pt bold,
top-left of each axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4

# Restrained palette: one neutral, one SparCC-family (blue), one
# Falcon-SR family (orange), one accent (green) for prior, plus a soft
# secondary grey for reference / baseline elements.
PALETTE = {
    "sparcc": "#3C6997",
    "pearson": "#88AED0",
    "sparxcc_iter": "#3C6997",
    "sparxcc_base": "#88AED0",
    "falcon_strict": "#A05030",
    "falcon_fast": "#D8753E",
    "falcon_calibrated": "#F2B36E",
    "falcon_prior": "#5A9F6D",
    "neutral": "#666666",
    "schematic_box": "#E7EEF5",
    "schematic_edge": "#3C6997",
    "schematic_prior_box": "#E6F0E7",
    "schematic_prior_edge": "#5A9F6D",
    "grid": "#DDDDDD",
}

METHOD_LABEL = {
    "sparcc_py": "SparCC",
    "pearson_clr": "Pearson(CLR)",
    "sparxcc_base": "SparXCC base",
    "sparxcc_iter": "SparXCC iter",
    "falcon_sr_strict": "Falcon-SR strict",
    "falcon_sr_fast": "Falcon-SR fast",
    "falcon_sr_fast_calibrated": "Falcon-SR fast + calibrate",
    "falcon_sr_cross_fast": "Falcon-SR fast",
    "falcon_sr_cross_fast_calibrated": "Falcon-SR fast + calibrate",
    "falcon_sr_cross_prior": "Falcon-SR fast + prior",
}

METHOD_COLOR = {
    "sparcc_py": PALETTE["sparcc"],
    "pearson_clr": PALETTE["pearson"],
    "sparxcc_iter": PALETTE["sparxcc_iter"],
    "sparxcc_base": PALETTE["sparxcc_base"],
    "falcon_sr_strict": PALETTE["falcon_strict"],
    "falcon_sr_fast": PALETTE["falcon_fast"],
    "falcon_sr_fast_calibrated": PALETTE["falcon_calibrated"],
    "falcon_sr_cross_fast": PALETTE["falcon_fast"],
    "falcon_sr_cross_fast_calibrated": PALETTE["falcon_calibrated"],
    "falcon_sr_cross_prior": PALETTE["falcon_prior"],
}

METHOD_MARKER = {
    "sparcc_py": "o",
    "pearson_clr": "v",
    "sparxcc_iter": "o",
    "sparxcc_base": "v",
    "falcon_sr_strict": "D",
    "falcon_sr_fast": "s",
    "falcon_sr_fast_calibrated": "^",
    "falcon_sr_cross_fast": "s",
    "falcon_sr_cross_fast_calibrated": "^",
    "falcon_sr_cross_prior": "P",
}


def install_pub_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6.5,
        "legend.frameon": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def panel_letter(ax, letter: str, *, dx: float = -0.08, dy: float = 1.02) -> None:
    ax.text(
        dx, dy, letter,
        transform=ax.transAxes,
        fontsize=8.5, fontweight="bold",
        ha="left", va="bottom",
    )


def save_pub(fig, stem: Path | str, *, tiff_dpi: int = 600) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.tiff", dpi=tiff_dpi)
