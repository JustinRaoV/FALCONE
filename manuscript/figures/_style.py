"""Shared matplotlib style + colour palette + CSV loader for FALCON figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
FIG_DIR = REPO / "manuscript" / "figures"


# ---------------------------------------------------------------------------
# Style — calibrated for a single-column manuscript (~150 mm column width)
# ---------------------------------------------------------------------------

def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
    })


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C = {
    "blue_deep":  "#08306b",
    "blue":       "#2166ac",
    "blue_mid":   "#4393c3",
    "blue_light": "#92c5de",
    "red":        "#b2182b",
    "red_light":  "#f4a582",
    "green":      "#1b7837",
    "gray":       "#5a5a5a",
    "gray_light": "#bdbdbd",
    "cream":      "#f7f7f7",
    "yellow":     "#fce8b3",
    "domain_x":   "#2166ac",
    "domain_y":   "#b2182b",
    "bias":       "#fddbc7",
    "highlight":  "#f4a582",
}


# ---------------------------------------------------------------------------
# CSV reader (no pandas dependency — keeps figures importable on minimal envs)
# ---------------------------------------------------------------------------

def load_csv(name: str) -> list[dict]:
    """Read data/<name>.csv into a list of dicts with float coercion.

    Returns [] if the file is missing or empty, so figure code can degrade
    gracefully when only some benchmarks have been run.
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
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


def save_fig(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
