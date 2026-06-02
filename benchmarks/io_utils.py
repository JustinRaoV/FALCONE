"""CSV I/O for Falcon-SR feasibility benchmarks.

Two canonical tables, one per inference family, under ``data/`` at the
repository root:

  data/falcon_sr_single_feasibility.csv
  data/falcon_sr_cross_feasibility.csv

All times are seconds; all fractions are in [0, 1]; peak memory is bytes
from ``tracemalloc.get_traced_memory``. Each row is a single
(method, cell, replicate) measurement so downstream aggregation can stay
in pandas-or-equivalent territory rather than baking summary statistics
into the CSV schema.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


COLUMNS = {
    "falcon_sr_single_feasibility": [
        "method", "replicate", "n", "p", "density", "top_k",
        "candidate_count", "candidate_recall",
        "edge_overlap_vs_sparcc", "sign_accuracy_vs_truth",
        "auroc_vs_truth", "recall_at_K_vs_truth",
        "wallclock_seconds", "peak_bytes",
        "fallback_reason", "calibration_method",
    ],
    "falcon_sr_cross_feasibility": [
        "method", "replicate", "n", "p", "q", "density", "top_k",
        "candidate_count", "candidate_recall",
        "edge_overlap_vs_sparxcc_iter", "sign_accuracy_vs_truth",
        "auroc_vs_truth", "recall_at_K_vs_truth",
        "wallclock_seconds", "peak_bytes",
        "fallback_reason", "calibration_method",
        "prior_count", "data_disagreed_with_prior_count",
    ],
}


def write_table(table_name: str, rows: Iterable[dict]) -> Path:
    if table_name not in COLUMNS:
        raise KeyError(
            f"Unknown table {table_name!r}; expected one of {sorted(COLUMNS)}"
        )
    cols = COLUMNS[table_name]
    out_path = DATA_DIR / f"{table_name}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    return out_path


def read_table(table_name: str) -> list[dict]:
    if table_name not in COLUMNS:
        raise KeyError(f"Unknown table {table_name!r}")
    path = DATA_DIR / f"{table_name}.csv"
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            out.append(row)
    return out


def append_row(table_name: str, row: dict) -> Path:
    if table_name not in COLUMNS:
        raise KeyError(f"Unknown table {table_name!r}")
    cols = COLUMNS[table_name]
    path = DATA_DIR / f"{table_name}.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in cols})
    return path
