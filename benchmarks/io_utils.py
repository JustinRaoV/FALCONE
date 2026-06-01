"""
CSV I/O for FALCON benchmarks and figures.

We use CSV (not JSON) as the data-exchange format so that:
  * benchmark results can be computed remotely (e.g. on a HPC cluster
    with multiprocessing), `scp`-ed back, and consumed directly;
  * partial results are human-readable and easy to inspect;
  * `pandas.read_csv` makes plotting code trivial.

There are four canonical tables, each in ``data/`` at the repository root:

  data/scalability.csv     n, p, fastprop_sec, randprop_sec, host
  data/detection.csv       n, p, effect, n_reps, power_mean, power_std,
                           auroc_mean, auroc_std, recall_at_K_mean,
                           recall_at_K_std
  data/fdr_control.csv     alpha, scenario, n_reps, fpr_mean, fpr_std,
                           fdr_mean, fdr_std
  data/cross_domain.csv    method, replicate, corr, bias, sign_acc,
                           sensitivity, specificity

All times are seconds. All fractions are in [0, 1].
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# Canonical column orders — keep stable so figures don't break when CSVs are
# regenerated on the server.
COLUMNS = {
    "scalability": ["n", "p", "fastprop_sec", "randprop_sec", "host"],
    "detection": [
        "n", "p", "effect", "n_reps",
        "power_mean", "power_std",
        "auroc_mean", "auroc_std",
        "recall_at_K_mean", "recall_at_K_std",
    ],
    "fdr_control": [
        "alpha", "scenario", "n_reps", "fpr_mean", "fpr_std",
        "fdr_mean", "fdr_std",
    ],
    "cross_domain": [
        "method", "replicate",
        "corr", "bias", "sign_acc",
        "sensitivity", "specificity",
    ],
    "method_comparison": [
        "method", "n", "p", "effect", "n_reps",
        "time_sec_mean", "time_sec_std",
        "auroc_mean", "auroc_std",
        "recall_at_K_mean", "recall_at_K_std",
        "null_bias_mean", "null_bias_std",
    ],
}


def write_table(table_name: str, rows: Iterable[dict]) -> Path:
    """Write `rows` to `data/<table_name>.csv` with the canonical column order.

    Missing keys become empty strings; extra keys are ignored. Existing file
    is overwritten.
    """
    if table_name not in COLUMNS:
        raise KeyError(f"Unknown table {table_name!r}; "
                       f"expected one of {sorted(COLUMNS)}")
    cols = COLUMNS[table_name]
    out_path = DATA_DIR / f"{table_name}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    return out_path


def read_table(table_name: str) -> list[dict]:
    """Read `data/<table_name>.csv` and return a list of dicts with float
    coercion of numeric columns. Returns [] if the file does not exist
    (so figure scripts gracefully degrade when a benchmark hasn't been run).
    """
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
    """Append a single row; create file with header if it doesn't exist.

    Useful for rolling saves from long-running benchmarks: every cell that
    finishes is durable, even if the job is killed.
    """
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
