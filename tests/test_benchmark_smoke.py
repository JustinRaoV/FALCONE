"""Smoke test that the feasibility benchmark runners produce schema-valid rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from falcon_sr_single import run_cell as run_single
from falcon_sr_cross import run_cell as run_cross
from io_utils import COLUMNS


def test_single_runner_emits_schema_valid_rows():
    rows = run_single(n=40, p=20, density=0.02, top_k=3, reps=1)
    assert len(rows) == 5  # 5 methods per cell
    expected = set(COLUMNS["falcon_sr_single_feasibility"])
    for row in rows:
        assert set(row.keys()) == expected
        assert row["method"] in {
            "sparcc_py", "pearson_clr",
            "falcon_sr_strict", "falcon_sr_fast",
            "falcon_sr_fast_calibrated",
        }
        assert row["wallclock_seconds"] >= 0
        assert row["peak_bytes"] > 0


def test_cross_runner_emits_schema_valid_rows():
    rows = run_cross(n=40, p=15, q=15, density=0.02, top_k=3, reps=1)
    assert len(rows) == 5
    expected = set(COLUMNS["falcon_sr_cross_feasibility"])
    for row in rows:
        assert set(row.keys()) == expected
        assert row["method"] in {
            "sparxcc_iter", "sparxcc_base",
            "falcon_sr_cross_fast", "falcon_sr_cross_prior",
            "falcon_sr_cross_fast_calibrated",
        }
        assert row["wallclock_seconds"] >= 0
