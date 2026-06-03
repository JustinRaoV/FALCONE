"""Tests for the benchmark scaffolding (Python baselines + R adapter +
runner schema)."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from benchmarks.baselines import (
    AVAILABLE_PYTHON_BASELINES,
    BaselineResult,
    pearson_clr,
    sparcc_closed_form,
)
from benchmarks.r_adapters import R_BASELINES, RAdapterSkip, run_r_baseline
from benchmarks.run_benchmark import SCHEMA_FIELDS, run as run_benchmark


def _planted_counts(n=120, p=24, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.eye(p)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        cov[i, j] = cov[j, i] = 0.7
    log_basis = rng.multivariate_normal(np.zeros(p), cov, size=n)
    return np.round(np.exp(log_basis) / np.exp(log_basis).sum(axis=1, keepdims=True) * 5000).astype(np.int64)


# --- Python baselines --------------------------------------------------------


def test_pearson_clr_returns_unit_diagonal_correlation():
    counts = _planted_counts()
    out = pearson_clr(counts)
    assert isinstance(out, BaselineResult)
    assert out.method == "pearson_clr"
    np.testing.assert_allclose(np.diag(out.correlation), 1.0)
    np.testing.assert_allclose(out.correlation, out.correlation.T, atol=1e-12)
    assert (np.abs(out.correlation) <= 1.0 + 1e-9).all()
    assert out.estimand_family == "latent_log_abundance_correlation"


def test_sparcc_closed_form_returns_valid_correlation():
    counts = _planted_counts()
    out = sparcc_closed_form(counts)
    assert isinstance(out, BaselineResult)
    np.testing.assert_allclose(np.diag(out.correlation), 1.0)
    np.testing.assert_allclose(out.correlation, out.correlation.T, atol=1e-12)
    assert (np.abs(out.correlation) <= 1.0 + 1e-9).all()
    assert out.estimand_family == "latent_log_abundance_correlation"


def test_python_baselines_match_advertised_set():
    assert set(AVAILABLE_PYTHON_BASELINES) == {"pearson_clr", "sparcc_closed_form"}


# --- R adapter skip behavior -------------------------------------------------


def test_r_adapter_skips_when_rscript_missing():
    counts = _planted_counts()
    with mock.patch.object(shutil, "which", return_value=None):
        out = run_r_baseline("cclasso", counts)
    assert isinstance(out, RAdapterSkip)
    assert "Rscript" in out.reason


def test_r_adapter_skips_when_baseline_dir_missing(tmp_path: Path, monkeypatch):
    counts = _planted_counts()
    monkeypatch.setenv("FALCON_R_BASELINE_DIR", str(tmp_path / "no_such_dir"))
    with mock.patch.object(shutil, "which", return_value="/usr/local/bin/Rscript"):
        out = run_r_baseline("coat", counts)
    assert isinstance(out, RAdapterSkip)
    assert "not found" in out.reason


def test_r_adapter_secom_always_reports_skip_with_reason():
    counts = _planted_counts()
    out = run_r_baseline("secom", counts)
    assert isinstance(out, RAdapterSkip)
    assert "ANCOMBC" in out.reason or "BiocManager" in out.reason


def test_r_adapter_rejects_unknown_method():
    counts = _planted_counts()
    with pytest.raises(ValueError, match="unknown R baseline"):
        run_r_baseline("not_a_real_method", counts)


def test_r_baselines_advertised_list():
    assert set(R_BASELINES) == {"cclasso", "coat", "secom"}


# --- Runner schema -----------------------------------------------------------


def test_runner_writes_full_schema_to_csv(tmp_path: Path):
    out = tmp_path / "rows.csv"
    # Use a tiny method set so the run is fast even with the training
    # grid being multiple cells. We pick adaptive_threshold (cheap) + the
    # closed-form SparCC baseline.
    run_benchmark(
        [
            "--split",
            "training",
            "--output",
            str(out),
            "--methods",
            "falcon_adaptive_threshold,sparcc_closed_form",
            "--reps",
            "1",
        ]
    )
    assert out.exists()
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) > 0
    assert set(reader.fieldnames) == set(SCHEMA_FIELDS)
    methods = {row["method"] for row in rows}
    assert methods == {"falcon_adaptive_threshold", "sparcc_closed_form"}
    splits = {row["split"] for row in rows}
    assert splits == {"training"}
    families = {row["estimand_family"] for row in rows}
    assert families == {"latent_log_abundance_correlation"}
