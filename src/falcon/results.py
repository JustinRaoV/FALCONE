"""Public result types for the rebuilt single-domain estimator.

The schema follows
``docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md``
section 5. ``selection_probability`` is the primary uncertainty output;
approximate p- and q-values are populated only when a calibration procedure
whose simulation FDR behavior has been measured fills them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALID_ESTIMATORS = frozenset(
    {
        "weighted_sparse",
        "adaptive_threshold",
        "pd_sparse",
    }
)

VALID_CALIBRATIONS = frozenset(
    {
        "none",
        "permutation_base_only",
        "subsampling",
        "empirical_isotonic_per_scenario",
        "empirical_isotonic_pooled",
        "meinshausen_buhlmann_bound",
    }
)

VALID_UNCERTAINTY_INTERPRETATIONS = frozenset(
    {
        "no_uncertainty_reported",
        "selection_probability_only",
        "selection_probability_with_approx_fdr",
        "permutation_base_only",
        "permutation_max_statistic",
        "calibrated_posterior",
        "calibrated_posterior_pooled",
    }
)


@dataclass(frozen=True)
class EdgeTable:
    """Sparse edge table emitted by ``infer_network``.

    ``pairs`` has shape ``(n_edges, 2)`` and uses canonical ``i < j``
    ordering for single-domain inference. ``scores`` are the refined
    correlation values. The three optional uncertainty arrays are
    populated when the estimator and calibration support them.
    """

    pairs: np.ndarray
    scores: np.ndarray
    selection_probability: np.ndarray | None = None
    pvalue_approx: np.ndarray | None = None
    qvalue_approx: np.ndarray | None = None
    posterior_probability: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("EdgeTable.pairs must have two columns")
        n = self.pairs.shape[0]
        if self.scores.shape != (n,):
            raise ValueError("EdgeTable.scores length must match pairs length")
        for name in (
            "selection_probability",
            "pvalue_approx",
            "qvalue_approx",
            "posterior_probability",
        ):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (n,):
                raise ValueError(f"EdgeTable.{name} length must match pairs length")


@dataclass(frozen=True)
class EstimatorDiagnostics:
    """Honest record of how the estimator behaved on the given input."""

    estimator: str
    lambda_value: float
    converged: bool
    iterations: int
    min_eigenvalue: float
    calibration_method: str
    uncertainty_interpretation: str
    preprocess_report: object | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.estimator not in VALID_ESTIMATORS:
            raise ValueError(
                f"unknown estimator label {self.estimator!r}; "
                f"valid labels: {sorted(VALID_ESTIMATORS)}"
            )
        if self.calibration_method not in VALID_CALIBRATIONS:
            raise ValueError(
                f"unknown calibration_method {self.calibration_method!r}; "
                f"valid labels: {sorted(VALID_CALIBRATIONS)}"
            )
        if self.uncertainty_interpretation not in VALID_UNCERTAINTY_INTERPRETATIONS:
            raise ValueError(
                f"unknown uncertainty_interpretation {self.uncertainty_interpretation!r}; "
                f"valid labels: {sorted(VALID_UNCERTAINTY_INTERPRETATIONS)}"
            )


@dataclass(frozen=True)
class NetworkResult:
    """Top-level return value of ``infer_network``."""

    edges: EdgeTable
    diagnostics: EstimatorDiagnostics
    correlation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.correlation is not None:
            if self.correlation.ndim != 2 or self.correlation.shape[0] != self.correlation.shape[1]:
                raise ValueError("NetworkResult.correlation must be square when present")


@dataclass(frozen=True)
class CalibrationReport:
    """Per-cell or aggregate calibration evidence emitted by the
    Line B isotonic procedure. Lives alongside EdgeTable, never inside
    it (the cell-level report includes off-diagonal pairs not in the
    selected EdgeTable)."""

    cell_id: str
    scenario: str
    n: int
    p: int
    n_off_diagonal_pairs: int
    ece_aggregate: float
    ece_per_scenario: dict[str, float]
    brier_score: float
    reliability_bin_midpoints: np.ndarray
    reliability_observed_frequency: np.ndarray
    reliability_bin_counts: np.ndarray
    pi_train: float
    calibration_method: str

    def __post_init__(self) -> None:
        if self.calibration_method not in VALID_CALIBRATIONS:
            raise ValueError(
                f"invalid calibration_method {self.calibration_method!r}; "
                f"valid: {sorted(VALID_CALIBRATIONS)}"
            )
        for name in (
            "reliability_bin_midpoints",
            "reliability_observed_frequency",
            "reliability_bin_counts",
        ):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
        if not (
            self.reliability_bin_midpoints.shape
            == self.reliability_observed_frequency.shape
            == self.reliability_bin_counts.shape
        ):
            raise ValueError("reliability arrays must share shape")
