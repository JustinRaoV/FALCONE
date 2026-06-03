"""Calibration helpers for stability-selection output.

Line B (spec v2 §6.1) — converts `selection_probability` into a
calibrated posterior probability P̂(true_edge | sel_prob) via per-scenario
or pooled isotonic regression on training cells.

The output is a *calibrated posterior*, not a p-value or q-value. The
procedure does not claim FDR control. See `pfer_bound` for the
Meinshausen-Bühlmann family-level diagnostic, which is reported
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import isotonic_regression

VALID_MODES = ("per_scenario", "pooled")


@dataclass
class IsotonicCalibrator:
    """Fit a monotone non-decreasing mapping sel_prob -> P̂(true edge).

    In "per_scenario" mode, fit one isotonic curve per scenario string.
    In "pooled" mode, fit a single global curve regardless of scenario.
    """

    mode: Literal["per_scenario", "pooled"] = "per_scenario"
    _curves: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")

    def fit(
        self,
        sel_prob: np.ndarray,
        is_true_edge: np.ndarray,
        *,
        scenario: str,
    ) -> "IsotonicCalibrator":
        x = np.asarray(sel_prob, dtype=np.float64)
        y = np.asarray(is_true_edge, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"sel_prob shape {x.shape} != is_true_edge shape {y.shape}")
        if x.size == 0:
            raise ValueError("cannot fit isotonic on empty arrays")
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        # scipy.optimize.isotonic_regression returns an OptimizeResult-like
        # object with the fitted values on `.x`. Clip to [0, 1] since the
        # fit values are interpreted as probabilities.
        res = isotonic_regression(ys, increasing=True)
        fit_vals = getattr(res, "x", res) if not isinstance(res, np.ndarray) else res
        fit_vals = np.clip(np.asarray(fit_vals, dtype=np.float64), 0.0, 1.0)
        key = "*" if self.mode == "pooled" else scenario
        self._curves[key] = (xs, fit_vals)
        return self

    def predict(self, sel_prob: np.ndarray, *, scenario: str) -> np.ndarray:
        key = "*" if self.mode == "pooled" else scenario
        if key not in self._curves:
            raise KeyError(f"calibrator not fit for scenario {scenario!r}")
        xs, fit_vals = self._curves[key]
        return np.interp(
            np.asarray(sel_prob, dtype=np.float64), xs, fit_vals,
            left=fit_vals[0], right=fit_vals[-1],
        )

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._curves))


def reliability_diagram(
    sel_prob: np.ndarray,
    is_true_edge: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_midpoints, observed_frequency, bin_counts)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    # right=True so that predictions exactly on an interior edge fall into
    # the lower bin (e.g., 0.5 -> midpoint 0.45 rather than 0.55). Keeps
    # ECE small when predictions sit on bin boundaries.
    bins = np.clip(np.digitize(sel_prob, edges, right=True) - 1, 0, n_bins - 1)
    obs = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bins == b
        c = int(mask.sum())
        counts[b] = c
        obs[b] = float(is_true_edge[mask].astype(np.float64).mean()) if c > 0 else 0.0
    return mids, obs, counts


def expected_calibration_error(
    bin_midpoints: np.ndarray,
    observed_frequency: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """Weighted average |bin_midpoint - observed_frequency|."""
    total = int(bin_counts.sum())
    if total == 0:
        return 0.0
    weights = bin_counts / total
    return float(np.sum(weights * np.abs(observed_frequency - bin_midpoints)))


def brier_score(predicted: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.mean((p - t) ** 2))


def pfer_bound(*, q_avg: float, pi_thr: float, p_off: int) -> float:
    """Meinshausen-Bühlmann (2010) per-family error rate upper bound.

    PFER ≤ q_avg^2 / ((2 pi_thr − 1) × p_off), valid for pi_thr > 0.5.

    Reported as a family-level diagnostic alongside the calibrated
    posterior; never combined on the same axis.
    """
    if pi_thr <= 0.5:
        raise ValueError(f"pi_thr must be > 0.5 for M-B bound; got {pi_thr}")
    denominator = (2.0 * pi_thr - 1.0) * p_off
    return float((q_avg ** 2) / denominator)
