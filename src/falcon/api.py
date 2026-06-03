"""Public ``infer_network`` entrypoint for the rebuilt single-domain estimator.

Pipeline:

    counts
      -> validate finite non-negative matrix
      -> prevalence and total-count filter
      -> zero-handling policy
      -> row normalization and CLR transform
      -> estimator candidate
      -> sparse covariance and correlation
      -> stability selection by subsampling (optional)
      -> edge table, diagnostics, and source-data rows

The schema is fixed by
``docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md``
section 5. ``selection_probability`` is the primary uncertainty output.
Approximate p- and q-values stay ``None`` until a calibration procedure
whose simulation FDR behaviour has been measured fills them in.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from falcon.estimators.adaptive_threshold import estimate_adaptive_threshold
from falcon.estimators.pd_correction import apply_pd_correction
from falcon.estimators.weighted_sparse import estimate_weighted_sparse
from falcon.preprocessing import prepare_log_composition
from falcon.results import EdgeTable, EstimatorDiagnostics, NetworkResult
from falcon.stability import select_by_stability

VALID_ESTIMATORS = ("weighted_sparse", "adaptive_threshold", "pd_sparse")
VALID_SELECTIONS = ("none", "stability")


def _adaptive_lambda(n: int, p: int, c: float = 0.5) -> float:
    return float(c * np.sqrt(np.log(max(p, 2)) / max(n, 1)))


def _clr_centered(log_composition: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_composition)
    if not finite.all():
        # complete_case policy: rows have NaNs; subtract per-row mean of the
        # observed entries and zero-fill the rest so dot products treat
        # missing positions as no contribution.
        row_mean = np.where(
            finite,
            log_composition,
            0.0,
        ).sum(axis=1, keepdims=True) / np.maximum(finite.sum(axis=1, keepdims=True), 1)
        Z = np.where(finite, log_composition - row_mean, 0.0)
    else:
        Z = log_composition - log_composition.mean(axis=1, keepdims=True)
    return Z


def _build_estimator(
    estimator: str,
    *,
    lambda_value: float | None,
    threshold_constant: float,
    threshold_mode: str,
    pd_floor: float,
) -> tuple[Callable[[np.ndarray], "_EstResult"], Callable[[np.ndarray], np.ndarray]]:
    """Return (estimate_fn, support_fn). ``support_fn`` is a thin wrapper
    used by the stability-selection loop; it must return only the
    covariance matrix."""

    if estimator == "weighted_sparse":
        def estimate_fn(Z: np.ndarray) -> _EstResult:
            n, p = Z.shape
            lam = lambda_value if lambda_value is not None else _adaptive_lambda(n, p)
            r = estimate_weighted_sparse(Z, lambda_value=lam)
            return _EstResult(
                covariance=r.covariance,
                correlation=r.correlation,
                lambda_value=r.lambda_value,
                iterations=r.iterations,
                converged=r.converged,
                min_eigenvalue=r.min_eigenvalue,
                notes="",
            )
        def support_fn(Z: np.ndarray) -> np.ndarray:
            return estimate_fn(Z).covariance
    elif estimator == "adaptive_threshold":
        def estimate_fn(Z: np.ndarray) -> _EstResult:
            r = estimate_adaptive_threshold(
                Z, threshold_constant=threshold_constant, mode=threshold_mode
            )
            return _EstResult(
                covariance=r.covariance,
                correlation=r.correlation,
                lambda_value=r.lambda_value,
                iterations=r.iterations,
                converged=r.converged,
                min_eigenvalue=r.min_eigenvalue,
                notes=f"adaptive_threshold_mode={threshold_mode}",
            )
        def support_fn(Z: np.ndarray) -> np.ndarray:
            return estimate_fn(Z).covariance
    elif estimator == "pd_sparse":
        def estimate_fn(Z: np.ndarray) -> _EstResult:
            base = estimate_adaptive_threshold(
                Z, threshold_constant=threshold_constant, mode=threshold_mode
            )
            corrected = apply_pd_correction(base.covariance, floor=pd_floor)
            return _EstResult(
                covariance=corrected.covariance,
                correlation=corrected.correlation,
                lambda_value=base.lambda_value,
                iterations=base.iterations,
                converged=base.converged,
                min_eigenvalue=corrected.min_eigenvalue_after,
                notes=f"pd_corrected; shift={corrected.shift:.4g}",
            )
        def support_fn(Z: np.ndarray) -> np.ndarray:
            return estimate_fn(Z).covariance
    else:
        raise ValueError(
            f"unknown estimator {estimator!r}; valid options: {VALID_ESTIMATORS}"
        )
    return estimate_fn, support_fn


class _EstResult:
    __slots__ = (
        "covariance",
        "correlation",
        "lambda_value",
        "iterations",
        "converged",
        "min_eigenvalue",
        "notes",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_edge_table(
    correlation: np.ndarray,
    covariance: np.ndarray,
    selection_probability: np.ndarray | None,
) -> EdgeTable:
    p = correlation.shape[0]
    triu_i, triu_j = np.triu_indices(p, k=1)
    nonzero = covariance[triu_i, triu_j] != 0
    pairs_i = triu_i[nonzero]
    pairs_j = triu_j[nonzero]
    pairs = np.stack([pairs_i, pairs_j], axis=1).astype(np.int64)
    scores = correlation[pairs_i, pairs_j].astype(np.float64)

    if selection_probability is not None:
        sp = selection_probability[pairs_i, pairs_j].astype(np.float64)
    else:
        sp = None

    return EdgeTable(
        pairs=pairs,
        scores=scores,
        selection_probability=sp,
        pvalue_approx=None,
        qvalue_approx=None,
    )


def infer_network(
    counts: np.ndarray,
    *,
    estimator: str = "weighted_sparse",
    zero_policy: str = "multiplicative",
    selection: str = "stability",
    n_resamples: int = 100,
    subsample_fraction: float = 0.5,
    lambda_value: float | None = None,
    threshold_constant: float = 2.0,
    threshold_mode: str = "hard",
    pd_floor: float = 1e-4,
    min_prevalence: float = 0.0,
    min_total: float = 1.0,
    seed: int = 0,
) -> NetworkResult:
    """Infer a single-domain compositional network.

    Defaults frozen on the 2026-06-03 training grid and validated on the
    2026-06-03 holdout (see ``docs/acceptance-gate-report.md``):

    * ``estimator="weighted_sparse"`` — provides a sparse edge table
      plus stability-based ``selection_probability``. Ranks at or near
      ``sparcc_closed_form`` on every holdout scenario; clearly wins on
      hub-cluster data at ``p >= 500`` where every other tested method
      collapses to near-random AUROC.
    * ``zero_policy="multiplicative"`` — best when zero fraction <= 0.15;
      re-run with ``zero_policy="complete_case"`` when zero fraction > 0.20.
    * ``selection="stability"`` with ``n_resamples=100`` — primary
      uncertainty output is ``selection_probability``.

    Honest trade-off recorded in the README "When to use which estimator"
    table: the default is ~1000x slower than ``sparcc_closed_form``
    (available via ``benchmarks.baselines``) and ties it on
    AUROC / AP within rounding error on most scenarios. Use this default
    when you need sparse output or per-edge uncertainty; use
    ``sparcc_closed_form`` for fast dense ranking. The repository does
    not claim ``weighted_sparse`` outperforms ``sparcc_closed_form`` on
    accuracy.

    See ``docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md``
    section 5 for the schema and section 6 for the data flow.
    """
    if estimator not in VALID_ESTIMATORS:
        raise ValueError(
            f"unknown estimator {estimator!r}; valid options: {VALID_ESTIMATORS}"
        )
    if selection not in VALID_SELECTIONS:
        raise ValueError(
            f"unknown selection {selection!r}; valid options: {VALID_SELECTIONS}"
        )

    prepared = prepare_log_composition(
        counts,
        min_prevalence=min_prevalence,
        min_total=min_total,
        zero_policy=zero_policy,
    )
    Z = _clr_centered(prepared.log_composition)

    estimate_fn, support_fn = _build_estimator(
        estimator,
        lambda_value=lambda_value,
        threshold_constant=threshold_constant,
        threshold_mode=threshold_mode,
        pd_floor=pd_floor,
    )

    full = estimate_fn(Z)

    if selection == "stability":
        stab = select_by_stability(
            Z,
            support_fn,
            n_resamples=n_resamples,
            subsample_fraction=subsample_fraction,
            seed=seed,
        )
        sel_prob = stab.selection_probability
        uncertainty = "selection_probability_only"
    else:
        sel_prob = None
        uncertainty = "no_uncertainty_reported"

    edges = _build_edge_table(full.correlation, full.covariance, sel_prob)

    diagnostics = EstimatorDiagnostics(
        estimator=estimator,
        lambda_value=float(full.lambda_value),
        converged=bool(full.converged),
        iterations=int(full.iterations),
        min_eigenvalue=float(full.min_eigenvalue),
        calibration_method="none",
        uncertainty_interpretation=uncertainty,
        preprocess_report=prepared.report,
        notes=full.notes,
    )

    return NetworkResult(
        edges=edges,
        diagnostics=diagnostics,
        correlation=full.correlation,
    )
