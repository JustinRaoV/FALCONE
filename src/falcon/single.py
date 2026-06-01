from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falcon.preprocessing import PreprocessReport, prepare_log_composition


@dataclass(frozen=True)
class SingleBaseResult:
    variation: np.ndarray
    basis_variance: np.ndarray
    correlation: np.ndarray
    preprocess_report: PreprocessReport


def variation_matrix(log_composition: np.ndarray) -> np.ndarray:
    centered = log_composition - log_composition.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    diagonal = np.diag(covariance)
    variation = diagonal[:, None] + diagonal[None, :] - 2.0 * covariance
    np.fill_diagonal(variation, 0.0)
    return variation


def _dense_modifier(p: int) -> np.ndarray:
    modifier = np.ones((p, p), dtype=np.float64)
    np.fill_diagonal(modifier, p - 1.0)
    return modifier


def solve_basis_variance_dense(
    variation: np.ndarray,
    *,
    excluded: np.ndarray | None = None,
    min_variance: float = 1e-4,
) -> np.ndarray:
    p = variation.shape[0]
    modifier = _dense_modifier(p)
    rhs = variation.sum(axis=1).copy()
    if excluded is not None and excluded.size:
        for i, j in excluded:
            rhs[i] -= variation[i, j]
            rhs[j] -= variation[i, j]
            modifier[i, i] -= 1.0
            modifier[j, j] -= 1.0
            modifier[i, j] -= 1.0
            modifier[j, i] -= 1.0
    return np.maximum(np.linalg.solve(modifier, rhs), min_variance)


def correlations_from_basis(
    variation: np.ndarray,
    basis_variance: np.ndarray,
) -> np.ndarray:
    covariance = 0.5 * (
        basis_variance[:, None] + basis_variance[None, :] - variation
    )
    scale = np.sqrt(np.outer(basis_variance, basis_variance))
    correlation = np.clip(covariance / scale, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def single_base_score(counts: np.ndarray) -> SingleBaseResult:
    prepared = prepare_log_composition(counts)
    variation = variation_matrix(prepared.log_composition)
    basis_variance = solve_basis_variance_dense(variation)
    correlation = correlations_from_basis(variation, basis_variance)
    return SingleBaseResult(
        variation=variation,
        basis_variance=basis_variance,
        correlation=correlation,
        preprocess_report=prepared.report,
    )
