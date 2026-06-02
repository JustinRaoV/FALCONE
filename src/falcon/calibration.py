"""Permutation calibration for Falcon-SR candidate edges.

Reference: Falcon-SR design specification, sections 11 and 2.3 of the
2026-06-02 execution design.

We approximate the spec's permutation calibration by recomputing only the
*base score* per permutation, not the full sparse-refinement. The diagnostic
output labels the method ``permutation_base_only`` so downstream code never
silently treats the result as a calibration-tight test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falcon.preprocessing import prepare_log_composition


@dataclass(frozen=True)
class CalibrationResult:
    pvalue_approx: np.ndarray
    qvalue_approx: np.ndarray
    null_max_distribution: np.ndarray
    n_permutations: int
    method: str


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg q-values for an array of p-values.

    q[i] = min_{k >= rank(i)} sorted_p[k] * n / (k + 1), clipped to [0, 1].
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = pvalues.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(pvalues, kind="mergesort")
    sorted_p = pvalues[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw = sorted_p * n / ranks
    cummin = np.minimum.accumulate(raw[::-1])[::-1]
    sorted_q = np.clip(cummin, 0.0, 1.0)
    qvalues = np.empty(n, dtype=np.float64)
    qvalues[order] = sorted_q
    return qvalues


def _single_base_correlation_closed_form(log_composition: np.ndarray) -> np.ndarray:
    n, p = log_composition.shape
    centered = log_composition - log_composition.mean(axis=0, keepdims=True)
    cov_log = (centered.T @ centered) / (n - 1)
    diag = np.diag(cov_log)
    t_mat = diag[:, None] + diag[None, :] - 2.0 * cov_log
    np.fill_diagonal(t_mat, 0.0)
    row_sum = t_mat.sum(axis=1)
    total = row_sum.sum() / (2.0 * (p - 1))
    omega_sq = np.maximum((row_sum - total) / (p - 2), 1e-8)
    omega = np.sqrt(omega_sq)
    denom = 2.0 * np.outer(omega, omega)
    rho = (omega_sq[:, None] + omega_sq[None, :] - t_mat) / denom
    np.clip(rho, -1.0, 1.0, out=rho)
    np.fill_diagonal(rho, 1.0)
    return rho


def _column_permute(log_composition: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, p = log_composition.shape
    out = np.empty_like(log_composition)
    for j in range(p):
        out[:, j] = log_composition[rng.permutation(n), j]
    return out


def calibrate_single(
    counts: np.ndarray,
    candidate_pairs: np.ndarray,
    refined_scores: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = 0,
) -> CalibrationResult:
    prepared = prepare_log_composition(counts)
    log_comp = prepared.log_composition
    candidate_pairs = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
    refined_scores = np.asarray(refined_scores, dtype=np.float64)
    if candidate_pairs.shape[0] != refined_scores.size:
        raise ValueError("candidate_pairs and refined_scores must align")

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutations, dtype=np.float64)
    for r in range(n_permutations):
        permuted = _column_permute(log_comp, rng)
        rho = _single_base_correlation_closed_form(permuted)
        scores = np.abs(rho[candidate_pairs[:, 0], candidate_pairs[:, 1]])
        null_max[r] = float(scores.max()) if scores.size else 0.0

    abs_refined = np.abs(refined_scores)
    geq = null_max[None, :] >= abs_refined[:, None]
    pvals = (1.0 + geq.sum(axis=1)) / (1.0 + n_permutations)
    qvals = benjamini_hochberg(pvals)
    return CalibrationResult(
        pvalue_approx=pvals,
        qvalue_approx=qvals,
        null_max_distribution=null_max,
        n_permutations=n_permutations,
        method="permutation_base_only",
    )


def _cross_basis_omega(log_composition: np.ndarray) -> np.ndarray:
    """Within-domain SparCC basis standard deviations for the cross-domain
    estimator. Equivalent to the helper used by ``sparxcc_base``.
    """
    n, p = log_composition.shape
    centered = log_composition - log_composition.mean(axis=0, keepdims=True)
    cov_log = (centered.T @ centered) / (n - 1)
    diag = np.diag(cov_log)
    t_mat = diag[:, None] + diag[None, :] - 2.0 * cov_log
    np.fill_diagonal(t_mat, 0.0)
    row_sum = t_mat.sum(axis=1)
    total = row_sum.sum() / (2.0 * (p - 1))
    omega_sq = np.maximum((row_sum - total) / (p - 2), 1e-8)
    return np.sqrt(omega_sq)


def _cross_base_correlation(
    log_x: np.ndarray, log_y: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    n = log_x.shape[0]
    zx = log_x - log_x.mean(axis=0, keepdims=True)
    zy = log_y - log_y.mean(axis=0, keepdims=True)
    cov_xy = (zx.T @ zy) / (n - 1)
    rowm = cov_xy.mean(axis=1, keepdims=True)
    colm = cov_xy.mean(axis=0, keepdims=True)
    centered = cov_xy - rowm - colm + cov_xy.mean()
    denom = np.outer(alpha, beta)
    return np.clip(centered / denom, -1.0, 1.0)


def calibrate_cross(
    counts_x: np.ndarray,
    counts_y: np.ndarray,
    candidate_pairs: np.ndarray,
    refined_scores: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = 0,
) -> CalibrationResult:
    prepared_x = prepare_log_composition(counts_x)
    prepared_y = prepare_log_composition(counts_y)
    log_x = prepared_x.log_composition
    log_y = prepared_y.log_composition
    if log_x.shape[0] != log_y.shape[0]:
        raise ValueError("counts_x and counts_y must share sample rows")

    candidate_pairs = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
    refined_scores = np.asarray(refined_scores, dtype=np.float64)
    if candidate_pairs.shape[0] != refined_scores.size:
        raise ValueError("candidate_pairs and refined_scores must align")

    alpha = _cross_basis_omega(log_x)
    beta = _cross_basis_omega(log_y)
    n = log_x.shape[0]
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutations, dtype=np.float64)
    for r in range(n_permutations):
        permuted_y = log_y[rng.permutation(n)]
        rho = _cross_base_correlation(log_x, permuted_y, alpha, beta)
        scores = np.abs(rho[candidate_pairs[:, 0], candidate_pairs[:, 1]])
        null_max[r] = float(scores.max()) if scores.size else 0.0

    abs_refined = np.abs(refined_scores)
    geq = null_max[None, :] >= abs_refined[:, None]
    pvals = (1.0 + geq.sum(axis=1)) / (1.0 + n_permutations)
    qvals = benjamini_hochberg(pvals)
    return CalibrationResult(
        pvalue_approx=pvals,
        qvalue_approx=qvals,
        null_max_distribution=null_max,
        n_permutations=n_permutations,
        method="permutation_base_only",
    )
