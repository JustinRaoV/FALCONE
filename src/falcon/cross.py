"""Cross-domain Falcon-SR.

Implements SparXCC Case-C compatible base scoring and edge-driven sparse
refinement. The refinement geometry is described in section 2.1 of the
2026-06-02 execution design: each excluded candidate edge prunes one X row
and one Y column from the centering pool, preserving the H_p ⊗ H_q^T
identity that underlies SparXCC base and iter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falcon.preprocessing import PreprocessReport, prepare_log_composition
from falcon.single import solve_basis_variance_dense, variation_matrix
from falcon.types import CrossCandidateSet


@dataclass(frozen=True)
class CrossBaseResult:
    cov_xy: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    correlation: np.ndarray
    report_x: PreprocessReport
    report_y: PreprocessReport


@dataclass(frozen=True)
class CrossRefinementResult:
    pairs: np.ndarray
    scores: np.ndarray
    excluded_pairs: np.ndarray
    rounds: int
    pruned_x_count: int
    pruned_y_count: int
    fallback_to_base_centering: bool


def _basis_omega(log_composition: np.ndarray) -> np.ndarray:
    variation = variation_matrix(log_composition)
    basis_variance = solve_basis_variance_dense(variation)
    return np.sqrt(basis_variance)


def cross_base_score(counts_x: np.ndarray, counts_y: np.ndarray) -> CrossBaseResult:
    prepared_x = prepare_log_composition(counts_x)
    prepared_y = prepare_log_composition(counts_y)
    log_x = prepared_x.log_composition
    log_y = prepared_y.log_composition
    if log_x.shape[0] != log_y.shape[0]:
        raise ValueError(
            "counts_x and counts_y must share the same sample rows"
        )

    alpha = _basis_omega(log_x)
    beta = _basis_omega(log_y)
    n = log_x.shape[0]
    zx = log_x - log_x.mean(axis=0, keepdims=True)
    zy = log_y - log_y.mean(axis=0, keepdims=True)
    cov_xy = (zx.T @ zy) / (n - 1)

    row_mean = cov_xy.mean(axis=1, keepdims=True)
    col_mean = cov_xy.mean(axis=0, keepdims=True)
    grand = cov_xy.mean()
    centered = cov_xy - row_mean - col_mean + grand
    denom = np.outer(alpha, beta)
    correlation = np.clip(centered / denom, -1.0, 1.0)

    return CrossBaseResult(
        cov_xy=cov_xy,
        alpha=alpha,
        beta=beta,
        correlation=correlation,
        report_x=prepared_x.report,
        report_y=prepared_y.report,
    )


def _centered_score(
    cov_xy: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    excluded_rows: np.ndarray,
    excluded_cols: np.ndarray,
) -> tuple[np.ndarray, bool]:
    p, q = cov_xy.shape
    keep_row_mask = np.ones(p, dtype=bool)
    keep_col_mask = np.ones(q, dtype=bool)
    if excluded_rows.size:
        keep_row_mask[excluded_rows] = False
    if excluded_cols.size:
        keep_col_mask[excluded_cols] = False

    fallback = False
    if keep_row_mask.sum() < 3 or keep_col_mask.sum() < 3:
        keep_row_mask[:] = True
        keep_col_mask[:] = True
        fallback = True

    sub = cov_xy[np.ix_(keep_row_mask, keep_col_mask)]
    row_mean = cov_xy[keep_row_mask, :].mean(axis=0, keepdims=True)
    col_mean = cov_xy[:, keep_col_mask].mean(axis=1, keepdims=True)
    grand = sub.mean()
    centered = cov_xy - row_mean - col_mean + grand
    denom = np.outer(alpha, beta)
    return np.clip(centered / denom, -1.0, 1.0), fallback


def sparse_refine_cross(
    base: CrossBaseResult,
    candidates: CrossCandidateSet,
    *,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
) -> CrossRefinementResult:
    excluded_pairs: list[tuple[int, int]] = []
    excluded_indices: set[int] = set()
    fallback_seen = False
    row_excluded: set[int] = set()
    col_excluded: set[int] = set()
    candidate_pairs = candidates.pairs

    for _ in range(max_exclusions):
        excluded_rows = np.fromiter(row_excluded, dtype=np.int64, count=len(row_excluded))
        excluded_cols = np.fromiter(col_excluded, dtype=np.int64, count=len(col_excluded))
        rho, fallback = _centered_score(
            base.cov_xy, base.alpha, base.beta,
            excluded_rows, excluded_cols,
        )
        fallback_seen = fallback_seen or fallback
        scores = rho[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        abs_scores = np.abs(scores).astype(np.float64)
        if excluded_indices:
            mask = np.zeros(abs_scores.size, dtype=bool)
            mask[list(excluded_indices)] = True
            abs_scores[mask] = -np.inf
        best_idx = int(np.argmax(abs_scores))
        if abs_scores[best_idx] <= exclusion_threshold:
            break
        i, k = candidate_pairs[best_idx]
        excluded_pairs.append((int(i), int(k)))
        excluded_indices.add(best_idx)
        row_excluded.add(int(i))
        col_excluded.add(int(k))

    excluded_rows = np.fromiter(row_excluded, dtype=np.int64, count=len(row_excluded))
    excluded_cols = np.fromiter(col_excluded, dtype=np.int64, count=len(col_excluded))
    rho, fallback = _centered_score(
        base.cov_xy, base.alpha, base.beta,
        excluded_rows, excluded_cols,
    )
    fallback_seen = fallback_seen or fallback
    scores = rho[candidate_pairs[:, 0], candidate_pairs[:, 1]]

    return CrossRefinementResult(
        pairs=candidate_pairs,
        scores=scores,
        excluded_pairs=np.asarray(excluded_pairs, dtype=np.int64).reshape(-1, 2),
        rounds=len(excluded_pairs),
        pruned_x_count=len(row_excluded),
        pruned_y_count=len(col_excluded),
        fallback_to_base_centering=fallback_seen,
    )
