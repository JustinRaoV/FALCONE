"""Cross-domain Falcon-SR.

Implements SparXCC Case-C compatible base scoring and edge-driven sparse
refinement. The refinement geometry is described in section 2.1 of the
2026-06-02 execution design: each excluded candidate edge prunes one X row
and one Y column from the centering pool, preserving the H_p ⊗ H_q^T
identity that underlies SparXCC base and iter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from falcon.calibration import calibrate_cross
from falcon.preprocessing import PreprocessReport, prepare_log_composition
from falcon.prior import (
    PriorEdge,
    apply_prior_shrinkage,
    inject_prior_candidates,
    validate_cross_priors,
)
from falcon.screen import cross_candidates, edge_overlap
from falcon.single import solve_basis_variance_dense, variation_matrix
from falcon.types import (
    CrossCandidateSet,
    EdgeTable,
    NetworkResult,
    ScreenDiagnostics,
)


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


def cross_base_score(
    counts_x: np.ndarray,
    counts_y: np.ndarray,
    *,
    zero_policy: str = "multiplicative",
) -> CrossBaseResult:
    prepared_x = prepare_log_composition(counts_x, zero_policy=zero_policy)
    prepared_y = prepare_log_composition(counts_y, zero_policy=zero_policy)
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


def _strict_refine_cross(
    base: CrossBaseResult,
    *,
    exclusion_threshold: float,
    max_exclusions: int,
) -> CrossRefinementResult:
    """Reference refine path that considers all (p*q) pairs as candidates.

    Used by ``infer_cross(mode='strict')``; quadratic in pq so only
    sensible for small grids or feasibility checks.
    """
    p, q = base.correlation.shape
    rows, cols = np.meshgrid(np.arange(p), np.arange(q), indexing="ij")
    full_pairs = np.column_stack([rows.ravel(), cols.ravel()])
    full = CrossCandidateSet(
        pairs=full_pairs,
        scores=base.correlation.ravel(),
        top_k=q,
        n_features_x=p,
        n_features_y=q,
    )
    return sparse_refine_cross(
        base, full,
        exclusion_threshold=exclusion_threshold,
        max_exclusions=max_exclusions,
    )


def _attach_cross_calibration(
    pairs: np.ndarray,
    scores: np.ndarray,
    counts_x: np.ndarray,
    counts_y: np.ndarray,
    calibration: str | None,
    n_permutations: int,
    seed: int,
):
    if calibration in ("none", None):
        return EdgeTable(pairs=pairs, scores=scores), None, None, None
    if calibration != "permutation":
        raise ValueError(
            f"calibration must be 'permutation' or 'none'; got {calibration!r}"
        )
    cal = calibrate_cross(
        counts_x, counts_y, pairs, scores,
        n_permutations=n_permutations, seed=seed,
    )
    edges = EdgeTable(
        pairs=pairs, scores=scores,
        pvalue_approx=cal.pvalue_approx,
        qvalue_approx=cal.qvalue_approx,
    )
    return edges, cal, cal.method, cal.n_permutations


def _strong_cross_edges(pairs: np.ndarray, scores: np.ndarray, limit: int):
    count = min(limit, pairs.shape[0])
    order = np.argsort(-np.abs(scores))[:count]
    return pairs[order], scores[order]


def _cross_overlap(left_pairs, right_pairs):
    left = set(map(tuple, left_pairs.tolist()))
    right = set(map(tuple, right_pairs.tolist()))
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _cross_sign_stability(left_pairs, left_scores, right_pairs, right_scores):
    left_map = dict(zip(map(tuple, left_pairs.tolist()), np.sign(left_scores)))
    right_map = dict(zip(map(tuple, right_pairs.tolist()), np.sign(right_scores)))
    shared = left_map.keys() & right_map.keys()
    if not shared:
        return 1.0
    return float(np.mean([left_map[pr] == right_map[pr] for pr in shared]))


def infer_cross(
    counts_x: np.ndarray,
    counts_y: np.ndarray,
    *,
    mode: str = "fast",
    top_k: int = 50,
    max_top_k: int | None = None,
    min_abs_score: float | None = None,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
    stability_threshold: float = 0.95,
    zero_policy: str = "multiplicative",
    prior: Sequence[PriorEdge] | None = None,
    prior_weight: float = 0.0,
    prior_target_magnitude: float = 0.3,
    calibration: str | None = "permutation",
    n_permutations: int = 100,
    seed: int = 0,
) -> NetworkResult:
    base = cross_base_score(counts_x, counts_y, zero_policy=zero_policy)
    p, q = base.correlation.shape

    if prior is None:
        prior_pairs = np.empty((0, 2), dtype=np.int64)
        prior_signs = np.empty(0, dtype=np.int64)
        prior_confs = np.empty(0, dtype=np.float64)
    else:
        prior_pairs, prior_signs, prior_confs, _ = validate_cross_priors(
            prior, n_x=p, n_y=q,
        )

    if mode == "strict":
        strict = _strict_refine_cross(
            base,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        # Reconstruct dense matrix from refined scores
        dense = np.zeros((p, q), dtype=np.float64)
        dense[strict.pairs[:, 0], strict.pairs[:, 1]] = strict.scores

        scores = strict.scores
        if prior_weight > 0 and prior_pairs.shape[0]:
            scores, disagreed = apply_prior_shrinkage(
                strict.pairs, strict.scores,
                prior_pairs=prior_pairs,
                prior_signs=prior_signs,
                prior_confs=prior_confs,
                prior_weight=prior_weight,
                target_magnitude=prior_target_magnitude,
            )
            disagreed_count = int(disagreed.sum())
        else:
            disagreed_count = 0

        edges, cal, cal_method, cal_n = _attach_cross_calibration(
            strict.pairs, scores,
            counts_x, counts_y, calibration, n_permutations, seed,
        )
        return NetworkResult(
            edges=edges,
            diagnostics=ScreenDiagnostics(
                initial_top_k=min(q, p),
                final_top_k=min(q, p),
                candidate_count=strict.pairs.shape[0],
                candidate_density=1.0,
                growth_rounds=0,
                overlap_across_budgets=1.0,
                sign_stability_across_budgets=1.0,
                fallback_reason=None,
                calibration_method=cal_method,
                n_permutations=cal_n,
                pruned_x_count=strict.pruned_x_count,
                pruned_y_count=strict.pruned_y_count,
                fallback_to_base_centering=strict.fallback_to_base_centering,
                prior_count=int(prior_pairs.shape[0]),
                data_disagreed_with_prior_count=disagreed_count,
            ),
            initial_matrix=dense,
            calibration=cal,
        )
    if mode != "fast":
        raise ValueError("mode must be 'fast' or 'strict'")

    bound = min(p, q)
    budget = min(top_k, bound)
    if max_top_k is not None and max_top_k < budget:
        raise ValueError("max_top_k must be greater than or equal to top_k")
    max_top_k = min(max_top_k or max(budget, 2 * budget), bound)

    previous_pairs = None
    previous_scores = None
    growth_rounds = 0
    overlap = 1.0
    sign_stability = 1.0
    fallback_reason = None
    strong_edge_limit = p + q

    while True:
        candidates = cross_candidates(
            base.correlation,
            top_k=budget,
            min_abs_score=min_abs_score,
        )
        if prior_pairs.shape[0] and prior_weight > 0:
            candidates = inject_prior_candidates(
                candidates, base.correlation, prior_pairs,
            )
        refined = sparse_refine_cross(
            base, candidates,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        if previous_pairs is not None:
            prev_p, prev_s = _strong_cross_edges(
                previous_pairs, previous_scores, strong_edge_limit
            )
            cur_p, cur_s = _strong_cross_edges(
                refined.pairs, refined.scores, strong_edge_limit
            )
            overlap = _cross_overlap(prev_p, cur_p)
            sign_stability = _cross_sign_stability(prev_p, prev_s, cur_p, cur_s)
            if (
                overlap >= stability_threshold
                and sign_stability >= stability_threshold
            ):
                break
        if budget >= max_top_k:
            fallback_reason = "candidate budget reached before stability"
            break
        previous_pairs = refined.pairs
        previous_scores = refined.scores
        budget = min(2 * budget, max_top_k)
        growth_rounds += 1

    final_pairs = refined.pairs
    final_scores = refined.scores
    if prior_weight > 0 and prior_pairs.shape[0]:
        final_scores, disagreed = apply_prior_shrinkage(
            final_pairs, final_scores,
            prior_pairs=prior_pairs,
            prior_signs=prior_signs,
            prior_confs=prior_confs,
            prior_weight=prior_weight,
            target_magnitude=prior_target_magnitude,
        )
        disagreed_count = int(disagreed.sum())
    else:
        disagreed_count = 0

    edges, cal, cal_method, cal_n = _attach_cross_calibration(
        final_pairs, final_scores,
        counts_x, counts_y, calibration, n_permutations, seed,
    )

    return NetworkResult(
        edges=edges,
        diagnostics=ScreenDiagnostics(
            initial_top_k=min(top_k, bound),
            final_top_k=budget,
            candidate_count=edges.pairs.shape[0],
            candidate_density=candidates.density,
            growth_rounds=growth_rounds,
            overlap_across_budgets=overlap,
            sign_stability_across_budgets=sign_stability,
            fallback_reason=fallback_reason,
            calibration_method=cal_method,
            n_permutations=cal_n,
            pruned_x_count=refined.pruned_x_count,
            pruned_y_count=refined.pruned_y_count,
            fallback_to_base_centering=refined.fallback_to_base_centering,
            prior_count=int(prior_pairs.shape[0]),
            data_disagreed_with_prior_count=disagreed_count,
        ),
        initial_matrix=None,
        calibration=cal,
    )
