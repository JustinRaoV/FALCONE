from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from falcon.preprocessing import PreprocessReport, prepare_log_composition
from falcon.screen import edge_overlap, single_candidates
from falcon.types import (
    CandidateSet,
    EdgeTable,
    NetworkResult,
    ScreenDiagnostics,
)


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


@dataclass(frozen=True)
class StrictRefinementResult:
    correlation: np.ndarray
    basis_variance: np.ndarray
    excluded_pairs: np.ndarray
    rounds: int


def strict_refine_single(
    variation: np.ndarray,
    *,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
) -> StrictRefinementResult:
    excluded: list[tuple[int, int]] = []
    for _ in range(max_exclusions):
        excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
        basis_variance = solve_basis_variance_dense(
            variation,
            excluded=excluded_array,
        )
        correlation = correlations_from_basis(variation, basis_variance)
        absolute = np.abs(correlation)
        np.fill_diagonal(absolute, -np.inf)
        for i, j in excluded:
            absolute[i, j] = -np.inf
            absolute[j, i] = -np.inf
        flat_index = int(np.argmax(absolute))
        i, j = np.unravel_index(flat_index, absolute.shape)
        if absolute[i, j] <= exclusion_threshold:
            break
        excluded.append((min(i, j), max(i, j)))

    excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    basis_variance = solve_basis_variance_dense(
        variation,
        excluded=excluded_array,
    )
    correlation = correlations_from_basis(variation, basis_variance)
    return StrictRefinementResult(
        correlation=correlation,
        basis_variance=basis_variance,
        excluded_pairs=excluded_array,
        rounds=len(excluded),
    )


def solve_basis_variance_sparse(
    variation: np.ndarray,
    *,
    excluded: np.ndarray,
    min_variance: float = 1e-4,
) -> np.ndarray:
    p = variation.shape[0]
    excluded = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    rhs = variation.sum(axis=1).copy()
    if excluded.size:
        edge_variation = variation[excluded[:, 0], excluded[:, 1]]
        rhs -= np.bincount(
            np.concatenate([excluded[:, 0], excluded[:, 1]]),
            weights=np.concatenate([edge_variation, edge_variation]),
            minlength=p,
        )
    degree = np.bincount(excluded.ravel(), minlength=p).astype(np.float64)
    if excluded.size:
        rows = np.concatenate([excluded[:, 0], excluded[:, 1]])
        cols = np.concatenate([excluded[:, 1], excluded[:, 0]])
        adjacency = sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)),
            shape=(p, p),
        )
    else:
        adjacency = sparse.csr_matrix((p, p))

    def matvec(vector: np.ndarray) -> np.ndarray:
        return vector.sum() + (p - 2.0 - degree) * vector - adjacency @ vector

    operator = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    solution, info = cg(operator, rhs, rtol=1e-10, atol=1e-12, maxiter=10 * p)
    if info != 0:
        raise RuntimeError(f"sparse basis solve did not converge: info={info}")
    return np.maximum(solution, min_variance)


@dataclass(frozen=True)
class SparseRefinementResult:
    pairs: np.ndarray
    scores: np.ndarray
    basis_variance: np.ndarray
    excluded_pairs: np.ndarray
    rounds: int


def _candidate_scores(
    variation: np.ndarray,
    basis_variance: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    left = pairs[:, 0]
    right = pairs[:, 1]
    covariance = 0.5 * (
        basis_variance[left] + basis_variance[right] - variation[left, right]
    )
    return np.clip(
        covariance / np.sqrt(basis_variance[left] * basis_variance[right]),
        -1.0,
        1.0,
    )


def sparse_refine_single(
    variation: np.ndarray,
    candidates: CandidateSet,
    *,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
) -> SparseRefinementResult:
    excluded: list[tuple[int, int]] = []
    excluded_indices: set[int] = set()
    for _ in range(max_exclusions):
        excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
        basis_variance = solve_basis_variance_sparse(
            variation,
            excluded=excluded_array,
        )
        scores = _candidate_scores(variation, basis_variance, candidates.pairs)
        available = np.ones(scores.size, dtype=bool)
        if excluded_indices:
            available[list(excluded_indices)] = False
        masked = np.where(available, np.abs(scores), -np.inf)
        edge_index = int(np.argmax(masked))
        if masked[edge_index] <= exclusion_threshold:
            break
        excluded_indices.add(edge_index)
        excluded.append(tuple(candidates.pairs[edge_index]))

    excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    basis_variance = solve_basis_variance_sparse(
        variation,
        excluded=excluded_array,
    )
    scores = _candidate_scores(variation, basis_variance, candidates.pairs)
    return SparseRefinementResult(
        pairs=candidates.pairs,
        scores=scores,
        basis_variance=basis_variance,
        excluded_pairs=excluded_array,
        rounds=len(excluded),
    )


def _sign_stability(left: EdgeTable, right: EdgeTable) -> float:
    left_map = dict(zip(map(tuple, left.pairs), np.sign(left.scores)))
    right_map = dict(zip(map(tuple, right.pairs), np.sign(right.scores)))
    shared = left_map.keys() & right_map.keys()
    return 1.0 if not shared else float(
        np.mean([left_map[pair] == right_map[pair] for pair in shared])
    )


def _strong_edges(edges: EdgeTable, limit: int) -> EdgeTable:
    count = min(limit, edges.pairs.shape[0])
    indices = np.argsort(-np.abs(edges.scores))[:count]
    return EdgeTable(pairs=edges.pairs[indices], scores=edges.scores[indices])


def infer_single(
    counts: np.ndarray,
    *,
    mode: str = "fast",
    top_k: int = 50,
    max_top_k: int | None = None,
    min_abs_score: float | None = None,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
    stability_threshold: float = 0.95,
) -> NetworkResult:
    base = single_base_score(counts)
    p = base.correlation.shape[0]
    if mode == "strict":
        strict = strict_refine_single(
            base.variation,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        rows, cols = np.triu_indices(p, k=1)
        pairs = np.column_stack([rows, cols])
        return NetworkResult(
            edges=EdgeTable(pairs=pairs, scores=strict.correlation[rows, cols]),
            diagnostics=ScreenDiagnostics(
                initial_top_k=p - 1,
                final_top_k=p - 1,
                candidate_count=pairs.shape[0],
                candidate_density=1.0,
                growth_rounds=0,
                overlap_across_budgets=1.0,
                sign_stability_across_budgets=1.0,
                fallback_reason=None,
            ),
            initial_matrix=strict.correlation,
        )
    if mode != "fast":
        raise ValueError("mode must be 'fast' or 'strict'")

    budget = min(top_k, p - 1)
    if max_top_k is not None and max_top_k < budget:
        raise ValueError("max_top_k must be greater than or equal to top_k")
    max_top_k = min(max_top_k or max(budget, 2 * budget), p - 1)
    previous = None
    growth_rounds = 0
    overlap = 1.0
    sign_stability = 1.0
    fallback_reason = None
    strong_edge_limit = p
    while True:
        candidates = single_candidates(
            base.correlation,
            top_k=budget,
            min_abs_score=min_abs_score,
        )
        refined = sparse_refine_single(
            base.variation,
            candidates,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        edges = EdgeTable(pairs=refined.pairs, scores=refined.scores)
        if previous is not None:
            previous_strong = _strong_edges(previous, strong_edge_limit)
            current_strong = _strong_edges(edges, strong_edge_limit)
            overlap = edge_overlap(previous_strong.pairs, current_strong.pairs)
            sign_stability = _sign_stability(previous_strong, current_strong)
            if (
                overlap >= stability_threshold
                and sign_stability >= stability_threshold
            ):
                break
        if budget >= max_top_k:
            fallback_reason = "candidate budget reached before stability"
            break
        previous = edges
        budget = min(2 * budget, max_top_k)
        growth_rounds += 1

    return NetworkResult(
        edges=edges,
        diagnostics=ScreenDiagnostics(
            initial_top_k=min(top_k, p - 1),
            final_top_k=budget,
            candidate_count=edges.pairs.shape[0],
            candidate_density=candidates.density,
            growth_rounds=growth_rounds,
            overlap_across_budgets=overlap,
            sign_stability_across_budgets=sign_stability,
            fallback_reason=fallback_reason,
        ),
        initial_matrix=None,
    )
