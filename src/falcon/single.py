from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from falcon.preprocessing import PreprocessReport, prepare_log_composition
from falcon.types import CandidateSet


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
