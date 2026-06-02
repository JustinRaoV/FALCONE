from __future__ import annotations

import numpy as np

from falcon.types import CandidateSet, CrossCandidateSet


def _canonical_pairs(rows: np.ndarray, cols: np.ndarray, p: int) -> np.ndarray:
    left = np.minimum(rows, cols).astype(np.int64)
    right = np.maximum(rows, cols).astype(np.int64)
    keys = left * np.int64(p) + right
    order = np.unique(keys, return_index=True)[1]
    return np.column_stack([left[order], right[order]])


def single_candidates(
    correlation: np.ndarray,
    *,
    top_k: int,
    min_abs_score: float | None = None,
) -> CandidateSet:
    p = correlation.shape[0]
    if correlation.shape != (p, p):
        raise ValueError("correlation must be square")
    if not 1 <= top_k < p:
        raise ValueError("top_k must lie in [1, p)")

    absolute = np.abs(correlation).copy()
    np.fill_diagonal(absolute, -np.inf)
    columns = np.argpartition(-absolute, top_k - 1, axis=1)[:, :top_k]
    rows = np.repeat(np.arange(p), top_k)
    pairs = _canonical_pairs(rows, columns.ravel(), p)

    if min_abs_score is not None:
        extra_rows, extra_cols = np.where(
            np.triu(absolute >= min_abs_score, k=1)
        )
        pairs = _canonical_pairs(
            np.concatenate([pairs[:, 0], extra_rows]),
            np.concatenate([pairs[:, 1], extra_cols]),
            p,
        )

    scores = correlation[pairs[:, 0], pairs[:, 1]]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return CandidateSet(
        pairs=pairs[order],
        scores=scores[order],
        top_k=top_k,
        n_features=p,
    )


def edge_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(map(tuple, np.asarray(left).tolist()))
    right_set = set(map(tuple, np.asarray(right).tolist()))
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _dedupe_cross_pairs(rows: np.ndarray, cols: np.ndarray, q: int) -> np.ndarray:
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    keys = rows * np.int64(q) + cols
    order = np.unique(keys, return_index=True)[1]
    return np.column_stack([rows[order], cols[order]])


def cross_candidates(
    score: np.ndarray,
    *,
    top_k: int,
    min_abs_score: float | None = None,
) -> CrossCandidateSet:
    p, q = score.shape
    bound = min(p, q)
    if not 1 <= top_k <= bound:
        raise ValueError(
            f"top_k must lie in [1, min(p, q)] = [1, {bound}]; got {top_k}"
        )

    absolute = np.abs(score)

    # Top-k Y partners for every X row
    y_cols = np.argpartition(-absolute, top_k - 1, axis=1)[:, :top_k]
    x_rows = np.repeat(np.arange(p), top_k)

    # Top-k X partners for every Y column
    x_partners = np.argpartition(-absolute, top_k - 1, axis=0)[:top_k, :]
    y_partners = np.tile(np.arange(q), top_k)

    rows_all = np.concatenate([x_rows, x_partners.ravel(order="C")])
    cols_all = np.concatenate([y_cols.ravel(order="C"), y_partners])

    if min_abs_score is not None:
        extra_rows, extra_cols = np.where(absolute >= min_abs_score)
        rows_all = np.concatenate([rows_all, extra_rows])
        cols_all = np.concatenate([cols_all, extra_cols])

    pairs = _dedupe_cross_pairs(rows_all, cols_all, q)
    scores = score[pairs[:, 0], pairs[:, 1]]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return CrossCandidateSet(
        pairs=pairs[order],
        scores=scores[order],
        top_k=top_k,
        n_features_x=p,
        n_features_y=q,
    )
