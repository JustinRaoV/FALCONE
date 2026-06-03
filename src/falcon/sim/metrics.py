"""Edge-recovery metrics used by the benchmark runner.

All metrics operate on flattened off-diagonal upper-triangle pairs:
``scores`` are the absolute correlations the estimator assigns; ``truth``
is a boolean array marking the planted edges. ``recall_at_k`` and
``precision_at_k`` use the top ``k`` edges by absolute score.
"""

from __future__ import annotations

import numpy as np


def _flatten_upper(matrix: np.ndarray) -> np.ndarray:
    p = matrix.shape[0]
    iu, ju = np.triu_indices(p, k=1)
    return matrix[iu, ju]


def _scores_and_truth(score_matrix: np.ndarray, truth_matrix: np.ndarray):
    s = np.abs(_flatten_upper(score_matrix.astype(np.float64)))
    t = _flatten_upper(truth_matrix.astype(bool)).astype(np.int64)
    return s, t


def auroc_score(score_matrix: np.ndarray, truth_matrix: np.ndarray) -> float:
    """Mann-Whitney U based AUROC on the upper triangle."""
    scores, truth = _scores_and_truth(score_matrix, truth_matrix)
    if truth.sum() == 0 or truth.sum() == truth.size:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # Average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    pos_ranks = ranks[truth == 1].sum()
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(score_matrix: np.ndarray, truth_matrix: np.ndarray) -> float:
    """Step-function average precision (area under PR curve)."""
    scores, truth = _scores_and_truth(score_matrix, truth_matrix)
    if truth.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    truth_sorted = truth[order]
    cum_tp = np.cumsum(truth_sorted)
    precision = cum_tp / np.arange(1, truth_sorted.size + 1)
    recall = cum_tp / max(int(truth.sum()), 1)
    # Step-function AP = sum (recall_i - recall_{i-1}) * precision_i
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    return float((delta_recall * precision).sum())


def recall_at_k(score_matrix: np.ndarray, truth_matrix: np.ndarray, k: int) -> float:
    scores, truth = _scores_and_truth(score_matrix, truth_matrix)
    if truth.sum() == 0:
        return float("nan")
    if k <= 0:
        return 0.0
    k = min(k, scores.size)
    top = np.argpartition(-scores, k - 1)[:k]
    tp = int(truth[top].sum())
    return float(tp / int(truth.sum()))


def precision_at_k(score_matrix: np.ndarray, truth_matrix: np.ndarray, k: int) -> float:
    scores, truth = _scores_and_truth(score_matrix, truth_matrix)
    if k <= 0:
        return 0.0
    k = min(k, scores.size)
    top = np.argpartition(-scores, k - 1)[:k]
    tp = int(truth[top].sum())
    return float(tp / k)


def fdr_at_target(
    score_matrix: np.ndarray,
    truth_matrix: np.ndarray,
    selected_mask: np.ndarray,
) -> float:
    """Empirical FDR among the entries flagged by ``selected_mask``.

    ``selected_mask`` is a boolean ``(p, p)`` matrix with True for entries
    the procedure declared significant. Diagonals are ignored.
    """
    s = _flatten_upper(selected_mask.astype(bool))
    t = _flatten_upper(truth_matrix.astype(bool))
    n_selected = int(s.sum())
    if n_selected == 0:
        return float("nan")
    n_false = int((s & ~t).sum())
    return float(n_false / n_selected)
