"""Cross-domain Falcon-SR feasibility benchmark.

Sweeps `(n, (p, q), top_k)` cells, runs Falcon-SR fast / calibrated / prior
variants and SparXCC base / iter baselines, and writes per-method-per-cell
rows to ``data/falcon_sr_cross_feasibility.csv``.

Usage:

    uv run python benchmarks/falcon_sr_cross.py \\
        --n 100 250 --pq 100,100 500,500 --top-k 10 25 --reps 2
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

for _msg in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from comparison_methods import sparxcc_base, sparxcc_iter  # noqa: E402
from falcon import infer_cross  # noqa: E402
from falcon.prior import PriorEdge  # noqa: E402
from io_utils import append_row  # noqa: E402
from sim import generate_cross_domain  # noqa: E402


def _timed(fn):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak


def _fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return (
        float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)


def _recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k <= 0 or k > scores.size:
        return float("nan")
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return float(labels[top_idx].sum() / n_pos)


def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, pairs.tolist()))


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = _pair_set(left)
    b = _pair_set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _strong_pairs_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    p, q = matrix.shape
    flat_abs = np.abs(matrix).ravel()
    k = min(k, flat_abs.size)
    order = np.argpartition(-flat_abs, k - 1)[:k]
    rows = (order // q).astype(np.int64)
    cols = (order % q).astype(np.int64)
    return np.column_stack([rows, cols])


def _strong_pairs_edges(pairs: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(-np.abs(scores))[:min(k, pairs.shape[0])]
    return pairs[order]


def _flatten_dense(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, q = matrix.shape
    rows = np.repeat(np.arange(p), q)
    cols = np.tile(np.arange(q), p)
    return rows, cols


def _label_array(rows, cols, planted: set[tuple[int, int]]):
    labels = np.zeros(rows.size, dtype=np.int8)
    for idx, (i, j) in enumerate(zip(rows.tolist(), cols.tolist())):
        if (i, j) in planted:
            labels[idx] = 1
    return labels


def _sign_acc(pairs, scores, planted_signs):
    if pairs.shape[0] == 0:
        return float("nan")
    hits = total = 0
    for (i, j), s in zip(map(tuple, pairs.tolist()), scores):
        if (i, j) in planted_signs:
            total += 1
            if np.sign(s) == planted_signs[(i, j)]:
                hits += 1
    return hits / total if total else float("nan")


def _auroc_from_dense(matrix, planted):
    rows, cols = _flatten_dense(matrix)
    flat = np.abs(matrix.ravel())
    labels = _label_array(rows, cols, planted)
    return _fast_auroc(flat, labels)


def _recall_dense(matrix, planted, k):
    rows, cols = _flatten_dense(matrix)
    flat = np.abs(matrix.ravel())
    labels = _label_array(rows, cols, planted)
    return _recall_at_k(flat, labels, k)


def _auroc_from_edges(pairs, scores, planted, p, q):
    if pairs.shape[0] == 0:
        return float("nan")
    flat = np.zeros(p * q, dtype=np.float64)
    for (i, j), s in zip(map(tuple, pairs.tolist()), scores):
        flat[i * q + j] = abs(s)
    rows = np.repeat(np.arange(p), q)
    cols = np.tile(np.arange(q), p)
    labels = _label_array(rows, cols, planted)
    return _fast_auroc(flat, labels)


def _recall_at_k_edges(pairs, scores, planted, k):
    if pairs.shape[0] == 0:
        return float("nan")
    order = np.argsort(-np.abs(scores))
    cand = set(
        (int(pairs[idx, 0]), int(pairs[idx, 1]))
        for idx in order[: min(k, pairs.shape[0])]
    )
    return len(cand & planted) / len(planted) if planted else float("nan")


def _row(**kwargs):
    return {
        "method": kwargs["method"],
        "replicate": kwargs["replicate"],
        "n": kwargs["n"],
        "p": kwargs["p"],
        "q": kwargs["q"],
        "density": kwargs["density"],
        "top_k": kwargs["top_k"],
        "candidate_count": kwargs["candidate_count"],
        "candidate_recall": kwargs["candidate_recall"],
        "edge_overlap_vs_sparxcc_iter": kwargs["overlap"],
        "sign_accuracy_vs_truth": kwargs["sign_acc"],
        "auroc_vs_truth": kwargs["auroc"],
        "recall_at_K_vs_truth": kwargs["recall"],
        "wallclock_seconds": kwargs["seconds"],
        "peak_bytes": kwargs["peak"],
        "fallback_reason": kwargs["fallback_reason"] or "",
        "calibration_method": kwargs["calibration_method"] or "",
        "prior_count": kwargs["prior_count"],
        "data_disagreed_with_prior_count": kwargs["data_disagreed_with_prior_count"],
    }


def run_cell(*, n: int, p: int, q: int, density: float, top_k: int, reps: int):
    out = []
    for replicate in range(reps):
        seed = 2000 + replicate + p + q
        rng = np.random.default_rng(seed)
        counts_x, counts_y, _, planted = generate_cross_domain(
            rng, n, p, q, density=density,
        )
        planted_pairs = {(i, k) for i, k, _ in planted}
        planted_signs = {(i, k): np.sign(s) for i, k, s in planted}
        n_planted = len(planted_pairs)
        if n_planted == 0:
            continue

        # Reference: SparXCC iter, ranked
        iter_matrix, t_iter, peak_iter = _timed(
            lambda: sparxcc_iter(counts_x, counts_y)
        )
        iter_strong = _strong_pairs_matrix(iter_matrix, n_planted)
        iter_sign_acc = float(np.mean(
            [np.sign(iter_matrix[i, k]) == planted_signs[(i, k)]
             for (i, k) in planted_pairs]
        ))
        out.append(_row(
            method="sparxcc_iter", replicate=replicate,
            n=n, p=p, q=q, density=density, top_k=top_k,
            candidate_count=p * q, candidate_recall=1.0,
            overlap=1.0, sign_acc=iter_sign_acc,
            auroc=_auroc_from_dense(iter_matrix, planted_pairs),
            recall=_recall_dense(iter_matrix, planted_pairs, n_planted),
            seconds=t_iter, peak=peak_iter,
            fallback_reason=None, calibration_method=None,
            prior_count=0, data_disagreed_with_prior_count=0,
        ))

        base_matrix, t_base, peak_base = _timed(
            lambda: sparxcc_base(counts_x, counts_y)
        )
        base_strong = _strong_pairs_matrix(base_matrix, n_planted)
        base_sign_acc = float(np.mean(
            [np.sign(base_matrix[i, k]) == planted_signs[(i, k)]
             for (i, k) in planted_pairs]
        ))
        out.append(_row(
            method="sparxcc_base", replicate=replicate,
            n=n, p=p, q=q, density=density, top_k=top_k,
            candidate_count=p * q, candidate_recall=1.0,
            overlap=_jaccard(base_strong, iter_strong),
            sign_acc=base_sign_acc,
            auroc=_auroc_from_dense(base_matrix, planted_pairs),
            recall=_recall_dense(base_matrix, planted_pairs, n_planted),
            seconds=t_base, peak=peak_base,
            fallback_reason=None, calibration_method=None,
            prior_count=0, data_disagreed_with_prior_count=0,
        ))

        # Falcon-SR cross fast (no calibration, no prior)
        fast_res, t_fast, peak_fast = _timed(
            lambda: infer_cross(
                counts_x, counts_y, mode="fast", top_k=top_k,
                stability_threshold=0.0, calibration="none",
            )
        )
        fast_strong = _strong_pairs_edges(
            fast_res.edges.pairs, fast_res.edges.scores, n_planted
        )
        out.append(_row(
            method="falcon_sr_cross_fast", replicate=replicate,
            n=n, p=p, q=q, density=density, top_k=top_k,
            candidate_count=fast_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted_pairs(fast_res.edges.pairs, planted_pairs),
            overlap=_jaccard(fast_strong, iter_strong),
            sign_acc=_sign_acc(fast_res.edges.pairs, fast_res.edges.scores, planted_signs),
            auroc=_auroc_from_edges(
                fast_res.edges.pairs, fast_res.edges.scores, planted_pairs, p, q
            ),
            recall=_recall_at_k_edges(
                fast_res.edges.pairs, fast_res.edges.scores, planted_pairs, n_planted
            ),
            seconds=t_fast, peak=peak_fast,
            fallback_reason=fast_res.diagnostics.fallback_reason,
            calibration_method=fast_res.diagnostics.calibration_method,
            prior_count=0, data_disagreed_with_prior_count=0,
        ))

        # Falcon-SR cross with synthetic prior covering half the planted edges
        prior_subset = list(planted_pairs)[: max(1, n_planted // 2)]
        priors = [
            PriorEdge(
                source_feature=int(i),
                target_feature=int(k),
                expected_sign=int(planted_signs[(i, k)]),
                confidence=0.8,
                provenance="synthetic_50pct",
            )
            for (i, k) in prior_subset
        ]
        prior_res, t_prior, peak_prior = _timed(
            lambda: infer_cross(
                counts_x, counts_y, mode="fast", top_k=top_k,
                stability_threshold=0.0, calibration="none",
                prior=priors, prior_weight=0.5,
                prior_target_magnitude=0.4,
            )
        )
        prior_strong = _strong_pairs_edges(
            prior_res.edges.pairs, prior_res.edges.scores, n_planted
        )
        out.append(_row(
            method="falcon_sr_cross_prior", replicate=replicate,
            n=n, p=p, q=q, density=density, top_k=top_k,
            candidate_count=prior_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted_pairs(prior_res.edges.pairs, planted_pairs),
            overlap=_jaccard(prior_strong, iter_strong),
            sign_acc=_sign_acc(prior_res.edges.pairs, prior_res.edges.scores, planted_signs),
            auroc=_auroc_from_edges(
                prior_res.edges.pairs, prior_res.edges.scores, planted_pairs, p, q
            ),
            recall=_recall_at_k_edges(
                prior_res.edges.pairs, prior_res.edges.scores, planted_pairs, n_planted
            ),
            seconds=t_prior, peak=peak_prior,
            fallback_reason=prior_res.diagnostics.fallback_reason,
            calibration_method=prior_res.diagnostics.calibration_method,
            prior_count=prior_res.diagnostics.prior_count or 0,
            data_disagreed_with_prior_count=(
                prior_res.diagnostics.data_disagreed_with_prior_count or 0
            ),
        ))

        # Falcon-SR cross + permutation calibration (no prior)
        cal_res, t_cal, peak_cal = _timed(
            lambda: infer_cross(
                counts_x, counts_y, mode="fast", top_k=top_k,
                stability_threshold=0.0,
                calibration="permutation", n_permutations=100, seed=replicate,
            )
        )
        cal_strong = _strong_pairs_edges(
            cal_res.edges.pairs, cal_res.edges.scores, n_planted
        )
        out.append(_row(
            method="falcon_sr_cross_fast_calibrated", replicate=replicate,
            n=n, p=p, q=q, density=density, top_k=top_k,
            candidate_count=cal_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted_pairs(cal_res.edges.pairs, planted_pairs),
            overlap=_jaccard(cal_strong, iter_strong),
            sign_acc=_sign_acc(cal_res.edges.pairs, cal_res.edges.scores, planted_signs),
            auroc=_auroc_from_edges(
                cal_res.edges.pairs, cal_res.edges.scores, planted_pairs, p, q
            ),
            recall=_recall_at_k_edges(
                cal_res.edges.pairs, cal_res.edges.scores, planted_pairs, n_planted
            ),
            seconds=t_cal, peak=peak_cal,
            fallback_reason=cal_res.diagnostics.fallback_reason,
            calibration_method=cal_res.diagnostics.calibration_method,
            prior_count=0, data_disagreed_with_prior_count=0,
        ))

    return out


def _recall_planted_pairs(pairs, planted):
    if not planted:
        return float("nan")
    cand = _pair_set(pairs)
    return len(cand & planted) / len(planted)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--pq", nargs="+", default=["100,100", "500,500"],
                        help="Comma-separated p,q pairs")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 25])
    parser.add_argument("--reps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cells = []
    for n in args.n:
        for pq in args.pq:
            p_str, q_str = pq.split(",")
            for top_k in args.top_k:
                cells.append((n, int(p_str), int(q_str), top_k))

    for (n, p, q, top_k) in cells:
        rows = run_cell(
            n=n, p=p, q=q, density=args.density, top_k=top_k, reps=args.reps,
        )
        for row in rows:
            append_row("falcon_sr_cross_feasibility", row)
        print(
            f"n={n} p={p} q={q} k={top_k}: wrote {len(rows)} rows",
            flush=True,
        )
    print("Done. Output: data/falcon_sr_cross_feasibility.csv")
