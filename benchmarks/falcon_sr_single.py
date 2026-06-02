"""Single-domain Falcon-SR feasibility benchmark.

Sweeps `(n, p, top_k)` cells over `density = 0.02`, runs Falcon-SR strict,
fast, and permutation-calibrated fast modes, plus SparCC and Pearson(CLR)
baselines for head-to-head ranking comparison. Writes per-method-per-cell
rows to ``data/falcon_sr_single_feasibility.csv``.

Usage:

    uv run python benchmarks/falcon_sr_single.py \\
        --n 100 250 --p 100 500 --top-k 10 25 --reps 2

Cells finish in arbitrary order; each completed measurement is durable.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

# macOS Accelerate + NumPy 2.x emits spurious matmul warnings on valid
# finite inputs. Silence the false positive so benchmark stdout stays clean.
for _msg in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from comparison_methods import pearson_clr, sparcc_py  # noqa: E402
from falcon import infer_single  # noqa: E402
from io_utils import append_row  # noqa: E402
from sim import generate_basis_correlation, generate_single_domain  # noqa: E402


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
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = (i + j + 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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


def _strong_pairs_from_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    p = matrix.shape[0]
    rows, cols = np.triu_indices(p, k=1)
    scores = matrix[rows, cols]
    order = np.argsort(-np.abs(scores))[:k]
    return np.column_stack([rows[order], cols[order]])


def _strong_pairs_from_edges(pairs: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    if pairs.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(-np.abs(scores))[:min(k, pairs.shape[0])]
    return pairs[order]


def run_cell(*, n: int, p: int, density: float, top_k: int, reps: int):
    out = []
    for replicate in range(reps):
        seed = 1000 + replicate + p
        rng = np.random.default_rng(seed)
        sigma, planted = generate_basis_correlation(rng, p, density)
        counts = generate_single_domain(rng, n, p, sigma)
        planted_pairs = {
            (min(i, j), max(i, j)) for i, j, _ in planted
        }
        planted_signs = {
            (min(i, j), max(i, j)): np.sign(rho) for i, j, rho in planted
        }
        n_planted = len(planted_pairs)
        if n_planted == 0:
            continue

        # Reference: SparCC base score, ranked
        sparcc_matrix, t_sparcc, peak_sparcc = _timed(lambda: sparcc_py(counts))
        sparcc_strong = _strong_pairs_from_matrix(sparcc_matrix, n_planted)
        rows, cols = np.triu_indices(p, k=1)
        sparcc_scores_flat = np.abs(sparcc_matrix[rows, cols])
        labels = np.zeros(rows.size, dtype=np.int8)
        for idx, (i, j) in enumerate(zip(rows.tolist(), cols.tolist())):
            if (i, j) in planted_pairs:
                labels[idx] = 1
        sparcc_auroc = _fast_auroc(sparcc_scores_flat, labels)
        sparcc_recall = _recall_at_k(sparcc_scores_flat, labels, n_planted)
        sparcc_sign_acc = float(
            np.mean(
                [
                    np.sign(sparcc_matrix[i, j]) == planted_signs[(i, j)]
                    for (i, j) in planted_pairs
                ]
            )
        )
        out.append(_row(
            method="sparcc_py", replicate=replicate, n=n, p=p,
            density=density, top_k=top_k,
            candidate_count=p * (p - 1) // 2,
            candidate_recall=1.0,
            overlap=1.0, sign_acc=sparcc_sign_acc,
            auroc=sparcc_auroc, recall=sparcc_recall,
            seconds=t_sparcc, peak=peak_sparcc,
            fallback_reason=None, calibration_method=None,
        ))

        pearson_matrix, t_pearson, peak_pearson = _timed(lambda: pearson_clr(counts))
        pearson_strong = _strong_pairs_from_matrix(pearson_matrix, n_planted)
        pearson_scores_flat = np.abs(pearson_matrix[rows, cols])
        pearson_auroc = _fast_auroc(pearson_scores_flat, labels)
        pearson_recall = _recall_at_k(pearson_scores_flat, labels, n_planted)
        pearson_sign_acc = float(
            np.mean(
                [
                    np.sign(pearson_matrix[i, j]) == planted_signs[(i, j)]
                    for (i, j) in planted_pairs
                ]
            )
        )
        pearson_overlap = _jaccard(pearson_strong, sparcc_strong)
        out.append(_row(
            method="pearson_clr", replicate=replicate, n=n, p=p,
            density=density, top_k=top_k,
            candidate_count=p * (p - 1) // 2,
            candidate_recall=1.0,
            overlap=pearson_overlap, sign_acc=pearson_sign_acc,
            auroc=pearson_auroc, recall=pearson_recall,
            seconds=t_pearson, peak=peak_pearson,
            fallback_reason=None, calibration_method=None,
        ))

        # Falcon-SR strict
        strict_res, t_strict, peak_strict = _timed(
            lambda: infer_single(
                counts, mode="strict", max_exclusions=10, calibration="none",
            )
        )
        strict_pairs = strict_res.edges.pairs
        strict_scores = strict_res.edges.scores
        strict_strong = _strong_pairs_from_edges(strict_pairs, strict_scores, n_planted)
        out.append(_row(
            method="falcon_sr_strict", replicate=replicate, n=n, p=p,
            density=density, top_k=top_k,
            candidate_count=strict_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted(strict_pairs, planted_pairs),
            overlap=_jaccard(strict_strong, sparcc_strong),
            sign_acc=_sign_accuracy(strict_pairs, strict_scores, planted_signs),
            auroc=_auroc_from_edges(strict_pairs, strict_scores, planted_pairs, p),
            recall=_recall_at_k_edges(strict_pairs, strict_scores, planted_pairs, n_planted),
            seconds=t_strict, peak=peak_strict,
            fallback_reason=strict_res.diagnostics.fallback_reason,
            calibration_method=strict_res.diagnostics.calibration_method,
        ))

        # Falcon-SR fast (no calibration)
        fast_res, t_fast, peak_fast = _timed(
            lambda: infer_single(
                counts, mode="fast", top_k=top_k,
                stability_threshold=0.0, calibration="none",
            )
        )
        fast_pairs = fast_res.edges.pairs
        fast_scores = fast_res.edges.scores
        fast_strong = _strong_pairs_from_edges(fast_pairs, fast_scores, n_planted)
        out.append(_row(
            method="falcon_sr_fast", replicate=replicate, n=n, p=p,
            density=density, top_k=top_k,
            candidate_count=fast_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted(fast_pairs, planted_pairs),
            overlap=_jaccard(fast_strong, sparcc_strong),
            sign_acc=_sign_accuracy(fast_pairs, fast_scores, planted_signs),
            auroc=_auroc_from_edges(fast_pairs, fast_scores, planted_pairs, p),
            recall=_recall_at_k_edges(fast_pairs, fast_scores, planted_pairs, n_planted),
            seconds=t_fast, peak=peak_fast,
            fallback_reason=fast_res.diagnostics.fallback_reason,
            calibration_method=fast_res.diagnostics.calibration_method,
        ))

        # Falcon-SR fast + permutation calibration
        cal_res, t_cal, peak_cal = _timed(
            lambda: infer_single(
                counts, mode="fast", top_k=top_k,
                stability_threshold=0.0,
                calibration="permutation", n_permutations=100, seed=replicate,
            )
        )
        cal_pairs = cal_res.edges.pairs
        cal_scores = cal_res.edges.scores
        cal_strong = _strong_pairs_from_edges(cal_pairs, cal_scores, n_planted)
        out.append(_row(
            method="falcon_sr_fast_calibrated", replicate=replicate,
            n=n, p=p, density=density, top_k=top_k,
            candidate_count=cal_res.diagnostics.candidate_count,
            candidate_recall=_recall_planted(cal_pairs, planted_pairs),
            overlap=_jaccard(cal_strong, sparcc_strong),
            sign_acc=_sign_accuracy(cal_pairs, cal_scores, planted_signs),
            auroc=_auroc_from_edges(cal_pairs, cal_scores, planted_pairs, p),
            recall=_recall_at_k_edges(cal_pairs, cal_scores, planted_pairs, n_planted),
            seconds=t_cal, peak=peak_cal,
            fallback_reason=cal_res.diagnostics.fallback_reason,
            calibration_method=cal_res.diagnostics.calibration_method,
        ))

    return out


def _row(**kwargs):
    return {
        "method": kwargs["method"],
        "replicate": kwargs["replicate"],
        "n": kwargs["n"],
        "p": kwargs["p"],
        "density": kwargs["density"],
        "top_k": kwargs["top_k"],
        "candidate_count": kwargs["candidate_count"],
        "candidate_recall": kwargs["candidate_recall"],
        "edge_overlap_vs_sparcc": kwargs["overlap"],
        "sign_accuracy_vs_truth": kwargs["sign_acc"],
        "auroc_vs_truth": kwargs["auroc"],
        "recall_at_K_vs_truth": kwargs["recall"],
        "wallclock_seconds": kwargs["seconds"],
        "peak_bytes": kwargs["peak"],
        "fallback_reason": kwargs["fallback_reason"] or "",
        "calibration_method": kwargs["calibration_method"] or "",
    }


def _recall_planted(pairs: np.ndarray, planted: set[tuple[int, int]]) -> float:
    if not planted:
        return float("nan")
    cand = _pair_set(pairs)
    return len(cand & planted) / len(planted)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = _pair_set(left)
    b = _pair_set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _sign_accuracy(pairs, scores, planted_signs):
    if pairs.shape[0] == 0:
        return float("nan")
    hits = 0
    total = 0
    for (i, j), s in zip(map(tuple, pairs.tolist()), scores):
        if (i, j) in planted_signs:
            total += 1
            if np.sign(s) == planted_signs[(i, j)]:
                hits += 1
    return hits / total if total else float("nan")


def _auroc_from_edges(pairs, scores, planted, p):
    rows, cols = np.triu_indices(p, k=1)
    flat = np.zeros(rows.size, dtype=np.float64)
    for (i, j), s in zip(map(tuple, pairs.tolist()), scores):
        if i > j:
            i, j = j, i
        # Linear index of (i, j) in upper triangle ordering:
        # k = i*(2*p - i - 1)/2 + (j - i - 1)
        lin = i * (2 * p - i - 1) // 2 + (j - i - 1)
        flat[lin] = abs(s)
    labels = np.zeros(rows.size, dtype=np.int8)
    for idx, (i, j) in enumerate(zip(rows.tolist(), cols.tolist())):
        if (i, j) in planted:
            labels[idx] = 1
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--p", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--density", type=float, default=0.02)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--reps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for n in args.n:
        for p in args.p:
            for top_k in args.top_k:
                rows = run_cell(
                    n=n, p=p, density=args.density,
                    top_k=top_k, reps=args.reps,
                )
                for row in rows:
                    append_row("falcon_sr_single_feasibility", row)
                print(
                    f"n={n} p={p} k={top_k}: wrote {len(rows)} rows",
                    flush=True,
                )
    print("Done. Output: data/falcon_sr_single_feasibility.csv")
