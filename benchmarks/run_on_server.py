"""
FALCON benchmark runner — designed for HPC servers with multiprocessing.

Each benchmark task is independent and writes (or appends to) a CSV under
data/ at the repository root. The four tables are:

    data/scalability.csv     wall-clock per (n, p) cell
    data/detection.csv       power + AUROC + Recall@K per (n, p, rho) cell
    data/cross_domain.csv    one row per (method, replicate) pair
    data/fdr_control.csv     FPR + FDR at each nominal alpha and scenario

Cells finish in arbitrary order; CSVs grow incrementally so a killed job
loses at most the in-flight cell. Re-running this script with the same
arguments simply appends fresh measurements; deduplicate manually with
pandas if needed.

Usage examples
--------------
    # Run all four benchmarks, ~4 workers, default cell grids:
    python benchmarks/run_on_server.py --workers 4

    # Only the detection grid, up to p = 10000 and n = 5000:
    python benchmarks/run_on_server.py --task detection \\
        --workers 8 --reps 10 \\
        --p 100 500 1000 5000 10000 \\
        --n 500 1000 2000 5000 \\
        --rho 0.4 0.7

    # Only scalability (FastProp + RandProp wall-clock):
    python benchmarks/run_on_server.py --task scalability --workers 4

The script auto-detects the local hostname (`socket.gethostname()`) and
records it as the `host` column in `scalability.csv` so multiple machines'
measurements can co-exist.

Implementation notes
--------------------
* multiprocessing.Pool is used per task. Each worker imports falcon
  independently to avoid sharing NumPy thread pools.
* For very large cells, set OMP_NUM_THREADS=1 in the environment before
  invocation so the per-worker BLAS doesn't oversubscribe cores.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from io_utils import append_row  # noqa: E402

HOSTNAME = socket.gethostname()


# =============================================================================
# Shared simulation utilities
# =============================================================================


def nearest_pd(A: np.ndarray, min_eig: float = 0.01) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh((A + A.T) / 2)
    eigvals = np.maximum(eigvals, min_eig)
    return (eigvecs * eigvals) @ eigvecs.T


def generate_basis_correlation(rng, p, density, effect_lo, effect_hi,
                                neg_fraction=0.5):
    n_edges_target = int(density * p * (p - 1) / 2)
    Sigma = np.eye(p)
    edges = []
    chosen = set()
    while len(edges) < n_edges_target:
        i = int(rng.integers(0, p))
        j = int(rng.integers(0, p))
        if i == j or (i, j) in chosen or (j, i) in chosen:
            continue
        rho = float(rng.uniform(effect_lo, effect_hi))
        if rng.random() < neg_fraction:
            rho = -rho
        Sigma[i, j] = Sigma[j, i] = rho
        chosen.add((i, j))
        edges.append((i, j, rho))
    Sigma = nearest_pd(Sigma)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)
    return Sigma, edges


def generate_single_domain(rng, n, p, Sigma,
                           sequencing_depth=50_000, detection_limit=1e-4):
    L = cholesky(Sigma, lower=True)
    Z = rng.standard_normal((n, p)) @ L.T
    ranks = np.arange(1, p + 1)
    mean_log_abundance = 6.0 - 1.5 * np.log(ranks)
    rng.shuffle(mean_log_abundance)
    W = np.exp(Z + mean_log_abundance)
    proportions = W / W.sum(axis=1, keepdims=True)
    depths = rng.poisson(sequencing_depth, size=n)
    counts = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts[i] = rng.multinomial(int(depths[i]), proportions[i])
    relative = counts / counts.sum(axis=1, keepdims=True)
    counts[relative < detection_limit] = 0
    return counts


def fast_auroc(scores, labels):
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
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def recall_at_k(scores, labels, k):
    if k <= 0 or k > scores.size:
        return float("nan")
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return float(labels[top_idx].sum() / n_pos)


# =============================================================================
# Worker functions (one per task type)
# =============================================================================


def scalability_cell(args):
    n, p = args
    from falcon import fastprop, randprop  # noqa: E402

    rng = np.random.default_rng(42)
    counts = rng.integers(0, 200, size=(n, p))
    bytes_dense = p * p * 8
    if bytes_dense < 1.5e9:
        t0 = time.perf_counter()
        rho = fastprop(counts, shrinkage=True)
        t_fast = time.perf_counter() - t0
        del rho
    else:
        t_fast = float("nan")
    t0 = time.perf_counter()
    W = randprop(counts, k=50, seed=42)
    t_rand = time.perf_counter() - t0
    del W, counts
    return {"n": n, "p": p, "fastprop_sec": t_fast,
            "randprop_sec": t_rand, "host": HOSTNAME}


def detection_cell(args):
    n, p, rho_val, n_reps = args
    from falcon import (extract_network, fastprop,  # noqa: E402
                        fastprop_pvalues)

    bytes_dense = p * p * 8
    if bytes_dense > 1.5e9:
        return {"n": n, "p": p, "effect": rho_val, "n_reps": 0,
                "power_mean": float("nan"), "power_std": float("nan"),
                "auroc_mean": float("nan"), "auroc_std": float("nan"),
                "recall_at_K_mean": float("nan"),
                "recall_at_K_std": float("nan")}

    density = 0.02 if p <= 1000 else 0.005
    powers, aurocs, recalls = [], [], []
    for rep in range(n_reps):
        seed = rep * 17 + p * 3 + n
        rng = np.random.default_rng(seed)
        Sigma, edges = generate_basis_correlation(
            rng, p, density,
            effect_lo=rho_val - 0.05, effect_hi=rho_val + 0.05,
        )
        counts = generate_single_domain(rng, n, p, Sigma)
        rho = fastprop(counts, shrinkage=True)
        pvals = fastprop_pvalues(counts, rho)
        adj = extract_network(rho, pvals, alpha=0.05,
                              min_abs_corr=0.1, fdr_correct=True)
        true_edges = set((min(i, j), max(i, j)) for i, j, _ in edges)
        detected = set((min(i, j), max(i, j)) for i, j in zip(*adj.nonzero()))
        tp = len(true_edges & detected)
        powers.append(tp / len(true_edges) if true_edges else 0.0)

        iu = np.triu_indices(p, k=1)
        scores = np.abs(rho[iu])
        labels = np.zeros(scores.size, dtype=np.int8)
        for lin_idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            if (int(i), int(j)) in true_edges:
                labels[lin_idx] = 1
        aurocs.append(fast_auroc(scores, labels))
        recalls.append(recall_at_k(scores, labels, k=len(true_edges)))
        del rho, pvals, counts, Sigma, scores, labels, iu

    return {
        "n": n, "p": p, "effect": rho_val, "n_reps": n_reps,
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "recall_at_K_mean": float(np.nanmean(recalls)),
        "recall_at_K_std": float(np.nanstd(recalls)),
    }


def cross_domain_replicate(args):
    rep, n, p, q, n_interactions = args
    from falcon import crossnet  # noqa: E402

    sim_rng = np.random.default_rng(rep * 100)
    Omega_true = np.zeros((p, q))
    interactions = set()
    while len(interactions) < n_interactions:
        i = int(sim_rng.integers(0, p))
        j = int(sim_rng.integers(0, q))
        if (i, j) in interactions:
            continue
        interactions.add((i, j))
        sign = -1 if sim_rng.random() < 0.7 else 1
        Omega_true[i, j] = sign * sim_rng.uniform(0.3, 0.6)

    # Simulate cross-domain compositions with shared latent factors
    n_latent = min(n_interactions, 8)
    Lx = sim_rng.standard_normal((p, n_latent)) * 0.4
    Ly = sim_rng.standard_normal((q, n_latent)) * 0.4
    for (i, j) in interactions:
        f = int(sim_rng.integers(0, n_latent))
        Lx[i, f] = np.sign(Omega_true[i, j]) * abs(Omega_true[i, j]) * 0.9
        Ly[j, f] = 1.0
    scores = sim_rng.standard_normal((n, n_latent))
    Zx = sim_rng.standard_normal((n, p)) * 0.5 + scores @ Lx.T
    Zy = sim_rng.standard_normal((n, q)) * 0.5 + scores @ Ly.T
    Wx = np.exp(Zx + sim_rng.uniform(2, 4, size=p))
    Wy = np.exp(Zy + sim_rng.uniform(3, 5, size=q))
    Wx[sim_rng.random((n, p)) < 0.15] = 0
    Wy[sim_rng.random((n, q)) < 0.12] = 0
    Wx = Wx / np.where(Wx.sum(axis=1, keepdims=True) == 0, 1, Wx.sum(axis=1, keepdims=True))
    Wy = Wy / np.where(Wy.sum(axis=1, keepdims=True) == 0, 1, Wy.sum(axis=1, keepdims=True))
    X = (Wx * sim_rng.integers(20_000, 80_000, size=(n, 1))).astype(int)
    Y = (Wy * sim_rng.integers(30_000, 100_000, size=(n, 1))).astype(int)

    rows = []
    # FALCON's three cross-domain methods
    for method in ("naive_clr", "sparxcc_like", "bias_corrected"):
        C, pvals = crossnet(X, Y, method=method)
        true_mask = Omega_true != 0
        null_mask = ~true_mask
        corr_val = float(np.corrcoef(Omega_true[true_mask], C[true_mask])[0, 1])
        bias_val = float(C[null_mask].mean())
        sign_acc = float(np.mean(np.sign(C[true_mask]) == np.sign(Omega_true[true_mask])))
        sig = (pvals < 0.05) & (np.abs(C) > 0.1)
        sens = float(sig[true_mask].mean())
        spec = float((~sig[null_mask]).mean())
        rows.append({
            "method": method, "replicate": rep,
            "corr": corr_val, "bias": bias_val, "sign_acc": sign_acc,
            "sensitivity": sens, "specificity": spec,
        })

    # Jensen et al. (2024) SparXCC: faithful Python reimplementation.
    # We score it with the same significance criterion as the other
    # methods (|rho| > 0.1 and Fisher-z p < 0.05) for apples-to-apples.
    from comparison_methods import sparxcc_base, sparxcc_iter  # noqa: E402
    from scipy.stats import norm as _norm  # noqa: E402

    def _eval(C_est: np.ndarray, label: str):
        true_mask = Omega_true != 0
        null_mask = ~true_mask
        corr_val = float(np.corrcoef(Omega_true[true_mask], C_est[true_mask])[0, 1])
        bias_val = float(C_est[null_mask].mean())
        sign_acc = float(np.mean(np.sign(C_est[true_mask]) == np.sign(Omega_true[true_mask])))
        # Fisher-z p-values for compatibility with the other methods
        C_clip = np.clip(C_est, -1 + 1e-10, 1 - 1e-10)
        z = np.arctanh(C_clip)
        se = 1.0 / np.sqrt(n - 3)
        pvals = 2.0 * _norm.sf(np.abs(z) / se)
        sig = (pvals < 0.05) & (np.abs(C_est) > 0.1)
        sens = float(sig[true_mask].mean())
        spec = float((~sig[null_mask]).mean())
        return {
            "method": label, "replicate": rep,
            "corr": corr_val, "bias": bias_val, "sign_acc": sign_acc,
            "sensitivity": sens, "specificity": spec,
        }

    rows.append(_eval(sparxcc_base(X, Y), "sparxcc_base"))
    rows.append(_eval(sparxcc_iter(X, Y), "sparxcc_iter"))

    # SPIEC-EASI cross-domain (Kurtz 2015). Joint graphical lasso on
    # [CLR_X, CLR_Y]; extract the p x q cross block.
    from comparison_methods import spieceasi_cross_glasso  # noqa: E402
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            se_cross = spieceasi_cross_glasso(X, Y, alpha=0.05)
            rows.append(_eval(se_cross, "spieceasi_cross"))
        except Exception:
            pass
    return rows


def method_comparison_cell(args):
    """Head-to-head against the full Case-A baseline panel:
    FastProp / SparCC / Pearson(CLR) / Pearson(raw) / SPIEC-EASI-glasso
    / SPIEC-EASI-MB. Returns one row per method per cell."""
    n, p, rho_val, n_reps = args
    import warnings as _w
    _w.filterwarnings("ignore")
    from falcon import fastprop  # noqa: E402
    from comparison_methods import (pearson_clr, pearson_raw,  # noqa: E402
                                     sparcc_py, spieceasi_glasso,
                                     spieceasi_mb)

    bytes_dense = p * p * 8
    if bytes_dense > 1.5e9:
        return [{"method": m, "n": n, "p": p, "effect": rho_val, "n_reps": 0,
                 "time_sec_mean": float("nan"), "time_sec_std": float("nan"),
                 "auroc_mean": float("nan"), "auroc_std": float("nan"),
                 "recall_at_K_mean": float("nan"),
                 "recall_at_K_std": float("nan"),
                 "null_bias_mean": float("nan"),
                 "null_bias_std": float("nan")}
                for m in ("fastprop", "sparcc_py", "pearson_clr", "pearson_raw")]

    methods = {
        "fastprop":         lambda X: fastprop(X, shrinkage=True),
        "sparcc_py":        sparcc_py,
        "pearson_clr":      pearson_clr,
        "pearson_raw":      pearson_raw,
        "spieceasi_glasso": lambda X: spieceasi_glasso(X, alpha=0.05),
        "spieceasi_mb":     lambda X: spieceasi_mb(X, alpha=0.1, max_iter=200),
    }
    aggregates = {m: {"time": [], "auroc": [], "recall": [], "bias": []}
                  for m in methods}

    density = 0.02 if p <= 1000 else 0.005
    iu = np.triu_indices(p, k=1)
    for rep in range(n_reps):
        seed = rep * 31 + p * 7 + n + 1
        rng = np.random.default_rng(seed)
        Sigma, edges = generate_basis_correlation(
            rng, p, density,
            effect_lo=rho_val - 0.05, effect_hi=rho_val + 0.05,
        )
        counts = generate_single_domain(rng, n, p, Sigma)
        true_edges = set((min(i, j), max(i, j)) for i, j, _ in edges)
        labels = np.zeros(iu[0].size, dtype=np.int8)
        for lin_idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            if (int(i), int(j)) in true_edges:
                labels[lin_idx] = 1
        for mname, fn in methods.items():
            t0 = time.perf_counter()
            rho = fn(counts)
            t_run = time.perf_counter() - t0
            scores = np.abs(rho[iu])
            aggregates[mname]["time"].append(t_run)
            aggregates[mname]["auroc"].append(fast_auroc(scores, labels))
            aggregates[mname]["recall"].append(
                recall_at_k(scores, labels, k=len(true_edges)))
            null_mask = labels == 0
            aggregates[mname]["bias"].append(float(rho[iu][null_mask].mean()))
            del rho
        del counts, Sigma

    out = []
    for mname, vals in aggregates.items():
        out.append({
            "method": mname, "n": n, "p": p, "effect": rho_val,
            "n_reps": n_reps,
            "time_sec_mean": float(np.mean(vals["time"])),
            "time_sec_std": float(np.std(vals["time"])),
            "auroc_mean": float(np.nanmean(vals["auroc"])),
            "auroc_std": float(np.nanstd(vals["auroc"])),
            "recall_at_K_mean": float(np.nanmean(vals["recall"])),
            "recall_at_K_std": float(np.nanstd(vals["recall"])),
            "null_bias_mean": float(np.mean(vals["bias"])),
            "null_bias_std": float(np.std(vals["bias"])),
        })
    return out


def fdr_cell(args):
    """One (alpha, scenario, rep) triple for FDR control."""
    alpha, scenario, rep, n, p = args
    from falcon import (extract_network, fastprop,  # noqa: E402
                        fastprop_pvalues)

    rng = np.random.default_rng(rep)
    if scenario == "null":
        Sigma = np.eye(p)
        counts = generate_single_domain(rng, n, p, Sigma)
        rho = fastprop(counts, shrinkage=True)
        pvals = fastprop_pvalues(counts, rho)
        adj = extract_network(rho, pvals, alpha=alpha,
                              min_abs_corr=0.0, fdr_correct=True)
        total = p * (p - 1) // 2
        n_det = adj.nnz // 2
        return {"alpha": alpha, "scenario": scenario, "rep": rep,
                "fpr": n_det / total, "fdr": float("nan")}
    else:  # alternative
        Sigma, edges = generate_basis_correlation(rng, p, 0.03, 0.4, 0.6)
        counts = generate_single_domain(rng, n, p, Sigma)
        rho = fastprop(counts, shrinkage=True)
        pvals = fastprop_pvalues(counts, rho)
        adj = extract_network(rho, pvals, alpha=alpha,
                              min_abs_corr=0.0, fdr_correct=True)
        true_edges = set((min(i, j), max(i, j)) for i, j, _ in edges)
        detected = set((min(i, j), max(i, j)) for i, j in zip(*adj.nonzero()))
        tp = len(true_edges & detected)
        fp = len(detected - true_edges)
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        return {"alpha": alpha, "scenario": scenario, "rep": rep,
                "fpr": float("nan"), "fdr": fdr}


# =============================================================================
# Task runners (orchestrate workers and append to CSVs)
# =============================================================================


def run_scalability(n_list, p_list, workers):
    cells = [(n, p) for n in n_list for p in p_list]
    print(f"[scalability] {len(cells)} cells on {workers} workers")
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(scalability_cell, cells), 1):
            append_row("scalability", result)
            print(f"  ({i}/{len(cells)}) n={result['n']}, p={result['p']}: "
                  f"FastProp={result['fastprop_sec']:.3f}s, "
                  f"RandProp={result['randprop_sec']:.3f}s", flush=True)


def run_detection(n_list, p_list, rho_list, reps, workers):
    cells = [(n, p, r, reps) for n in n_list for p in p_list for r in rho_list]
    print(f"[detection] {len(cells)} cells x {reps} reps on {workers} workers")
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(detection_cell, cells), 1):
            append_row("detection", result)
            print(f"  ({i}/{len(cells)}) n={result['n']}, p={result['p']}, "
                  f"rho={result['effect']}: power={result['power_mean']:.3f}, "
                  f"AUROC={result['auroc_mean']:.3f}, "
                  f"Recall@K={result['recall_at_K_mean']:.3f}", flush=True)


def run_cross_domain(n, p, q, n_interactions, reps, workers):
    args_list = [(rep, n, p, q, n_interactions) for rep in range(reps)]
    print(f"[cross_domain] {reps} reps on {workers} workers")
    with Pool(workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(cross_domain_replicate, args_list), 1):
            for row in batch:
                append_row("cross_domain", row)
            print(f"  rep {i}/{reps} done", flush=True)


def run_method_comparison(n_list, p_list, rho_list, reps, workers):
    cells = [(n, p, r, reps) for n in n_list for p in p_list for r in rho_list]
    print(f"[method_comparison] {len(cells)} cells x {reps} reps "
          f"x 6 methods on {workers} workers")
    with Pool(workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(method_comparison_cell, cells), 1):
            for row in batch:
                append_row("method_comparison", row)
            cell = batch[0]
            line = (f"  ({i}/{len(cells)}) n={cell['n']}, p={cell['p']}, "
                    f"rho={cell['effect']}: ")
            line += "  ".join(
                f"{r['method']}=AUROC {r['auroc_mean']:.3f}/{r['time_sec_mean']*1000:.0f}ms"
                for r in batch
            )
            print(line, flush=True)


def run_fdr_control(alphas, n, p, reps, workers):
    cells = [(a, scen, rep, n, p)
             for a in alphas for scen in ("null", "alternative")
             for rep in range(reps)]
    print(f"[fdr_control] {len(cells)} cells on {workers} workers")
    # Aggregate per (alpha, scenario) before writing
    bucket = {}
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(fdr_cell, cells), 1):
            key = (result["alpha"], result["scenario"])
            bucket.setdefault(key, []).append(result)
            if i % 10 == 0:
                print(f"  ({i}/{len(cells)})", flush=True)
    for (alpha, scen), rows in sorted(bucket.items()):
        vals_fpr = [r["fpr"] for r in rows if not np.isnan(r["fpr"])]
        vals_fdr = [r["fdr"] for r in rows if not np.isnan(r["fdr"])]
        append_row("fdr_control", {
            "alpha": alpha, "scenario": scen, "n_reps": reps,
            "fpr_mean": float(np.mean(vals_fpr)) if vals_fpr else "",
            "fpr_std": float(np.std(vals_fpr)) if vals_fpr else "",
            "fdr_mean": float(np.mean(vals_fdr)) if vals_fdr else "",
            "fdr_std": float(np.std(vals_fdr)) if vals_fdr else "",
        })


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["scalability", "detection",
                                       "cross_domain", "fdr_control",
                                       "method_comparison", "all"],
                   default="all")
    p.add_argument("--workers", type=int,
                   default=max(1, min(8, cpu_count() - 1)))
    p.add_argument("--reps", type=int, default=10,
                   help="Replicates per cell (default 10)")
    p.add_argument("--n", type=int, nargs="+",
                   default=[100, 500, 1000, 2000, 5000],
                   help="Sample sizes")
    p.add_argument("--p", type=int, nargs="+",
                   default=[100, 500, 1000, 5000, 10000],
                   help="Feature dimensions")
    p.add_argument("--rho", type=float, nargs="+",
                   default=[0.4, 0.7],
                   help="Effect sizes for detection")
    p.add_argument("--alpha", type=float, nargs="+",
                   default=[0.01, 0.05, 0.10, 0.20],
                   help="Nominal FDR levels for fdr_control")
    p.add_argument("--cross-n", type=int, default=300)
    p.add_argument("--cross-p", type=int, default=60)
    p.add_argument("--cross-q", type=int, default=80)
    p.add_argument("--cross-edges", type=int, default=20)
    p.add_argument("--fdr-n", type=int, default=200)
    p.add_argument("--fdr-p", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"FALCON benchmark runner @ {HOSTNAME}  (workers={args.workers})")
    print(f"Output: {ROOT / 'data'}")

    if args.task in ("scalability", "all"):
        run_scalability(args.n, args.p, args.workers)
    if args.task in ("detection", "all"):
        run_detection(args.n, args.p, args.rho, args.reps, args.workers)
    if args.task in ("cross_domain", "all"):
        run_cross_domain(args.cross_n, args.cross_p, args.cross_q,
                         args.cross_edges, args.reps, args.workers)
    if args.task in ("fdr_control", "all"):
        run_fdr_control(args.alpha, args.fdr_n, args.fdr_p,
                        args.reps, args.workers)
    if args.task in ("method_comparison", "all"):
        run_method_comparison(args.n, args.p, args.rho,
                              args.reps, args.workers)
    print("\nDone. CSVs in", ROOT / "data")
