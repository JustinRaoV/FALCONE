"""Simulators for Falcon-SR feasibility benchmarks.

* ``generate_basis_correlation`` — sparse symmetric basis correlation matrix.
* ``generate_single_domain``     — compositional count matrix for SparCC-style
                                  single-domain inference.
* ``generate_cross_domain``      — two independently normalised compositional
                                  count matrices with planted bipartite
                                  cross-correlations for SparXCC Case-C style
                                  inference.

These are deliberately simple data-generating processes scoped to the
feasibility benchmark, not the full simulation grid in spec §13.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


def nearest_pd(matrix: np.ndarray, min_eig: float = 0.01) -> np.ndarray:
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, min_eig)
    return (eigvecs * eigvals) @ eigvecs.T


def generate_basis_correlation(
    rng: np.random.Generator,
    p: int,
    density: float,
    *,
    effect_lo: float = 0.35,
    effect_hi: float = 0.65,
    neg_fraction: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    n_edges_target = int(density * p * (p - 1) / 2)
    sigma = np.eye(p)
    edges: list[tuple[int, int, float]] = []
    chosen: set[tuple[int, int]] = set()
    while len(edges) < n_edges_target:
        i = int(rng.integers(0, p))
        j = int(rng.integers(0, p))
        if i == j or (i, j) in chosen or (j, i) in chosen:
            continue
        rho = float(rng.uniform(effect_lo, effect_hi))
        if rng.random() < neg_fraction:
            rho = -rho
        sigma[i, j] = sigma[j, i] = rho
        chosen.add((i, j))
        edges.append((i, j, rho))
    sigma = nearest_pd(sigma)
    diag = np.sqrt(np.diag(sigma))
    sigma = sigma / np.outer(diag, diag)
    return sigma, edges


def generate_single_domain(
    rng: np.random.Generator,
    n: int,
    p: int,
    sigma: np.ndarray,
    *,
    sequencing_depth: int = 50_000,
    detection_limit: float = 1e-4,
) -> np.ndarray:
    chol = cholesky(sigma, lower=True)
    z = rng.standard_normal((n, p)) @ chol.T
    ranks = np.arange(1, p + 1)
    mean_log_abundance = 6.0 - 1.5 * np.log(ranks)
    rng.shuffle(mean_log_abundance)
    abundance = np.exp(z + mean_log_abundance)
    proportions = abundance / abundance.sum(axis=1, keepdims=True)
    depths = rng.poisson(sequencing_depth, size=n)
    counts = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        counts[i] = rng.multinomial(int(depths[i]), proportions[i])
    relative = counts / counts.sum(axis=1, keepdims=True)
    counts[relative < detection_limit] = 0
    return counts


def generate_cross_domain(
    rng: np.random.Generator,
    n: int,
    p: int,
    q: int,
    *,
    density: float = 0.01,
    effect_lo: float = 0.35,
    effect_hi: float = 0.65,
    neg_fraction: float = 0.5,
    sequencing_depth: int = 50_000,
    detection_limit: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """Generate two independently normalised compositions with planted
    bipartite cross-correlations.

    Returns ``(counts_x, counts_y, cross_correlation_truth, planted_edges)``
    where ``cross_correlation_truth`` is the (p, q) Pearson-correlation matrix
    of the latent log-abundances after the joint PD projection, and
    ``planted_edges`` is the list of intended (i, k, sign * magnitude)
    triples (their realised magnitudes may shrink after the PD projection).
    """
    n_edges_target = max(1, int(density * p * q))
    omega = np.zeros((p, q))
    edges: list[tuple[int, int, float]] = []
    chosen: set[tuple[int, int]] = set()
    while len(edges) < n_edges_target:
        i = int(rng.integers(0, p))
        k = int(rng.integers(0, q))
        if (i, k) in chosen:
            continue
        magnitude = float(rng.uniform(effect_lo, effect_hi))
        if rng.random() < neg_fraction:
            magnitude = -magnitude
        omega[i, k] = magnitude
        edges.append((i, k, magnitude))
        chosen.add((i, k))

    sigma_full = np.eye(p + q)
    sigma_full[:p, p:] = omega
    sigma_full[p:, :p] = omega.T
    sigma_full = nearest_pd(sigma_full)
    diag = np.sqrt(np.diag(sigma_full))
    sigma_full = sigma_full / np.outer(diag, diag)
    cross_truth = sigma_full[:p, p:]

    chol = cholesky(sigma_full, lower=True)
    z = rng.standard_normal((n, p + q)) @ chol.T
    mean_x = rng.uniform(2.0, 6.0, size=p)
    mean_y = rng.uniform(2.0, 6.0, size=q)
    w_x = np.exp(z[:, :p] + mean_x)
    w_y = np.exp(z[:, p:] + mean_y)
    prop_x = w_x / w_x.sum(axis=1, keepdims=True)
    prop_y = w_y / w_y.sum(axis=1, keepdims=True)
    depths_x = rng.poisson(sequencing_depth, size=n)
    depths_y = rng.poisson(sequencing_depth, size=n)
    counts_x = np.zeros((n, p), dtype=np.int64)
    counts_y = np.zeros((n, q), dtype=np.int64)
    for s in range(n):
        counts_x[s] = rng.multinomial(int(depths_x[s]), prop_x[s])
        counts_y[s] = rng.multinomial(int(depths_y[s]), prop_y[s])
    rel_x = counts_x / counts_x.sum(axis=1, keepdims=True)
    rel_y = counts_y / counts_y.sum(axis=1, keepdims=True)
    counts_x[rel_x < detection_limit] = 0
    counts_y[rel_y < detection_limit] = 0
    return counts_x, counts_y, cross_truth, edges
