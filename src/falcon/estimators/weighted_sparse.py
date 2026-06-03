"""Weighted sparse covariance estimator (fastCCLasso-style).

Clean-room implementation derived from the published method description
in Zhang, Fang, and Hu (2024) "fastCCLasso". No reference R/LGPL code is
consulted.

Model:
    S_clr ~ G_p Sigma_basis G_p
where ``S_clr`` is the sample CLR covariance and ``G_p = I - 1 1' / p``.
Equivalently, off the diagonal,
    Sigma_basis_{ij} = S_clr_{ij} + f_i + f_j
for some offset vector ``f``. The estimator minimizes

    L(Sigma, f) = (1/2) || W ⊙ (Sigma - 1 f' - f 1' - S_clr) ||_F^2
                  + lambda * sum_{i != j} | Sigma_{ij} | / w_{ij}**2_eff

by alternating between

    Step A (offset update, fixed Sigma):
        f = (R 1 - 1' R 1 / (2 p) * 1) / p,   R = Sigma - S_clr
        (closed form; ignores W and uses the symmetric residual.)

    Step B (covariance update, fixed f):
        M = S_clr + f 1' + 1 f'
        Sigma_{ii} = M_{ii}                           (diagonal unpenalized)
        Sigma_{ij} = sign(M_{ij}) * max(|M_{ij}| - lambda / w_{ij}^2, 0)
                                                       (off-diagonal soft thresh)

The default weight matrix uses entry-specific variance estimates
``theta_{ij} = (1/n) sum_k (Z_ki Z_kj - sigma_ij)^2``; weights are
``w_{ij} = 1 / sqrt(theta_{ij})`` then renormalized to mean 1 across
off-diagonal entries so ``lambda`` retains its usual scale.

Complexity is ``O(p^2)`` per iteration with no eigendecomposition or
linear solve. Convergence is monitored on the Frobenius change of
``Sigma`` between iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedSparseResult:
    covariance: np.ndarray
    correlation: np.ndarray
    lambda_value: float
    iterations: int
    converged: bool
    min_eigenvalue: float
    offset: np.ndarray
    weights: np.ndarray


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(np.float64, copy=True)
    diag[diag <= 0] = 1.0
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _default_weights(Zc: np.ndarray, S_clr: np.ndarray) -> np.ndarray:
    n, p = Zc.shape
    if p == 1:
        return np.ones((p, p))
    # Avoid the (n, p, p) outer-product tensor used by the naive form;
    # use the GEMM identity E[(Z_i Z_j)^2] = (Zc^2).T @ (Zc^2) / n. See
    # estimators/adaptive_threshold.py for the derivation. This keeps
    # memory at O(p^2) instead of O(n p^2).
    z_sq = Zc * Zc
    expected_sq_product = (z_sq.T @ z_sq) / n
    theta = np.maximum(expected_sq_product - S_clr ** 2, 1e-12)
    weights = 1.0 / np.sqrt(theta)
    np.fill_diagonal(weights, 1.0)
    off_mask = ~np.eye(p, dtype=bool)
    mean_off = weights[off_mask].mean()
    if mean_off > 0:
        weights = weights / mean_off
    return weights


def estimate_weighted_sparse(
    Z: np.ndarray,
    *,
    lambda_value: float,
    weights: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    support_only: bool = False,
) -> WeightedSparseResult:
    """Estimate the basis covariance via weighted soft thresholding."""
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional matrix")
    n, p = Z.shape
    if n < 3:
        raise ValueError("estimator needs at least three samples")

    Zc = Z - Z.mean(axis=0, keepdims=True)
    S_clr = (Zc.T @ Zc) / (n - 1)

    if weights is None:
        W = _default_weights(Zc, S_clr)
    else:
        W = np.asarray(weights, dtype=np.float64)
        if W.shape != (p, p):
            raise ValueError("weights must have shape (p, p)")

    # Per-entry threshold strength: lambda_eff_{ij} = lambda / w_{ij}^2.
    # Larger weight (smaller theta) -> smaller threshold (we trust it more).
    threshold_off = lambda_value / np.maximum(W ** 2, 1e-12)
    np.fill_diagonal(threshold_off, 0.0)

    Sigma = S_clr.copy()
    f = np.zeros(p)
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        Sigma_prev = Sigma.copy()

        # Step A — closed-form offset update from R = Sigma - S_clr.
        R = Sigma - S_clr
        R_sum = R.sum(axis=1)
        total = R.sum()
        f = (R_sum - total / (2 * p)) / p

        # Step B — soft-threshold the off-diagonal of M = S_clr + f1' + 1f';
        # diagonal stays unpenalized.
        M = S_clr + f[:, None] + f[None, :]
        Sigma = np.sign(M) * np.maximum(np.abs(M) - threshold_off, 0.0)
        np.fill_diagonal(Sigma, np.diag(M))
        Sigma = 0.5 * (Sigma + Sigma.T)

        delta = np.linalg.norm(Sigma - Sigma_prev)
        scale = max(np.linalg.norm(Sigma), 1e-12)
        iterations = it
        if delta / scale < tol:
            converged = True
            break

    if support_only:
        # Skip the O(p^3) eigvalsh and the unused correlation extraction.
        # min_eigenvalue is recorded as NaN to signal "not computed".
        correlation = Sigma
        min_eig = float("nan")
    else:
        correlation = _correlation_from_covariance(Sigma)
        min_eig = float(np.linalg.eigvalsh(Sigma).min())

    return WeightedSparseResult(
        covariance=Sigma,
        correlation=correlation,
        lambda_value=float(lambda_value),
        iterations=int(iterations),
        converged=bool(converged),
        min_eigenvalue=min_eig,
        offset=f,
        weights=W,
    )
