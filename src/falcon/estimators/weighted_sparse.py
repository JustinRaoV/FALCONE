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

from falcon.estimators import _weighted_sparse_kernel as _kernel


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

    # Preallocated buffers for the alternating loop. We rotate
    # (Sigma, Sigma_new) between iterations to avoid copying.
    Sigma = S_clr.copy()
    Sigma_new = np.empty_like(Sigma)
    delta_buf = np.empty_like(Sigma)
    M_buf = np.empty_like(Sigma)
    abs_buf = np.empty_like(Sigma)
    sign_buf = np.empty_like(Sigma)

    f = np.zeros(p)
    converged = False
    iterations = 0
    tol_sq = tol * tol  # compare squared norms to avoid the sqrt
    # Path is chosen per call (not at import) so the fallback test can
    # monkeypatch _NUMBA_OK between calls.
    use_jit = _kernel.is_available()

    for it in range(1, max_iter + 1):
        if use_jit:
            delta_sq, scale_sq = _kernel._alternating_step(
                S_clr, threshold_off, Sigma, Sigma_new, f, p,
            )
        else:
            # Pure-NumPy in-place fallback (the A4 implementation).
            # Step A — closed-form offset update from R = Sigma - S_clr.
            # Reuse delta_buf as R.
            np.subtract(Sigma, S_clr, out=delta_buf)
            R_sum = delta_buf.sum(axis=1)
            total = delta_buf.sum()
            # In-place write so f stays the same array across iterations,
            # matching the JIT kernel which mutates f in place.
            f[:] = (R_sum - total / (2 * p)) / p

            # Step B — soft-threshold the off-diagonal of M = S_clr + f1' + 1f';
            # diagonal stays unpenalized.
            np.add(S_clr, f[:, None], out=M_buf)
            M_buf += f[None, :]
            np.abs(M_buf, out=abs_buf)
            np.subtract(abs_buf, threshold_off, out=abs_buf)
            np.maximum(abs_buf, 0.0, out=abs_buf)
            np.sign(M_buf, out=sign_buf)
            np.multiply(sign_buf, abs_buf, out=Sigma_new)
            np.fill_diagonal(Sigma_new, np.diag(M_buf))
            # Symmetrize via temporary to avoid aliasing on the in-place add
            # of Sigma_new + Sigma_new.T.
            np.add(Sigma_new, Sigma_new.T, out=delta_buf)
            np.multiply(delta_buf, 0.5, out=Sigma_new)

            # Frobenius delta_sq and scale_sq via einsum (allocation-free).
            np.subtract(Sigma_new, Sigma, out=delta_buf)
            delta_sq = float(np.einsum("ij,ij->", delta_buf, delta_buf))
            scale_sq = max(float(np.einsum("ij,ij->", Sigma_new, Sigma_new)), 1e-24)

        iterations = it
        # Rotate buffers: Sigma becomes the just-computed value; the old
        # Sigma buffer is reused next iteration as Sigma_new.
        Sigma, Sigma_new = Sigma_new, Sigma

        if delta_sq / max(scale_sq, 1e-24) < tol_sq:
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
