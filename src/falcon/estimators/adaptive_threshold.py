"""COAT-style adaptive threshold covariance estimator.

This is a clean-room implementation derived from the published method
description in Cao, Lin, and Li (2019), "Large covariance estimation for
compositional data via composition-adjusted thresholding". No reference R
code is consulted.

Algorithm overview:

1. Take the CLR-centered log composition ``Z`` (the caller is responsible
   for centering rows of ``log(composition)``).
2. Form the sample covariance ``S = cov(Z)``.
3. Estimate the per-entry variance ``theta_ij`` of the off-diagonal sample
   covariance entries.
4. Apply entry-specific thresholding ``T_lambda(S_ij)`` with threshold
   ``lambda_ij = c * sqrt(theta_ij * log p / n)``. The diagonal is
   preserved untouched.

Hard and soft thresholding are exposed; the benchmark freezes one mode
before the holdout grid is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_MODES = ("hard", "soft")


@dataclass(frozen=True)
class AdaptiveThresholdResult:
    covariance: np.ndarray
    correlation: np.ndarray
    lambda_value: float
    iterations: int
    converged: bool
    min_eigenvalue: float
    threshold_matrix: np.ndarray
    mode: str


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(np.float64, copy=True)
    diag[diag <= 0] = 1.0
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def estimate_adaptive_threshold(
    Z: np.ndarray,
    *,
    threshold_constant: float = 2.0,
    mode: str = "hard",
) -> AdaptiveThresholdResult:
    """Threshold the CLR sample covariance with entry-adaptive lambdas.

    Parameters
    ----------
    Z : (n, p) array
        CLR-centered log composition, i.e. each row should sum to zero.
        The caller (``preprocessing`` + ``infer_network``) prepares this.
    threshold_constant : float
        Multiplier ``c`` on ``sqrt(log p / n)`` in the threshold rule.
    mode : {"hard", "soft"}
        Hard or soft thresholding applied to off-diagonal entries.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if threshold_constant < 0:
        raise ValueError("threshold_constant must be non-negative")
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional matrix")

    n, p = Z.shape
    if n < 3:
        raise ValueError("estimator needs at least three samples")

    Zc = Z - Z.mean(axis=0, keepdims=True)
    sample_cov = (Zc.T @ Zc) / (n - 1)

    # Per-entry variance of the sample covariance: theta_ij = (1/n) Σ_k
    # (Z_ki Z_kj - sigma_ij)^2 with the convention that the centered
    # product matrix ``Y_kij = Zc_ki Zc_kj`` has empirical mean ~ sigma_ij.
    products = (Zc[:, :, None] * Zc[:, None, :])  # (n, p, p)
    theta = ((products - sample_cov[None, :, :]) ** 2).mean(axis=0)

    lambda_matrix = threshold_constant * np.sqrt(
        np.maximum(theta, 0.0) * np.log(max(p, 2)) / n
    )
    np.fill_diagonal(lambda_matrix, 0.0)

    if mode == "hard":
        thresholded = np.where(np.abs(sample_cov) >= lambda_matrix, sample_cov, 0.0)
    else:  # soft
        sign = np.sign(sample_cov)
        magnitude = np.maximum(np.abs(sample_cov) - lambda_matrix, 0.0)
        thresholded = sign * magnitude

    # Preserve diagonal exactly.
    np.fill_diagonal(thresholded, np.diag(sample_cov))
    # Symmetrize to remove tiny numerical asymmetries.
    thresholded = 0.5 * (thresholded + thresholded.T)

    correlation = _correlation_from_covariance(thresholded)
    lambda_value = float(lambda_matrix[~np.eye(p, dtype=bool)].mean()) if p > 1 else 0.0
    min_eig = float(np.linalg.eigvalsh(thresholded).min())

    return AdaptiveThresholdResult(
        covariance=thresholded,
        correlation=correlation,
        lambda_value=lambda_value,
        iterations=1,
        converged=True,
        min_eigenvalue=min_eig,
        threshold_matrix=lambda_matrix,
        mode=mode,
    )
