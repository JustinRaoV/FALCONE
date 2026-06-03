"""Positive-definite correction that preserves selected-edge support.

The standard Higham nearest-PD projection touches every off-diagonal
entry. That silently destroys the support produced by a sparse
estimator. This module instead uses diagonal loading: shift the diagonal
upward until the minimum eigenvalue reaches the configured floor. Off-
diagonal entries — which carry the selected support — are left exactly
as the caller produced them.

If the input is already PD with margin above the floor, the correction
is a no-op (shift == 0). The function is idempotent under a fixed floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PDCorrectionResult:
    covariance: np.ndarray
    correlation: np.ndarray
    shift: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(np.float64, copy=True)
    diag[diag <= 0] = 1.0
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def apply_pd_correction(
    covariance: np.ndarray,
    *,
    floor: float = 1e-4,
) -> PDCorrectionResult:
    """Force a covariance matrix to be PD by diagonal loading."""
    if floor < 0:
        raise ValueError("floor must be non-negative")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")

    sym = 0.5 * (cov + cov.T)
    min_eig_before = float(np.linalg.eigvalsh(sym).min())

    # Treat min_eig within rounding error of the floor as already satisfied
    # so a second call is a strict no-op.
    tol = max(1e-12, 100 * np.finfo(np.float64).eps * max(1.0, abs(floor)))
    if min_eig_before >= floor - tol:
        return PDCorrectionResult(
            covariance=cov,
            correlation=_correlation_from_covariance(cov),
            shift=0.0,
            min_eigenvalue_before=min_eig_before,
            min_eigenvalue_after=min_eig_before,
        )

    shift = floor - min_eig_before
    corrected = cov.copy()
    diag_idx = np.arange(cov.shape[0])
    corrected[diag_idx, diag_idx] = corrected[diag_idx, diag_idx] + shift
    min_eig_after = float(np.linalg.eigvalsh(0.5 * (corrected + corrected.T)).min())

    return PDCorrectionResult(
        covariance=corrected,
        correlation=_correlation_from_covariance(corrected),
        shift=float(shift),
        min_eigenvalue_before=min_eig_before,
        min_eigenvalue_after=min_eig_after,
    )
