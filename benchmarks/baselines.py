"""Self-contained Python baselines for the rebuild benchmark.

Each baseline operates on raw counts and returns a ``BaselineResult``
with ``correlation`` (a ``(p, p)`` matrix; entries on the diagonal are 1
or unset), ``estimand_family``, ``converged``, and ``iterations``.
Baselines are responsible for their own preprocessing so they remain
honest snapshots of the original method, not Falcon-shaped variants.

``estimand_family`` is recorded so the benchmark report can avoid
incorrectly comparing matched and adjacent estimands. Methods that
target the latent log-abundance correlation are labeled
``latent_log_abundance_correlation``; precision-matrix methods would
use ``inverse_covariance``; nonlinear methods would use
``nonlinear_dependence``. Adjacent-estimand baselines are reported as
context, not as match-evidence for an advantage claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    correlation: np.ndarray
    estimand_family: str
    method: str
    converged: bool
    iterations: int


def _multiplicative_replacement(composition: np.ndarray) -> np.ndarray:
    p = composition.shape[1]
    delta = 0.65 / (p * p)
    zero_mask = composition == 0
    zero_count = zero_mask.sum(axis=1, keepdims=True)
    scale = 1.0 - zero_count * delta
    return np.where(zero_mask, delta, composition * scale)


def _to_log_composition(counts: np.ndarray) -> np.ndarray:
    composition = counts / counts.sum(axis=1, keepdims=True)
    composition = _multiplicative_replacement(composition)
    return np.log(composition)


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(np.float64, copy=True)
    diag[diag <= 0] = 1.0
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def pearson_clr(counts: np.ndarray) -> BaselineResult:
    """Naive baseline: Pearson correlation of the CLR-transformed log
    composition. Carries an implicit closure bias which is exactly why
    SparCC and friends exist."""
    log_x = _to_log_composition(counts)
    Z = log_x - log_x.mean(axis=1, keepdims=True)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)
    corr = _correlation_from_covariance(cov)
    return BaselineResult(
        correlation=corr,
        estimand_family="latent_log_abundance_correlation",
        method="pearson_clr",
        converged=True,
        iterations=1,
    )


def sparcc_closed_form(counts: np.ndarray) -> BaselineResult:
    """SparCC closed-form basis-correlation estimate.

    Implements the sparse-average-correlation closed form for the basis
    variances:

        omega_i^2 = (sum_j t_ij - mean_total) / (p - 1)
                    (approximation that assumes few strong correlations)

    where ``t_ij = Var(log(x_i / x_j))`` is the Aitchison variation
    matrix. The basis correlation is then
    ``(omega_i^2 + omega_j^2 - t_ij) / (2 omega_i omega_j)``.
    """
    log_x = _to_log_composition(counts)
    n, p = log_x.shape
    centered = log_x - log_x.mean(axis=0, keepdims=True)
    cov_log = (centered.T @ centered) / max(n - 1, 1)
    diag = np.diag(cov_log)
    t = diag[:, None] + diag[None, :] - 2 * cov_log
    np.fill_diagonal(t, 0.0)

    # SparCC closed-form basis variance.
    row_sums = t.sum(axis=1)
    total = t.sum()
    mean_total = total / max(p * (p - 1), 1)
    omega_sq = np.maximum((row_sums / max(p - 1, 1)) - mean_total, 1e-9)
    omega = np.sqrt(omega_sq)

    rho = (omega_sq[:, None] + omega_sq[None, :] - t) / (2.0 * np.outer(omega, omega))
    np.fill_diagonal(rho, 1.0)
    rho = np.clip(rho, -1.0, 1.0)

    return BaselineResult(
        correlation=rho,
        estimand_family="latent_log_abundance_correlation",
        method="sparcc_closed_form",
        converged=True,
        iterations=1,
    )


AVAILABLE_PYTHON_BASELINES = {
    "pearson_clr": pearson_clr,
    "sparcc_closed_form": sparcc_closed_form,
}
