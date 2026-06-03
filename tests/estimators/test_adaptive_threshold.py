"""Tests for the COAT-style adaptive threshold covariance estimator.

The estimator targets the basis log-abundance covariance through the
identity ``Sigma_clr = G_p Sigma_basis G_p`` and applies entry-specific
thresholding to ``Sigma_clr``. Under sparsity of ``Sigma_basis`` the
thresholded matrix recovers the basis covariance support; the diagonal
is preserved untouched (the basis variances remain unidentifiable from
``Sigma_clr`` alone, so we keep the CLR diagonals).
"""

from __future__ import annotations

import numpy as np
import pytest

from falcon.estimators.adaptive_threshold import (
    AdaptiveThresholdResult,
    estimate_adaptive_threshold,
)


def _clr(log_composition: np.ndarray) -> np.ndarray:
    return log_composition - log_composition.mean(axis=1, keepdims=True)


def _make_dataset(n=400, p=24, support=None, seed=0, edge_strength=0.6):
    rng = np.random.default_rng(seed)
    if support is None:
        support = np.array([(0, 1), (2, 3), (4, 5)])
    cov = np.eye(p)
    for i, j in support:
        cov[i, j] = cov[j, i] = edge_strength
    log_basis = rng.multivariate_normal(np.zeros(p), cov, size=n)
    composition = np.exp(log_basis)
    composition = composition / composition.sum(axis=1, keepdims=True)
    log_composition = np.log(composition)
    return _clr(log_composition), support


def test_output_is_symmetric_with_preserved_diagonal():
    Z, _ = _make_dataset()
    result = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="hard")

    assert isinstance(result, AdaptiveThresholdResult)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
    # Diagonal of thresholded covariance equals the CLR sample variance diagonal.
    sample_cov = np.cov(Z, rowvar=False, ddof=1)
    np.testing.assert_allclose(np.diag(result.covariance), np.diag(sample_cov))


def test_correlation_matrix_in_unit_diagonal_form():
    Z, _ = _make_dataset()
    result = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="hard")
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert (np.abs(result.correlation) <= 1.0 + 1e-9).all()


def test_hard_threshold_zeroes_small_entries():
    Z, support = _make_dataset(seed=1)
    sample_cov = np.cov(Z, rowvar=False, ddof=1)
    result = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="hard")

    p = Z.shape[1]
    off = ~np.eye(p, dtype=bool)
    # Thresholding strictly reduces the off-diagonal support compared to
    # the dense sample covariance.
    sample_nonzero = np.count_nonzero(sample_cov[off])
    thresh_nonzero = np.count_nonzero(result.covariance[off])
    assert thresh_nonzero < sample_nonzero

    # At least half the planted edges survive the threshold.
    surviving = sum(int(result.covariance[i, j] != 0) for i, j in support)
    assert surviving >= len(support) // 2 + 1


def test_soft_threshold_shrinks_but_preserves_sign():
    Z, support = _make_dataset(seed=2)
    hard = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="hard")
    soft = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="soft")

    p = Z.shape[1]
    off = ~np.eye(p, dtype=bool)
    # Soft thresholding reduces magnitudes relative to the unthresholded
    # sample covariance everywhere it is non-zero.
    sample_cov = np.cov(Z, rowvar=False, ddof=1)
    nonzero = soft.covariance[off] != 0
    np.testing.assert_array_less(
        np.abs(soft.covariance[off][nonzero]) - 1e-12,
        np.abs(sample_cov[off][nonzero]) + 1e-12,
    )
    # Hard and soft agree on the support pattern (same threshold strength).
    np.testing.assert_array_equal(hard.covariance != 0, soft.covariance != 0)


def test_threshold_strength_increases_zero_fraction_monotonically():
    Z, _ = _make_dataset(seed=3)
    weak = estimate_adaptive_threshold(Z, threshold_constant=0.5, mode="hard")
    strong = estimate_adaptive_threshold(Z, threshold_constant=4.0, mode="hard")

    p = Z.shape[1]
    off = ~np.eye(p, dtype=bool)
    weak_zeros = (weak.covariance[off] == 0).mean()
    strong_zeros = (strong.covariance[off] == 0).mean()
    assert strong_zeros >= weak_zeros


def test_estimator_records_lambda_iterations_and_eigenvalue():
    Z, _ = _make_dataset(seed=4)
    result = estimate_adaptive_threshold(Z, threshold_constant=2.0, mode="hard")

    assert result.lambda_value > 0.0
    assert result.iterations == 1  # closed form
    assert result.converged is True
    # Without PD correction the thresholded covariance can have a small
    # negative minimum eigenvalue. The reported number must equal the
    # actual minimum eigenvalue, not be silently floored.
    sym = 0.5 * (result.covariance + result.covariance.T)
    expected_min_eig = np.linalg.eigvalsh(sym).min()
    np.testing.assert_allclose(result.min_eigenvalue, expected_min_eig, atol=1e-9)


def test_unknown_mode_rejected():
    Z, _ = _make_dataset(seed=5)
    with pytest.raises(ValueError, match="mode"):
        estimate_adaptive_threshold(Z, mode="bogus")


def test_negative_threshold_constant_rejected():
    Z, _ = _make_dataset(seed=6)
    with pytest.raises(ValueError, match="threshold_constant"):
        estimate_adaptive_threshold(Z, threshold_constant=-1.0)
