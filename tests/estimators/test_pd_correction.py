"""Tests for the positive-definite correction.

The correction must:

* preserve symmetry,
* preserve the off-diagonal selected-edge support exactly,
* bound the minimum eigenvalue at or above the configured floor,
* leave already-PD inputs essentially untouched, and
* be idempotent.
"""

from __future__ import annotations

import numpy as np
import pytest

from falcon.estimators.pd_correction import PDCorrectionResult, apply_pd_correction


def _indefinite_thresholded():
    """Build a symmetric matrix with sparse off-diagonal support and a
    negative minimum eigenvalue."""
    sigma = np.array(
        [
            [1.0, 0.95, 0.0, 0.0],
            [0.95, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.95],
            [0.0, 0.0, 0.95, 1.0],
        ]
    )
    # min eigenvalue here is positive (0.05), so add a perturbation to make
    # it indefinite while keeping the support.
    sigma[0, 0] = 0.5
    sigma[1, 1] = 0.5
    return sigma


def test_returns_symmetric_matrix_with_floored_min_eigenvalue():
    sigma = _indefinite_thresholded()
    result = apply_pd_correction(sigma, floor=1e-3)
    assert isinstance(result, PDCorrectionResult)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
    min_eig = float(np.linalg.eigvalsh(result.covariance).min())
    assert min_eig >= 1e-3 - 1e-9


def test_off_diagonal_support_is_preserved_exactly():
    sigma = _indefinite_thresholded()
    p = sigma.shape[0]
    off_mask = ~np.eye(p, dtype=bool)
    nonzero_before = sigma[off_mask] != 0

    result = apply_pd_correction(sigma, floor=1e-3)
    nonzero_after = result.covariance[off_mask] != 0
    np.testing.assert_array_equal(nonzero_before, nonzero_after)
    # Every entry that was zero stays zero.
    np.testing.assert_array_equal(
        result.covariance[off_mask][~nonzero_before],
        sigma[off_mask][~nonzero_before],
    )


def test_off_diagonal_values_are_unchanged_for_diagonal_loading_path():
    sigma = _indefinite_thresholded()
    result = apply_pd_correction(sigma, floor=1e-3)
    p = sigma.shape[0]
    off_mask = ~np.eye(p, dtype=bool)
    np.testing.assert_array_equal(result.covariance[off_mask], sigma[off_mask])


def test_already_pd_input_returns_zero_shift():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    sigma = A @ A.T + 0.5 * np.eye(6)
    assert np.linalg.eigvalsh(sigma).min() > 0.5

    result = apply_pd_correction(sigma, floor=1e-6)
    assert result.shift == 0.0
    np.testing.assert_allclose(result.covariance, sigma)


def test_correction_is_idempotent_under_same_floor():
    sigma = _indefinite_thresholded()
    once = apply_pd_correction(sigma, floor=1e-3)
    twice = apply_pd_correction(once.covariance, floor=1e-3)
    assert twice.shift == 0.0
    np.testing.assert_allclose(twice.covariance, once.covariance)


def test_records_shift_and_min_eigenvalues():
    sigma = _indefinite_thresholded()
    pre_min = float(np.linalg.eigvalsh(sigma).min())
    result = apply_pd_correction(sigma, floor=1e-3)
    np.testing.assert_allclose(result.min_eigenvalue_before, pre_min)
    assert result.min_eigenvalue_after >= 1e-3 - 1e-9
    assert result.shift == pytest.approx(max(0.0, 1e-3 - pre_min))


def test_negative_floor_rejected():
    sigma = _indefinite_thresholded()
    with pytest.raises(ValueError, match="floor"):
        apply_pd_correction(sigma, floor=-0.1)


def test_non_square_input_rejected():
    with pytest.raises(ValueError, match="square"):
        apply_pd_correction(np.zeros((3, 4)), floor=1e-3)


def test_correlation_recomputed_after_correction():
    sigma = _indefinite_thresholded()
    result = apply_pd_correction(sigma, floor=1e-3)
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    # Correlation supports are derived from the corrected covariance and
    # should still match the input off-diagonal support pattern.
    p = sigma.shape[0]
    off_mask = ~np.eye(p, dtype=bool)
    np.testing.assert_array_equal(
        result.correlation[off_mask] != 0,
        sigma[off_mask] != 0,
    )
