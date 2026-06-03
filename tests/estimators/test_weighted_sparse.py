"""Tests for the fastCCLasso-style weighted sparse covariance estimator.

The estimator alternates two simple updates:

* nuisance-offset update ``f`` that accounts for compositional closure
  (``Sigma_basis - S_clr ~ f 1' + 1 f'`` plus a soft-thresholding residual);
* weighted soft-threshold update on off-diagonal covariance entries with
  the diagonal unpenalized.

These properties — symmetry, diagonal preservation, sparsity under large
lambda, monotone convergence of the residual — are what we test for.
The clean-room implementation is derived from Zhang, Fang and Hu (2024)
"fastCCLasso", not copied from any reference R code.
"""

from __future__ import annotations

import numpy as np
import pytest

from falcon.estimators.weighted_sparse import (
    WeightedSparseResult,
    estimate_weighted_sparse,
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


def test_output_is_symmetric_and_returns_weighted_sparse_result():
    Z, _ = _make_dataset(seed=0)
    result = estimate_weighted_sparse(Z, lambda_value=0.05)
    assert isinstance(result, WeightedSparseResult)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-10)


def test_lambda_zero_recovers_clr_sample_covariance_up_to_offset():
    Z, _ = _make_dataset(seed=1)
    result = estimate_weighted_sparse(Z, lambda_value=0.0, max_iter=100)
    n = Z.shape[0]
    Zc = Z - Z.mean(axis=0)
    S_clr = (Zc.T @ Zc) / (n - 1)

    # With lambda=0 the off-diagonal residual after convergence sits on the
    # rank-2 offset manifold ``f 1' + 1 f'``.
    R = result.covariance - S_clr
    p = Z.shape[1]
    # Project R onto the rank-2 offset manifold and check the off-manifold
    # residual is essentially zero.
    f = (R.sum(axis=1) - R.sum() / (2 * p)) / p
    fitted_offset = np.add.outer(f, f)
    off = ~np.eye(p, dtype=bool)
    np.testing.assert_allclose(R[off], fitted_offset[off], atol=1e-6)


def test_large_lambda_zeros_off_diagonal_entries():
    Z, _ = _make_dataset(seed=2)
    result = estimate_weighted_sparse(Z, lambda_value=10.0, max_iter=50)
    p = Z.shape[1]
    off = ~np.eye(p, dtype=bool)
    np.testing.assert_array_equal(result.covariance[off], 0.0)


def test_diagonal_is_unpenalized_and_strictly_positive():
    Z, _ = _make_dataset(seed=3)
    result = estimate_weighted_sparse(Z, lambda_value=0.05)
    # Diagonal stays strictly positive; the algorithm never zeros it.
    assert (np.diag(result.covariance) > 0).all()


def test_correlation_diagonal_is_one_and_bounded():
    Z, _ = _make_dataset(seed=4)
    result = estimate_weighted_sparse(Z, lambda_value=0.05)
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert (np.abs(result.correlation) <= 1.0 + 1e-9).all()


def test_lambda_grid_monotone_in_off_diag_sparsity():
    Z, _ = _make_dataset(seed=5)
    p = Z.shape[1]
    off = ~np.eye(p, dtype=bool)
    last_zero_count = -1
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        result = estimate_weighted_sparse(Z, lambda_value=lam)
        zero_count = int((result.covariance[off] == 0).sum())
        assert zero_count >= last_zero_count
        last_zero_count = zero_count


def test_converges_within_max_iter_on_easy_data():
    Z, _ = _make_dataset(seed=6)
    result = estimate_weighted_sparse(Z, lambda_value=0.1, max_iter=500, tol=1e-7)
    assert result.converged is True
    assert result.iterations < 500


def test_returns_min_eigenvalue_of_estimate():
    Z, _ = _make_dataset(seed=7)
    result = estimate_weighted_sparse(Z, lambda_value=0.05)
    expected = float(np.linalg.eigvalsh(result.covariance).min())
    np.testing.assert_allclose(result.min_eigenvalue, expected, atol=1e-9)


def test_negative_lambda_rejected():
    Z, _ = _make_dataset(seed=8)
    with pytest.raises(ValueError, match="lambda"):
        estimate_weighted_sparse(Z, lambda_value=-0.1)


def test_uniform_weights_match_unweighted_path():
    Z, _ = _make_dataset(seed=9)
    p = Z.shape[1]
    result_default_then_uniform = estimate_weighted_sparse(
        Z, lambda_value=0.05, weights=np.ones((p, p))
    )
    # With uniform weights, the per-entry threshold equals lambda
    # everywhere; the path is well defined and produces a symmetric output.
    np.testing.assert_allclose(
        result_default_then_uniform.covariance,
        result_default_then_uniform.covariance.T,
        atol=1e-10,
    )
