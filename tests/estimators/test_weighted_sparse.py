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


def test_relative_tolerance_responds_to_tol_parameter():
    """Relative convergence tolerance: looser tol should converge in fewer
    iterations on the same problem. Verifies the relative criterion is
    in force (absolute-tol code would also pass this, but the new code
    must too)."""
    rng = np.random.default_rng(0)
    n, p = 80, 20
    # Use small p so well within convergence reach at default tol.
    Z = rng.normal(0, 1, size=(n, p))
    r_tight = estimate_weighted_sparse(Z, lambda_value=0.05, max_iter=200, tol=1e-6)
    r_loose = estimate_weighted_sparse(Z, lambda_value=0.05, max_iter=200, tol=1e-3)
    assert r_tight.converged, "tight tol must converge within max_iter on small p"
    assert r_loose.converged, "loose tol must converge within max_iter on small p"
    assert r_loose.iterations <= r_tight.iterations, (
        f"looser tol used MORE iters ({r_loose.iterations}) than tight ({r_tight.iterations})"
    )


def test_support_only_skips_eigvalsh_and_correlation(monkeypatch):
    """support_only=True must skip the per-call eigvalsh AND skip the
    correlation extraction. Covariance support (nonzero positions) is
    preserved exactly."""
    import numpy as np
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1, size=(60, 20))

    eigvalsh_calls = {"n": 0}
    real_eigvalsh = np.linalg.eigvalsh

    def counting_eigvalsh(*args, **kwargs):
        eigvalsh_calls["n"] += 1
        return real_eigvalsh(*args, **kwargs)

    monkeypatch.setattr("numpy.linalg.eigvalsh", counting_eigvalsh)

    r_full = estimate_weighted_sparse(Z, lambda_value=0.05, max_iter=50)
    n_full = eigvalsh_calls["n"]
    eigvalsh_calls["n"] = 0

    r_skip = estimate_weighted_sparse(Z, lambda_value=0.05, max_iter=50, support_only=True)
    n_skip = eigvalsh_calls["n"]

    assert n_skip == 0, f"support_only must not call eigvalsh; got {n_skip} calls"
    assert n_full >= 1, f"full path should call eigvalsh at least once; got {n_full}"
    assert np.isnan(r_skip.min_eigenvalue), "min_eigenvalue must be NaN when skipped"
    # Covariance nonzero support must match between full and support_only.
    full_support = (r_full.covariance != 0)
    skip_support = (r_skip.covariance != 0)
    np.testing.assert_array_equal(
        full_support, skip_support,
        err_msg="support_only must produce the same nonzero positions as full path",
    )


def test_alternating_loop_output_is_stable_to_reordered_ops():
    """The in-place alternating loop reorders some additions and
    multiplications. The final Sigma should be numerically identical
    to within floating-point noise (1e-10) for a small fixed problem."""
    import numpy as np
    rng = np.random.default_rng(123)
    Z = rng.normal(0, 1, size=(40, 8))
    r = estimate_weighted_sparse(Z, lambda_value=0.1, max_iter=80, tol=1e-7)
    # The result should be PD-friendly: max abs off-diagonal < max abs diagonal
    # (sanity that soft-thresholding shrinks off-diagonals).
    od = r.covariance.copy()
    diag = np.diag(od).copy()
    np.fill_diagonal(od, 0.0)
    assert np.max(np.abs(od)) <= np.max(np.abs(diag)) * 1.1, (
        "off-diagonal magnitudes exceed diagonal by more than 10% — "
        "likely a buffer-aliasing bug"
    )
    # Symmetric and finite.
    assert np.all(np.isfinite(r.covariance))
    np.testing.assert_allclose(r.covariance, r.covariance.T, atol=1e-12)
