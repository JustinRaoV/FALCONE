"""Tests for subsampling-based stability selection.

The output ``selection_probability`` records, per off-diagonal entry, the
fraction of subsamples in which the estimator returned a non-zero value
for that entry. The diagonal is reported as 1.0 (the diagonal is never
penalized by any estimator). The procedure must be exactly reproducible
under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from falcon.estimators.adaptive_threshold import estimate_adaptive_threshold
from falcon.stability import StabilityResult, select_by_stability


def _clr(log_composition: np.ndarray) -> np.ndarray:
    return log_composition - log_composition.mean(axis=1, keepdims=True)


def _make_dataset(n=200, p=12, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.eye(p)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        cov[i, j] = cov[j, i] = 0.7
    log_basis = rng.multivariate_normal(np.zeros(p), cov, size=n)
    composition = np.exp(log_basis)
    composition = composition / composition.sum(axis=1, keepdims=True)
    return _clr(np.log(composition))


def _hard_threshold_estimator(Z):
    return estimate_adaptive_threshold(Z, threshold_constant=2.5, mode="hard").covariance


def test_output_shape_and_bounds():
    Z = _make_dataset()
    result = select_by_stability(
        Z, _hard_threshold_estimator, n_resamples=20, seed=0
    )
    assert isinstance(result, StabilityResult)
    p = Z.shape[1]
    assert result.selection_probability.shape == (p, p)
    np.testing.assert_array_equal(np.diag(result.selection_probability), 1.0)
    assert (result.selection_probability >= 0).all()
    assert (result.selection_probability <= 1).all()


def test_output_is_symmetric():
    Z = _make_dataset(seed=1)
    result = select_by_stability(
        Z, _hard_threshold_estimator, n_resamples=15, seed=42
    )
    np.testing.assert_array_equal(
        result.selection_probability, result.selection_probability.T
    )


def test_reproducible_under_fixed_seed():
    Z = _make_dataset(seed=2)
    a = select_by_stability(Z, _hard_threshold_estimator, n_resamples=30, seed=7)
    b = select_by_stability(Z, _hard_threshold_estimator, n_resamples=30, seed=7)
    np.testing.assert_array_equal(a.selection_probability, b.selection_probability)


def test_different_seed_produces_different_subsamples():
    Z = _make_dataset(seed=3)
    a = select_by_stability(Z, _hard_threshold_estimator, n_resamples=30, seed=1)
    b = select_by_stability(Z, _hard_threshold_estimator, n_resamples=30, seed=2)
    # Probabilities should differ on at least some entry.
    assert not np.allclose(a.selection_probability, b.selection_probability)


def test_strong_edges_have_higher_stability_than_random_pairs():
    Z = _make_dataset(seed=4)
    result = select_by_stability(
        Z, _hard_threshold_estimator, n_resamples=50, seed=0
    )
    p = Z.shape[1]

    planted = [(0, 1), (2, 3), (4, 5)]
    planted_probs = [result.selection_probability[i, j] for i, j in planted]

    # Pick a few definitely-not-planted pairs.
    other_pairs = [(0, 5), (1, 4), (2, 7), (8, 11)]
    other_probs = [result.selection_probability[i, j] for i, j in other_pairs]

    assert min(planted_probs) > max(other_probs)


def test_records_n_resamples_and_subsample_size():
    Z = _make_dataset(seed=5)
    result = select_by_stability(
        Z,
        _hard_threshold_estimator,
        n_resamples=10,
        subsample_fraction=0.6,
        seed=0,
    )
    assert result.n_resamples_used == 10
    assert result.subsample_fraction == 0.6
    assert result.subsample_size == int(round(Z.shape[0] * 0.6))


def test_zero_resamples_rejected():
    Z = _make_dataset(seed=6)
    with pytest.raises(ValueError, match="n_resamples"):
        select_by_stability(Z, _hard_threshold_estimator, n_resamples=0, seed=0)


def test_invalid_subsample_fraction_rejected():
    Z = _make_dataset(seed=7)
    with pytest.raises(ValueError, match="subsample_fraction"):
        select_by_stability(
            Z,
            _hard_threshold_estimator,
            n_resamples=5,
            subsample_fraction=1.5,
            seed=0,
        )


def test_estimator_returning_non_square_rejected():
    Z = _make_dataset(seed=8)

    def bad(Z):
        return np.zeros((Z.shape[1], Z.shape[1] + 1))

    with pytest.raises(ValueError, match="square"):
        select_by_stability(Z, bad, n_resamples=2, seed=0)
