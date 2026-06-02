import numpy as np
import pytest

from falcon.calibration import (
    benjamini_hochberg,
    calibrate_single,
    calibrate_cross,
)


def test_benjamini_hochberg_known_example():
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    q = benjamini_hochberg(pvals)
    assert q.shape == pvals.shape
    assert q[0] < q[-1]
    assert (q >= 0).all() and (q <= 1).all()
    assert q[0] == pytest.approx(0.008, abs=1e-12)


def test_benjamini_hochberg_monotone_in_sorted_order():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, size=200)
    q = benjamini_hochberg(pvals)
    order = np.argsort(pvals)
    sorted_q = q[order]
    assert (np.diff(sorted_q) >= -1e-12).all()


def test_calibrate_single_is_reproducible_under_fixed_seed():
    rng = np.random.default_rng(11)
    counts = rng.integers(1, 200, size=(40, 12))
    candidate_pairs = np.array([[0, 1], [2, 3], [4, 5]])
    refined_scores = np.array([0.8, 0.2, -0.05])

    result_a = calibrate_single(
        counts, candidate_pairs, refined_scores,
        n_permutations=25, seed=7,
    )
    result_b = calibrate_single(
        counts, candidate_pairs, refined_scores,
        n_permutations=25, seed=7,
    )

    np.testing.assert_array_equal(result_a.pvalue_approx, result_b.pvalue_approx)
    np.testing.assert_array_equal(
        result_a.null_max_distribution, result_b.null_max_distribution
    )
    assert result_a.method == "permutation_base_only"


def test_calibrate_single_pvalues_lie_in_unit_interval():
    rng = np.random.default_rng(13)
    counts = rng.integers(1, 200, size=(30, 10))
    pairs = np.array([[0, 1], [1, 2], [3, 4]])
    scores = np.array([0.99, 0.0, -0.5])
    result = calibrate_single(counts, pairs, scores, n_permutations=20, seed=3)
    assert (result.pvalue_approx > 0).all()
    assert (result.pvalue_approx <= 1).all()
    assert (result.qvalue_approx >= 0).all()
    assert (result.qvalue_approx <= 1).all()


def test_calibrate_single_strong_edge_gets_smaller_pvalue_than_weak_edge():
    # Construct counts with a planted high-correlation pair.
    rng = np.random.default_rng(21)
    n, p = 200, 8
    z = rng.standard_normal(n)
    w = np.full((n, p), 5.0)
    w[:, 0] += 3.0 * z
    w[:, 1] += 3.0 * z
    counts = np.maximum(np.exp(w), 0).round().astype(np.int64) + 1

    pairs = np.array([[0, 1], [2, 3]])
    # Approximate refined scores: strong for (0,1), weak for (2,3)
    refined = np.array([0.95, 0.05])
    result = calibrate_single(counts, pairs, refined, n_permutations=50, seed=42)
    assert result.pvalue_approx[0] < result.pvalue_approx[1]


def test_calibrate_cross_is_reproducible_and_returns_unit_interval_pvalues():
    rng = np.random.default_rng(31)
    counts_x = rng.integers(1, 200, size=(40, 8))
    counts_y = rng.integers(1, 200, size=(40, 10))
    pairs = np.array([[0, 1], [3, 7], [5, 0]])
    scores = np.array([0.7, -0.1, 0.3])

    res_a = calibrate_cross(counts_x, counts_y, pairs, scores,
                            n_permutations=15, seed=5)
    res_b = calibrate_cross(counts_x, counts_y, pairs, scores,
                            n_permutations=15, seed=5)

    np.testing.assert_array_equal(res_a.pvalue_approx, res_b.pvalue_approx)
    assert (res_a.pvalue_approx > 0).all()
    assert (res_a.pvalue_approx <= 1).all()
