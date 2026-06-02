import numpy as np
import pytest

from falcon import infer_cross
from falcon.prior import PriorEdge


def test_infer_cross_fast_returns_sparse_edges_and_diagnostics():
    rng = np.random.default_rng(61)
    counts_x = rng.integers(1, 300, size=(60, 8))
    counts_y = rng.integers(1, 300, size=(60, 10))

    result = infer_cross(
        counts_x, counts_y,
        mode="fast", top_k=2, max_top_k=4,
        stability_threshold=0.0,
        calibration="none",
    )

    assert result.edges.pairs.shape[1] == 2
    assert (result.edges.pairs[:, 0] < 8).all()
    assert (result.edges.pairs[:, 1] < 10).all()
    assert result.diagnostics.initial_top_k == 2
    assert result.diagnostics.final_top_k in {2, 4}


def test_infer_cross_strict_returns_dense_matrix():
    rng = np.random.default_rng(67)
    counts_x = rng.integers(1, 300, size=(80, 6))
    counts_y = rng.integers(1, 300, size=(80, 7))

    result = infer_cross(counts_x, counts_y, mode="strict",
                          max_exclusions=2, calibration="none")
    assert result.initial_matrix.shape == (6, 7)


def test_infer_cross_with_prior_zero_weight_is_identical_to_no_prior():
    rng = np.random.default_rng(71)
    counts_x = rng.integers(1, 300, size=(50, 6))
    counts_y = rng.integers(1, 300, size=(50, 8))

    no_prior = infer_cross(counts_x, counts_y, mode="fast",
                           top_k=2, max_top_k=4, stability_threshold=0.0,
                           calibration="none", seed=3)
    with_zero = infer_cross(
        counts_x, counts_y, mode="fast", top_k=2, max_top_k=4,
        stability_threshold=0.0, calibration="none", seed=3,
        prior=[PriorEdge(0, 1, expected_sign=-1, confidence=1.0)],
        prior_weight=0.0,
    )
    np.testing.assert_array_equal(no_prior.edges.pairs, with_zero.edges.pairs)
    np.testing.assert_allclose(no_prior.edges.scores, with_zero.edges.scores,
                                atol=1e-12)


def test_infer_cross_with_high_prior_weight_pulls_score_toward_target():
    rng = np.random.default_rng(73)
    counts_x = rng.integers(1, 300, size=(50, 5))
    counts_y = rng.integers(1, 300, size=(50, 6))

    result = infer_cross(
        counts_x, counts_y, mode="fast", top_k=1, max_top_k=2,
        stability_threshold=0.0, calibration="none", seed=7,
        prior=[PriorEdge(0, 0, expected_sign=-1, confidence=1.0)],
        prior_weight=1e6, prior_target_magnitude=0.5,
    )
    pairs_list = list(map(tuple, result.edges.pairs.tolist()))
    assert (0, 0) in pairs_list
    idx = pairs_list.index((0, 0))
    assert result.edges.scores[idx] == pytest.approx(-0.5, abs=1e-5)


def test_infer_cross_permutation_calibration_populates_pvalues():
    rng = np.random.default_rng(83)
    counts_x = rng.integers(1, 300, size=(40, 5))
    counts_y = rng.integers(1, 300, size=(40, 7))

    result = infer_cross(
        counts_x, counts_y, mode="fast",
        top_k=2, max_top_k=4, stability_threshold=0.0,
        calibration="permutation", n_permutations=15, seed=4,
    )
    assert result.calibration is not None
    assert result.edges.pvalue_approx is not None
    assert result.edges.qvalue_approx is not None
