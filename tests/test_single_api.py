import numpy as np
import pytest

from falcon import infer_single


def test_infer_single_fast_returns_unique_sparse_edges_and_diagnostics():
    rng = np.random.default_rng(29)
    counts = rng.integers(1, 300, size=(80, 24))

    result = infer_single(
        counts,
        mode="fast",
        top_k=2,
        max_top_k=4,
        stability_threshold=0.0,
        calibration="none",
    )

    assert result.edges.pairs.shape[1] == 2
    assert (result.edges.pairs[:, 0] < result.edges.pairs[:, 1]).all()
    assert result.diagnostics.initial_top_k == 2
    assert result.diagnostics.final_top_k in {2, 4}
    assert result.initial_matrix is None


def test_infer_single_strict_returns_dense_matrix():
    rng = np.random.default_rng(31)
    counts = rng.integers(1, 300, size=(80, 12))

    result = infer_single(counts, mode="strict", max_exclusions=3,
                          calibration="none")

    assert result.initial_matrix.shape == (12, 12)
    np.testing.assert_allclose(result.initial_matrix, result.initial_matrix.T)


def test_infer_single_rejects_max_top_k_below_initial_budget():
    rng = np.random.default_rng(37)
    counts = rng.integers(1, 300, size=(80, 12))

    with pytest.raises(ValueError, match="max_top_k"):
        infer_single(counts, mode="fast", top_k=4, max_top_k=2)


def test_infer_single_permutation_calibration_populates_pvalues():
    rng = np.random.default_rng(43)
    counts = rng.integers(1, 300, size=(60, 12))

    result = infer_single(
        counts, mode="fast", top_k=2, max_top_k=4,
        stability_threshold=0.0,
        calibration="permutation", n_permutations=20, seed=11,
    )

    assert result.calibration is not None
    assert result.edges.pvalue_approx is not None
    assert result.edges.qvalue_approx is not None
    assert result.edges.pvalue_approx.shape == result.edges.scores.shape
    assert result.diagnostics.calibration_method == "permutation_base_only"


def test_infer_single_no_calibration_leaves_pvalues_unset():
    rng = np.random.default_rng(47)
    counts = rng.integers(1, 300, size=(40, 10))

    result = infer_single(counts, mode="fast", top_k=2, max_top_k=4,
                          stability_threshold=0.0, calibration="none")

    assert result.calibration is None
    assert result.edges.pvalue_approx is None


def test_infer_single_is_reproducible_under_same_seed():
    rng = np.random.default_rng(53)
    counts = rng.integers(1, 300, size=(40, 10))

    r1 = infer_single(counts, mode="fast", top_k=2, max_top_k=4,
                      stability_threshold=0.0,
                      calibration="permutation", n_permutations=10, seed=99)
    r2 = infer_single(counts, mode="fast", top_k=2, max_top_k=4,
                      stability_threshold=0.0,
                      calibration="permutation", n_permutations=10, seed=99)

    np.testing.assert_array_equal(r1.edges.pvalue_approx, r2.edges.pvalue_approx)
