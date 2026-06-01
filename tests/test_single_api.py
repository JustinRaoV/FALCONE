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
    )

    assert result.edges.pairs.shape[1] == 2
    assert (result.edges.pairs[:, 0] < result.edges.pairs[:, 1]).all()
    assert result.diagnostics.initial_top_k == 2
    assert result.diagnostics.final_top_k in {2, 4}
    assert result.initial_matrix is None


def test_infer_single_strict_returns_dense_matrix():
    rng = np.random.default_rng(31)
    counts = rng.integers(1, 300, size=(80, 12))

    result = infer_single(counts, mode="strict", max_exclusions=3)

    assert result.initial_matrix.shape == (12, 12)
    np.testing.assert_allclose(result.initial_matrix, result.initial_matrix.T)


def test_infer_single_rejects_max_top_k_below_initial_budget():
    rng = np.random.default_rng(37)
    counts = rng.integers(1, 300, size=(80, 12))

    with pytest.raises(ValueError, match="max_top_k"):
        infer_single(counts, mode="fast", top_k=4, max_top_k=2)
