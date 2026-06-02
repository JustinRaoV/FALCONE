import numpy as np

from falcon.cross import (
    CrossRefinementResult,
    cross_base_score,
    sparse_refine_cross,
)
from falcon.screen import cross_candidates
from falcon.types import CrossCandidateSet


def test_sparse_refine_cross_with_zero_exclusions_matches_base():
    rng = np.random.default_rng(201)
    counts_x = rng.integers(1, 200, size=(40, 6))
    counts_y = rng.integers(1, 200, size=(40, 7))

    base = cross_base_score(counts_x, counts_y)
    pairs = np.column_stack(np.where(np.abs(base.correlation) > 100.0))
    if pairs.size == 0:
        pairs = np.array([[0, 0]])
    candidates = CrossCandidateSet(
        pairs=pairs,
        scores=base.correlation[pairs[:, 0], pairs[:, 1]],
        top_k=1,
        n_features_x=base.correlation.shape[0],
        n_features_y=base.correlation.shape[1],
    )

    refined = sparse_refine_cross(
        base, candidates,
        exclusion_threshold=10.0,  # impossible threshold — no exclusions
        max_exclusions=0,
    )
    assert isinstance(refined, CrossRefinementResult)
    assert refined.rounds == 0
    assert refined.excluded_pairs.shape == (0, 2)
    np.testing.assert_allclose(
        refined.scores,
        base.correlation[refined.pairs[:, 0], refined.pairs[:, 1]],
        atol=1e-12,
    )


def test_sparse_refine_cross_excludes_strong_candidate_pairs():
    # Construct a Y matrix with one strong link to X.
    rng = np.random.default_rng(211)
    n, p, q = 300, 5, 6
    z = rng.standard_normal(n)
    base_x = rng.standard_normal((n, p))
    base_y = rng.standard_normal((n, q))
    base_x[:, 0] = base_x[:, 0] + 3.0 * z
    base_y[:, 0] = base_y[:, 0] + 3.0 * z
    wx = np.exp(base_x + 5.0)
    wy = np.exp(base_y + 5.0)
    counts_x = (wx / wx.sum(axis=1, keepdims=True) * 50_000).round().astype(np.int64) + 1
    counts_y = (wy / wy.sum(axis=1, keepdims=True) * 50_000).round().astype(np.int64) + 1

    base = cross_base_score(counts_x, counts_y)
    candidates = cross_candidates(base.correlation, top_k=2)
    refined = sparse_refine_cross(
        base, candidates,
        exclusion_threshold=0.3,
        max_exclusions=5,
    )
    assert refined.rounds >= 1
    assert tuple(refined.excluded_pairs[0]) == (0, 0)


def test_sparse_refine_cross_falls_back_when_subset_too_small():
    rng = np.random.default_rng(221)
    # Tiny dimensions so any exclusion empties S or T
    n, p, q = 30, 4, 4
    counts_x = rng.integers(1, 200, size=(n, p))
    counts_y = rng.integers(1, 200, size=(n, q))
    base = cross_base_score(counts_x, counts_y)
    candidates = cross_candidates(base.correlation, top_k=2)
    # Force several exclusions until fallback kicks in
    refined = sparse_refine_cross(
        base, candidates,
        exclusion_threshold=0.0,
        max_exclusions=p,
    )
    assert isinstance(refined, CrossRefinementResult)
    assert refined.rounds <= p
    assert refined.scores.shape == refined.pairs.shape[:1]
