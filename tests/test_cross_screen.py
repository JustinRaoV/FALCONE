import numpy as np

from falcon.screen import cross_candidates
from falcon.types import CrossCandidateSet


def test_cross_candidates_includes_top_k_in_both_directions():
    # Score: rows = X features, cols = Y features
    score = np.array(
        [
            [0.9, 0.1, 0.2, 0.1],
            [0.1, 0.8, 0.1, 0.2],
            [0.2, 0.1, 0.7, 0.6],
        ]
    )
    candidates = cross_candidates(score, top_k=1)
    assert isinstance(candidates, CrossCandidateSet)
    pairs = set(map(tuple, candidates.pairs.tolist()))
    # Top per X: (0,0), (1,1), (2,2)
    # Top per Y: (0,0), (1,1), (2,2), (2,3)  -- col 3 best is (2,3)=0.6
    assert pairs == {(0, 0), (1, 1), (2, 2), (2, 3)}


def test_cross_candidates_grow_monotonically_with_top_k():
    rng = np.random.default_rng(7)
    score = rng.uniform(-1, 1, size=(6, 8))
    a = cross_candidates(score, top_k=1)
    b = cross_candidates(score, top_k=3)
    a_set = set(map(tuple, a.pairs.tolist()))
    b_set = set(map(tuple, b.pairs.tolist()))
    assert a_set <= b_set


def test_cross_candidates_threshold_adds_extra_pairs():
    score = np.array(
        [
            [0.5, 0.4, 0.05],
            [0.05, 0.05, 0.3],
        ]
    )
    pairs_only_topk = cross_candidates(score, top_k=1)
    pairs_with_thresh = cross_candidates(score, top_k=1, min_abs_score=0.39)
    assert pairs_with_thresh.pairs.shape[0] >= pairs_only_topk.pairs.shape[0]
    assert (0, 1) in set(map(tuple, pairs_with_thresh.pairs.tolist()))


def test_cross_candidates_density_reports_fraction_of_bipartite_pairs():
    score = np.zeros((4, 5))
    score[0, 0] = score[1, 1] = score[2, 2] = score[3, 3] = 1.0
    candidates = cross_candidates(score, top_k=1)
    # 4 pairs from X-side, plus best of (col 4 across X rows) = (0,4) tied at 0
    # top_k=1 picks one per Y; ties broken by argpartition
    expected_density = candidates.pairs.shape[0] / (4 * 5)
    assert candidates.density == expected_density


def test_cross_candidates_rejects_invalid_top_k():
    import pytest

    score = np.zeros((3, 4))
    with pytest.raises(ValueError, match="top_k"):
        cross_candidates(score, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        cross_candidates(score, top_k=5)
