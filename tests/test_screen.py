import numpy as np

from falcon.screen import edge_overlap, single_candidates


def test_single_candidates_returns_symmetric_union_without_self_edges():
    score = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.8, 0.1],
            [0.1, 0.8, 1.0, 0.7],
            [0.2, 0.1, 0.7, 1.0],
        ]
    )

    candidates = single_candidates(score, top_k=1)

    assert candidates.pairs.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert (candidates.pairs[:, 0] < candidates.pairs[:, 1]).all()


def test_single_candidates_grow_monotonically_with_budget():
    rng = np.random.default_rng(7)
    score = rng.uniform(-1.0, 1.0, size=(12, 12))
    score = (score + score.T) / 2.0
    np.fill_diagonal(score, 1.0)

    small = single_candidates(score, top_k=1)
    large = single_candidates(score, top_k=3)

    assert set(map(tuple, small.pairs)) <= set(map(tuple, large.pairs))


def test_edge_overlap_uses_jaccard_similarity():
    left = np.array([[0, 1], [1, 2]])
    right = np.array([[1, 2], [2, 3]])

    assert edge_overlap(left, right) == 1 / 3
