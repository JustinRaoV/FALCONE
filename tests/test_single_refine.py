import numpy as np

from falcon.single import (
    solve_basis_variance_dense,
    solve_basis_variance_sparse,
    sparse_refine_single,
    strict_refine_single,
)
from falcon.types import CandidateSet


def test_strict_refine_single_excludes_at_most_one_new_pair_per_round():
    variation = np.array(
        [
            [0.0, 0.1, 1.7, 1.8],
            [0.1, 0.0, 1.6, 1.7],
            [1.7, 1.6, 0.0, 0.2],
            [1.8, 1.7, 0.2, 0.0],
        ]
    )

    result = strict_refine_single(
        variation,
        exclusion_threshold=0.1,
        max_exclusions=1,
    )

    assert result.excluded_pairs.shape == (1, 2)
    assert result.rounds == 1


def test_strict_refine_single_stops_when_threshold_is_not_crossed():
    variation = np.full((5, 5), 2.0)
    np.fill_diagonal(variation, 0.0)

    result = strict_refine_single(
        variation,
        exclusion_threshold=0.2,
        max_exclusions=10,
    )

    assert result.rounds == 0
    assert result.excluded_pairs.shape == (0, 2)


def test_sparse_basis_solve_matches_dense_excluded_solve():
    rng = np.random.default_rng(17)
    raw = rng.uniform(0.1, 2.0, size=(12, 12))
    variation = (raw + raw.T) / 2.0
    np.fill_diagonal(variation, 0.0)
    excluded = np.array([[0, 1], [2, 3], [3, 4], [8, 11]])

    expected = solve_basis_variance_dense(variation, excluded=excluded)
    actual = solve_basis_variance_sparse(variation, excluded=excluded)

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_sparse_refinement_matches_strict_when_candidates_cover_all_pairs():
    rng = np.random.default_rng(23)
    raw = rng.uniform(0.1, 2.0, size=(10, 10))
    variation = (raw + raw.T) / 2.0
    np.fill_diagonal(variation, 0.0)
    rows, cols = np.triu_indices(10, k=1)
    candidates = CandidateSet(
        pairs=np.column_stack([rows, cols]),
        scores=np.zeros(rows.size),
        top_k=9,
        n_features=10,
    )

    strict = strict_refine_single(
        variation,
        exclusion_threshold=0.1,
        max_exclusions=6,
    )
    sparse_result = sparse_refine_single(
        variation,
        candidates,
        exclusion_threshold=0.1,
        max_exclusions=6,
    )

    np.testing.assert_array_equal(
        sparse_result.excluded_pairs,
        strict.excluded_pairs,
    )
    np.testing.assert_allclose(
        sparse_result.basis_variance,
        strict.basis_variance,
        atol=1e-8,
    )
