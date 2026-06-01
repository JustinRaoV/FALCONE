import numpy as np

from falcon.single import strict_refine_single


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
