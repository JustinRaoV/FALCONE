import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from comparison_methods import sparcc_py

from falcon.single import single_base_score, variation_matrix


def _pair_loop_variation(log_composition):
    p = log_composition.shape[1]
    result = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            result[i, j] = np.var(
                log_composition[:, i] - log_composition[:, j], ddof=1
            )
    return result


def test_variation_matrix_matches_pair_loop():
    log_composition = np.log(
        np.array(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.10, 0.40, 0.30],
                [0.15, 0.30, 0.25, 0.30],
                [0.30, 0.20, 0.10, 0.40],
            ]
        )
    )

    np.testing.assert_allclose(
        variation_matrix(log_composition),
        _pair_loop_variation(log_composition),
        atol=1e-12,
    )


def test_single_base_score_is_symmetric_bounded_and_has_unit_diagonal():
    counts = np.array(
        [
            [30, 10, 20, 40],
            [25, 15, 30, 30],
            [15, 25, 35, 25],
            [20, 20, 10, 50],
            [35, 10, 25, 30],
        ],
        dtype=float,
    )

    result = single_base_score(counts)

    np.testing.assert_allclose(result.correlation, result.correlation.T)
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert (np.abs(result.correlation) <= 1.0).all()


def test_single_base_score_matches_existing_sparcc_baseline():
    rng = np.random.default_rng(11)
    counts = rng.integers(1, 200, size=(40, 12))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = sparcc_py(counts)
    actual = single_base_score(counts).correlation

    np.testing.assert_allclose(actual, expected, atol=1e-10)
