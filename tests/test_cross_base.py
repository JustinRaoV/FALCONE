import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from falcon.cross import (
    CrossBaseResult,
    cross_base_score,
)


def test_cross_base_score_matches_existing_sparxcc_baseline():
    from comparison_methods import sparxcc_base

    rng = np.random.default_rng(101)
    counts_x = rng.integers(1, 300, size=(40, 8))
    counts_y = rng.integers(1, 300, size=(40, 10))

    actual = cross_base_score(counts_x, counts_y)
    expected = sparxcc_base(counts_x, counts_y)

    assert isinstance(actual, CrossBaseResult)
    np.testing.assert_allclose(actual.correlation, expected, atol=1e-10)
    assert actual.correlation.shape == (8, 10)
    assert (np.abs(actual.correlation) <= 1.0).all()


def test_cross_base_score_records_basis_variance_per_domain():
    rng = np.random.default_rng(103)
    counts_x = rng.integers(1, 300, size=(60, 6))
    counts_y = rng.integers(1, 300, size=(60, 7))

    result = cross_base_score(counts_x, counts_y)

    assert result.alpha.shape == (6,)
    assert result.beta.shape == (7,)
    assert (result.alpha > 0).all()
    assert (result.beta > 0).all()


def test_cross_base_score_rejects_mismatched_sample_rows():
    counts_x = np.ones((10, 5), dtype=np.int64)
    counts_y = np.ones((11, 5), dtype=np.int64)
    with pytest.raises(ValueError, match="sample"):
        cross_base_score(counts_x, counts_y)
