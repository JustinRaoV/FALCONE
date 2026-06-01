import numpy as np
import pytest

from falcon.preprocessing import prepare_log_composition


def test_prepare_log_composition_preserves_rows_and_reports_zeros():
    counts = np.array([[10, 0, 30, 20], [0, 5, 5, 10]], dtype=float)

    prepared = prepare_log_composition(counts, zero_policy="multiplicative")

    np.testing.assert_allclose(prepared.composition.sum(axis=1), 1.0)
    assert np.isfinite(prepared.log_composition).all()
    assert prepared.report.zero_count == 2
    assert prepared.report.n_features_in == 4
    assert prepared.report.n_features_out == 4


@pytest.mark.parametrize(
    "counts, message",
    [
        (np.array([[1.0, -1.0], [1.0, 2.0]]), "non-negative"),
        (np.array([[1.0, np.nan], [1.0, 2.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 2.0]]), "positive row total"),
    ],
)
def test_prepare_log_composition_rejects_invalid_counts(counts, message):
    with pytest.raises(ValueError, match=message):
        prepare_log_composition(counts)


def test_prepare_log_composition_filters_low_prevalence_features():
    counts = np.array(
        [
            [10, 0, 1, 0, 7, 8],
            [10, 2, 0, 0, 7, 8],
            [10, 3, 0, 0, 7, 8],
            [10, 4, 0, 1, 7, 8],
        ],
        dtype=float,
    )

    prepared = prepare_log_composition(counts, min_prevalence=0.5)

    np.testing.assert_array_equal(prepared.report.kept_indices, [0, 1, 4, 5])
    assert prepared.composition.shape == (4, 4)
