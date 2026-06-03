"""Integration tests for the public ``infer_network`` API.

These tests exercise the full pipeline:

    counts -> preprocess -> CLR -> estimator -> (optional) stability -> edges

The schema is fixed in design section 5: a ``NetworkResult`` with an
``EdgeTable`` carrying ``selection_probability`` (when selection is on)
and an ``EstimatorDiagnostics`` carrying the estimator label, lambda,
convergence, iterations, minimum eigenvalue, calibration method,
uncertainty interpretation, and the preprocessing report.
"""

from __future__ import annotations

import numpy as np
import pytest

from falcon import infer_network
from falcon.results import EdgeTable, EstimatorDiagnostics, NetworkResult


def _planted_counts(n=200, p=20, seed=0, edge_strength=0.7):
    rng = np.random.default_rng(seed)
    cov = np.eye(p)
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        cov[i, j] = cov[j, i] = edge_strength
    log_basis = rng.multivariate_normal(np.zeros(p), cov, size=n)
    counts = np.exp(log_basis) * 1000
    return counts


@pytest.mark.parametrize(
    "estimator", ["weighted_sparse", "adaptive_threshold", "pd_sparse"]
)
def test_returns_network_result_with_required_diagnostics(estimator):
    counts = _planted_counts()
    result = infer_network(
        counts,
        estimator=estimator,
        zero_policy="multiplicative",
        selection="none",
        seed=0,
    )
    assert isinstance(result, NetworkResult)
    assert isinstance(result.edges, EdgeTable)
    assert isinstance(result.diagnostics, EstimatorDiagnostics)
    assert result.diagnostics.estimator == estimator
    assert result.diagnostics.preprocess_report is not None
    assert result.correlation is not None
    assert result.correlation.shape == (counts.shape[1], counts.shape[1])


def test_edge_table_uses_canonical_i_lt_j_ordering():
    counts = _planted_counts(seed=1)
    result = infer_network(
        counts, estimator="adaptive_threshold", selection="none", seed=0
    )
    pairs = result.edges.pairs
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_edges_correspond_to_nonzero_correlation_entries():
    counts = _planted_counts(seed=2)
    result = infer_network(
        counts, estimator="adaptive_threshold", selection="none", seed=0
    )
    corr = result.correlation
    # Every reported edge has a non-zero correlation entry.
    for (i, j), s in zip(result.edges.pairs, result.edges.scores):
        assert corr[i, j] != 0
        np.testing.assert_allclose(corr[i, j], s)


def test_stability_selection_populates_selection_probability():
    counts = _planted_counts(seed=3)
    result = infer_network(
        counts,
        estimator="adaptive_threshold",
        selection="stability",
        n_resamples=20,
        seed=0,
    )
    assert result.edges.selection_probability is not None
    assert result.edges.selection_probability.shape == result.edges.scores.shape
    assert (result.edges.selection_probability >= 0).all()
    assert (result.edges.selection_probability <= 1).all()
    assert (
        result.diagnostics.uncertainty_interpretation == "selection_probability_only"
    )


def test_no_selection_leaves_uncertainty_unreported():
    counts = _planted_counts(seed=4)
    result = infer_network(counts, estimator="adaptive_threshold", selection="none", seed=0)
    assert result.edges.selection_probability is None
    assert result.diagnostics.uncertainty_interpretation == "no_uncertainty_reported"


def test_pd_sparse_estimator_floors_eigenvalues():
    counts = _planted_counts(seed=5)
    result = infer_network(counts, estimator="pd_sparse", selection="none", seed=0)
    assert result.diagnostics.min_eigenvalue >= 0
    # The pd_sparse estimator records that PD correction was applied.
    assert "pd" in result.diagnostics.notes or result.diagnostics.min_eigenvalue >= 1e-6


def test_seed_makes_stability_run_reproducible():
    counts = _planted_counts(seed=6)
    a = infer_network(
        counts,
        estimator="adaptive_threshold",
        selection="stability",
        n_resamples=10,
        seed=42,
    )
    b = infer_network(
        counts,
        estimator="adaptive_threshold",
        selection="stability",
        n_resamples=10,
        seed=42,
    )
    np.testing.assert_array_equal(a.edges.pairs, b.edges.pairs)
    np.testing.assert_allclose(a.edges.scores, b.edges.scores)
    np.testing.assert_allclose(
        a.edges.selection_probability, b.edges.selection_probability
    )


def test_unknown_estimator_rejected():
    counts = _planted_counts(seed=7)
    with pytest.raises(ValueError, match="estimator"):
        infer_network(counts, estimator="bogus", seed=0)


def test_unknown_selection_rejected():
    counts = _planted_counts(seed=8)
    with pytest.raises(ValueError, match="selection"):
        infer_network(counts, estimator="adaptive_threshold", selection="bogus", seed=0)


def test_complete_case_zero_policy_supported():
    counts = _planted_counts(seed=9)
    # Force a few zeros into the data.
    counts[0, 0] = 0
    counts[1, 0] = 0
    counts[2, 1] = 0
    result = infer_network(
        counts,
        estimator="adaptive_threshold",
        zero_policy="complete_case",
        selection="none",
        seed=0,
    )
    assert result.diagnostics.preprocess_report.zero_policy == "complete_case"


def test_legacy_exports_are_removed():
    import falcon

    assert not hasattr(falcon, "infer_single")
    assert not hasattr(falcon, "infer_cross")
    assert not hasattr(falcon, "PriorEdge")
