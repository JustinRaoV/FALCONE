"""Schema tests for the new public NetworkResult / EdgeTable / EstimatorDiagnostics."""

import numpy as np
import pytest

from falcon.results import EdgeTable, EstimatorDiagnostics, NetworkResult


def test_edge_table_supports_selection_probability_and_pvalues():
    pairs = np.array([[0, 1], [0, 2]], dtype=int)
    scores = np.array([0.4, -0.3])
    sel_prob = np.array([0.95, 0.10])
    p = np.array([0.001, 0.4])
    q = np.array([0.005, 0.5])

    edges = EdgeTable(
        pairs=pairs,
        scores=scores,
        selection_probability=sel_prob,
        pvalue_approx=p,
        qvalue_approx=q,
    )

    assert edges.pairs.shape == (2, 2)
    assert edges.scores.shape == (2,)
    np.testing.assert_array_equal(edges.selection_probability, sel_prob)
    np.testing.assert_array_equal(edges.pvalue_approx, p)
    np.testing.assert_array_equal(edges.qvalue_approx, q)


def test_edge_table_optional_uncertainty_defaults_to_none():
    edges = EdgeTable(
        pairs=np.zeros((0, 2), dtype=int),
        scores=np.zeros(0),
    )
    assert edges.selection_probability is None
    assert edges.pvalue_approx is None
    assert edges.qvalue_approx is None


def test_edge_table_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="length"):
        EdgeTable(
            pairs=np.zeros((3, 2), dtype=int),
            scores=np.zeros(2),
        )


def test_edge_table_rejects_pair_shape_other_than_two_columns():
    with pytest.raises(ValueError, match="two columns"):
        EdgeTable(
            pairs=np.zeros((3, 3), dtype=int),
            scores=np.zeros(3),
        )


def test_estimator_diagnostics_records_required_fields():
    diag = EstimatorDiagnostics(
        estimator="weighted_sparse",
        lambda_value=0.12,
        converged=True,
        iterations=5,
        min_eigenvalue=0.001,
        calibration_method="none",
        uncertainty_interpretation="selection_probability_only",
    )
    assert diag.estimator == "weighted_sparse"
    assert diag.converged is True
    assert diag.iterations == 5
    assert diag.uncertainty_interpretation == "selection_probability_only"
    # Optional preprocess_report defaults to None
    assert diag.preprocess_report is None


def test_estimator_diagnostics_rejects_unknown_estimator_label():
    with pytest.raises(ValueError, match="estimator"):
        EstimatorDiagnostics(
            estimator="not_a_real_estimator",
            lambda_value=0.0,
            converged=True,
            iterations=0,
            min_eigenvalue=1.0,
            calibration_method="none",
            uncertainty_interpretation="selection_probability_only",
        )


def test_network_result_correlation_field_optional():
    edges = EdgeTable(pairs=np.zeros((0, 2), dtype=int), scores=np.zeros(0))
    diag = EstimatorDiagnostics(
        estimator="adaptive_threshold",
        lambda_value=0.0,
        converged=True,
        iterations=0,
        min_eigenvalue=1.0,
        calibration_method="none",
        uncertainty_interpretation="selection_probability_only",
    )
    result = NetworkResult(edges=edges, diagnostics=diag, correlation=None)
    assert result.correlation is None
    assert result.edges is edges
    assert result.diagnostics is diag


def test_network_result_correlation_must_be_square_when_present():
    edges = EdgeTable(pairs=np.zeros((0, 2), dtype=int), scores=np.zeros(0))
    diag = EstimatorDiagnostics(
        estimator="adaptive_threshold",
        lambda_value=0.0,
        converged=True,
        iterations=0,
        min_eigenvalue=1.0,
        calibration_method="none",
        uncertainty_interpretation="selection_probability_only",
    )
    with pytest.raises(ValueError, match="square"):
        NetworkResult(
            edges=edges,
            diagnostics=diag,
            correlation=np.zeros((3, 4)),
        )
