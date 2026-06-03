import numpy as np
import pytest

from falcon.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    pfer_bound,
    reliability_diagram,
)


def test_isotonic_is_monotone_and_in_unit_interval():
    rng = np.random.default_rng(0)
    sel_prob = rng.uniform(0.0, 1.0, size=500)
    truth = (sel_prob + rng.normal(0, 0.2, size=500)) > 0.5
    cal = IsotonicCalibrator().fit(sel_prob, truth, scenario="sparse_random")
    grid = np.linspace(0.0, 1.0, 51)
    g = cal.predict(grid, scenario="sparse_random")
    assert np.all(np.diff(g) >= -1e-12), "isotonic must be monotone non-decreasing"
    assert np.all((g >= 0.0) & (g <= 1.0))


def test_isotonic_pooled_vs_per_scenario():
    rng = np.random.default_rng(1)
    sel = rng.uniform(0, 1, size=1000)
    truth = sel > 0.5
    cal = IsotonicCalibrator(mode="pooled").fit(sel, truth, scenario="any")
    out_a = cal.predict(np.array([0.1, 0.9]), scenario="ignored")
    cal2 = IsotonicCalibrator(mode="per_scenario").fit(sel, truth, scenario="sparse_random")
    out_b = cal2.predict(np.array([0.1, 0.9]), scenario="sparse_random")
    assert out_a.shape == out_b.shape == (2,)


def test_isotonic_per_scenario_raises_on_missing_scenario():
    cal = IsotonicCalibrator(mode="per_scenario")
    cal.fit(np.array([0.1, 0.9]), np.array([False, True]), scenario="sparse_random")
    with pytest.raises(KeyError, match="hub"):
        cal.predict(np.array([0.5]), scenario="hub")


def test_reliability_diagram_returns_arrays_of_n_bins():
    sel = np.linspace(0.0, 1.0, 200)
    truth = sel > 0.5
    mids, obs, counts = reliability_diagram(sel, truth, n_bins=10)
    assert mids.shape == obs.shape == counts.shape == (10,)
    assert counts.sum() == 200


def test_ece_is_zero_when_predictions_match_truth_frequency():
    sel = np.array([0.5] * 1000)
    truth = np.random.default_rng(0).random(1000) < 0.5
    mids, obs, counts = reliability_diagram(sel, truth, n_bins=10)
    e = expected_calibration_error(mids, obs, counts)
    assert e < 0.05


def test_pfer_bound_basic_formula():
    p_off = 1225
    q_avg = 10
    pi_thr = 0.8
    bound = pfer_bound(q_avg=q_avg, pi_thr=pi_thr, p_off=p_off)
    expected = (q_avg ** 2) / ((2 * pi_thr - 1) * p_off)
    assert bound == pytest.approx(expected)


def test_pfer_bound_undefined_below_half_threshold():
    with pytest.raises(ValueError, match="pi_thr"):
        pfer_bound(q_avg=10, pi_thr=0.4, p_off=1000)


def test_brier_score_matches_sklearn_formula():
    pred = np.array([0.2, 0.7, 0.9])
    truth = np.array([0, 1, 1])
    # MSE: ((0.2-0)^2 + (0.7-1)^2 + (0.9-1)^2) / 3 = (0.04 + 0.09 + 0.01) / 3
    expected = (0.04 + 0.09 + 0.01) / 3.0
    assert brier_score(pred, truth) == pytest.approx(expected)
