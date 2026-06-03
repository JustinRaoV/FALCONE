"""Tests for the simulation harness skeleton."""

from __future__ import annotations

import numpy as np
import pytest

from falcon.sim import (
    auroc_score,
    available_scenarios,
    average_precision_score,
    fdr_at_target,
    generate_scenario,
    holdout_grid,
    precision_at_k,
    recall_at_k,
    training_grid,
)
from falcon.sim.scenarios import VALID_SCENARIOS


def test_six_scenarios_advertised():
    scenarios = available_scenarios()
    assert len(scenarios) == 6
    assert set(scenarios) == set(VALID_SCENARIOS)


@pytest.mark.parametrize("name", list(VALID_SCENARIOS))
def test_each_scenario_returns_valid_counts_and_support(name):
    scenario = generate_scenario(name, n=80, p=20, seed=0)
    assert scenario.counts.shape == (80, 20)
    assert scenario.counts.dtype.kind in {"i", "u"}
    assert (scenario.counts >= 0).all()
    assert scenario.counts.sum(axis=1).min() > 0
    # Support is symmetric, square, zero diagonal.
    assert scenario.support.shape == (20, 20)
    np.testing.assert_array_equal(scenario.support, scenario.support.T)
    assert not np.diag(scenario.support).any()
    # Metadata records the scenario name and seed.
    assert scenario.metadata["scenario"] == name
    assert scenario.metadata["seed"] == 0
    assert "zero_fraction" in scenario.metadata


def test_scenario_reproducible_under_seed():
    a = generate_scenario("sparse_random", n=50, p=12, seed=7)
    b = generate_scenario("sparse_random", n=50, p=12, seed=7)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.support, b.support)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="scenario"):
        generate_scenario("not_a_scenario", n=50, p=12, seed=0)


def test_training_and_holdout_grids_are_disjoint_in_seeds():
    train = training_grid()
    hold = holdout_grid()
    train_ids = {(c.scenario, c.n, c.p, c.seed) for c in train}
    hold_ids = {(c.scenario, c.n, c.p, c.seed) for c in hold}
    # Holdout uses seeds 10+ which we never use in training.
    assert train_ids.isdisjoint(hold_ids)
    # Splits are labeled correctly.
    assert {c.split for c in train} == {"training"}
    assert {c.split for c in hold} == {"holdout"}


def test_grid_metadata_has_all_required_fields():
    cell = training_grid()[0]
    md = cell.metadata()
    for key in ("scenario", "split", "n", "p", "seed", "density"):
        assert key in md


def test_auroc_perfect_separation():
    scores = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.1],
            [0.0, 0.1, 0.0],
        ]
    )
    truth = np.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    assert auroc_score(scores, truth) == 1.0


def test_auroc_random_scoring_returns_half():
    rng = np.random.default_rng(0)
    p = 20
    scores = rng.standard_normal((p, p))
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0)
    truth = rng.random((p, p)) > 0.7
    truth = truth | truth.T
    np.fill_diagonal(truth, False)
    auc = auroc_score(scores, truth)
    assert 0.3 < auc < 0.7  # by chance ~ 0.5


def test_average_precision_perfect_ordering():
    scores = np.array(
        [
            [0.0, 0.9, 0.05],
            [0.9, 0.0, 0.04],
            [0.05, 0.04, 0.0],
        ]
    )
    truth = np.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    assert average_precision_score(scores, truth) == pytest.approx(1.0)


def test_recall_and_precision_at_k():
    scores = np.array(
        [
            [0.0, 0.9, 0.6, 0.05],
            [0.9, 0.0, 0.04, 0.7],
            [0.6, 0.04, 0.0, 0.02],
            [0.05, 0.7, 0.02, 0.0],
        ]
    )
    truth = np.array(
        [
            [False, True, False, False],
            [True, False, False, True],
            [False, False, False, False],
            [False, True, False, False],
        ]
    )
    # Top-2 by absolute score: (0,1) score 0.9 (TP), (1,3) score 0.7 (TP)
    assert recall_at_k(scores, truth, 2) == pytest.approx(1.0)
    assert precision_at_k(scores, truth, 2) == pytest.approx(1.0)


def test_fdr_at_target_counts_false_among_selected():
    truth = np.array(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    # Select two pairs: (0,1) is TP, (0,2) is FP -> FDR = 0.5
    selected = np.array(
        [
            [False, True, True],
            [True, False, False],
            [True, False, False],
        ]
    )
    assert fdr_at_target(np.zeros_like(truth, dtype=float), truth, selected) == 0.5
