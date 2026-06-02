import numpy as np
import pytest

from falcon.prior import (
    PriorEdge,
    apply_prior_shrinkage,
    inject_prior_candidates,
    validate_cross_priors,
)
from falcon.types import CrossCandidateSet


def test_validate_cross_priors_rejects_out_of_range_indices():
    priors = [PriorEdge(source_feature=2, target_feature=10,
                        expected_sign=1, confidence=0.5)]
    with pytest.raises(ValueError, match="target_feature"):
        validate_cross_priors(priors, n_x=5, n_y=5)


def test_validate_cross_priors_rejects_invalid_sign_or_confidence():
    with pytest.raises(ValueError, match="expected_sign"):
        validate_cross_priors(
            [PriorEdge(0, 0, expected_sign=2, confidence=0.5)],
            n_x=2, n_y=2,
        )
    with pytest.raises(ValueError, match="confidence"):
        validate_cross_priors(
            [PriorEdge(0, 0, expected_sign=1, confidence=-0.1)],
            n_x=2, n_y=2,
        )


def test_apply_prior_shrinkage_with_zero_weight_leaves_scores_unchanged():
    edge_pairs = np.array([[0, 1], [2, 3]])
    edge_scores = np.array([0.7, -0.3])
    prior_pairs = np.array([[0, 1]])
    out_scores, disagreed = apply_prior_shrinkage(
        edge_pairs, edge_scores,
        prior_pairs=prior_pairs,
        prior_signs=np.array([-1]),
        prior_confs=np.array([0.8]),
        prior_weight=0.0,
        target_magnitude=0.5,
    )
    np.testing.assert_array_equal(out_scores, edge_scores)
    assert not disagreed.any()


def test_apply_prior_shrinkage_drives_strong_prior_toward_target():
    edge_pairs = np.array([[0, 1]])
    edge_scores = np.array([0.1])
    out_scores, _ = apply_prior_shrinkage(
        edge_pairs, edge_scores,
        prior_pairs=np.array([[0, 1]]),
        prior_signs=np.array([-1]),
        prior_confs=np.array([1.0]),
        prior_weight=1e6,
        target_magnitude=0.4,
    )
    assert out_scores[0] == pytest.approx(-0.4, abs=1e-6)


def test_apply_prior_shrinkage_records_disagreement_with_data_sign():
    edge_pairs = np.array([[0, 1], [2, 3]])
    edge_scores = np.array([0.7, -0.5])
    out_scores, disagreed = apply_prior_shrinkage(
        edge_pairs, edge_scores,
        prior_pairs=np.array([[0, 1], [2, 3]]),
        prior_signs=np.array([-1, -1]),
        prior_confs=np.array([0.5, 0.5]),
        prior_weight=0.5,
        target_magnitude=0.3,
    )
    # Edge 0: data +0.7 vs prior -1 → disagreement
    # Edge 1: data -0.5 vs prior -1 → agreement
    assert disagreed[0]
    assert not disagreed[1]


def test_inject_prior_candidates_adds_missing_pairs_and_keeps_existing():
    base_score = np.zeros((4, 5))
    base_score[0, 0] = 0.9
    base_score[3, 4] = 0.05
    candidates = CrossCandidateSet(
        pairs=np.array([[0, 0]]),
        scores=np.array([0.9]),
        top_k=1,
        n_features_x=4,
        n_features_y=5,
    )
    prior_pairs = np.array([[0, 0], [3, 4]])

    augmented = inject_prior_candidates(candidates, base_score, prior_pairs)

    keys = set(map(tuple, augmented.pairs.tolist()))
    assert (0, 0) in keys
    assert (3, 4) in keys
    assert augmented.pairs.shape[0] == 2
