"""Cross-domain signed biological priors.

Implements section 10 of the Falcon-SR design specification and section 2.2
of the 2026-06-02 execution design. A prior contributes two things:

1. The prior's (source, target) edge is forced into the candidate set, even
   when the statistical screen would omit it (spec §10 bullet 1).
2. After sparse refinement, the candidate edge score is replaced by the
   analytic closed-form minimiser of
     (rho - rho_data)^2 + lambda_prior * confidence * (rho - sign * target)^2.
   This is a soft direction, not an invented effect size: with
   prior_weight = 0 the score equals rho_data exactly.

Priors deliberately do not influence the iterative exclusion choices inside
refinement; that would risk hiding wrong-sign data signals behind a prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from falcon.types import CrossCandidateSet


@dataclass(frozen=True)
class PriorEdge:
    source_feature: int
    target_feature: int
    expected_sign: int
    confidence: float
    provenance: str = ""


def validate_cross_priors(
    priors: Sequence[PriorEdge],
    *,
    n_x: int,
    n_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pairs = []
    signs = []
    confidences = []
    provenances = []
    for p in priors:
        if not (0 <= p.source_feature < n_x):
            raise ValueError(
                f"source_feature {p.source_feature} out of range [0, {n_x})"
            )
        if not (0 <= p.target_feature < n_y):
            raise ValueError(
                f"target_feature {p.target_feature} out of range [0, {n_y})"
            )
        if p.expected_sign not in (-1, 0, 1):
            raise ValueError(
                f"expected_sign must be -1, 0, or 1; got {p.expected_sign}"
            )
        if not (0.0 <= p.confidence <= 1.0):
            raise ValueError(
                f"confidence must lie in [0, 1]; got {p.confidence}"
            )
        pairs.append((p.source_feature, p.target_feature))
        signs.append(p.expected_sign)
        confidences.append(p.confidence)
        provenances.append(p.provenance)
    if not pairs:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            [],
        )
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        np.asarray(confidences, dtype=np.float64),
        provenances,
    )


def inject_prior_candidates(
    candidates: CrossCandidateSet,
    base_score: np.ndarray,
    prior_pairs: np.ndarray,
) -> CrossCandidateSet:
    if prior_pairs.shape[0] == 0:
        return candidates
    q = candidates.n_features_y
    cand_keys = (
        candidates.pairs[:, 0].astype(np.int64) * q
        + candidates.pairs[:, 1].astype(np.int64)
    )
    prior_keys = (
        prior_pairs[:, 0].astype(np.int64) * q
        + prior_pairs[:, 1].astype(np.int64)
    )
    missing = np.setdiff1d(prior_keys, cand_keys, assume_unique=False)
    if missing.size == 0:
        return candidates
    new_rows = (missing // q).astype(np.int64)
    new_cols = (missing % q).astype(np.int64)
    new_pairs = np.column_stack([new_rows, new_cols])
    new_scores = base_score[new_rows, new_cols]

    all_pairs = np.vstack([candidates.pairs, new_pairs])
    all_scores = np.concatenate([candidates.scores, new_scores])
    order = np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))
    return CrossCandidateSet(
        pairs=all_pairs[order],
        scores=all_scores[order],
        top_k=candidates.top_k,
        n_features_x=candidates.n_features_x,
        n_features_y=candidates.n_features_y,
    )


def apply_prior_shrinkage(
    edge_pairs: np.ndarray,
    edge_scores: np.ndarray,
    *,
    prior_pairs: np.ndarray,
    prior_signs: np.ndarray,
    prior_confs: np.ndarray,
    prior_weight: float,
    target_magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    edge_pairs = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
    edge_scores = np.asarray(edge_scores, dtype=np.float64)
    out_scores = edge_scores.copy()
    disagreed = np.zeros(edge_pairs.shape[0], dtype=bool)

    if prior_weight == 0.0 or prior_pairs.shape[0] == 0:
        return out_scores, disagreed

    # Encode pairs as int64 keys (assumes feature indices < 2^31)
    BASE = np.int64(1) << 32
    edge_keys = edge_pairs[:, 0].astype(np.int64) * BASE + edge_pairs[:, 1].astype(np.int64)
    prior_keys = prior_pairs[:, 0].astype(np.int64) * BASE + prior_pairs[:, 1].astype(np.int64)

    order = np.argsort(prior_keys, kind="mergesort")
    sorted_prior_keys = prior_keys[order]
    sorted_signs = prior_signs[order]
    sorted_confs = prior_confs[order]
    pos = np.searchsorted(sorted_prior_keys, edge_keys)
    valid = (pos < sorted_prior_keys.size) & (
        sorted_prior_keys[np.clip(pos, 0, sorted_prior_keys.size - 1)] == edge_keys
    )
    if not valid.any():
        return out_scores, disagreed

    matched_pos = pos[valid]
    signs = sorted_signs[matched_pos].astype(np.float64)
    confs = sorted_confs[matched_pos]
    data_scores = edge_scores[valid]
    new_scores = (
        data_scores + prior_weight * confs * signs * target_magnitude
    ) / (1.0 + prior_weight * confs)
    out_scores[valid] = new_scores

    data_sign = np.sign(data_scores)
    disagree_mask = (signs != 0) & (data_sign != 0) & (data_sign != signs)
    valid_indices = np.flatnonzero(valid)
    disagreed[valid_indices[disagree_mask]] = True

    return out_scores, disagreed
