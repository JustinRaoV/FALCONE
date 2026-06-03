"""Preprocess raw count matrices into log compositions.

Three zero-handling policies are exposed and reported as a sensitivity axis:

* ``multiplicative`` — Aitchison multiplicative replacement; the default for
  estimators that need finite log compositions everywhere.
* ``pseudocount`` — add 0.5 to counts then renormalize.
* ``complete_case`` — keep zeros as zeros and emit ``NaN`` in the log
  composition at those positions; downstream estimators that support
  pairwise complete cases must handle the NaNs explicitly.

The default may only be chosen after the training grid is evaluated; the
benchmark records ``zero_policy`` per row so the choice never sneaks in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_ZERO_POLICIES = ("multiplicative", "pseudocount", "complete_case")


@dataclass(frozen=True)
class PreprocessReport:
    n_samples: int
    n_features_in: int
    n_features_out: int
    zero_count: int
    zero_fraction: float
    zero_policy: str
    kept_indices: np.ndarray
    zero_fraction_per_feature: np.ndarray
    nonzero_mask: np.ndarray


@dataclass(frozen=True)
class PreparedComposition:
    composition: np.ndarray
    log_composition: np.ndarray
    report: PreprocessReport


def _validated_counts(counts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("counts must contain only finite values")
    if (matrix < 0).any():
        raise ValueError("counts must be non-negative")
    if (matrix.sum(axis=1) <= 0).any():
        raise ValueError("every sample must have a positive row total")
    return matrix


def _multiplicative_replacement(composition: np.ndarray) -> np.ndarray:
    p = composition.shape[1]
    delta = 0.65 / (p * p)
    zero_mask = composition == 0
    zero_count = zero_mask.sum(axis=1, keepdims=True)
    scale = 1.0 - zero_count * delta
    return np.where(zero_mask, delta, composition * scale)


def prepare_log_composition(
    counts: np.ndarray,
    *,
    min_prevalence: float = 0.0,
    min_total: float = 1.0,
    zero_policy: str = "multiplicative",
) -> PreparedComposition:
    matrix = _validated_counts(counts)
    if not 0.0 <= min_prevalence <= 1.0:
        raise ValueError("min_prevalence must lie in [0, 1]")
    if min_total < 0:
        raise ValueError("min_total must be non-negative")
    if zero_policy not in VALID_ZERO_POLICIES:
        raise ValueError(
            f"unknown zero_policy: {zero_policy!r}; "
            f"valid options: {VALID_ZERO_POLICIES}"
        )

    prevalence = (matrix > 0).mean(axis=0)
    totals = matrix.sum(axis=0)
    kept = np.flatnonzero((prevalence >= min_prevalence) & (totals >= min_total))
    if kept.size < 4:
        raise ValueError("at least four features must remain after filtering")

    filtered = matrix[:, kept]
    composition_raw = filtered / filtered.sum(axis=1, keepdims=True)
    nonzero_mask = composition_raw > 0
    zero_count = int((~nonzero_mask).sum())
    zero_fraction_per_feature = (~nonzero_mask).mean(axis=0)

    if zero_policy == "multiplicative":
        composition = _multiplicative_replacement(composition_raw)
        log_composition = np.log(composition)
    elif zero_policy == "pseudocount":
        adjusted = filtered + 0.5
        composition = adjusted / adjusted.sum(axis=1, keepdims=True)
        log_composition = np.log(composition)
    else:  # complete_case
        composition = composition_raw
        log_composition = np.full_like(composition, np.nan)
        np.log(composition, out=log_composition, where=nonzero_mask)

    return PreparedComposition(
        composition=composition,
        log_composition=log_composition,
        report=PreprocessReport(
            n_samples=matrix.shape[0],
            n_features_in=matrix.shape[1],
            n_features_out=kept.size,
            zero_count=zero_count,
            zero_fraction=zero_count / max(filtered.size, 1),
            zero_policy=zero_policy,
            kept_indices=kept,
            zero_fraction_per_feature=zero_fraction_per_feature,
            nonzero_mask=nonzero_mask,
        ),
    )
