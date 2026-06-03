"""Subsampling-based stability selection (Meinshausen & Buhlmann 2010).

For each of ``n_resamples`` subsamples drawn at size ``subsample_size``,
the supplied estimator is run and the off-diagonal selected support is
recorded. The output ``selection_probability`` is the fraction of
subsamples in which an entry was non-zero. By convention the diagonal
is reported as 1.0 (no estimator penalizes the diagonal).

Reproducibility: under a fixed seed the resampling sequence and thus
the output are deterministic. This is a tested invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    selection_probability: np.ndarray
    n_resamples_used: int
    subsample_fraction: float
    subsample_size: int


def select_by_stability(
    Z: np.ndarray,
    estimator_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_resamples: int = 100,
    subsample_fraction: float = 0.5,
    seed: int = 0,
) -> StabilityResult:
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("subsample_fraction must lie in (0, 1]")

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional matrix")
    n, p = Z.shape
    subsample_size = int(round(n * subsample_fraction))
    subsample_size = max(3, min(n, subsample_size))

    rng = np.random.default_rng(seed)
    accumulator = np.zeros((p, p), dtype=np.int64)

    for _ in range(n_resamples):
        idx = rng.choice(n, size=subsample_size, replace=False)
        idx.sort()
        sub = Z[idx]
        cov = np.asarray(estimator_fn(sub), dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != p:
            raise ValueError(
                "estimator_fn must return a square matrix matching Z's feature count"
            )
        nonzero = (cov != 0).astype(np.int64)
        np.fill_diagonal(nonzero, 0)
        accumulator += nonzero

    probability = accumulator / float(n_resamples)
    probability = 0.5 * (probability + probability.T)
    np.fill_diagonal(probability, 1.0)

    return StabilityResult(
        selection_probability=probability,
        n_resamples_used=n_resamples,
        subsample_fraction=subsample_fraction,
        subsample_size=subsample_size,
    )
