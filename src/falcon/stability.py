"""Subsampling-based stability selection (Meinshausen & Bühlmann 2010).

For each of ``n_resamples`` subsamples drawn at size ``subsample_size``,
the supplied estimator is run and the off-diagonal selected support is
recorded. The output ``selection_probability`` is the fraction of
subsamples in which an entry was non-zero. By convention the diagonal
is reported as 1.0 (no estimator penalizes the diagonal).

Per-subsample seeds are spawned via ``np.random.SeedSequence(seed).spawn``
so the resampling sequence is bit-reproducible under (seed, n_jobs) and
each subsample uses a statistically independent stream (the prior
single-stream code introduced correlations that affected calibration).

When ``n_jobs > 1``, subsamples run in a ``ThreadPoolExecutor``. Workers
inherit any thread-count environment variables; callers concerned about
BLAS over-subscription should set ``OMP_NUM_THREADS=1``, etc. before
spawning.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    selection_probability: np.ndarray
    n_resamples_used: int
    subsample_fraction: float
    subsample_size: int


def _one_subsample(Z, idx, estimator_fn, p):
    """Run estimator_fn on a subsample and return a (p, p) int matrix
    of {0, 1} support indicators (zero diagonal).

    estimator_fn may return either:
    - a (p, p) covariance matrix (the legacy interface), or
    - a (p*(p-1)/2,) boolean upper-triangle support mask (faster).

    The upper-triangle order matches np.triu_indices(p, k=1).
    """
    sub = Z[idx]
    out = np.asarray(estimator_fn(sub))
    if out.ndim == 1:
        # Boolean upper-triangle mask. Expand to full (p, p) int matrix.
        expected = p * (p - 1) // 2
        if out.size != expected:
            raise ValueError(
                f"1-D support mask has length {out.size}, expected {expected} "
                f"for p={p}"
            )
        triu_i, triu_j = np.triu_indices(p, k=1)
        nonzero = np.zeros((p, p), dtype=np.int64)
        nonzero[triu_i, triu_j] = out.astype(np.int64)
        nonzero[triu_j, triu_i] = nonzero[triu_i, triu_j]
        return nonzero
    # 2-D covariance path: keep behaviour identical to A3.
    cov = out.astype(np.float64, copy=False)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != p:
        raise ValueError(
            "estimator_fn must return a square matrix matching Z's feature count"
        )
    nonzero = (cov != 0).astype(np.int64)
    np.fill_diagonal(nonzero, 0)
    return nonzero


def select_by_stability(
    Z: np.ndarray,
    estimator_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_resamples: int = 100,
    subsample_fraction: float = 0.5,
    seed: int = 0,
    n_jobs: int = 1,
) -> StabilityResult:
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("subsample_fraction must lie in (0, 1]")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional matrix")
    n, p = Z.shape
    subsample_size = int(round(n * subsample_fraction))
    subsample_size = max(3, min(n, subsample_size))

    # Spawn per-subsample SeedSequences. Independent streams; deterministic
    # under fixed (seed, n_jobs).
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_resamples)

    # Pre-draw all subsample index arrays so workers don't need RNG state.
    indices_list = []
    for child in child_seeds:
        rng = np.random.default_rng(child)
        idx = rng.choice(n, size=subsample_size, replace=False)
        idx.sort()
        indices_list.append(idx)

    accumulator = np.zeros((p, p), dtype=np.int64)
    if n_jobs == 1:
        for idx in indices_list:
            accumulator += _one_subsample(Z, idx, estimator_fn, p)
    else:
        # ThreadPoolExecutor: estimator_fn closures from api._build_estimator
        # are not pickle-safe, so a ProcessPoolExecutor isn't usable here
        # without extra plumbing. Threads work because NumPy releases the
        # GIL during BLAS calls.
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_one_subsample, Z, idx, estimator_fn, p)
                for idx in indices_list
            ]
            for fut in futures:
                accumulator += fut.result()

    probability = accumulator / float(n_resamples)
    probability = 0.5 * (probability + probability.T)
    np.fill_diagonal(probability, 1.0)

    return StabilityResult(
        selection_probability=probability,
        n_resamples_used=n_resamples,
        subsample_fraction=subsample_fraction,
        subsample_size=subsample_size,
    )
