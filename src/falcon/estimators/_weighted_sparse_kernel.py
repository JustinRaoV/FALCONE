"""Optional Numba-JIT inner kernel for the weighted_sparse alternating loop.

Imported by weighted_sparse.py. If numba is unavailable, _NUMBA_OK
is False and weighted_sparse uses its pure-NumPy in-place fallback
unchanged.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
    _NUMBA_OK = True
except ImportError:  # pragma: no cover - covered when extras-not-installed
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _alternating_step(S_clr, threshold_off, Sigma, Sigma_new, f, p):
    """One Step A + Step B iteration; mutates Sigma_new and f in place.
    Returns (delta_sq, scale_sq) for the convergence check."""
    # Step A: offset update from R = Sigma - S_clr.
    R_sum = np.zeros(p)
    total = 0.0
    for i in range(p):
        s = 0.0
        for j in range(p):
            r = Sigma[i, j] - S_clr[i, j]
            s += r
            total += r
        R_sum[i] = s
    total_div = total / (2.0 * p)
    for i in range(p):
        f[i] = (R_sum[i] - total_div) / p

    # Step B: soft-threshold on M = S_clr + f1' + 1f' for off-diagonal;
    # diagonal preserved as M_ii.
    for i in range(p):
        for j in range(p):
            m = S_clr[i, j] + f[i] + f[j]
            if i == j:
                Sigma_new[i, j] = m
            else:
                a = abs(m) - threshold_off[i, j]
                if a < 0.0:
                    Sigma_new[i, j] = 0.0
                else:
                    # sign(m) * a
                    if m > 0.0:
                        Sigma_new[i, j] = a
                    elif m < 0.0:
                        Sigma_new[i, j] = -a
                    else:
                        Sigma_new[i, j] = 0.0

    # Symmetrize in place (upper triangle averaging).
    for i in range(p):
        for j in range(i + 1, p):
            v = 0.5 * (Sigma_new[i, j] + Sigma_new[j, i])
            Sigma_new[i, j] = v
            Sigma_new[j, i] = v

    # Frobenius delta_sq and scale_sq.
    delta_sq = 0.0
    scale_sq = 0.0
    for i in range(p):
        for j in range(p):
            d = Sigma_new[i, j] - Sigma[i, j]
            delta_sq += d * d
            scale_sq += Sigma_new[i, j] * Sigma_new[i, j]
    return delta_sq, scale_sq


def is_available() -> bool:
    return _NUMBA_OK
