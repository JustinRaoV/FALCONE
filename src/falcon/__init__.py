"""Falcon-SR: latent log-abundance correlation networks for compositional data.

Public API:

    infer_single(counts, *, mode="fast", ...)
    infer_cross(counts_x, counts_y, *, mode="fast", prior=None, ...)
    PriorEdge(source_feature, target_feature, expected_sign, confidence,
              provenance="")

Falcon-SR estimates the latent log-abundance Pearson correlations targeted by
SparCC (single-domain) and SparXCC Case-C (cross-domain), but reaches the same
estimand through a screen-refine pipeline: a SparCC-compatible dense base score,
a top-k candidate union, and a sparse refinement that updates only
candidate-incident equations. Optional permutation calibration produces
approximate p-values; optional signed biological priors enter as candidate
injections plus a post-hoc analytic shrinkage.

See ``docs/superpowers/specs/2026-06-01-falcon-sr-design.md`` for the
algorithmic specification and ``docs/superpowers/specs/2026-06-02-falcon-sr-rewrite-execution-design.md``
for the execution / migration details.
"""

import warnings as _warnings

# macOS Accelerate + NumPy 2.x emits spurious "matmul" RuntimeWarnings on
# valid finite inputs. Silence the false positive at import time so users
# see real numerical warnings instead of platform noise. Real numerical
# issues remain visible through explicit isfinite assertions in the code.
for _msg in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    _warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

from falcon.cross import infer_cross
from falcon.prior import PriorEdge
from falcon.single import infer_single

__all__ = ["infer_single", "infer_cross", "PriorEdge"]
