"""FALCON: single-domain compositional network estimator.

Public API:

    infer_network(
        counts,
        *,
        estimator="weighted_sparse",
        zero_policy="multiplicative",
        selection="stability",
        n_resamples=100,
        seed=0,
    ) -> NetworkResult

Three Python-only estimator candidates share the entrypoint:

    * ``adaptive_threshold`` — composition-adjusted thresholding (COAT-style)
    * ``weighted_sparse`` — weighted soft-thresholded covariance (fastCCLasso-style)
    * ``pd_sparse`` — adaptive threshold + positive-definite diagonal-loading

The selected estimator must clear the acceptance gates listed in
``docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md``
section 14 before any production claim is made. Until those gates are
evaluated on a frozen holdout grid, this package does not assert that any
estimator outperforms the matched-estimand baselines.
"""

from __future__ import annotations

import warnings as _warnings

# macOS Accelerate + NumPy 2.x emits spurious "matmul" RuntimeWarnings on
# valid finite inputs. Filter the false positive at import time so users
# see real numerical warnings instead of platform noise. Real numerical
# issues are caught by explicit isfinite assertions in the code.
for _msg in (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
):
    _warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

from falcon.api import infer_network
from falcon.results import EdgeTable, EstimatorDiagnostics, NetworkResult

__all__ = ["infer_network", "EdgeTable", "EstimatorDiagnostics", "NetworkResult"]
