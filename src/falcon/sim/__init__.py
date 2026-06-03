"""Frozen simulation harness for the single-domain estimator rebuild.

Six scenarios per design section 8:

* ``sparse_random`` — Erdos-Renyi sparse correlation graph
* ``hub`` — small set of hubs each connected to many leaves
* ``block`` — block-diagonal community structure
* ``heavy_tailed`` — heavy-tailed latent log abundances (Student-t)
* ``negative_binomial_zi`` — NB counts with zero inflation
* ``np_ratio`` — sensitivity over n/p ratios

Training and holdout grids are split before any tuning. Holdout rows are
never used to select thresholds, lambda rules, or the winning estimator.
"""

from falcon.sim.metrics import (
    average_precision_score,
    auroc_score,
    fdr_at_target,
    precision_at_k,
    recall_at_k,
)
from falcon.sim.scenarios import (
    Scenario,
    available_scenarios,
    generate_scenario,
)
from falcon.sim.grid import (
    GridCell,
    holdout_grid,
    training_grid,
)

__all__ = [
    "Scenario",
    "available_scenarios",
    "generate_scenario",
    "GridCell",
    "training_grid",
    "holdout_grid",
    "average_precision_score",
    "auroc_score",
    "fdr_at_target",
    "precision_at_k",
    "recall_at_k",
]
