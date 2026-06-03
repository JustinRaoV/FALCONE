"""Frozen training and holdout grid configurations.

Training cells are small and fast enough to run on a laptop. Holdout
cells include larger ``n`` and ``p`` and are intended to run through a
generated server script. The two grids share schema fields so a single
benchmark runner consumes both.

Holdout cells are NOT used during method tuning. The split here is
authoritative; do not add training-grid configs to the holdout list or
vice-versa during a session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GridCell:
    scenario: str
    split: str  # "training" or "holdout"
    n: int
    p: int
    seed: int
    density: float = 0.05
    edge_strength: float = 0.6
    depth: int = 5000

    def metadata(self) -> dict:
        return {
            "scenario": self.scenario,
            "split": self.split,
            "n": int(self.n),
            "p": int(self.p),
            "seed": int(self.seed),
            "density": float(self.density),
            "edge_strength": float(self.edge_strength),
            "depth": int(self.depth),
        }


def _build(scenarios, sizes, density, seeds, split, edge_strength=0.6, depth=5000):
    cells = []
    for scenario in scenarios:
        for n, p in sizes:
            for seed in seeds:
                cells.append(
                    GridCell(
                        scenario=scenario,
                        split=split,
                        n=n,
                        p=p,
                        seed=seed,
                        density=density,
                        edge_strength=edge_strength,
                        depth=depth,
                    )
                )
    return cells


def training_grid() -> tuple[GridCell, ...]:
    """Small cells for local training-time tuning."""
    primary = ["sparse_random", "hub", "block"]
    sensitivity = ["heavy_tailed", "negative_binomial_zi"]
    sizes = [(100, 50), (200, 100)]
    seeds = [0, 1, 2]
    cells = _build(primary, sizes, density=0.05, seeds=seeds, split="training")
    cells += _build(sensitivity, sizes, density=0.05, seeds=seeds, split="training")
    np_ratio_sizes = [(50, 100), (100, 100), (200, 100)]
    cells += _build(["np_ratio"], np_ratio_sizes, density=0.05, seeds=seeds, split="training")
    return tuple(cells)


def holdout_grid() -> tuple[GridCell, ...]:
    """Larger cells reserved for the frozen holdout."""
    primary = ["sparse_random", "hub", "block"]
    sensitivity = ["heavy_tailed", "negative_binomial_zi"]
    sizes = [(250, 200), (500, 500), (500, 1000)]
    seeds = [10, 11, 12]
    cells = _build(primary, sizes, density=0.03, seeds=seeds, split="holdout")
    cells += _build(sensitivity, sizes, density=0.03, seeds=seeds, split="holdout")
    np_ratio_sizes = [(100, 500), (250, 500), (500, 500)]
    cells += _build(["np_ratio"], np_ratio_sizes, density=0.03, seeds=seeds, split="holdout")
    return tuple(cells)
