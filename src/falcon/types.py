from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateSet:
    pairs: np.ndarray
    scores: np.ndarray
    top_k: int
    n_features: int

    @property
    def density(self) -> float:
        possible = self.n_features * (self.n_features - 1) / 2
        return self.pairs.shape[0] / possible
