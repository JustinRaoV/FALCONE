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


@dataclass(frozen=True)
class CrossCandidateSet:
    pairs: np.ndarray
    scores: np.ndarray
    top_k: int
    n_features_x: int
    n_features_y: int

    @property
    def density(self) -> float:
        possible = self.n_features_x * self.n_features_y
        return self.pairs.shape[0] / possible


@dataclass(frozen=True)
class EdgeTable:
    pairs: np.ndarray
    scores: np.ndarray
    pvalue_approx: np.ndarray | None = None
    qvalue_approx: np.ndarray | None = None


@dataclass(frozen=True)
class ScreenDiagnostics:
    initial_top_k: int
    final_top_k: int
    candidate_count: int
    candidate_density: float
    growth_rounds: int
    overlap_across_budgets: float
    sign_stability_across_budgets: float
    fallback_reason: str | None
    calibration_method: str | None = None
    n_permutations: int | None = None
    pruned_x_count: int | None = None
    pruned_y_count: int | None = None
    fallback_to_base_centering: bool | None = None
    prior_count: int | None = None
    data_disagreed_with_prior_count: int | None = None


@dataclass(frozen=True)
class NetworkResult:
    edges: EdgeTable
    diagnostics: ScreenDiagnostics
    initial_matrix: np.ndarray | None
    calibration: object | None = None
