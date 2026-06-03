"""Six frozen scenario generators.

Each generator returns counts ``(n, p)``, the true off-diagonal support
matrix ``(p, p)`` (boolean, symmetric, zero diagonal), and a dictionary
of recorded metadata. Every random draw uses ``numpy.random.default_rng``
so seeded runs are reproducible across machines and Python versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

VALID_SCENARIOS = (
    "sparse_random",
    "hub",
    "block",
    "heavy_tailed",
    "negative_binomial_zi",
    "np_ratio",
)


@dataclass(frozen=True)
class Scenario:
    counts: np.ndarray
    support: np.ndarray  # (p, p) boolean
    metadata: dict


def available_scenarios() -> tuple[str, ...]:
    return VALID_SCENARIOS


def _ensure_pd(cov: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    eig = np.linalg.eigvalsh(cov).min()
    if eig < floor:
        cov = cov + (floor - eig) * np.eye(cov.shape[0])
    return cov


def _gaussian_basis(cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)


def _to_counts(log_basis: np.ndarray, depth: int = 5000) -> np.ndarray:
    abundance = np.exp(log_basis)
    composition = abundance / abundance.sum(axis=1, keepdims=True)
    counts = np.round(composition * depth).astype(np.int64)
    counts = np.maximum(counts, 0)
    return counts


def _sparse_random_cov(p: int, density: float, edge_strength: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cov = np.eye(p)
    support = np.zeros((p, p), dtype=bool)
    triu_i, triu_j = np.triu_indices(p, k=1)
    n_pairs = triu_i.size
    n_edges = int(round(density * n_pairs))
    if n_edges > 0:
        idx = rng.choice(n_pairs, size=n_edges, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_edges)
        for k, sgn in zip(idx, signs):
            i, j = triu_i[k], triu_j[k]
            cov[i, j] = cov[j, i] = sgn * edge_strength
            support[i, j] = support[j, i] = True
    return _ensure_pd(cov), support


def _hub_cov(p: int, n_hubs: int, edges_per_hub: int, edge_strength: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cov = np.eye(p)
    support = np.zeros((p, p), dtype=bool)
    hubs = rng.choice(p, size=n_hubs, replace=False)
    for h in hubs:
        candidates = np.array([k for k in range(p) if k != h])
        leaves = rng.choice(candidates, size=min(edges_per_hub, candidates.size), replace=False)
        for leaf in leaves:
            cov[h, leaf] = cov[leaf, h] = edge_strength
            support[h, leaf] = support[leaf, h] = True
    return _ensure_pd(cov), support


def _block_cov(p: int, block_size: int, within_strength: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cov = np.eye(p)
    support = np.zeros((p, p), dtype=bool)
    perm = rng.permutation(p)
    for start in range(0, p, block_size):
        members = perm[start : start + block_size]
        for a in members:
            for b in members:
                if a != b:
                    cov[a, b] = within_strength
                    support[a, b] = True
    return _ensure_pd(cov), support


def _heavy_tail_basis(cov: np.ndarray, n: int, rng: np.random.Generator, df: float = 4.0) -> np.ndarray:
    p = cov.shape[0]
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, p))
    g = rng.chisquare(df, size=(n, 1)) / df
    t = z / np.sqrt(g)
    return t @ L.T


def _nb_zi_counts(
    cov: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    depth: int,
    dispersion: float,
    zero_inflation: float,
) -> np.ndarray:
    log_basis = _gaussian_basis(cov, n, rng)
    composition = np.exp(log_basis)
    composition = composition / composition.sum(axis=1, keepdims=True)
    means = composition * depth
    # NB(mean=mu, dispersion=k): variance = mu + mu^2 / k
    p_par = dispersion / (dispersion + means)
    counts = rng.negative_binomial(dispersion, p_par)
    if zero_inflation > 0:
        zeros = rng.random(counts.shape) < zero_inflation
        counts = np.where(zeros, 0, counts)
    return counts.astype(np.int64)


def generate_scenario(
    name: str,
    *,
    n: int,
    p: int,
    seed: int,
    density: float = 0.05,
    edge_strength: float = 0.6,
    depth: int = 5000,
    dispersion: float = 5.0,
    zero_inflation: float = 0.2,
) -> Scenario:
    if name not in VALID_SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; valid: {VALID_SCENARIOS}")
    rng = np.random.default_rng(seed)
    metadata = {
        "scenario": name,
        "n": int(n),
        "p": int(p),
        "seed": int(seed),
        "density": float(density),
        "edge_strength": float(edge_strength),
        "depth": int(depth),
    }

    if name == "sparse_random":
        cov, support = _sparse_random_cov(p, density, edge_strength, rng)
        counts = _to_counts(_gaussian_basis(cov, n, rng), depth=depth)
    elif name == "hub":
        n_hubs = max(1, p // 25)
        edges_per_hub = max(2, p // 10)
        cov, support = _hub_cov(p, n_hubs, edges_per_hub, edge_strength, rng)
        counts = _to_counts(_gaussian_basis(cov, n, rng), depth=depth)
        metadata.update(n_hubs=n_hubs, edges_per_hub=edges_per_hub)
    elif name == "block":
        block_size = max(2, p // 8)
        cov, support = _block_cov(p, block_size, edge_strength, rng)
        counts = _to_counts(_gaussian_basis(cov, n, rng), depth=depth)
        metadata.update(block_size=block_size)
    elif name == "heavy_tailed":
        cov, support = _sparse_random_cov(p, density, edge_strength, rng)
        log_basis = _heavy_tail_basis(cov, n, rng, df=4.0)
        counts = _to_counts(log_basis, depth=depth)
        metadata.update(distribution="student_t", df=4.0)
    elif name == "negative_binomial_zi":
        cov, support = _sparse_random_cov(p, density, edge_strength, rng)
        counts = _nb_zi_counts(
            cov,
            n,
            rng,
            depth=depth,
            dispersion=dispersion,
            zero_inflation=zero_inflation,
        )
        metadata.update(
            distribution="nb_zi",
            dispersion=dispersion,
            zero_inflation=zero_inflation,
        )
    elif name == "np_ratio":
        cov, support = _sparse_random_cov(p, density, edge_strength, rng)
        counts = _to_counts(_gaussian_basis(cov, n, rng), depth=depth)
        metadata.update(np_ratio=float(n) / max(p, 1))
    else:  # pragma: no cover
        raise AssertionError(f"unhandled scenario {name!r}")

    metadata["zero_fraction"] = float((counts == 0).mean())
    return Scenario(counts=counts, support=support, metadata=metadata)


def iterate_scenarios(scenarios: Iterable[str], **kwargs) -> Iterable[Scenario]:
    for name in scenarios:
        yield generate_scenario(name, **kwargs)
