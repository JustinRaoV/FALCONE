# Falcon-SR

**Falcon-SR** is a screen-refine algorithm for inferring latent log-abundance
Pearson correlations from compositional sequencing data. It targets the same
estimand as SparCC (single-domain) and SparXCC Case-C (cross-domain) but
reaches it through a sparse pipeline: a SparCC-compatible dense base score, a
top-k candidate union, and a sparse refinement that updates only candidate-
incident equations. Optional permutation calibration produces approximate
p-values; optional signed biological priors enter as a candidate injection
plus an analytic post-hoc shrinkage.

This repository implements Falcon-SR end-to-end (`src/falcon/`), reproduces it
against external baselines (`benchmarks/`), and documents the design
(`docs/superpowers/specs/`).

> Status: experimental. The single- and cross-domain APIs are implemented
> and feasibility-tested; the published acceptance gates in the design
> specification must be satisfied by a full benchmark run before the
> method is presented as validated.

---

## Quickstart

```bash
uv sync
uv run pytest -q
```

```python
import numpy as np
from falcon import infer_single, infer_cross, PriorEdge

rng = np.random.default_rng(0)

# Single-domain inference
counts = rng.integers(1, 200, size=(100, 50))
result = infer_single(
    counts,
    mode="fast",
    top_k=10,
    calibration="permutation",
    n_permutations=100,
    seed=0,
)
print(result.edges.pairs.shape, result.edges.scores[:5])
print("approx q-values:", result.edges.qvalue_approx[:5])

# Cross-domain inference with optional signed biological prior
counts_x = rng.integers(1, 200, size=(100, 50))
counts_y = rng.integers(1, 200, size=(100, 60))
priors = [
    PriorEdge(source_feature=3, target_feature=12,
              expected_sign=-1, confidence=0.8,
              provenance="crispr_spacer"),
]
cross = infer_cross(
    counts_x, counts_y,
    mode="fast", top_k=10,
    prior=priors, prior_weight=0.5,
    calibration="permutation", n_permutations=100, seed=0,
)
print(cross.edges.pairs.shape, cross.edges.scores[:5])
```

`infer_single` and `infer_cross` both return a `NetworkResult` with:

- `edges.pairs` — `(n_edges, 2)` array of `(i, j)` feature indices
  (canonical `i < j` for single-domain; `i ∈ X, j ∈ Y` for cross-domain)
- `edges.scores` — refined Pearson correlation per edge
- `edges.pvalue_approx`, `edges.qvalue_approx` — populated only when
  `calibration="permutation"`
- `diagnostics` — adaptive growth metadata, candidate density, prior
  bookkeeping, and the explicit calibration method tag
- `initial_matrix` — full dense matrix when `mode="strict"`, else `None`
- `calibration` — `CalibrationResult` with the full null distribution,
  or `None` when `calibration="none"`

---

## Feasibility benchmarks

The repository ships two benchmark runners. They write per-method-per-cell
rows to `data/falcon_sr_*_feasibility.csv`.

```bash
# Single-domain feasibility grid
uv run python benchmarks/falcon_sr_single.py \
    --n 100 500 --p 100 500 1000 --top-k 10 25 50 --reps 3

# Cross-domain feasibility grid (SparXCC Case-C style simulator)
uv run python benchmarks/falcon_sr_cross.py \
    --n 100 500 --pq 100,100 500,500 --top-k 10 25 --reps 3

# Run both with defaults
./benchmarks/run_all.sh
```

Each cell runs Falcon-SR alongside SparCC / SparXCC base / SparXCC iter /
Pearson(CLR) and reports candidate recall, edge overlap, sign accuracy,
AUROC, Recall@K, wall-clock, and peak memory. The runners write rows as
each method finishes, so partial output is durable.

---

## Design

- `docs/superpowers/specs/2026-06-01-falcon-sr-design.md` — algorithmic
  specification (single + cross + priors + calibration).
- `docs/superpowers/specs/2026-06-02-falcon-sr-rewrite-execution-design.md`
  — execution / migration design that captures the cross-domain refinement
  geometry, prior closed form, and base-only permutation approximation.
- `docs/methodology.md` — narrative method description.
- `docs/decision-log.md` — design decisions and their motivation.

---

## License

MIT. See `pyproject.toml`.
