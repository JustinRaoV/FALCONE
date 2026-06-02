# Falcon-SR Single-Domain Feasibility Implementation Plan

> **SUPERSEDED 2026-06-02.** The single-domain feasibility scope captured
> below has been folded into a single end-to-end rewrite covering
> single + cross + priors + calibration + benchmarks + manuscript
> skeleton, executed against
> `docs/superpowers/specs/2026-06-02-falcon-sr-rewrite-execution-design.md`.
> Tasks 1–6 of this plan landed before the rewrite; tasks 7–8 are now
> subsumed by `benchmarks/falcon_sr_single.py` and the refreshed
> documentation. This file is retained for historical context only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a tested single-domain Falcon-SR prototype that estimates the same latent log-abundance Pearson correlations as SparCC, screens a sparse candidate set, and refines candidates without repeating dense exclusion solves.

**Architecture:** Preserve the current prototype API while adding focused modules under `src/falcon/`. The new path computes a SparCC-compatible dense base score once, selects a symmetric top-k candidate union, and updates excluded-edge equations with a sparse `LinearOperator + cg` solve. This plan is deliberately limited to the single-domain research hypothesis: prove candidate recall, reference agreement, and runtime behavior before implementing cross-domain inference, priors, or publication claims.

**Tech Stack:** Python 3.12, uv, NumPy, SciPy sparse linear algebra, pytest, stdlib CSV, stdlib `time`, stdlib `tracemalloc`.

---

## Scope And Sequencing

The approved specification covers a unified framework, but implementation must
proceed through testable research gates:

1. **This plan:** single-domain strict reference, sparse screening, sparse
   refinement, adaptive growth, and feasibility benchmark.
2. **Next plan after feasibility acceptance:** SparXCC-compatible cross-domain
   screening and refinement.
3. **Third plan after cross-domain acceptance:** optional signed priors,
   permutation calibration, expanded baseline adapters, public-data workflows,
   manuscript figures, and prose revision.

Do not modify the manuscript or regenerate publication figures during this
plan. The existing manuscript contains claims based on a different estimand and
must remain visibly stale until Falcon-SR benchmarks exist.

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | pytest development dependency |
| `src/falcon/preprocessing.py` | validated compositional preprocessing |
| `src/falcon/types.py` | immutable edge, diagnostic, and result containers |
| `src/falcon/single.py` | dense base score, strict reference exclusion, sparse candidate refinement, public single-domain API |
| `src/falcon/screen.py` | symmetric top-k candidate construction and overlap metrics |
| `src/falcon/__init__.py` | export the new API without deleting legacy prototype functions |
| `tests/test_preprocessing.py` | preprocessing contract |
| `tests/test_single_base.py` | dense base-score equivalence |
| `tests/test_screen.py` | candidate-set invariants |
| `tests/test_single_refine.py` | strict and sparse-refinement equivalence |
| `tests/test_single_api.py` | adaptive-growth and result-schema behavior |
| `benchmarks/single_feasibility.py` | reproducible research-gate benchmark |
| `benchmarks/io_utils.py` | feasibility CSV schema |
| `data/falcon_sr_single_feasibility.csv` | generated raw replicate rows, committed only after a deliberate benchmark run |
| `README.md` | mark Falcon-SR as an experimental API and preserve legacy API notes |

## Task 1: Add Test Runner And Validated Preprocessing

**Files:**
- Modify: `pyproject.toml`
- Create: `src/falcon/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Add pytest as a uv development dependency**

Run:

```bash
uv add --dev pytest
```

Expected: `pyproject.toml` and `uv.lock` change, and `uv run pytest --version`
prints a pytest version.

- [ ] **Step 2: Write failing preprocessing tests**

Create `tests/test_preprocessing.py`:

```python
import numpy as np
import pytest

from falcon.preprocessing import prepare_log_composition


def test_prepare_log_composition_preserves_rows_and_reports_zeros():
    counts = np.array([[10, 0, 30, 20], [0, 5, 5, 10]], dtype=float)

    prepared = prepare_log_composition(counts, zero_policy="multiplicative")

    np.testing.assert_allclose(prepared.composition.sum(axis=1), 1.0)
    assert np.isfinite(prepared.log_composition).all()
    assert prepared.report.zero_count == 2
    assert prepared.report.n_features_in == 4
    assert prepared.report.n_features_out == 4


@pytest.mark.parametrize(
    "counts, message",
    [
        (np.array([[1.0, -1.0], [1.0, 2.0]]), "non-negative"),
        (np.array([[1.0, np.nan], [1.0, 2.0]]), "finite"),
        (np.array([[0.0, 0.0], [1.0, 2.0]]), "positive row total"),
    ],
)
def test_prepare_log_composition_rejects_invalid_counts(counts, message):
    with pytest.raises(ValueError, match=message):
        prepare_log_composition(counts)


def test_prepare_log_composition_filters_low_prevalence_features():
    counts = np.array(
        [
            [10, 0, 1, 0, 7, 8],
            [10, 2, 0, 0, 7, 8],
            [10, 3, 0, 0, 7, 8],
            [10, 4, 0, 1, 7, 8],
        ],
        dtype=float,
    )

    prepared = prepare_log_composition(counts, min_prevalence=0.5)

    np.testing.assert_array_equal(prepared.report.kept_indices, [0, 1, 4, 5])
    assert prepared.composition.shape == (4, 4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_preprocessing.py -q
```

Expected: FAIL during import because `falcon.preprocessing` does not exist.

- [ ] **Step 4: Implement validated preprocessing**

Create `src/falcon/preprocessing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessReport:
    n_samples: int
    n_features_in: int
    n_features_out: int
    zero_count: int
    zero_fraction: float
    zero_policy: str
    kept_indices: np.ndarray


@dataclass(frozen=True)
class PreparedComposition:
    composition: np.ndarray
    log_composition: np.ndarray
    report: PreprocessReport


def _validated_counts(counts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("counts must contain only finite values")
    if (matrix < 0).any():
        raise ValueError("counts must be non-negative")
    if (matrix.sum(axis=1) <= 0).any():
        raise ValueError("every sample must have a positive row total")
    return matrix


def _multiplicative_replacement(composition: np.ndarray) -> np.ndarray:
    p = composition.shape[1]
    delta = 0.65 / (p * p)
    zero_mask = composition == 0
    zero_count = zero_mask.sum(axis=1, keepdims=True)
    scale = 1.0 - zero_count * delta
    return np.where(zero_mask, delta, composition * scale)


def prepare_log_composition(
    counts: np.ndarray,
    *,
    min_prevalence: float = 0.0,
    min_total: float = 1.0,
    zero_policy: str = "multiplicative",
) -> PreparedComposition:
    matrix = _validated_counts(counts)
    if not 0.0 <= min_prevalence <= 1.0:
        raise ValueError("min_prevalence must lie in [0, 1]")
    if min_total < 0:
        raise ValueError("min_total must be non-negative")

    prevalence = (matrix > 0).mean(axis=0)
    totals = matrix.sum(axis=0)
    kept = np.flatnonzero((prevalence >= min_prevalence) & (totals >= min_total))
    if kept.size < 4:
        raise ValueError("at least four features must remain after filtering")

    filtered = matrix[:, kept]
    composition = filtered / filtered.sum(axis=1, keepdims=True)
    zero_count = int((composition == 0).sum())
    if zero_policy == "multiplicative":
        composition = _multiplicative_replacement(composition)
    elif zero_policy == "pseudocount":
        filtered = filtered + 0.5
        composition = filtered / filtered.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown zero_policy: {zero_policy}")

    return PreparedComposition(
        composition=composition,
        log_composition=np.log(composition),
        report=PreprocessReport(
            n_samples=matrix.shape[0],
            n_features_in=matrix.shape[1],
            n_features_out=kept.size,
            zero_count=zero_count,
            zero_fraction=zero_count / filtered.size,
            zero_policy=zero_policy,
            kept_indices=kept,
        ),
    )
```

- [ ] **Step 5: Run preprocessing tests**

Run:

```bash
uv run pytest tests/test_preprocessing.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit preprocessing**

```bash
git add pyproject.toml uv.lock src/falcon/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add validated compositional preprocessing"
```

## Task 2: Add SparCC-Compatible Dense Base Score

**Files:**
- Create: `src/falcon/single.py`
- Create: `tests/test_single_base.py`

- [ ] **Step 1: Write failing dense-base tests**

Create `tests/test_single_base.py`:

```python
import numpy as np

from falcon.single import single_base_score, variation_matrix


def _pair_loop_variation(log_composition):
    p = log_composition.shape[1]
    result = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            result[i, j] = np.var(
                log_composition[:, i] - log_composition[:, j], ddof=1
            )
    return result


def test_variation_matrix_matches_pair_loop():
    log_composition = np.log(
        np.array(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.10, 0.40, 0.30],
                [0.15, 0.30, 0.25, 0.30],
                [0.30, 0.20, 0.10, 0.40],
            ]
        )
    )

    np.testing.assert_allclose(
        variation_matrix(log_composition),
        _pair_loop_variation(log_composition),
        atol=1e-12,
    )


def test_single_base_score_is_symmetric_bounded_and_has_unit_diagonal():
    counts = np.array(
        [
            [30, 10, 20, 40],
            [25, 15, 30, 30],
            [15, 25, 35, 25],
            [20, 20, 10, 50],
            [35, 10, 25, 30],
        ],
        dtype=float,
    )

    result = single_base_score(counts)

    np.testing.assert_allclose(result.correlation, result.correlation.T)
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert (np.abs(result.correlation) <= 1.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_single_base.py -q
```

Expected: FAIL during import because `falcon.single` does not exist.

- [ ] **Step 3: Implement dense variation and base score**

Create `src/falcon/single.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falcon.preprocessing import PreprocessReport, prepare_log_composition


@dataclass(frozen=True)
class SingleBaseResult:
    variation: np.ndarray
    basis_variance: np.ndarray
    correlation: np.ndarray
    preprocess_report: PreprocessReport


def variation_matrix(log_composition: np.ndarray) -> np.ndarray:
    centered = log_composition - log_composition.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    diagonal = np.diag(covariance)
    variation = diagonal[:, None] + diagonal[None, :] - 2.0 * covariance
    np.fill_diagonal(variation, 0.0)
    return variation


def _dense_modifier(p: int) -> np.ndarray:
    modifier = np.ones((p, p), dtype=np.float64)
    np.fill_diagonal(modifier, p - 1.0)
    return modifier


def solve_basis_variance_dense(
    variation: np.ndarray,
    *,
    excluded: np.ndarray | None = None,
    min_variance: float = 1e-4,
) -> np.ndarray:
    p = variation.shape[0]
    modifier = _dense_modifier(p)
    rhs = variation.sum(axis=1).copy()
    if excluded is not None and excluded.size:
        for i, j in excluded:
            rhs[i] -= variation[i, j]
            rhs[j] -= variation[i, j]
            modifier[i, i] -= 1.0
            modifier[j, j] -= 1.0
            modifier[i, j] -= 1.0
            modifier[j, i] -= 1.0
    return np.maximum(np.linalg.solve(modifier, rhs), min_variance)


def correlations_from_basis(
    variation: np.ndarray,
    basis_variance: np.ndarray,
) -> np.ndarray:
    covariance = 0.5 * (
        basis_variance[:, None] + basis_variance[None, :] - variation
    )
    scale = np.sqrt(np.outer(basis_variance, basis_variance))
    correlation = np.clip(covariance / scale, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def single_base_score(counts: np.ndarray) -> SingleBaseResult:
    prepared = prepare_log_composition(counts)
    variation = variation_matrix(prepared.log_composition)
    basis_variance = solve_basis_variance_dense(variation)
    correlation = correlations_from_basis(variation, basis_variance)
    return SingleBaseResult(
        variation=variation,
        basis_variance=basis_variance,
        correlation=correlation,
        preprocess_report=prepared.report,
    )
```

- [ ] **Step 4: Run dense-base tests**

Run:

```bash
uv run pytest tests/test_single_base.py -q
```

Expected: PASS.

- [ ] **Step 5: Add equivalence test against the existing vectorized SparCC baseline**

Append to `tests/test_single_base.py`:

```python
from benchmarks.comparison_methods import sparcc_py


def test_single_base_score_matches_existing_sparcc_baseline():
    rng = np.random.default_rng(11)
    counts = rng.integers(1, 200, size=(40, 12))

    expected = sparcc_py(counts)
    actual = single_base_score(counts).correlation

    np.testing.assert_allclose(actual, expected, atol=1e-10)
```

- [ ] **Step 6: Run equivalence test**

Run:

```bash
uv run pytest tests/test_single_base.py::test_single_base_score_matches_existing_sparcc_baseline -q
```

Expected: PASS.

- [ ] **Step 7: Commit dense base score**

```bash
git add src/falcon/single.py tests/test_single_base.py
git commit -m "feat: add SparCC-compatible dense base score"
```

## Task 3: Add Symmetric Top-K Candidate Screening

**Files:**
- Create: `src/falcon/types.py`
- Create: `src/falcon/screen.py`
- Create: `tests/test_screen.py`

- [ ] **Step 1: Write failing candidate-screen tests**

Create `tests/test_screen.py`:

```python
import numpy as np

from falcon.screen import edge_overlap, single_candidates


def test_single_candidates_returns_symmetric_union_without_self_edges():
    score = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.8, 0.1],
            [0.1, 0.8, 1.0, 0.7],
            [0.2, 0.1, 0.7, 1.0],
        ]
    )

    candidates = single_candidates(score, top_k=1)

    assert candidates.pairs.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert (candidates.pairs[:, 0] < candidates.pairs[:, 1]).all()


def test_single_candidates_grow_monotonically_with_budget():
    rng = np.random.default_rng(7)
    score = rng.uniform(-1.0, 1.0, size=(12, 12))
    score = (score + score.T) / 2.0
    np.fill_diagonal(score, 1.0)

    small = single_candidates(score, top_k=1)
    large = single_candidates(score, top_k=3)

    assert set(map(tuple, small.pairs)) <= set(map(tuple, large.pairs))


def test_edge_overlap_uses_jaccard_similarity():
    left = np.array([[0, 1], [1, 2]])
    right = np.array([[1, 2], [2, 3]])

    assert edge_overlap(left, right) == 1 / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_screen.py -q
```

Expected: FAIL during import because `falcon.screen` does not exist.

- [ ] **Step 3: Implement candidate type and symmetric top-k union**

Create `src/falcon/types.py`:

```python
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
```

Create `src/falcon/screen.py`:

```python
from __future__ import annotations

import numpy as np

from falcon.types import CandidateSet


def _canonical_pairs(rows: np.ndarray, cols: np.ndarray, p: int) -> np.ndarray:
    left = np.minimum(rows, cols).astype(np.int64)
    right = np.maximum(rows, cols).astype(np.int64)
    keys = left * np.int64(p) + right
    order = np.unique(keys, return_index=True)[1]
    return np.column_stack([left[order], right[order]])


def single_candidates(
    correlation: np.ndarray,
    *,
    top_k: int,
    min_abs_score: float | None = None,
) -> CandidateSet:
    p = correlation.shape[0]
    if correlation.shape != (p, p):
        raise ValueError("correlation must be square")
    if not 1 <= top_k < p:
        raise ValueError("top_k must lie in [1, p)")

    absolute = np.abs(correlation).copy()
    np.fill_diagonal(absolute, -np.inf)
    columns = np.argpartition(-absolute, top_k - 1, axis=1)[:, :top_k]
    rows = np.repeat(np.arange(p), top_k)
    pairs = _canonical_pairs(rows, columns.ravel(), p)

    if min_abs_score is not None:
        extra_rows, extra_cols = np.where(
            np.triu(absolute >= min_abs_score, k=1)
        )
        pairs = _canonical_pairs(
            np.concatenate([pairs[:, 0], extra_rows]),
            np.concatenate([pairs[:, 1], extra_cols]),
            p,
        )

    scores = correlation[pairs[:, 0], pairs[:, 1]]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return CandidateSet(
        pairs=pairs[order],
        scores=scores[order],
        top_k=top_k,
        n_features=p,
    )


def edge_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(map(tuple, np.asarray(left).tolist()))
    right_set = set(map(tuple, np.asarray(right).tolist()))
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)
```

- [ ] **Step 4: Run screen tests**

Run:

```bash
uv run pytest tests/test_screen.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit candidate screening**

```bash
git add src/falcon/types.py src/falcon/screen.py tests/test_screen.py
git commit -m "feat: add symmetric Falcon-SR candidate screening"
```

## Task 4: Add Strict Iterative Exclusion Reference

**Files:**
- Modify: `src/falcon/single.py`
- Create: `tests/test_single_refine.py`

- [ ] **Step 1: Write failing strict-reference test**

Create `tests/test_single_refine.py`:

```python
import numpy as np

from falcon.single import strict_refine_single


def test_strict_refine_single_excludes_at_most_one_new_pair_per_round():
    variation = np.array(
        [
            [0.0, 0.1, 1.7, 1.8],
            [0.1, 0.0, 1.6, 1.7],
            [1.7, 1.6, 0.0, 0.2],
            [1.8, 1.7, 0.2, 0.0],
        ]
    )

    result = strict_refine_single(
        variation,
        exclusion_threshold=0.1,
        max_exclusions=1,
    )

    assert result.excluded_pairs.shape == (1, 2)
    assert result.rounds == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_single_refine.py::test_strict_refine_single_excludes_at_most_one_new_pair_per_round -q
```

Expected: FAIL because `strict_refine_single` does not exist.

- [ ] **Step 3: Implement strict dense exclusion reference**

Append to `src/falcon/single.py`:

```python
@dataclass(frozen=True)
class StrictRefinementResult:
    correlation: np.ndarray
    basis_variance: np.ndarray
    excluded_pairs: np.ndarray
    rounds: int


def strict_refine_single(
    variation: np.ndarray,
    *,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
) -> StrictRefinementResult:
    excluded: list[tuple[int, int]] = []
    p = variation.shape[0]
    for _ in range(max_exclusions):
        excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
        basis_variance = solve_basis_variance_dense(
            variation,
            excluded=excluded_array,
        )
        correlation = correlations_from_basis(variation, basis_variance)
        absolute = np.abs(correlation)
        np.fill_diagonal(absolute, -np.inf)
        for i, j in excluded:
            absolute[i, j] = -np.inf
            absolute[j, i] = -np.inf
        flat_index = int(np.argmax(absolute))
        i, j = np.unravel_index(flat_index, absolute.shape)
        if absolute[i, j] <= exclusion_threshold:
            break
        excluded.append((min(i, j), max(i, j)))

    excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    basis_variance = solve_basis_variance_dense(
        variation,
        excluded=excluded_array,
    )
    correlation = correlations_from_basis(variation, basis_variance)
    return StrictRefinementResult(
        correlation=correlation,
        basis_variance=basis_variance,
        excluded_pairs=excluded_array,
        rounds=len(excluded),
    )
```

- [ ] **Step 4: Run strict-reference test**

Run:

```bash
uv run pytest tests/test_single_refine.py::test_strict_refine_single_excludes_at_most_one_new_pair_per_round -q
```

Expected: PASS.

- [ ] **Step 5: Add strict-reference fixed-point test**

Append to `tests/test_single_refine.py`:

```python
def test_strict_refine_single_stops_when_threshold_is_not_crossed():
    variation = np.full((5, 5), 2.0)
    np.fill_diagonal(variation, 0.0)

    result = strict_refine_single(
        variation,
        exclusion_threshold=0.2,
        max_exclusions=10,
    )

    assert result.rounds == 0
    assert result.excluded_pairs.shape == (0, 2)
```

- [ ] **Step 6: Run strict-reference tests**

Run:

```bash
uv run pytest tests/test_single_refine.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit strict reference**

```bash
git add src/falcon/single.py tests/test_single_refine.py
git commit -m "feat: add strict SparCC exclusion reference"
```

## Task 5: Add Sparse Candidate Refinement

**Files:**
- Modify: `src/falcon/single.py`
- Modify: `tests/test_single_refine.py`

- [ ] **Step 1: Write failing sparse-solve equivalence test**

Append to `tests/test_single_refine.py`:

```python
from falcon.single import (
    solve_basis_variance_dense,
    solve_basis_variance_sparse,
    sparse_refine_single,
)
from falcon.types import CandidateSet


def test_sparse_basis_solve_matches_dense_excluded_solve():
    rng = np.random.default_rng(17)
    raw = rng.uniform(0.1, 2.0, size=(12, 12))
    variation = (raw + raw.T) / 2.0
    np.fill_diagonal(variation, 0.0)
    excluded = np.array([[0, 1], [2, 3], [3, 4], [8, 11]])

    expected = solve_basis_variance_dense(variation, excluded=excluded)
    actual = solve_basis_variance_sparse(variation, excluded=excluded)

    np.testing.assert_allclose(actual, expected, atol=1e-8)
```

- [ ] **Step 2: Run sparse-solve test to verify it fails**

Run:

```bash
uv run pytest tests/test_single_refine.py::test_sparse_basis_solve_matches_dense_excluded_solve -q
```

Expected: FAIL because `solve_basis_variance_sparse` does not exist.

- [ ] **Step 3: Implement sparse excluded-edge solve**

Add imports near the top of `src/falcon/single.py`:

```python
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
```

Append:

```python
def solve_basis_variance_sparse(
    variation: np.ndarray,
    *,
    excluded: np.ndarray,
    min_variance: float = 1e-4,
) -> np.ndarray:
    p = variation.shape[0]
    excluded = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    rhs = variation.sum(axis=1).copy()
    if excluded.size:
        rhs -= np.bincount(
            np.concatenate([excluded[:, 0], excluded[:, 1]]),
            weights=np.concatenate(
                [
                    variation[excluded[:, 0], excluded[:, 1]],
                    variation[excluded[:, 0], excluded[:, 1]],
                ]
            ),
            minlength=p,
        )
    degree = np.bincount(excluded.ravel(), minlength=p).astype(np.float64)
    if excluded.size:
        rows = np.concatenate([excluded[:, 0], excluded[:, 1]])
        cols = np.concatenate([excluded[:, 1], excluded[:, 0]])
        adjacency = sparse.csr_matrix(
            (np.ones(rows.size), (rows, cols)),
            shape=(p, p),
        )
    else:
        adjacency = sparse.csr_matrix((p, p))

    def matvec(vector: np.ndarray) -> np.ndarray:
        return vector.sum() + (p - 2.0 - degree) * vector - adjacency @ vector

    operator = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    solution, info = cg(operator, rhs, rtol=1e-10, atol=1e-12, maxiter=10 * p)
    if info != 0:
        raise RuntimeError(f"sparse basis solve did not converge: info={info}")
    return np.maximum(solution, min_variance)
```

- [ ] **Step 4: Run sparse-solve equivalence test**

Run:

```bash
uv run pytest tests/test_single_refine.py::test_sparse_basis_solve_matches_dense_excluded_solve -q
```

Expected: PASS.

- [ ] **Step 5: Write failing candidate-refinement equivalence test**

Append to `tests/test_single_refine.py`:

```python
def test_sparse_refinement_matches_strict_when_candidates_cover_all_pairs():
    rng = np.random.default_rng(23)
    raw = rng.uniform(0.1, 2.0, size=(10, 10))
    variation = (raw + raw.T) / 2.0
    np.fill_diagonal(variation, 0.0)
    rows, cols = np.triu_indices(10, k=1)
    candidates = CandidateSet(
        pairs=np.column_stack([rows, cols]),
        scores=np.zeros(rows.size),
        top_k=9,
        n_features=10,
    )

    strict = strict_refine_single(
        variation,
        exclusion_threshold=0.1,
        max_exclusions=6,
    )
    sparse_result = sparse_refine_single(
        variation,
        candidates,
        exclusion_threshold=0.1,
        max_exclusions=6,
    )

    np.testing.assert_array_equal(
        sparse_result.excluded_pairs,
        strict.excluded_pairs,
    )
    np.testing.assert_allclose(
        sparse_result.basis_variance,
        strict.basis_variance,
        atol=1e-8,
    )
```

- [ ] **Step 6: Run candidate-refinement test to verify it fails**

Run:

```bash
uv run pytest tests/test_single_refine.py::test_sparse_refinement_matches_strict_when_candidates_cover_all_pairs -q
```

Expected: FAIL because `sparse_refine_single` does not exist.

- [ ] **Step 7: Implement sparse candidate refinement**

Append to `src/falcon/single.py`:

```python
@dataclass(frozen=True)
class SparseRefinementResult:
    pairs: np.ndarray
    scores: np.ndarray
    basis_variance: np.ndarray
    excluded_pairs: np.ndarray
    rounds: int


def _candidate_scores(
    variation: np.ndarray,
    basis_variance: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    left = pairs[:, 0]
    right = pairs[:, 1]
    covariance = 0.5 * (
        basis_variance[left] + basis_variance[right] - variation[left, right]
    )
    return np.clip(
        covariance / np.sqrt(basis_variance[left] * basis_variance[right]),
        -1.0,
        1.0,
    )


def sparse_refine_single(
    variation: np.ndarray,
    candidates,
    *,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
) -> SparseRefinementResult:
    excluded: list[tuple[int, int]] = []
    excluded_indices: set[int] = set()
    for _ in range(max_exclusions):
        excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
        basis_variance = solve_basis_variance_sparse(
            variation,
            excluded=excluded_array,
        )
        scores = _candidate_scores(variation, basis_variance, candidates.pairs)
        available = np.ones(scores.size, dtype=bool)
        if excluded_indices:
            available[list(excluded_indices)] = False
        masked = np.where(available, np.abs(scores), -np.inf)
        edge_index = int(np.argmax(masked))
        if masked[edge_index] <= exclusion_threshold:
            break
        excluded_indices.add(edge_index)
        excluded.append(tuple(candidates.pairs[edge_index]))

    excluded_array = np.asarray(excluded, dtype=np.int64).reshape(-1, 2)
    basis_variance = solve_basis_variance_sparse(
        variation,
        excluded=excluded_array,
    )
    scores = _candidate_scores(variation, basis_variance, candidates.pairs)
    return SparseRefinementResult(
        pairs=candidates.pairs,
        scores=scores,
        basis_variance=basis_variance,
        excluded_pairs=excluded_array,
        rounds=len(excluded),
    )
```

- [ ] **Step 8: Run refinement tests**

Run:

```bash
uv run pytest tests/test_single_refine.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit sparse refinement**

```bash
git add src/falcon/single.py tests/test_single_refine.py
git commit -m "feat: refine Falcon-SR candidates with sparse solves"
```

## Task 6: Add Adaptive Growth And Public Single-Domain API

**Files:**
- Modify: `src/falcon/types.py`
- Modify: `src/falcon/single.py`
- Modify: `src/falcon/__init__.py`
- Create: `tests/test_single_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_single_api.py`:

```python
import numpy as np

from falcon import infer_single


def test_infer_single_fast_returns_unique_sparse_edges_and_diagnostics():
    rng = np.random.default_rng(29)
    counts = rng.integers(1, 300, size=(80, 24))

    result = infer_single(
        counts,
        mode="fast",
        top_k=2,
        max_top_k=4,
        stability_threshold=0.0,
    )

    assert result.edges.pairs.shape[1] == 2
    assert (result.edges.pairs[:, 0] < result.edges.pairs[:, 1]).all()
    assert result.diagnostics.initial_top_k == 2
    assert result.diagnostics.final_top_k in {2, 4}
    assert result.initial_matrix is None


def test_infer_single_strict_returns_dense_matrix():
    rng = np.random.default_rng(31)
    counts = rng.integers(1, 300, size=(80, 12))

    result = infer_single(counts, mode="strict", max_exclusions=3)

    assert result.initial_matrix.shape == (12, 12)
    np.testing.assert_allclose(result.initial_matrix, result.initial_matrix.T)
```

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
uv run pytest tests/test_single_api.py -q
```

Expected: FAIL because `infer_single` is not exported.

- [ ] **Step 3: Add result containers**

Append to `src/falcon/types.py`:

```python
@dataclass(frozen=True)
class EdgeTable:
    pairs: np.ndarray
    scores: np.ndarray


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


@dataclass(frozen=True)
class NetworkResult:
    edges: EdgeTable
    diagnostics: ScreenDiagnostics
    initial_matrix: np.ndarray | None
```

- [ ] **Step 4: Implement adaptive fast mode and strict mode**

Add imports near the top of `src/falcon/single.py`:

```python
from falcon.screen import edge_overlap, single_candidates
from falcon.types import EdgeTable, NetworkResult, ScreenDiagnostics
```

Append:

```python
def _sign_stability(left, right) -> float:
    left_map = dict(zip(map(tuple, left.pairs), np.sign(left.scores)))
    right_map = dict(zip(map(tuple, right.pairs), np.sign(right.scores)))
    shared = left_map.keys() & right_map.keys()
    return 1.0 if not shared else float(
        np.mean([left_map[pair] == right_map[pair] for pair in shared])
    )


def _strong_edges(edges: EdgeTable, limit: int) -> EdgeTable:
    count = min(limit, edges.pairs.shape[0])
    indices = np.argsort(-np.abs(edges.scores))[:count]
    return EdgeTable(pairs=edges.pairs[indices], scores=edges.scores[indices])


def infer_single(
    counts: np.ndarray,
    *,
    mode: str = "fast",
    top_k: int = 50,
    max_top_k: int | None = None,
    min_abs_score: float | None = None,
    exclusion_threshold: float = 0.1,
    max_exclusions: int = 10,
    stability_threshold: float = 0.95,
) -> NetworkResult:
    base = single_base_score(counts)
    p = base.correlation.shape[0]
    if mode == "strict":
        strict = strict_refine_single(
            base.variation,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        rows, cols = np.triu_indices(p, k=1)
        pairs = np.column_stack([rows, cols])
        return NetworkResult(
            edges=EdgeTable(pairs=pairs, scores=strict.correlation[rows, cols]),
            diagnostics=ScreenDiagnostics(
                initial_top_k=p - 1,
                final_top_k=p - 1,
                candidate_count=pairs.shape[0],
                candidate_density=1.0,
                growth_rounds=0,
                overlap_across_budgets=1.0,
                sign_stability_across_budgets=1.0,
                fallback_reason=None,
            ),
            initial_matrix=strict.correlation,
        )
    if mode != "fast":
        raise ValueError("mode must be 'fast' or 'strict'")

    max_top_k = min(max_top_k or max(top_k, 2 * top_k), p - 1)
    budget = min(top_k, p - 1)
    previous = None
    growth_rounds = 0
    overlap = 1.0
    sign_stability = 1.0
    fallback_reason = None
    strong_edge_limit = p
    while True:
        candidates = single_candidates(
            base.correlation,
            top_k=budget,
            min_abs_score=min_abs_score,
        )
        refined = sparse_refine_single(
            base.variation,
            candidates,
            exclusion_threshold=exclusion_threshold,
            max_exclusions=max_exclusions,
        )
        edges = EdgeTable(pairs=refined.pairs, scores=refined.scores)
        if previous is not None:
            previous_strong = _strong_edges(previous, strong_edge_limit)
            current_strong = _strong_edges(edges, strong_edge_limit)
            overlap = edge_overlap(previous_strong.pairs, current_strong.pairs)
            sign_stability = _sign_stability(previous_strong, current_strong)
            if (
                overlap >= stability_threshold
                and sign_stability >= stability_threshold
            ):
                break
        if budget >= max_top_k:
            fallback_reason = "candidate budget reached before stability"
            break
        previous = edges
        budget = min(2 * budget, max_top_k)
        growth_rounds += 1

    return NetworkResult(
        edges=edges,
        diagnostics=ScreenDiagnostics(
            initial_top_k=min(top_k, p - 1),
            final_top_k=budget,
            candidate_count=edges.pairs.shape[0],
            candidate_density=candidates.density,
            growth_rounds=growth_rounds,
            overlap_across_budgets=overlap,
            sign_stability_across_budgets=sign_stability,
            fallback_reason=fallback_reason,
        ),
        initial_matrix=None,
    )
```

- [ ] **Step 5: Export the experimental API without removing legacy exports**

Append to `src/falcon/__init__.py`:

```python
# Experimental Falcon-SR API. Legacy prototype functions remain exported while
# the screen-refine research gates are being evaluated.
from falcon.single import infer_single
```

- [ ] **Step 6: Run API tests**

Run:

```bash
uv run pytest tests/test_single_api.py -q
```

Expected: PASS.

- [ ] **Step 7: Run the complete test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit public API**

```bash
git add src/falcon/types.py src/falcon/single.py src/falcon/__init__.py tests/test_single_api.py
git commit -m "feat: add adaptive Falcon-SR single-domain API"
```

## Task 7: Add The Single-Domain Research-Gate Benchmark

**Files:**
- Modify: `benchmarks/io_utils.py`
- Create: `benchmarks/single_feasibility.py`
- Create: `tests/test_single_benchmark.py`

- [ ] **Step 1: Write failing benchmark smoke test**

Create `tests/test_single_benchmark.py`:

```python
from benchmarks.single_feasibility import run_cell


def test_single_feasibility_cell_reports_research_gate_metrics():
    rows = run_cell(n=80, p=40, density=0.03, top_k=4, reps=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 80
    assert row["p"] == 40
    assert 0.0 <= row["candidate_recall"] <= 1.0
    assert 0.0 <= row["edge_overlap"] <= 1.0
    assert 0.0 <= row["sign_accuracy"] <= 1.0
    assert row["base_seconds"] >= 0.0
    assert row["screen_seconds"] >= 0.0
    assert row["fast_refine_seconds"] >= 0.0
    assert row["strict_refine_seconds"] >= 0.0
    assert row["fast_seconds"] >= 0.0
    assert row["strict_seconds"] >= 0.0
```

- [ ] **Step 2: Run benchmark smoke test to verify it fails**

Run:

```bash
uv run pytest tests/test_single_benchmark.py -q
```

Expected: FAIL because `benchmarks.single_feasibility` does not exist.

- [ ] **Step 3: Add the feasibility CSV schema**

Add to `COLUMNS` in `benchmarks/io_utils.py`:

```python
    "falcon_sr_single_feasibility": [
        "replicate", "n", "p", "density", "top_k",
        "candidate_count", "candidate_recall",
        "edge_overlap", "sign_accuracy",
        "base_seconds", "screen_seconds",
        "fast_refine_seconds", "strict_refine_seconds",
        "fast_seconds", "strict_seconds",
        "base_peak_allocated_bytes", "screen_peak_allocated_bytes",
        "fast_refine_peak_allocated_bytes", "strict_refine_peak_allocated_bytes",
    ],
```

- [ ] **Step 4: Implement the feasibility benchmark**

Create `benchmarks/single_feasibility.py`:

```python
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from io_utils import write_table
from run_on_server import generate_basis_correlation, generate_single_domain
from falcon.screen import edge_overlap, single_candidates
from falcon.single import (
    single_base_score,
    sparse_refine_single,
    strict_refine_single,
)


def _timed_peak(function):
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak


def _pair_set(pairs):
    return set(map(tuple, pairs.tolist()))


def _sign_accuracy(reference_pairs, reference_scores, actual_pairs, actual_scores):
    reference = dict(zip(map(tuple, reference_pairs.tolist()), np.sign(reference_scores)))
    actual = dict(zip(map(tuple, actual_pairs.tolist()), np.sign(actual_scores)))
    shared = reference.keys() & actual.keys()
    return 1.0 if not shared else float(
        np.mean([reference[pair] == actual[pair] for pair in shared])
    )


def run_cell(*, n, p, density, top_k, reps):
    rows = []
    for replicate in range(reps):
        rng = np.random.default_rng(1000 + replicate + p)
        sigma, planted = generate_basis_correlation(
            rng,
            p,
            density,
            effect_lo=0.35,
            effect_hi=0.65,
        )
        counts = generate_single_domain(rng, n, p, sigma)

        base, base_seconds, base_peak = _timed_peak(
            lambda: single_base_score(counts)
        )
        candidates, screen_seconds, screen_peak = _timed_peak(
            lambda: single_candidates(base.correlation, top_k=top_k)
        )
        fast, fast_refine_seconds, fast_refine_peak = _timed_peak(
            lambda: sparse_refine_single(
                base.variation,
                candidates,
                max_exclusions=10,
            )
        )
        strict, strict_refine_seconds, strict_refine_peak = _timed_peak(
            lambda: strict_refine_single(
                base.variation,
                max_exclusions=10,
            )
        )
        fast_seconds = base_seconds + screen_seconds + fast_refine_seconds
        strict_seconds = base_seconds + strict_refine_seconds

        planted_pairs = {
            (min(i, j), max(i, j))
            for i, j, _ in planted
        }
        candidate_pairs = _pair_set(candidates.pairs)
        strict_rows, strict_cols = np.triu_indices(p, k=1)
        strict_pairs = np.column_stack([strict_rows, strict_cols])
        strict_scores = strict.correlation[strict_rows, strict_cols]
        strict_order = np.argsort(-np.abs(strict_scores))
        strong_count = max(1, len(planted_pairs))
        strong_index = strict_order[:strong_count]
        reference_pairs = strict_pairs[strong_index]
        reference_scores = strict_scores[strong_index]
        fast_pairs = fast.pairs
        fast_scores = fast.scores
        rows.append(
            {
                "replicate": replicate,
                "n": n,
                "p": p,
                "density": density,
                "top_k": top_k,
                "candidate_count": candidates.pairs.shape[0],
                "candidate_recall": len(candidate_pairs & _pair_set(reference_pairs))
                / strong_count,
                "edge_overlap": edge_overlap(
                    reference_pairs,
                    fast_pairs[np.argsort(-np.abs(fast_scores))[:strong_count]],
                ),
                "sign_accuracy": _sign_accuracy(
                    reference_pairs,
                    reference_scores,
                    fast_pairs,
                    fast_scores,
                ),
                "base_seconds": base_seconds,
                "screen_seconds": screen_seconds,
                "fast_refine_seconds": fast_refine_seconds,
                "strict_refine_seconds": strict_refine_seconds,
                "fast_seconds": fast_seconds,
                "strict_seconds": strict_seconds,
                "base_peak_allocated_bytes": base_peak,
                "screen_peak_allocated_bytes": screen_peak,
                "fast_refine_peak_allocated_bytes": fast_refine_peak,
                "strict_refine_peak_allocated_bytes": strict_refine_peak,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--p", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--reps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = []
    for n in args.n:
        for p in args.p:
            for top_k in args.top_k:
                rows.extend(
                    run_cell(
                        n=n,
                        p=p,
                        density=args.density,
                        top_k=top_k,
                        reps=args.reps,
                    )
                )
    path = write_table("falcon_sr_single_feasibility", rows)
    print(path)
```

- [ ] **Step 5: Run benchmark smoke test**

Run:

```bash
uv run pytest tests/test_single_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Run a small feasibility benchmark**

Run:

```bash
uv run python benchmarks/single_feasibility.py \
  --n 100 250 \
  --p 100 500 \
  --top-k 10 25 \
  --reps 2
```

Expected: creates `data/falcon_sr_single_feasibility.csv` with 16 raw
replicate rows.

- [ ] **Step 7: Inspect research-gate metrics**

Run:

```bash
uv run python - <<'PY'
from benchmarks.io_utils import read_table

rows = read_table("falcon_sr_single_feasibility")
for row in rows:
    print(
        f"n={int(row['n'])} p={int(row['p'])} k={int(row['top_k'])} "
        f"candidate_recall={row['candidate_recall']:.3f} "
        f"overlap={row['edge_overlap']:.3f} "
        f"sign={row['sign_accuracy']:.3f} "
        f"fast={row['fast_seconds']:.4f}s strict={row['strict_seconds']:.4f}s"
    )
PY
```

Expected: each row prints measured candidate recall, overlap, sign accuracy,
and same-host times. Do not claim feasibility success unless candidate recall
is at least `0.99`, overlap is at least `0.95`, and sign accuracy is at least
`0.95` on the primary cells.

- [ ] **Step 8: Commit benchmark tooling but decide separately whether to commit generated data**

```bash
git add benchmarks/io_utils.py benchmarks/single_feasibility.py tests/test_single_benchmark.py
git commit -m "bench: add Falcon-SR single-domain feasibility gate"
```

Keep `data/falcon_sr_single_feasibility.csv` unstaged until its measured rows
have been reviewed.

## Task 8: Document Experimental API And Run Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add an experimental Falcon-SR section after the existing Quickstart usage block**

Add:

````markdown
### Experimental: Falcon-SR single-domain feasibility API

`Falcon-SR` estimates the same latent log-abundance Pearson-correlation target
as SparCC. Its experimental fast path computes one dense SparCC-compatible base
score, screens a sparse top-k candidate union, and refines candidates with a
sparse excluded-edge solve.

```python
from falcon import infer_single

result = infer_single(counts, mode="fast", top_k=50)
edge_pairs = result.edges.pairs
edge_scores = result.edges.scores
diagnostics = result.diagnostics
```

Use `mode="strict"` for small reference runs. The Falcon-SR API is experimental
until the committed feasibility benchmarks satisfy the candidate-recall,
strong-edge-overlap, and sign-accuracy gates in the design specification.
````

- [ ] **Step 2: Run the complete unit and smoke suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run import smoke test**

Run:

```bash
uv run python - <<'PY'
import numpy as np
from falcon import infer_single

counts = np.random.default_rng(41).integers(1, 100, size=(40, 12))
result = infer_single(counts, mode="fast", top_k=2, max_top_k=4)
print(result.edges.pairs.shape)
print(result.diagnostics)
PY
```

Expected: prints an edge-table shape and diagnostics without warnings or
tracebacks.

- [ ] **Step 4: Review the worktree**

Run:

```bash
git status --short
git diff --check
```

Expected: only reviewed generated benchmark data may remain unstaged;
`git diff --check` prints no whitespace errors.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe experimental Falcon-SR API"
```

## Final Research-Gate Review

After Task 8, stop before cross-domain implementation and report:

1. Unit and smoke-test results.
2. Candidate recall, edge overlap, and sign accuracy from the small feasibility
   grid.
3. Ranking-only wall-clock and Python-tracked peak-allocation measurements.
4. Whether sparse refinement is faster than strict refinement after the shared
   dense base score.
5. Any cells that trigger adaptive-growth warnings.
6. Whether the research hypothesis is supported strongly enough to write the
   SparXCC-compatible cross-domain implementation plan.

Do not update manuscript claims or figures until this review passes.
