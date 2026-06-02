# Falcon-SR Rewrite Execution Design

Date: 2026-06-02

Status: approved by user instruction "carefully self-review, then execute"

## 0. Authority

The algorithmic design is fixed by
`docs/superpowers/specs/2026-06-01-falcon-sr-design.md` (the canonical spec).
This document captures the **execution / migration design** that the canonical
spec leaves underspecified, plus the few mathematical refinements needed to
make the spec implementable without ambiguity.

This rewrite supersedes the legacy FastProp / RandProp / CrossNet
implementation because that code estimates *proportionality* `rho_p`, a
different estimand from the latent log-abundance Pearson correlation that
Falcon-SR (and SparCC / SparXCC) target. Spec §3 forbids treating these as
synonyms.

## 1. Scope

In scope for this single execution pass:

1. Permutation calibration module (spec §11).
2. Cross-domain `cross.py` module (spec §5.2, §7.3, §9.2).
3. Optional signed prior `prior.py` module (spec §10).
4. Public APIs `infer_single` (extended) and `infer_cross` (new).
5. Feasibility-grade benchmark suite (single + cross), with head-to-head
   comparison against SparCC, SparXCC base, SparXCC iter, and Pearson(CLR).
6. Deletion of all legacy FastProp / RandProp / CrossNet code and any
   benchmark / data / manuscript material whose meaning depended on the
   legacy estimand.
7. Refreshed `docs/methodology.md`, `docs/decision-log.md`, `README.md`,
   and a new minimal `manuscript/main.tex` skeleton plus a Python figure
   script reading benchmark CSV.

Out of scope:

1. Random-projection screening for `p > 5000` (spec §3 non-goal).
2. Selective-inference correction for candidate-only permutation
   (spec §19 risk — surfaced as diagnostic, not silently fixed).
3. Public real-data validation runs (spec §14 — left to a later plan).
4. Manuscript prose beyond the skeleton.

## 2. Mathematical Refinements

### 2.1 Cross-domain sparse refinement (spec §9.2 elaboration)

Spec §9.2 says cross-domain sparse refine "restricts updates to candidate
edges and affected row-column bias terms" but does not pin the
edge-exclusion geometry. We adopt **feature-level pruning driven by the
candidate edge set**, which preserves the `H_p ⊗ H_q^T` double-centering
identity and therefore reduces to SparXCC base/iter when candidates cover
all pairs.

Algorithm:

```
inputs:  cov_xy, alpha (sqrt(basis_var_X)), beta (sqrt(basis_var_Y)),
         candidates (set of (i, k) pairs), threshold tau, max_rounds R
state:   E  (set of excluded strong pairs, subset of candidates)

repeat at most R times:
    rho = sparxcc_centered(cov_xy, alpha, beta, exclude=E)
    rank candidate pairs by |rho|; pick top (i, k) not in E
    if |rho_{i,k}| <= tau: break
    add (i, k) to E

return rho, E
```

where `sparxcc_centered(cov_xy, alpha, beta, exclude=E)` is:

```
S = {i in [p] : no (i, k) in E}         # rows with no strong edge
T = {k in [q] : no (i, k) in E}         # cols with no strong edge
if |S| < 3 or |T| < 3:
    fall back to base centering (S = [p], T = [q]) and emit diagnostic
row_mean[i, :] = mean over j in S of cov_xy[j, :]
col_mean[:, k] = mean over l in T of cov_xy[:, l]
grand          = mean of cov_xy[S, T]
centered       = cov_xy - row_mean - col_mean + grand
return centered / outer(alpha, beta) clipped to [-1, 1]
```

Equivalence properties (tested):
- `E = empty set` -> equals `sparxcc_base`.
- Reusing the candidate set `{(i, k) : i in S^c or k in T^c}` produced by
  SparXCC iter's threshold gate, the centered matrix matches SparXCC iter
  up to clipping.

### 2.2 Prior penalty (spec §10 elaboration)

Spec §10 specifies the prior penalty
`lambda_prior * confidence * (rho - sign * target_magnitude)^2`
but leaves how it composes with the per-edge data fit unspecified.

We use a post-hoc analytic shrinkage: each refined candidate edge with a
prior entry receives

```
rho_combined = (rho_data + lambda_prior * confidence * sign * target_mag)
             / (1 + lambda_prior * confidence)
```

which is the closed-form minimizer of
`(rho - rho_data)^2 + lambda_prior * confidence * (rho - sign * target_mag)^2`.

This deliberately does **not** influence which edges sparse refine excludes,
matching spec §10 "soft direction, not an invented effect size". A prior on
an edge also forces it into the candidate set (spec §10 bullet 1).

When `prior_weight = 0` the formula reduces to `rho_data` exactly, so the
prior path is opt-in with zero penalty for users that ignore it.

### 2.3 Permutation calibration (spec §11 simplification)

Spec §11 requires permutation calibration of retained edges. Exact
calibration requires re-running base + refine per permutation, which on
`p = 1000, R = 100` exceeds the feasibility wall-clock budget.

For the feasibility gate we adopt a **base-only permutation null** and
mark it as approximate in diagnostics:

```
for each of R permutations (default R = 100):
    shuffle each column of the log composition matrix independently
    recompute base correlation (skips refinement)
    record max |rho| over the candidate edge set
empirical p-value(edge e) = (1 + #{perm : max_perm >= |rho_e^refined|}) / (1 + R)
empirical q-value via BH on candidate p-values
```

The cross-domain analogue shuffles rows of one of the two count matrices
relative to the other, breaking cross-sample alignment while preserving
within-domain marginals.

We do **not** claim the approximation is calibration-tight; the result
schema labels these columns `pvalue_approx`, `qvalue_approx` and the
diagnostics record `calibration = "permutation_base_only"`. Spec §19
research-risk 4 stays explicit.

## 3. Module Plan

| File | Action | Notes |
|---|---|---|
| `src/falcon/__init__.py` | rewrite | export only Falcon-SR symbols |
| `src/falcon/preprocessing.py` | keep | already shared by single/cross |
| `src/falcon/types.py` | extend | add `CrossEdgeTable`, `PriorEdge`, `CrossNetworkResult` |
| `src/falcon/screen.py` | extend | add `cross_candidates` (bidirectional top-k) |
| `src/falcon/single.py` | extend | accept `calibration`, `prior`, `seed`; integrate calibration |
| `src/falcon/cross.py` | new | dense base score, sparse refine, public `infer_cross` |
| `src/falcon/calibration.py` | new | column-permutation null, BH q-values |
| `src/falcon/prior.py` | new | `PriorEdge` validation, candidate injection, post-hoc shrinkage |
| `tests/test_preprocessing.py` | keep | already TDD-covered |
| `tests/test_screen.py` | extend | add cross-screen tests |
| `tests/test_single_base.py` | keep | already TDD-covered |
| `tests/test_single_refine.py` | keep | already TDD-covered |
| `tests/test_single_api.py` | extend | calibration smoke test |
| `tests/test_cross_base.py` | new | equivalence vs SparXCC base |
| `tests/test_cross_refine.py` | new | equivalence vs SparXCC iter when candidates full |
| `tests/test_calibration.py` | new | reproducibility, monotonicity |
| `tests/test_prior.py` | new | zero-weight neutrality, dominance, candidate injection |
| `tests/test_falcon_sr_benchmark.py` | new | smoke test of benchmark cells |
| `benchmarks/comparison_methods.py` | edit | inline CLR/multiplicative-replacement helpers, drop falcon import |
| `benchmarks/io_utils.py` | rewrite | new COLUMNS for `single_feasibility`, `cross_feasibility`, `head_to_head_single`, `head_to_head_cross` |
| `benchmarks/run_on_server.py` | delete | depends on legacy API throughout |
| `benchmarks/sim.py` | new | simulators for single + cross (reuse current `generate_basis_correlation`, `generate_single_domain`; add `generate_cross_domain`) |
| `benchmarks/falcon_sr_single.py` | new | single feasibility + head-to-head runner |
| `benchmarks/falcon_sr_cross.py` | new | cross feasibility + head-to-head runner |
| `benchmarks/run_all.sh` | rewrite | call the two new runners |
| `data/scalability.csv`, `data/detection.csv`, `data/cross_domain.csv`, `data/fdr_control.csv`, `data/method_comparison.csv` | delete | legacy-estimand outputs |
| `manuscript/main.tex`, `manuscript/main.pdf`, `manuscript/main.fls`, `manuscript/main.fdb_latexmk`, `manuscript/figures/*`, `manuscript/supplementary/*`, `manuscript/references.bib` | delete | start fresh |
| `manuscript/main.tex` | new skeleton | Abstract / Methods / Results / Discussion stubs referencing the new benchmarks |
| `manuscript/references.bib` | new | only papers cited by the new skeleton |
| `manuscript/figures/figure1.py` | new | reads benchmark CSV, writes SVG + PDF |
| `docs/methodology.md` | rewrite | describe Falcon-SR, drop FastProp/RandProp/CrossNet |
| `docs/decision-log.md` | rewrite | record estimand-change rationale, prior-shrinkage closed form, calibration approximation |
| `docs/superpowers/plans/2026-06-01-...md` | annotate | append "superseded by the 2026-06-02 plan" note |
| `README.md` | rewrite | quickstart for `infer_single` / `infer_cross` only |
| `pyproject.toml` | unchanged | dependencies already cover numpy/scipy/sklearn/pytest |

## 4. Test Strategy

Implementation follows TDD: every new module begins with failing tests, then
implementation, then a final commit when the suite is green.

Specific equivalence tests that gate correctness:

1. **Cross base = SparXCC base** at small `(p, q)` to within `1e-10`.
2. **Cross refine == SparXCC iter** when candidate set covers all pairs and
   threshold matches.
3. **`prior_weight = 0`** leaves the refined edge table identical to the
   non-prior call (byte-for-byte on the score array).
4. **`prior_weight = infinity`** drives prior edges to `sign * target_mag`
   regardless of data.
5. **Permutation null** with fixed seed reproduces identical `pvalue_approx`
   across two calls.
6. **Single-domain adaptive growth** still emits `fallback_reason` when the
   `2k` budget never crosses the stability threshold.

## 5. Benchmark Plan

### 5.1 Single-domain feasibility cells

```
n        in {100, 500}
p        in {100, 500, 1000}
top_k    in {10, 25, 50}
density  = 0.02
reps     = 3
methods  = falcon_sr_strict, falcon_sr_fast, falcon_sr_fast_calibrated,
           sparcc_py, pearson_clr
```

`falcon_sr_fast` runs with `calibration="none"` so its wall-clock is
comparable to SparCC and Pearson(CLR). `falcon_sr_fast_calibrated` runs
with `calibration="permutation"` (default `R = 100`) and records the
calibration overhead separately.

Output CSV: `data/falcon_sr_single_feasibility.csv`.

Reports per cell-method: candidate recall (vs strict-Falcon-SR reference),
edge overlap, sign accuracy, AUROC vs planted truth, Recall@K, wall-clock,
peak resident memory (via `tracemalloc`).

### 5.2 Cross-domain feasibility cells

```
n        in {100, 500}
(p, q)   in {(100, 100), (500, 500)}
top_k    in {10, 25}
density  = 0.01 (bipartite)
reps     = 3
methods  = falcon_sr_cross_fast, falcon_sr_cross_fast_calibrated,
           falcon_sr_cross_prior, sparxcc_base, sparxcc_iter,
           pearson_clr_cross
```

`falcon_sr_cross_fast` runs with `calibration="none"`; the
`_calibrated` variant records calibration overhead separately. The
`_prior` variant runs with `calibration="none"` plus a synthetic prior
that covers half the planted edges with the correct sign.

Output CSV: `data/falcon_sr_cross_feasibility.csv`.

Reports per cell-method: candidate recall, edge overlap, sign accuracy,
AUROC, Recall@K, wall-clock, peak memory.

### 5.3 Acceptance gates (spec §2 success criteria)

The benchmark **runs** in all cases. We report results faithfully. We claim
the research hypothesis succeeded only if:

- candidate recall >= 0.99 on the primary cells,
- edge overlap vs SparCC >= 0.95 in single-domain,
- edge overlap vs SparXCC iter >= 0.95 in cross-domain,
- sign accuracy >= 0.95.

If a gate fails the decision-log records the failure and the manuscript
skeleton flags the open issue. No silent metric inflation.

## 6. Migration Order

1. Add calibration module + tests; integrate into `infer_single`.
2. Add cross module + tests; build `infer_cross` API.
3. Add prior module + tests; integrate into both APIs.
4. Inline CLR/MR helpers in `comparison_methods.py`; remove falcon import.
5. New simulators (`benchmarks/sim.py`) and benchmark runners.
6. Smoke-run the two new benchmark runners on the smallest cells; commit
   schemas and a smoke output.
7. Delete legacy code from `src/falcon/__init__.py` and any function it
   re-exports; delete `benchmarks/run_on_server.py`; delete stale data
   CSVs; delete `manuscript/`.
8. Write new `manuscript/main.tex` skeleton, `figure1.py`,
   `references.bib`, refreshed `README.md`, `docs/methodology.md`,
   `docs/decision-log.md`.
9. Final feasibility benchmark run; commit CSVs and figures.
10. Verification: `uv run pytest -q` green, `uv run python -c "from falcon
    import infer_single, infer_cross"` succeeds, decision-log summarizes
    gates and any failures.

Each step commits independently with the existing prefix conventions
(`feat:`, `test:`, `bench:`, `docs:`, `chore: remove`).

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Cross sparse refine diverges from SparXCC base when `|S| < 3` or `|T| < 3` | Fall back to base centering and record `fallback_reason` in diagnostics |
| Permutation null too conservative under refinement inflation | Label as approximate; spec §19 risk stays open; record `calibration = "permutation_base_only"` |
| Prior shrinkage hides a wrong-sign data signal | Output records prior_weight, prior_target, prior_provenance, and a flag `data_disagreed_with_prior` |
| Legacy benchmark deletion accidentally drops still-useful baseline code | `comparison_methods.py` retains SparCC/SparXCC/SPIEC-EASI/Pearson; only `run_on_server.py` is removed |
| Manuscript skeleton becomes the source of unsupported claims | Skeleton sections explicitly mark "Results pending benchmark run YYYY-MM-DD" and reference the gate file |

## 8. Definition of Done

- All new and existing tests pass under `uv run pytest -q`.
- `from falcon import infer_single, infer_cross` works; nothing else is
  exported from the package.
- `benchmarks/falcon_sr_single.py` and `benchmarks/falcon_sr_cross.py`
  each run a tiny smoke grid and write a schema-valid CSV.
- `data/falcon_sr_single_feasibility.csv` and
  `data/falcon_sr_cross_feasibility.csv` exist for the configured cells.
- Legacy code, legacy data CSVs, and old manuscript are removed from the
  working tree and from git.
- `docs/methodology.md`, `docs/decision-log.md`, `README.md`, and
  `manuscript/main.tex` describe Falcon-SR only.
- Decision log explicitly records the estimand change, the prior closed
  form, and the calibration approximation.
