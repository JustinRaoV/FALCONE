# Decision Log — Single-Domain Estimator Rebuild

This log records design decisions taken during the 2026-06-02 rebuild. The
authoritative design is in
[`superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md`](superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md).
The pre-rebuild log was deleted with the screen-refine implementation; the
git history retains the historical entries.

## 2026-06-02 — Rebuild scope

**Decision.** Drop the screen-refine architecture and rebuild the package
around a single-domain compositional network estimator with three Python-only
candidates: `adaptive_threshold`, `weighted_sparse`, and `pd_sparse`.
Cross-domain inference and signed biological priors are explicitly deferred.

**Why.** The committed feasibility tables exposed an architectural
mismatch: the screen-refine fast path computed a full dense SparCC-compatible
base matrix before screening, so the screen and sparse-refine stages added
work after the quadratic bottleneck rather than removing it. The hard cells
(`n=100, p=1000`) were 6–25× slower than the SparCC closed-form baseline at
worse candidate recall.

**Where it shows up.** `manuscript/`, `data/falcon_sr_*_feasibility.csv`,
`benchmarks/{falcon_sr_single,falcon_sr_cross,comparison_methods}.py`, and the
old `src/falcon/{single,cross,prior,screen,calibration,types}.py` modules are
removed. The previous design and execution-design specs are removed because
they describe an algorithm we no longer ship; the new design lives in
`docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md`.

## 2026-06-02 — Three estimator candidates behind one interface

**Decision.** All three candidates share the `infer_network` entrypoint and
return the same `NetworkResult`. The selected production estimator is chosen
on the training grid only, and is frozen before the holdout grid is touched.

**Why.** A single public surface keeps the call sites stable across the
acceptance evaluation. Different candidate paths sharing one schema avoid
schema drift between training and holdout reports.

## 2026-06-02 — Stability selection is the primary uncertainty output

**Decision.** `selection_probability` (subsample frequency of non-zero
support) is the default uncertainty output. `pvalue_approx` and
`qvalue_approx` stay `None` unless a calibration procedure whose simulation
FDR behaviour has been measured fills them in. The
`uncertainty_interpretation` field on `EstimatorDiagnostics` names the regime
explicitly.

**Why.** The previous architecture exposed permutation p-values that were
labeled `permutation_base_only` to flag that calibration was approximate.
That is too easy to forget. Selection probability under a fixed seed is
bit-reproducible and has a clear interpretation; q-values come back only
when we have measured FDR control on holdout cells.

## 2026-06-02 — Diagonal loading for PD correction

**Decision.** The `pd_sparse` candidate uses diagonal loading
(`Sigma += (floor - min_eig) * I` when `min_eig < floor`) rather than
Higham's nearest-PD projection.

**Why.** Higham's projection touches every off-diagonal entry and silently
destroys the support recovered by adaptive thresholding. Diagonal loading
preserves the support exactly and is idempotent under a fixed floor.

## 2026-06-02 — R baselines run only in benchmark adapters

**Decision.** `fastCCLasso`, `COAT`, and `SECOM` are invoked through
subprocess adapters in `benchmarks/r_adapters.py`. The production package
does not import or invoke R. When R or the named package is missing the
adapter returns an `RAdapterSkip` with an explicit reason; the benchmark
records a "skipped" row instead of a fake numeric result.

**Why.** Vendoring or wrapping LGPL R code at runtime in the production
package would impose licence constraints far beyond the benchmark scope.
Subprocess adapters keep the licence boundary clean and let the benchmark
runner produce schema-valid rows even when R is not installed.

## 2026-06-02 — Frozen schema with `estimand_family` labeling

**Decision.** Every benchmark row records `estimand_family`. Methods with
adjacent estimands (precision matrix, nonlinear dependence) are tagged so
they cannot be silently promoted into match-evidence for an advantage claim.

**Why.** Comparing matched-estimand and adjacent-estimand methods on the
same axis is the most common way honest-looking benchmark rows mislead
reviewers. The label makes the comparison rule explicit at row level.

## 2026-06-02 — Acceptance gates are required before any advantage claim

**Decision.** The repository must report `not yet evaluated` for the six
acceptance gates listed in design §14 until the holdout grid is run and
reviewed. Negative results remain valid outputs; the rebuild does not assume
victory.

**Why.** This is the constraint that drove the rebuild in the first place.
The previous implementation made performance claims its own feasibility
tables contradicted. Naming the gates and refusing to remove the
"not yet evaluated" status until the holdout grid is in is the cheapest
forcing function to keep that from recurring.

## 2026-06-03 — Production estimator frozen as `weighted_sparse` (training only)

**Decision.** The production estimator selected from `infer_network` for
the holdout evaluation is `weighted_sparse`. Locked on training-grid
evidence only. The holdout grid has not been touched.

**Why.** From `data/bench_training_local.csv` (39 cells × 3 reps × 5
methods, n_resamples=30):

| method                       | AUROC          | AP             | wall (med, s) |
|------------------------------|----------------|----------------|---------------|
| `falcon_weighted_sparse`     | 0.897 ± 0.133  | 0.646 ± 0.308  | 0.300         |
| `pearson_clr`                | 0.895 ± 0.131  | 0.625 ± 0.297  | 0.0003        |
| `sparcc_closed_form`         | 0.895 ± 0.131  | 0.624 ± 0.296  | 0.0004        |
| `falcon_adaptive_threshold`  | 0.609 ± 0.169  | 0.263 ± 0.321  | 0.041         |
| `falcon_pd_sparse`           | 0.609 ± 0.169  | 0.263 ± 0.321  | 0.055         |

`weighted_sparse` is the only candidate that ranks at or above every
matched-estimand baseline on every scenario. The largest gap to SparCC
is on the `hub` scenario (AP 0.874 vs 0.748).

**Risks carried into holdout.**

1. Average AUROC delta vs SparCC is 0.002 — within 1 sigma. Holdout may
   not reproduce the win.
2. Wallclock is ~750× the SparCC closed form at the training cell sizes
   (`p ≤ 100`). The screen-refine architecture was removed precisely
   because it lost on runtime; the rebuild risks repeating that on the
   holdout if the median wall does not fall as `p` grows. Acceptance
   gate 3 (runtime/memory improvement) is the most likely to fail.
3. Convergence: 0/117 non-converged training runs. If holdout cells with
   `p ∈ {500, 1000}` produce non-convergence, the diagnostic must be
   surfaced rather than silently retried.

**How to apply.** The holdout runner uses
`--methods falcon_weighted_sparse,sparcc_closed_form,pearson_clr,fastCCLasso,COAT,SECOM`
and treats `weighted_sparse` as the production estimator. The other two
falcon candidates ship as part of the public API but are reported as
context, not primary evidence.

## 2026-06-03 — Default `zero_policy` frozen as `multiplicative`

**Decision.** The default `zero_policy` is `multiplicative`. The other
two policies (`pseudocount`, `complete_case`) stay exposed as the
documented sensitivity axis.

**Why.** From `data/zero_policy_sensitivity_local.csv` (15 cells across
sparse_random, heavy_tailed, negative_binomial_zi):

| zero regime | best policy        | second        | gap    |
|-------------|--------------------|---------------|--------|
| zf < 0.05   | `multiplicative`   | `pseudocount` | < 0.01 |
| zf ~ 0.13   | `multiplicative`   | `pseudocount` | < 0.01 |
| zf > 0.20   | `complete_case` *  | mult/pseudo   | +0.15  |

(*) On `negative_binomial_zi` (zf ≈ 0.21–0.25), `complete_case` raised
AUROC from 0.65 to 0.81. This is a real signal — high zero-fraction data
should be re-run with `complete_case`. But it is not the right default:
on lower zero-fraction data, `complete_case` *underperforms* slightly
(`heavy_tailed`: 0.891 vs 0.968).

`multiplicative` is the safe default; the README and methodology note
that high-zero datasets should be re-run with `complete_case` and the
choice recorded per study. The benchmark runner records `zero_policy`
per row so the choice is never silent.

**How to apply.** `infer_network` default stays `zero_policy="multiplicative"`.
Any future advantage claim that depends on a non-default policy must
report the policy choice explicitly.

## 2026-06-03 — `adaptive_threshold` and `pd_sparse` retained but demoted

**Decision.** Both estimators stay in the public API. They are no longer
candidates for the production estimator on the training grid.

**Why.**

* `adaptive_threshold` (hard mode, `threshold_constant=2.0`) zeros out
  too many true edges at training cell sizes (`p ∈ {50, 100}`). AUROC
  ties many real and noise pairs at zero. The mode/constant could be
  retuned, but the design rule is to freeze the production estimator on
  training-only evidence and `weighted_sparse` already wins every
  scenario at the current training defaults. Re-tuning thresholds and
  re-running risks polluting the holdout's statistical guarantee.
* `pd_sparse` produces identical AUROC/AP to `adaptive_threshold` (its
  diagonal-loading PD correction does not change off-diagonal ranking).
  It still has a niche when the downstream consumer needs a PD
  covariance — e.g., for log-likelihood evaluation. We keep the entry
  point but document that it does not improve edge ranking.

**How to apply.** README and methodology describe the three candidates
honestly: `weighted_sparse` is the production default, the other two
are auxiliary modes with documented use cases. The acceptance-gate
report names `weighted_sparse` as the candidate under evaluation.

## Open questions

* Whether `weighted_sparse` clears acceptance gate 3 (runtime / peak
  memory improvement vs the strongest accurate baseline) at the
  holdout cell sizes (`p ∈ {200, 500, 1000}`). Training data shows
  ~750× slower than SparCC at `p=100`; this must reverse as `p` grows
  or the gate fails.
* Whether the small AUROC delta vs SparCC (~0.002) survives the
  holdout's larger cells.
* Approximate q-values. Exposed only if observed simulation FDR is
  defensible across holdout scenarios; otherwise these stay `None`.
* Whether to retune `adaptive_threshold` for a future release. Possible
  improvements: lower `threshold_constant`, switch to soft mode by
  default. Out of scope for this release because the production
  estimator is already chosen.
