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

## Open questions

* Default zero policy. `multiplicative`, `pseudocount`, and `complete_case`
  are exposed as a sensitivity axis. The default may be chosen only after
  the training grid records FDR and runtime under each policy.
* Default thresholding mode for `adaptive_threshold` (hard vs. soft). To be
  frozen on the training grid before any holdout cell is touched.
* Whether `pd_sparse` ships in the public API. Retained only if PD
  correction improves numerical reliability without losing accuracy or
  efficiency on the training grid.
* Approximate q-values. Exposed only if observed simulation FDR is
  defensible across holdout scenarios; otherwise these stay `None`.
