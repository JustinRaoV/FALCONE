# Method Optimization Design — Pre-Paper Push (v2)

Date: 2026-06-03 (rev 2)

Status: approved for implementation planning

Supersedes nothing in code. Builds on
[`2026-06-02-single-domain-estimator-rebuild-design.md`](2026-06-02-single-domain-estimator-rebuild-design.md).
The frozen-grid protocol and estimator candidate definitions from that
design remain in force.

## 0. Revision history

**v1 → v2** (2026-06-03). Self-review against academic and Nature-style
rigor produced 10 P0 clusters, of which 4 traced to spec assumptions
that were never measured. Two measurement scripts were written and run
under the brainstorming flow before any main-package code was touched
(`benchmarks/profile_stability.py`, `benchmarks/bench_gap_n100.py`,
commit `480e52f`). The measurements rewrote three sections:

* eigvalsh is **3.2 %** of wallclock at p=1000, not the 20–40 % v1
  claimed. The alternating loop's per-iteration NumPy broadcasts hold
  87.2 % of wallclock self-time. Line A is restructured around that.
* The true production-default gap to `sparcc_closed_form` at
  `n_resamples=100` is **2 826× – 11 618×** (median ≈ 4 800×), not the
  2 600–6 000× quoted from the `n_resamples=30` holdout. The Line A
  Primary target is reset to ≤ 500× SparCC.
* RSS-based memory ratio is 1.4×–2.8× SparCC (gate-3 PASS at ≤ 2×);
  tracemalloc-based ratio is 3.15×–3.40× (gate-3 FAIL). The choice of
  instrumentation flips the verdict. The Primary metric is switched to
  RSS, and the gate-3 work becomes "swap the instrumentation, validate"
  rather than "shrink the memory footprint".
* The Line B isotonic + BH approach was mathematically wrong (prior
  shift; `(1 − g)` is not a p-value; conservative-max of B-α and B-β is
  scale-incoherent). Replaced with a calibrated-posterior framing
  reported with Brier score + Expected Calibration Error. The
  procedure no longer claims FDR control.
* B-γ knockoff filter is dropped from the spec entirely (CLR features
  violate Gaussian-knockoff exchangeability assumptions; no FDR proof
  exists for the procedure on this data). B-α theoretical bound is
  reported as a separate diagnostic, not maxed against B-β.
* The Lin et al. (2022) SECOM "published network recall" Primary
  metric is demoted to descriptive context (multi-source flag:
  concordance, not validity).
* The escape hatch was tightened: any single unmet Primary triggers
  user consultation, not ≥ 2 as in v1.
* A new §13 names the academic-pipeline Stage 2.5 / 4.5 hand-off
  contract explicitly so the integrity gates cannot be silently
  skipped.

The v1 text is preserved in git history at commit `f077380`; the
measurement evidence sits at commit `480e52f`. The v2 file is the
current commit.

## 1. Objective

Push the rebuilt single-domain estimator — and the supporting
evaluation record — to a state that supports the trade-off paper
without overclaim. The 2026-06-03 holdout produced 2 of 6 acceptance
gates passing ([`docs/acceptance-gate-report.md`](../../acceptance-gate-report.md))
and the repository committed to a trade-off paper framing rather than
an advantage claim. The trade-off paper is materially stronger if the
following are true before writing begins:

1. The Gate 2 calibration is shipped as a calibrated posterior
   probability with reliability evaluation, replacing the current
   `selection_probability` reported without calibration.
2. The Gate 3 runtime gap to `sparcc_closed_form` shrinks to ≤ 500×
   at p ∈ {500, 1000} under the production default `n_resamples=100`.
3. The Gate 3 memory metric is reported in RSS rather than tracemalloc
   (the instrumentation change is itself the bulk of the work, since
   tracemalloc misses LAPACK workspaces).
4. The Gate 5 public-data subsampling stability is produced on at
   least two 16S resources with a reproducibility receipt.

No work in this design touches estimator accuracy on the existing
holdout. Re-tuning estimators on holdout cells destroys the frozen-grid
statistical guarantee that the eventual trade-off paper depends on
(decision-log 2026-06-03 "Open questions"). Gate 1 remains FAIL under
the strict reading and the paper acknowledges it.

## 2. Why optimize before writing

The trade-off paper's main claims are
([`docs/acceptance-gate-report.md`](../../acceptance-gate-report.md) §
"What the evaluation does establish"):

1. Versus same-class sparse baselines `cclasso` and `coat`,
   `weighted_sparse` improves edge ranking at 10–100× lower wallclock.
2. On hub-cluster data at p ≥ 500, `weighted_sparse` is the only tested
   method that recovers signal above near-random.
3. Sparse output and per-edge stability-based uncertainty are
   capabilities `sparcc_closed_form` does not provide.

Claim 3 is the weakest as currently shipped, because
`selection_probability` is reported without calibration. A
Bioinformatics reviewer is likely to read claim 3 as "they say
uncertainty but they do not actually calibrate it." Producing a
calibrated posterior is the clearest available strengthening
(comparative evidence: gate-2 PENDING vs gate-1/gate-3 FAIL/PENDING).

Claim 1's runtime margin is real against `cclasso` and `coat` but is
juxtaposed in the paper against a 4 800× median runtime loss to
`sparcc_closed_form` at production defaults. Bringing that to a
500× ceiling shifts the language from "alarming" to "acceptable
trade-off for the capabilities".

Public-data evidence (Gate 5) is the standard Bioinformatics
expectation for any network-inference method paper. A trade-off paper
without it is likely to receive predictable reviewer pushback.

## 3. Scope

### 3.1 In scope

| Line | Workstream |
|---|---|
| A | Speed and memory optimization of `weighted_sparse` and its stability-selection wrapper. Pure engineering. |
| B | Calibrate `selection_probability` → posterior `P̂(true | sel_prob)` (no FDR claim). Report Brier score + Expected Calibration Error on a holdout panel. |
| D | Public-data subsampling stability evaluation on SECOM v1.0.0 and HMP 16S. Concordance with Lin et al. (2022) SECOM network reported descriptively. |

### 3.2 Explicitly out of scope

| Item | Reason |
|---|---|
| Re-tuning of `weighted_sparse` / `adaptive_threshold` / `pd_sparse` thresholds, lambdas, or convergence rules on existing holdout cells | Destroys frozen-grid statistical guarantee (decision-log 2026-06-03). |
| New estimator candidates beyond the existing three | Out of scope for this push. |
| Cross-domain inference, signed priors, nonlinear correlation as production estimators | Deferred per rebuild design §3.2. |
| Algorithm research aimed at outranking `sparcc_closed_form` on AUROC/AP | Out of scope. The trade-off framing is locked. **Exception:** if a Line A optimization produces an incidental AUROC/AP improvement ≥ 0.005 on any scenario, the observed improvement is reported alongside the speed/memory result — the frame is not changed, the numbers are reported as observed. |
| Reduction of `n_resamples` default from 100 in production | Out of scope. The differential-test harness may use n_resamples=30 to keep per-commit overhead low; the reporting holdout uses n_resamples=100. |
| LaTeX, Pandoc, or any manuscript-rendering work | Belongs to the academic-pipeline. |
| Knockoff filter (B-γ in v1) | Dropped. Gaussian-knockoff exchangeability does not hold for CLR-transformed compositional features (rank-deficient covariance), and there is no published FDR proof for stability-selection composed with knockoffs on CLR data. Punted to a future spec if anyone wants to write the math note. |
| Meinshausen-Bühlmann (2010) PFER bound as headline FDR claim | Dropped as headline. Reported as a separate family-level diagnostic ("at this stability cutoff, the M–B bound says E[V] ≤ X"), not maxed against B-β. |

### 3.3 Conditional scope

| Item | Trigger |
|---|---|
| A-γ active-set / restart heuristics for `weighted_sparse` | Triggered only if A-α + A-β fail to reach the Line A Primary target. If A-γ ships, the resulting estimator is renamed `weighted_sparse_v2` and the holdout is re-evaluated end-to-end (including AUROC) for that estimator. |

## 4. Stopping Criteria

Targets are split into Primary (must hit before triggering the
academic-pipeline) and Stretch (best-effort). Any Primary that remains
unmet at the hard cap of seven weeks triggers a §12 user consultation;
the orchestrator does not auto-advance.

### 4.1 Line A — Speed and memory

| Metric | Primary | Stretch | Evaluation set |
|---|---|---|---|
| `wallclock_median` ratio to `sparcc_closed_form`, p=500 | ≤ 500× | ≤ 200× | 54-cell holdout grid at `n_resamples=100` (production default) |
| `wallclock_median` ratio to `sparcc_closed_form`, p=1000 | ≤ 500× | ≤ 250× | 54-cell holdout grid at `n_resamples=100` |
| `peak_memory_median_rss` ratio to `sparcc_closed_form`, p=500/1000 | ≤ 2× | ≤ 1.5× | 54-cell holdout grid at `n_resamples=100`; **psutil RSS, not tracemalloc** |
| Accuracy regression (AUROC, AP) | per-cell |Δ| ≤ 0.005 AND mean |Δ| ≤ 0.001 across the 39 training cells | 0 detected change on training | 39-cell training grid only; holdout AUROC is NOT recomputed under A-α/A-β (frozen-grid contract) |
| Convergence rate | ≥ 51 / 54 cells (≥ current) on holdout re-run | 54 / 54 | 54-cell holdout grid |

**Aggregation rules** (all four wallclock/memory rows):
* "median" is the median of per-cell `(method_wallclock / sparcc_wallclock)` ratios, computed per-cell first then medianed across cells.
* "≤ 80 % of cells" rounds up: 0.8 × 54 = 43.2 → at least 44 cells must pass.
* Cells where `sparcc_closed_form` reports wallclock < 0.001 s collapse to a floor of 0.001 s in the denominator to avoid degenerate ratios.
* Borderline values within ± 5 % of the threshold are reported as "at threshold" not "PASS"; an at-threshold result triggers a one-sentence note in the gate report, not an auto-PASS.

### 4.2 Line B — Calibrated posterior (no FDR claim)

| Metric | Primary | Stretch | Evaluation set |
|---|---|---|---|
| Reliability diagram (10 bins of sel_prob ∈ [0, 1]) | per-bin |observed_freq − bin_midpoint| ≤ 0.15 on **≥ 7 of 10 bins** AND ≤ 0.30 on **all 10 bins** | ≤ 0.10 on ≥ 7 bins | 54-cell holdout |
| Expected Calibration Error (ECE) | ≤ 0.10 on aggregate; per-scenario ECE ≤ 0.20 on each of the 6 scenarios | ECE ≤ 0.05 aggregate | 54-cell holdout, stratified by scenario |
| Brier score | reported (descriptive, no pass/fail threshold) | improvement vs. raw `sel_prob` as posterior | 54-cell holdout |
| Calibration coverage | All 54 holdout cells produce a reliability diagram and ECE; pi_train is reported per cell and per aggregate | + per-scenario reliability diagrams | 54-cell holdout (= 6 scenarios × 3 (n,p) sizes × 3 seeds) |
| Calibration constant tuning | training cells only (B-1 protocol §6.4); cell-level cross-validation, not pair-level | + leave-one-scenario-out diagnostic | 39-cell training grid |
| Out-of-family generalization | report Brier + ECE on SECOM (real data, unknown ground truth — use Lin et al. (2022) concordance as a proxy reference for the calibration check, with explicit acknowledgement of circularity) | + per-body-site stratification on HMP | SECOM only |

**What was dropped from v1**:
* "empirical FDR ≤ 1.5× nominal at q ∈ {0.01, 0.05, 0.10}" — gone. The procedure no longer claims FDR control; `(1 − g(sel_prob))` is not a p-value and BH on it has no FDR guarantee.
* "Power@q=0.05" — gone. Without an FDR claim, there is no operationally-defined Power metric.
* Conservative-max(B-α, B-β) headline — gone. B-α is reported separately as a family-level PFER diagnostic.

### 4.3 Line D — Real-data evaluation

| Metric | Primary | Stretch | Data |
|---|---|---|---|
| Dataset coverage | SECOM v1.0.0 + HMP 16S (acknowledged: both 16S, both gut-dominated; this is not a cross-environment generalization test) | + a third 16S resource from a different body site or environment | `data/public/{secom_v1.0.0,hmp_16s}.md` |
| Subsampling stability report | High-stability edge count, density, sel-prob distribution at thresholds {0.6, 0.7, 0.8, 0.9} reported for both datasets | + Jaccard(subsample halves) | SECOM + HMP |
| Sample-holdout CV | 5 × (50/50 split by sample); edge agreement (Jaccard) at sel-prob ≥ 0.8 with mean ± SD across the 5 splits | + 10-fold CV | SECOM + HMP |
| Reproducibility receipt | Dataset DOI/identifier + SHA-256 of archive + preprocessing parameters + seed + `n_resamples` + threshold + git hash | + per-edge selection traces | SECOM + HMP |
| Lin et al. (2022) SECOM network concordance | **Descriptive only — not a Primary gate.** Reports overlap, Jaccard, and Spearman rank correlation against the published top edges at sel-prob ∈ {0.7, 0.8, 0.9}; pre-registered K and cutoff before Line D execution; concordance, not validity | — | SECOM only |

**Demoted from v1 Primary**: "Top-10 Lin et al. SECOM edges recalled at sel-prob ≥ 0.8". Reason: the published network is itself an inference output on the same SECOM archive; treating it as a benchmark target is circular and the framing is concordance, not validity. The metric remains reported as descriptive context.

### 4.4 Trigger for the academic-pipeline

The pipeline (Stage 2 WRITE) is triggered when **every** Line A, B, D
Primary target is hit on the post-A holdout re-run. Any single unmet
Primary triggers the §12 user consultation. The Acknowledged
Limitations escape hatch is not a unilateral orchestrator decision —
it is invoked only after the user has reviewed and chosen it.

## 5. Line A — Speed and Memory Optimization

### 5.1 Measured hot path (A-0, done 2026-06-03)

Measurements committed in commit `480e52f`. cProfile on
`infer_network(estimator="weighted_sparse", selection="stability",
n_resamples=100, seed=200)` for `sparse_random, n=500, p=1000,
density=0.002` (a fresh holdout-equivalent cell, not a frozen holdout
cell):

| Hot function | Cumulative s | Self s | % wallclock (self) | Notes |
|---|---|---|---|---|
| `estimate_weighted_sparse` (alternating loop) | 253.1 | **221.0** | **87.2** | The alternating Step A / Step B updates on (p, p) matrices, repeated ~137 iterations × 101 estimator calls = 13,917 iters |
| `numpy.ufunc.reduce` (sum) | 11.4 | 11.4 | 4.5 | `R.sum(axis=1)`, `R.sum()` in Step A offset update |
| `numpy.linalg.eigvalsh` | 8.1 | 8.1 | 3.2 | Once per estimator call at the end; **not the dominant cost** |
| `ndarray.copy` | 6.0 | 6.0 | 2.4 | `Sigma_prev = Sigma.copy()` per iteration |
| `ndarray.dot` (inside norm) | 3.8 | 3.8 | 1.5 | `np.linalg.norm(Sigma - Sigma_prev)` per iteration |
| `_default_weights` GEMM | 1.05 | 0.97 | 0.4 | Already optimized with GEMM identity per the existing comment |
| `_correlation_from_covariance` | 0.29 | 0.11 | < 0.1 | Wasted work on the support path (not used by stability) |
| `infer_network` setup | 0.45 | 0.001 | < 0.1 | preprocessing, edge-table build, etc. |

The full pstats dump is at
`data/profile_weighted_sparse_p1000_n100.txt`; the parsed summary is at
`data/profile_weighted_sparse_p1000_n100.summary.json`.

**Implication:** Line A's leverage is overwhelmingly in the alternating
loop, not in eigvalsh. Optimizations are ordered by measured leverage.

### 5.2 A-α — Quick wins on the alternating loop (~1.5 weeks)

In approximate order of measured impact:

1. **Reduce iteration count.** The current 137-iter median across the
   profiled cell is high. The tolerance and stop criterion are revised
   per training-cell evidence only (not holdout): switch to a relative
   tolerance `||Sigma_new − Sigma||_F / max(||Sigma||_F, 1e-12) < tol`
   with `tol = 1e-5`. Keep `max_iter = 200` as the absolute cap; do
   not tune it on holdout-observed convergence. Expected leverage:
   1.5–2× on cells that previously hit 100+ iter.
2. **Parallelize stability subsamples** with `concurrent.futures` /
   `joblib`. Use `np.random.SeedSequence(seed).spawn(n_resamples)` for
   per-subsample independent streams — **not** `seed + k`, which
   produces correlated streams (devil-13 / feas-1). Workers are spawned
   with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1` set in the subprocess environment to avoid
   BLAS oversubscription. Default `n_jobs = max(1, os.cpu_count() - 1)`;
   expose `n_jobs` on `infer_network`. Expected leverage on an 8-core
   laptop: 3–5× (lower than naive cpu_count due to per-cell work
   variation and pickling overhead, but real). Determinism contract is
   re-stated: outputs are reproducible under (seed, n_jobs); the v1
   `data/bench_holdout_local.csv` is reproducible only under the
   sequential code path (which remains available as `n_jobs=1`).
3. **In-place delta accumulation** to remove `Sigma.copy()` (2.4 %) and
   the `norm` allocation+dot (1.5 %). Pattern: maintain `delta_buf` as
   a preallocated buffer; compute `np.subtract(Sigma_new, Sigma,
   out=delta_buf); delta_sq = np.einsum('ij,ij->', delta_buf, delta_buf)`.
   Expected leverage: 3–4 %.
4. **Skip `_correlation_from_covariance` on the support path.** Support
   path does not need the correlation matrix. Refactor
   `_build_estimator` so the support callable invokes a
   `support_only=True` variant of `estimate_weighted_sparse` that
   skips both the eigvalsh and the correlation computation. Expected
   leverage: ~3 % (the eigvalsh + correlation combined).
5. **Return a boolean upper-triangle mask from support_fn** instead of
   the dense covariance matrix (feas-9). At p=1000 this drops the
   per-worker return payload from 8 MB to 62 kB, reducing pickle/copy
   overhead in the joblib path. Expected leverage: 1–3 % wallclock,
   substantial peak-memory reduction.

**A-α expected combined leverage**: 8–15× wallclock at p=1000 under
parallelism; less at p=500. Hot path measurement updates are committed
to `data/profile_*_post_a_alpha.json` after A-α ships.

### 5.3 A-β — Numba JIT on the alternating kernel (~1 week)

The alternating-loop self-time of 221 s at p=1000 is dominated by NumPy
broadcasts that Numba can fuse into element-wise machine code without
the intermediate (p, p) allocations.

1. JIT the Step A + Step B kernel with `@numba.njit(cache=True, parallel=False)`
   (single-threaded; parallelism is at the subsample level via A-α).
2. Maintain `M = S_clr + f[:, None] + f[None, :]` incrementally with
   `M += delta_f[:, None] + delta_f[None, :]` rather than recomputing
   from scratch.
3. Add `numba>=0.61,<0.63` to `[project.optional-dependencies]` group
   `accel`. The `infer_network` import checks for Numba; if absent, it
   falls back to the pure-NumPy A-α path with a runtime warning. CI
   tests both paths.

Expected leverage on A-α baseline: 1.5–2×. The first-call JIT compile
time (~2–5 s) is amortized across `n_resamples` subsamples and across
the 54 holdout cells via `cache=True`. CI cold-cache runtime is
documented in the differential-test harness budget.

### 5.4 A-γ — Active-set restart (conditional only) (~1.5 weeks)

Triggered only if A-α + A-β fail to reach the Line A Primary on the
post-A holdout. **A-γ changes the algorithm's iteration trajectory; if
shipped, it becomes a separate estimator `weighted_sparse_v2`** and the
holdout is re-evaluated end-to-end for that name, including AUROC.

1. Active-set tracking: maintain a boolean `active_mask` of entries
   whose magnitude exceeds the per-entry threshold; entries that have
   been zero for `k=10` consecutive iterations are excluded from the
   update loop. Re-evaluate every 25 iterations.
2. Restart heuristics: warm-start subsequent stability subsamples from
   the previous subsample's Sigma when the subsamples overlap by
   ≥ 40 % (the realistic ceiling at `subsample_fraction=0.5`).

A-γ acceptance gate (stricter than v1):

* Per-training-cell edge-set Jaccard at sel-prob ≥ 0.8 of
  `weighted_sparse_v2` vs. `weighted_sparse` (A-α + A-β baseline)
  ≥ 0.99, AND
* Per-cell AUROC delta ≤ 0.001 on training, AND
* If A-γ ships, the B-β calibration is **re-fit** on
  `weighted_sparse_v2` output (Line B is partially redone for the new
  estimator).

If any A-γ gate fails, A-γ is dropped and the Line A Primary moves to
the §12 user-consult path.

### 5.5 Verification protocol

1. **Differential-test harness (must exist before A-α step 2 ships)**.
   `tests/test_weighted_sparse_differential.py` runs the full 39
   training cells at `n_resamples=30` (kept low to fit a 10-minute CI
   budget) and compares per-cell AUROC/AP against a pinned baseline
   CSV at `tests/baselines/weighted_sparse_baseline.csv`. Fails the
   commit if per-cell |ΔAUROC| > 0.005 OR mean |ΔAUROC| > 0.001.
   Building this harness is an explicit week-1 deliverable.
2. **Holdout re-run with new instrumentation**. After A-α complete,
   re-run the 54-cell holdout at `n_resamples=100` with both
   tracemalloc and RSS (psutil) instrumented. Write the result to
   `data/bench_holdout_local_v2.csv` (the v1 CSV at
   `data/bench_holdout_local.csv` is preserved verbatim). Compare
   wallclock and `peak_memory_median_rss`.
3. **Per-A-step profile re-run**. After A-α complete, re-profile the
   same `(sparse_random, n=500, p=1000, seed=200)` cell and commit to
   `data/profile_*_post_a_alpha.json`. Repeat after A-β.
4. **AUROC/AP on holdout is NOT recomputed under A-α/A-β** to preserve
   the frozen-grid contract. The training-grid differential test is the
   only AUROC check during A development. If A-γ triggers, this rule
   changes (§ 5.4).

## 6. Line B — Calibrated Posterior

### 6.1 B-β — Empirical isotonic calibration (main path) (~2 weeks)

Procedure:

1. On the 39 training cells, collect tuples `(scenario, n, p,
   sel_prob_ij, is_true_edge_ij, pi_cell)` for every off-diagonal entry
   across all cells and seeds. Record `pi_cell` = density × p(p-1)/2 /
   p(p-1)/2 = `density` per cell.
2. Fit a **scenario-stratified** monotone isotonic regression `g_s:
   sel_prob → P̂(is_true_edge | sel_prob)` per scenario `s`. Pooled
   isotonic is reported in parallel as a degraded baseline. With ~6.5
   training cells per scenario, the per-scenario isotonic uses
   leave-one-cell-out CV for variance estimation.
3. Convert `g_s(sel_prob_ij)` to a **calibrated posterior probability**
   (not a q-value, not a p-value). Schema field
   `EdgeTable.posterior_probability` (new) carries this number. The
   old `pvalue_approx` and `qvalue_approx` fields **stay `None`** —
   they reserve their semantic for an actual p-value / FDR-controlled
   q-value if a future spec ships one.
4. At runtime, the user passes a `scenario_hint` to `infer_network`
   that selects which `g_s` is applied (defaults to the pooled `g`
   when no hint is given). The choice is recorded in
   `EstimatorDiagnostics.calibration_method`.

### 6.2 B-α — PFER bound (diagnostic only)

The Meinshausen-Bühlmann (2010) PFER bound is implemented and reported
as a **family-level diagnostic**, never combined with B-β on the same
axis.

* Formula: at stability cutoff `pi_thr`, `PFER ≤ q_avg² / ((2 pi_thr − 1) × p_off)` where `p_off = p(p-1)/2`.
* Reporting: "at cutoff π = 0.8 and average selected set size q_avg = X, the M–B bound says E[V] ≤ Y."
* No max, no min, no aggregation with B-β. The two answer different
  questions and live on different axes.

### 6.3 Evaluation protocol (B-1)

| Phase | Cells used | Purpose |
|---|---|---|
| Development | 39 training cells | Fit `g_s` per scenario; tune isotonic knot count via leave-one-cell-out CV (cell-level, not pair-level) |
| Reporting | 54 holdout cells | Report reliability diagram (10 bins), ECE (aggregate + per-scenario), Brier score |
| Out-of-family check | SECOM real data | Brier + ECE on SECOM using Lin et al. (2022) network as a proxy reference (with explicit circularity caveat in the paper) |

The paper carries a one-sentence explicit declaration: "The
calibration procedure was developed on `data/bench_training_local.csv`
(39 cells) and reported on `data/bench_holdout_local_v2.csv` (54
cells). The holdout grid was not used to tune the calibration."

A pre-registered prediction is committed to `docs/decision-log.md`
before the holdout calibration is run: "We expect aggregate ECE in the
range [0.05, 0.12]. Aggregate ECE > 0.15 triggers spec amendment and
user consultation; we will not silently move the gate."

### 6.4 Output schema additions

Three additive changes to `src/falcon/results.py`:

* `EdgeTable.posterior_probability: np.ndarray | None` — new field,
  shape `(n_edges,)`, defaulting to `None`. Populated when
  `selection="stability"` and calibration assets are available.
* `EstimatorDiagnostics.calibration_method` enum extended **additively**
  (no removals): `{"none", "permutation_base_only", "subsampling",
  "empirical_isotonic_per_scenario", "empirical_isotonic_pooled",
  "meinshausen_buhlmann_bound"}`. The first three remain valid;
  existing callers do not break.
* `EstimatorDiagnostics.uncertainty_interpretation` enum extended:
  add `"calibrated_posterior"` and `"calibrated_posterior_pooled"`.

A separate `CalibrationReport` dataclass holds per-cell aggregates
(`ece_aggregate`, `ece_per_scenario`, `brier_score`, `reliability_bins`)
and is written to `data/calibration_holdout_v2.csv`.

### 6.5 What is no longer claimed

The Line B work product does not claim:

* FDR control at any nominal level.
* That the posterior probability is a q-value or a p-value.
* That the posterior is calibrated outside the training distribution
  family (the SECOM check is a partial test; HMP is a sanity check).

The Line B work product does claim:

* A reliability diagram showing per-bin agreement between sel_prob and
  empirical P(true edge | sel_prob) on the holdout.
* An aggregate and per-scenario ECE number.
* A Brier score for the calibration mapping.
* The pi_train under which calibration was fit, so downstream consumers
  can re-scale if their data has a different edge prior.

## 7. Line D — Real-data Evaluation

### 7.1 D.0 — Build dataset extractors (~1 week, week 1–2)

`scripts/process_public_data.py` currently has
`DATASET_EXTRACTORS: dict[str, Callable] = {}`. Build the two
extractors before D.1 runs:

1. **SECOM extractor**: download Zenodo archive
   (`10.5281/zenodo.6809029`); record SHA-256; locate OTU table and
   metadata; harmonize taxa to genus level; export
   `data/public/secom_v1.0.0/{counts.npz, taxa.csv, samples.csv}`.
2. **HMP 16S extractor**: download BioProject PRJNA48489 OTU table;
   parse `.biom` (adds `biom-format>=2.1,<3.0` dependency); rarefy
   per the documented protocol; export
   `data/public/hmp_16s/{counts.npz, taxa.csv, samples.csv}`.
3. **Tests**: `tests/test_public_data_extractors.py` with small fixture
   archives (committed under `tests/fixtures/`) verifies the extractor
   correctness independent of network access.

### 7.2 D.1–D.4 SECOM workflow

1. Run `infer_network(estimator="weighted_sparse", n_resamples=100,
   seed=0)` on the full preprocessed SECOM matrix.
2. Report: high-stability edge counts at thresholds
   {0.6, 0.7, 0.8, 0.9}; density; sel-prob distribution histogram (50 bins).
3. Sample-holdout CV: 5 × (50/50 sample split) with seeds 0–4. Report
   Jaccard at sel-prob ≥ 0.8 mean ± SD.
4. Lin et al. (2022) concordance: extract published top edges (K and
   sel-prob cutoff pre-registered in decision-log before Line D run).
   Report overlap, Jaccard, and Spearman rank correlation as
   **descriptive only** — not a Primary gate.

### 7.3 HMP 16S workflow

Same as §7.2 but with the HMP rarefied OTU table. No published-network
concordance (no comparable network published on the same processing).

### 7.4 Reproducibility receipt

Every Line D output ships with:

* Dataset DOI or BioProject identifier
* SHA-256 of the downloaded archive (recorded at extractor time)
* Preprocessing parameters (prevalence threshold, total-count cutoff,
  rarefaction depth where applicable)
* Random seed
* `n_resamples` and `subsample_fraction`
* Edge stability thresholds reported
* Git commit hash of the package version
* A `data/{secom,hmp}_results.csv` source-data file backing every
  numerical claim

### 7.5 Honest framing language

The Line D writeup uses the phrase "two gut-dominated 16S resources",
not "two independent datasets". The paper explicitly acknowledges that
the two datasets share substantial taxa and similar compositional
artifacts; this is reproducibility within a family, not
cross-environment generalization.

## 8. Execution Timeline

Hard cap: 7 weeks from 2026-06-04. Three workstreams run in parallel
with weekly check-ins on Friday. Each line is on its own feature
branch (`feat/line-a-speed`, `feat/line-b-calibration`,
`feat/line-d-realdata`); the schema-touching Line B branch merges
first (week 2), then Lines A and D rebase.

| Week | Line A | Line B | Line D |
|---|---|---|---|
| 1 | Build differential-test harness; pin baseline CSV under tests/baselines/ | Schema additions (additive only): posterior_probability field, calibration_method enum, CalibrationReport dataclass | D.0 build SECOM extractor; pyproject.toml adds biom-format |
| 2 | A-α steps 1 + 4 (iteration count + support_only skip) | B-β isotonic fit on 39 training cells, per-scenario | D.0 build HMP extractor; D.1 SECOM stability run |
| 3 | A-α steps 2 + 3 + 5 (parallelism + delta + bool mask) | B-α PFER bound; leave-one-cell-out CV on training | D.2 SECOM CV; D.3 SECOM Lin concordance |
| 4 | A-β JIT + first holdout re-run with new instrumentation | B-β holdout reliability report; pre-register prediction | D repeat on HMP |
| 5 | Triage: if A-α + A-β missed Primary, decide A-γ go/no-go with user | B holdout final report + SECOM out-of-family check | Reproducibility receipts |
| 6 | A-γ (conditional) OR hardening; final A holdout re-run | Schema integration tests; CalibrationReport CSV | HMP final report |
| 7 | Acceptance-gate v2 report; decision-log update; **§12 user-consult on any unmet Primary**; integration check (#10) unblocks academic-pipeline | | |

**Benchmark wall-time accounting**: every weekly check-in records the
wall-time used by overnight benchmark runs separately from coding time.
The two overnight benchmarks (full holdout × 2) plus the SECOM/HMP
runs total an estimated 40–60 hours of compute over the 7 weeks; this
is overnight time, not daytime time.

If any line slips past its week-N target by more than 3 days, the
weekly check-in escalates to the §12 user consultation.

## 9. Risk Register

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | A-α + A-β fail to reach ≤ 500× SparCC at p ∈ {500, 1000} | low–medium (measurements suggest 200–500× is reachable) | high | Trigger A-γ with the stricter v2 acceptance gate; if A-γ also misses, §12 user consult |
| R2 | B-β per-scenario isotonic does not generalize because only ~6.5 training cells per scenario | medium | medium | Pooled isotonic reported alongside; SECOM out-of-family check exposes the generalization issue if present; pi_train documented |
| R3 | SECOM Lin et al. concordance is low, exposing the trade-off paper to "your method disagrees with the only published reference" | medium | medium | Already demoted from Primary to descriptive; reported honestly with overlap/Jaccard/Spearman; interpreted as concordance not validity in §4.3 and the paper text |
| R4 | A optimization introduces numerical drift causing per-cell AUROC regression > 0.005 on training | low | high | Per-commit differential-test harness (§5.5 step 1); commit reverted if regression detected |
| R5 | Hard cap slips beyond 7 weeks | medium | high | Weekly check-in escalation; §12 user-consult mandatory at any Primary miss (not just multi-miss) |
| R6 | Parallelism breaks bit-exact reproducibility with the v1 holdout CSV | **certain** (this is by design) | low | Documented in §5.2 step 2; v1 CSV preserved; new v2 CSV is the post-A baseline; pre-A reproducibility remains available at `n_jobs=1` with `np.random.SeedSequence(seed).spawn(n_resamples)` |
| R7 | Public-data download fails or DOI deprecated | low | medium | Manifest contains SHA-256; cached snapshot recorded; reproducibility receipt records cache hash |
| R8 | Numba install fails on operator's environment (Python/NumPy pin conflict) | low–medium | medium | Numba is `[project.optional-dependencies].accel`; A-β path checks `import numba` and falls back to A-α pure-NumPy with a warning. CI tests both paths |
| R9 | tracemalloc-vs-RSS metric divergence makes the published gate-3 number look inconsistent with v1 | certain | low | Explicit "metric switch" subsection in the paper's Methods; both numbers reported, RSS as headline |
| R10 | Operator's machine has fewer than 8 physical cores; parallelism leverage is < 3× | low | medium | `infer_network(n_jobs=N)` is user-configurable; documented in §5.2 step 2; the Primary target's "≤ 500×" leaves headroom for sub-8-core machines |

## 10. Decision References

Brainstorming dialog 2026-06-03 (the v1 → v2 conversation):

| Decision | Outcome |
|---|---|
| Goals to pursue | A (speed/memory) + B (FDR) + D (real-data); C (accuracy) excluded |
| Depth tier | Middle — Strong-trade-off (~5–6 weeks; 7-week hard cap) |
| B evaluation protocol | B-1 (39 training cells for tuning, 54 holdout cells for reporting) |
| Stopping criteria | v1: as proposed; v2: RESET per measurements (Line A from ≤100× to ≤500×; Line B from FDR control to calibrated posterior; Line D published-net Primary demoted) |
| Spec posture (post-self-review) | Measure-first, then rewrite spec v2 |
| Measurement done | commit `480e52f` (profile + gap at n_resamples=100) |

Cross-references:

* Rebuild design: `2026-06-02-single-domain-estimator-rebuild-design.md`
* Acceptance evidence: `docs/acceptance-gate-report.md`
* Decision history: `docs/decision-log.md`
* Measurement artifacts: `data/profile_weighted_sparse_p1000_n100.{txt,summary.json}`, `data/bench_gap_n100.csv`
* Trade-off paper framing: `README.md` §"When to use which estimator"
* Methodology: `docs/methodology.md`

## 11. Out of Scope / Deferred

The following are explicit non-goals of this push. Each is recorded so
later contributors do not silently expand the scope.

1. Cross-domain inference, signed biological priors, nonlinear
   correlation as production estimators (rebuild design §3.2).
2. Re-tuning of any of the three estimator candidates'
   selection-critical parameters on holdout cells.
3. New estimator candidates beyond `weighted_sparse`,
   `adaptive_threshold`, `pd_sparse`.
4. Algorithm research aimed at outranking `sparcc_closed_form` on
   AUROC/AP. The trade-off framing is locked. **Exception**: if a
   Line A change produces a holdout AUROC/AP improvement ≥ 0.005 on
   any scenario, the improvement is reported alongside the
   speed/memory result. The frame is not changed; the numbers are
   reported as observed.
5. LaTeX, Pandoc, or any manuscript-rendering work. That belongs to
   academic-pipeline Stage 5 FINALIZE.
6. Reduction of `n_resamples` from 100 in production. The
   differential-test harness uses `n_resamples=30` to keep per-commit
   CI time low; the reporting holdout uses `n_resamples=100`.
7. Knockoff filter as an FDR procedure on this data. The mathematical
   foundations are not in place; punted to a future spec.
8. Meinshausen-Bühlmann PFER bound as a headline FDR claim. Reported
   as a separate family-level diagnostic only.

## 12. Hard-cap escalation contract

At week 7 day 1, the orchestrator runs the integration check
(`Verify A+B+D Primary Targets Met → unblock pipeline`, task #10).

If **all** Primary targets pass: write the gate-v2 report; update
decision-log; mark task #10 complete; the academic-pipeline (Stage 2
WRITE, task #1) becomes the next pending task.

If **any** Primary target is unmet: STOP. Do not auto-advance. Surface
the unmet target(s) to the user with the four options:

1. Extend the cap with additional budget (state how much).
2. Accept the unmet Primary as an Acknowledged Limitation and proceed
   to writing.
3. Drop a Stretch target and retry the unmet Primary with the freed
   budget.
4. Abort the writing stage; the trade-off paper waits for a future
   push.

The orchestrator does not pick. The user picks. The choice is recorded
in `docs/decision-log.md` with the date and reasoning.

The §9 R5 escalation order (B-γ first, then A-γ, then D-stretch) was
removed in v2. Escalation order is the user's call, not the
orchestrator's preference.

## 13. Hand-off to academic-pipeline

When the integration check passes (or §12 chooses to proceed), the
academic-pipeline takes over. The hand-off contract is:

| Pipeline stage | What this spec hands off | What the pipeline runs |
|---|---|---|
| Stage 2 WRITE | Optimized estimator, calibrated posterior, real-data evidence, gate-v2 report, decision-log entries | `academic-paper` skill in full mode |
| **Stage 2.5 INTEGRITY (MANDATORY, BLOCKING)** | The draft from Stage 2 | `integrity_verification_agent` (pre-review mode) — 5-phase protocol + 7-mode AI failure-mode checklist. Must PASS. |
| Stage 3 REVIEW | Verified paper | `academic-paper-reviewer` full mode (5-reviewer panel) |
| Stage 4 REVISE | Revision Roadmap | `academic-paper` revision mode |
| Stage 3' RE-REVIEW | Revised draft | `academic-paper-reviewer` re-review mode |
| Stage 4' RE-REVISE (conditional) | Re-review residuals | `academic-paper` revision mode (final round) |
| **Stage 4.5 FINAL INTEGRITY (MANDATORY, BLOCKING)** | Revised paper | `integrity_verification_agent` (final-check mode) — verifies from scratch, runs the 7-mode AI failure-mode checklist again. Must PASS with zero issues. |
| Stage 5 FINALIZE | Verified final paper | `academic-paper` format-convert mode (MD → DOCX → LaTeX → PDF on OUP template) |
| Stage 6 PROCESS SUMMARY | Full pipeline history | Orchestrator generates process record |

This spec's completion **does not** authorize skipping Stage 2.5 or
4.5. Stage 2.5 / 4.5 are blocking gates owned by
`integrity_verification_agent`, not by this spec. The 7-mode AI
failure-mode checklist
(`academic-pipeline/references/ai_research_failure_modes.md`) is run
at each integrity stage; suspected failures block the pipeline and
require user acknowledgement.
