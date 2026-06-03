# Method Optimization Design — Pre-Paper Push

Date: 2026-06-03

Status: approved for implementation planning

Supersedes nothing. Builds on
[`2026-06-02-single-domain-estimator-rebuild-design.md`](2026-06-02-single-domain-estimator-rebuild-design.md).
The frozen-grid protocol and estimator candidate definitions from that
design remain in force.

## 1. Objective

Push the rebuilt single-domain estimator — and the supporting evaluation
record — to the strongest honest state that the existing methodology
contract permits before opening the academic-pipeline writing stage.
The 2026-06-03 holdout produced 2 of 6 acceptance gates passing
([`docs/acceptance-gate-report.md`](../../acceptance-gate-report.md)) and
the repository committed to a trade-off paper framing rather than an
advantage claim. The trade-off paper becomes meaningfully stronger if the
following are true before writing begins:

1. The Gate 2 calibration is shipped and validated, so `selection_probability`
   converts into honest q-values rather than `None`.
2. The Gate 3 runtime gap to `sparcc_closed_form` shrinks from the
   2 600–6 000× reported in the holdout to ≤ 100× at p ∈ {500, 1000}.
3. The Gate 5 public-data subsampling stability is produced on at least
   two independent datasets with edge-overlap evidence against the
   published reference network.

No work in this design touches estimator accuracy on the existing
holdout. Re-tuning estimators on holdout cells destroys the frozen-grid
statistical guarantee that the eventual trade-off paper depends on; this
is recorded explicitly in [`docs/decision-log.md`](../../decision-log.md)
under the 2026-06-03 entry "Open questions". Gate 1 remains FAIL under
the strict reading and the paper acknowledges it.

## 2. Why Optimize Before Writing

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
`selection_probability` is reported without a calibrated q-value. A
Bioinformatics reviewer will read claim 3 as "they say uncertainty but
they don't actually quantify FDR control." Closing this is the
single-highest-leverage strengthening available.

Claim 1's runtime margin is real against `cclasso` and `coat` but is
juxtaposed in the paper against a 2 600–6 000× runtime loss to
`sparcc_closed_form`. Shrinking that loss makes the cost half of the
trade-off acceptable rather than alarming.

Public-data evidence (Gate 5) is the standard Bioinformatics expectation
for any network-inference method paper. A trade-off paper without it
will receive predictable reviewer pushback.

## 3. Scope

### 3.1 In scope

| Line | Workstream |
|---|---|
| A | Speed and memory optimization of `weighted_sparse` and its stability-selection wrapper. Pure engineering. |
| B | Calibrate `selection_probability` → q-value. Validate empirical FDR on holdout. |
| D | Public-data subsampling stability evaluation on SECOM v1.0.0 and HMP 16S. Includes published-network overlap comparison. |

### 3.2 Explicitly out of scope

| Item | Reason |
|---|---|
| Re-tuning of `weighted_sparse` / `adaptive_threshold` / `pd_sparse` thresholds, lambdas, or convergence rules on existing holdout cells | Destroys frozen-grid statistical guarantee (decision-log 2026-06-03 "Open questions"). |
| New estimator candidates beyond the existing three | Out of scope for this push; would invalidate the holdout's role. |
| Cross-domain inference, signed priors, nonlinear correlation | Deferred per rebuild design §3.2. Not revisited. |
| Algorithm research aimed at outranking `sparcc_closed_form` on AUROC/AP | Out of scope. The trade-off framing is locked. |
| Reduction of `n_resamples` default from 100 | Out of scope. Lower `n_resamples` reduces `selection_probability` resolution and would partly hide the speed gap rather than close it. |
| LaTeX / Pandoc / manuscript work | Belongs to the academic-pipeline stage, gated on this design's completion. |

### 3.3 Conditional scope

| Item | Trigger |
|---|---|
| A-γ active-set / restart heuristics for `weighted_sparse` | Triggered only if A-α + A-β fail to reach the Line A Primary target. |
| B-γ Knockoff filter | Triggered only if Lines A, B, D are on schedule and B-α + B-β have completed and met Primary targets. |

## 4. Stopping Criteria

Targets are split into Primary (must hit before triggering the
academic-pipeline) and Stretch (best-effort). Any Primary target that
remains unmet at the hard cap of seven weeks must be recorded in
`docs/decision-log.md` with the reason and documented in the paper's
Acknowledged Limitations section.

### 4.1 Line A — Speed and memory

| Metric | Primary | Stretch | Evaluation set |
|---|---|---|---|
| `wallclock_median` (p=500) | ≤ 100× `sparcc_closed_form` | ≤ 50× | 54-cell holdout grid |
| `wallclock_median` (p=1000) | ≤ 100× `sparcc_closed_form` | ≤ 75× | 54-cell holdout grid |
| `peak_memory_median` (p=500/1000) | ≤ 2× `sparcc_closed_form` | ≤ 1.5× | 54-cell holdout grid |
| Accuracy regression (AUROC, AP) | ≤ 0.005 vs current `weighted_sparse` | 0.0 (no regression) | 54-cell holdout grid |
| Convergence rate | ≥ 51 / 54 cells (≥ current) | 54 / 54 | 54-cell holdout grid |

### 4.2 Line B — FDR calibration

| Metric | Primary | Stretch | Evaluation set |
|---|---|---|---|
| Empirical FDR @ nominal q ∈ {0.01, 0.05, 0.10} | ≤ 1.5× nominal on ≥ 80 % of holdout cells | ≤ 1.0× nominal on ≥ 90 % | 54-cell holdout grid |
| Power @ q = 0.05 | ≥ 0.50 on `sparse_random`, `block`, `heavy_tailed` | ≥ 0.70 on each | 54-cell holdout grid |
| Calibration coverage | All 6 scenarios × 3 n × 3 p × 3 seed report curves | + per-scenario stratified reports | 54-cell holdout grid |
| Calibration constant tuning | Training cells only (B-1 protocol §6.4) | + cross-validation on training | 39-cell training grid |

### 4.3 Line D — Real-data evaluation

| Metric | Primary | Stretch | Data |
|---|---|---|---|
| Dataset coverage | SECOM v1.0.0 + HMP 16S | + a third independent dataset | `data/public/{secom_v1.0.0,hmp_16s}.md` |
| Subsampling stability report | High-stability edges (≥ 0.8) count, density, sel-prob distribution | + Jaccard(subsample halves) | SECOM + HMP |
| Sample-holdout CV | 50/50 split, edge agreement reported with mean ± SD | + 10-fold CV | SECOM + HMP |
| Reproducibility receipt | Dataset DOI/identifier + seed + resample count + threshold + git hash | + per-edge selection traces | SECOM + HMP |
| Published-network comparison | Top-10 Lin et al. (2022) SECOM edges recalled at sel-prob ≥ 0.8 | + full edge overlap | SECOM only |

### 4.4 Trigger for the academic-pipeline

The pipeline (Stage 2 WRITE) is triggered when **every** Line A, B, D
Primary target is hit, or each unmet target is recorded in
`docs/decision-log.md` with a reason and an explicit Acknowledged
Limitations entry queued for the paper. Stretch targets are reported
but never block.

## 5. Line A — Speed and Memory Optimization

### 5.1 Hot-path findings

Reading `src/falcon/api.py` and `src/falcon/estimators/weighted_sparse.py`
identifies two structural wastes that account for the bulk of the
runtime gap to `sparcc_closed_form`:

1. **`support_fn` calls full estimator including eigendecomposition.**
   `src/falcon/api.py:86-87` defines `support_fn(Z) = estimate_fn(Z).covariance`.
   `estimate_weighted_sparse` ends with
   `min_eig = float(np.linalg.eigvalsh(Sigma).min())` at
   `src/falcon/estimators/weighted_sparse.py:147`. The stability loop calls
   `support_fn` `n_resamples` times (default 100). At p = 1000 each
   `eigvalsh` is `O(p^3)` ≈ 1 × 10^9 flops; 100 invocations cost
   ≈ 1 × 10^11 flops — about 200× the entire work `sparcc_closed_form`
   does, none of it used by stability. This is a single-line removal
   from the support path.
2. **Sequential subsample loop.** `src/falcon/stability.py:52-63` loops
   `n_resamples` subsamples serially. The work is embarrassingly
   parallel: each subsample is independent. On an 8-core laptop a
   `joblib` / `concurrent.futures` rewrite delivers a 5–8× wallclock
   speedup with no algorithmic change.

The remaining inner-loop allocations and the
`Sigma_prev = Sigma.copy()` per iteration are real but secondary; in-place
delta updates take them off the critical path without a JIT.

### 5.2 A-α — Quick wins (no algorithmic change)

1. Remove `eigvalsh` from the `support_fn` path. Keep it on the full
   `estimate_fn` path where `min_eigenvalue` is reported as a
   diagnostic. The cleanest refactor introduces a `support_only=True`
   keyword on `estimate_weighted_sparse` that skips the eigendecomposition.
2. Parallelize `select_by_stability` over subsamples. Default to
   `n_jobs = max(1, os.cpu_count() - 1)`; expose `n_jobs` on
   `infer_network`. Determinism preserved by seeding per-subsample with
   `seed + k`.
3. Replace `Sigma_prev = Sigma.copy()` with incremental Frobenius
   delta: keep a running `delta_sq` accumulator updated as
   `np.subtract(Sigma_new, Sigma, out=delta_buf)` then
   `np.einsum('ij,ij->', delta_buf, delta_buf)`.
4. Collapse the Step B `np.sign × np.abs × np.maximum` chain into a
   single `np.copyto` with an in-place `np.where` using a preallocated
   `M_buf`, `Sigma_new_buf` pair.
5. Replace the `np.linalg.norm` call with `np.sqrt(delta_sq)` from
   step 3.

A-α requires no new dependencies and no algorithmic change. The
estimator's mathematical behaviour is unchanged.

### 5.3 A-β — Hot-loop JIT (Numba)

1. JIT the alternating Step A + Step B loop with `@numba.njit(cache=True)`.
   Numba removes Python overhead in the iteration loop and allows
   straight-line BLAS-style fused ops on `M`, `R`, `Sigma`.
2. Maintain `M = S_clr + f[:, None] + f[None, :]` incrementally rather
   than recomputing from scratch: `M += delta_f[:, None] + delta_f[None, :]`.
3. Re-evaluate `max_iter` from the default 200. Most cells converge in
   ≤ 60 iterations on the holdout grid; tightening to 80 with the
   existing tolerance produces no convergence regression in the
   training-cell smoke test and saves wallclock on slow cells.

A-β requires Numba (already common in scientific Python; add to
`pyproject.toml` extras). The estimator's mathematical behaviour is
unchanged.

### 5.4 A-γ — Algorithmic restructuring (conditional)

Triggered only if A-α + A-β do not reach the Line A Primary target.

1. Active-set tracking: maintain a boolean `active_mask` of entries
   whose magnitude exceeds the per-entry threshold; entries that have
   been zero for k consecutive iterations are excluded from the update
   loop. Re-evaluate every restart_period iterations.
2. Restart heuristics: warm-start subsequent stability subsamples from
   the previous subsample's Sigma when the subsamples overlap by ≥ 70 %.

A-γ does change the algorithm's iteration trajectory. The mathematical
fixed point is unchanged, but the numerical path is. A-γ ships only if
its differential test on the training grid produces AUROC change
≤ 0.001 against the A-α + A-β baseline. If A-γ ships, the spec is
updated with a 2026-06-1x decision-log entry.

### 5.5 Verification protocol

1. Per-step differential test on the 39 training cells: AUROC and AP
   must not move by more than 0.001 against the baseline `weighted_sparse`
   on each step's commit. If any cell exceeds, the commit is reverted.
2. After A-α complete, re-run the full 54-cell holdout grid with three
   reps and update `docs/acceptance-gate-report.md` with the new wallclock
   and peak_memory numbers. Do not change the AUROC/AP reporting; that
   is held over from the original holdout.
3. After A-β complete, same protocol.
4. A-γ (if triggered) follows the same protocol.

The pre-A holdout numbers are preserved as `data/bench_holdout_local.csv`
verbatim. The post-A numbers are written to a new
`data/bench_holdout_local_v2.csv` so the comparison is auditable.

## 6. Line B — FDR Calibration

### 6.1 B-β — Empirical calibration (main path)

Procedure:

1. On the 39 training cells, collect tuples
   `(scenario, n, p, sel_prob_ij, is_true_edge_ij)` for every
   off-diagonal entry across all cells and seeds.
2. Fit a monotone isotonic regression
   `g: sel_prob → P̂(is_true_edge | sel_prob)`. Restrict to off-diagonal
   training pairs to avoid the trivial diagonal.
3. Define the calibrated per-edge q-value by Benjamini-Hochberg
   adjustment on `1 - g(sel_prob_ij)` across all reported edges.
4. Tune the calibration's smoothing constant only on training cells via
   k-fold cross-validation within training; the chosen constant is
   frozen before holdout is touched.

The schema in `src/falcon/results.py::EdgeTable` already carries
`pvalue_approx` and `qvalue_approx` fields kept as `None`. B-β fills
these in and sets the diagnostic's `uncertainty_interpretation` to
`"empirical_isotonic_BH"` with the calibration metadata.

### 6.2 B-α — Theoretical bound (cross-check)

Procedure:

1. Implement the Meinshausen-Bühlmann (2010) per-family error rate
   bound: under sub-Gaussian noise assumptions, for stability cutoff
   `pi_thr` and average selected support size `q_avg`,
   `PFER ≤ q_avg^2 / ((2 * pi_thr − 1) * p_off)` where `p_off = p (p − 1) / 2`.
2. Report the theoretical bound's q-value mapping alongside B-β. They
   are co-tabulated in the calibration report.
3. The paper's headline q-value is the **conservative max** of B-α and
   B-β at every cutoff. This is a documented choice, not a hidden
   default.

### 6.3 B-γ — Knockoff filter (stretch only)

Triggered only if A, B-α, B-β all hit Primary on schedule:

1. Construct column-permuted knockoff features as in Barber & Candès
   (2015) §2 with the Gaussian-knockoff construction adapted to the CLR
   covariance.
2. Define the knockoff statistic
   `W_ij = sel_prob_real_ij − sel_prob_knockoff_ij`.
3. Apply Knockoff+ procedure for finite-sample FDR control.

B-γ ships only with its own dedicated `data/bench_holdout_knockoff_v2.csv`
and reproducibility receipt.

### 6.4 Evaluation protocol (B-1)

The full protocol selected at brainstorming
([2026-06-03 decision in §10](#10-decision-references)):

| Phase | Cells used | Purpose |
|---|---|---|
| Development | 39 training cells | Fit isotonic regression, tune smoothing constant, pick conservative-max rule between B-α and B-β |
| Reporting | 54 holdout cells | Report empirical FDR at q ∈ {0.01, 0.05, 0.10}, report Power@0.05 |

The paper will carry a one-sentence explicit declaration:
"The FDR calibration procedure was developed on `data/bench_training_local.csv`
and reported on `data/bench_holdout_local.csv`. The holdout grid was
not used to tune the calibration."

### 6.5 Output schema additions

`EdgeTable.pvalue_approx` and `EdgeTable.qvalue_approx` populated when
`selection="stability"` and the calibration assets are present.
`EstimatorDiagnostics.calibration_method` becomes one of
`{"none", "empirical_isotonic_BH", "meinshausen_buhlmann_bound", "max_conservative"}`.

## 7. Line D — Real-data Evaluation

### 7.1 SECOM v1.0.0 workflow

1. Download via the existing `scripts/process_public_data.py` recipe in
   `data/public/secom_v1.0.0.md`. Record SHA-256 of the downloaded
   archive in the manifest.
2. Apply the documented preprocessing pipeline: prevalence ≥ 0.1, total
   count ≥ 1000 reads/sample, drop singleton features.
3. Run `infer_network(estimator="weighted_sparse", n_resamples=100, seed=0)`
   on the full preprocessed matrix.
4. Report: high-stability edge count at thresholds {0.7, 0.8, 0.9};
   density; selection_probability distribution histogram (50 bins);
   converged flag.

### 7.2 HMP 16S subproject workflow

Same as §7.1 but with the HMP 16S 4.5K rarefied OTU table and the
prevalence/total-count thresholds from `data/public/hmp_16s.md`.

### 7.3 Sample-holdout cross-validation

1. For each dataset: 5 × (50/50 random split by sample); seeds 0–4.
2. Run `infer_network` on each half; record high-stability edge sets
   (≥ 0.8 selection_probability).
3. Compute Jaccard agreement between the two halves; report mean ± SD
   across the 5 splits.

### 7.4 Published-network comparison (SECOM only)

1. From Lin et al. (2022) "Linear and nonlinear correlation estimators
   unveil undescribed taxa interactions in microbiome data" (Nat Commun,
   DOI 10.1038/s41467-022-32243-x), extract the published SECOM linear
   correlation network's top edges by absolute strength. The exact table
   identifier and edge count is fixed at Line D kickoff after reading
   the paper; the target is the top-10 edges if the published table
   reports ≥ 10, otherwise the full published list.
2. Map taxa identifiers between the SECOM archive (genus level after
   prevalence filter) and the published reference. Drop any reference
   edge whose endpoints do not survive our prevalence filter and record
   the count of dropped edges in the reproducibility receipt.
3. Report which of the surviving reference edges `weighted_sparse`
   recovers at selection_probability ≥ 0.8. Recall and per-edge evidence
   printed.
4. The reporting does not claim that this is "ground truth" — it is
   "concordance with the published reference network".

### 7.5 Reproducibility receipts

Every Line D output ships with:

- Dataset DOI or BioProject identifier
- SHA-256 of the downloaded archive
- Preprocessing parameters (prevalence threshold, total-count cutoff)
- Random seed
- `n_resamples` and `subsample_fraction`
- Edge stability threshold used
- Git commit hash of the package version
- A `data/public/{secom,hmp}_results.csv` source-data file backing every
  numerical claim

## 8. Execution Timeline

Hard cap: 7 weeks from 2026-06-04. Three workstreams run in parallel
with weekly check-ins on Friday.

| Week | Line A | Line B | Line D |
|---|---|---|---|
| 1 | A-α start: support_fn eigvalsh removal + joblib parallelism | B framework: schema additions, isotonic harness skeleton | D.1 data download + SHA-256 manifests |
| 2 | A-α done + differential test on training | B-β fit isotonic on training | D.2 SECOM subsampling stability |
| 3 | A-β start (Numba JIT + incremental M) | B-β tune smoothing + B-α theoretical bound | D.3 SECOM sample-holdout CV |
| 4 | A-β done + first holdout re-run | B-β + B-α validation on holdout | D.4 SECOM published-network compare |
| 5 | A-γ if Primary unmet, else hardening | B output schema integration | D repeat on HMP 16S |
| 6 | Final A holdout re-run, source-data v2 commit | B holdout final report | D HMP final report |
| 7 | Acceptance-gate v2 report + decision-log update; integration check (`Verify A+B+D Primary Targets Met`) — unblock academic-pipeline | | |

If any line slips past its week-N target by more than 3 days, the
weekly check-in escalates: drop a Stretch target, or invoke the
Acknowledged-Limitation path for an unmet Primary.

## 9. Risk Register

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | A-α + A-β fail to reach 100× SparCC on p ∈ {500, 1000} | medium | high | Trigger A-γ; if still unmet, document gap in decision-log and the paper's Acknowledged Limitations; do not move the target |
| R2 | B-β empirical isotonic overfits training scenarios | medium | medium | B-α theoretical bound serves as floor; conservative-max rule keeps reported FDR honest; report scenario-stratified empirical FDR to expose overfit if present |
| R3 | D published-network comparison recall is low | medium | medium | Do not hide; report raw numbers; interpret as "weighted_sparse ranks similarly to SparCC and recovers the published network's strong-edge core, missing weaker edges expected from a sparse estimator" |
| R4 | A optimization introduces numerical drift causing AUROC/AP regression > 0.005 | low | high | Per-commit differential test on training (§5.5 step 1); commit reverted if regression detected |
| R5 | Overrun beyond 7-week hard cap | medium | high | Hard cap is hard. Triggered escalation drops B-γ first, then A-γ, then a D-stretch (HMP). A primary Line A/B/D unmet at cap goes to Acknowledged Limitations rather than pushing the cap |
| R6 | Stability-selection parallelism breaks determinism | low | medium | Per-subsample seeding with `seed + k`; explicit determinism test added to `tests/test_stability.py` checking identical outputs across `n_jobs ∈ {1, 4}` |
| R7 | Public-data download fails (Zenodo outage, deprecated DOI) | low | medium | Manifest contains SHA-256 of the expected archive; if download fails, fall back to a cached snapshot recorded on a separate disk; reproducibility receipt records the cache hash |

## 10. Decision References

Brainstorming dialog 2026-06-03 (this design's parent conversation):

| Decision | Outcome |
|---|---|
| Goals to pursue | A (speed/memory) + B (FDR) + D (real-data); C (accuracy) **excluded** to preserve frozen-grid statistical guarantee |
| Depth tier | Middle — Strong-trade-off (~5–6 weeks; 7-week hard cap) |
| B evaluation protocol | B-1 (39 training cells for tuning, 54 holdout cells for reporting) |
| Stopping criteria | Accept all targets in §4 as Primary/Stretch as proposed; no adjustments |

Cross-references:

- Rebuild design: `2026-06-02-single-domain-estimator-rebuild-design.md`
- Acceptance evidence: `docs/acceptance-gate-report.md`
- Decision history: `docs/decision-log.md` (the 2026-06-03 entries set
  the constraints this design respects)
- Trade-off paper framing: `README.md` §"When to use which estimator"
- Methodology: `docs/methodology.md`

## 11. Out of Scope / Deferred

The following are explicit non-goals of this push. Each is recorded so
later contributors do not silently expand the scope.

1. Cross-domain inference, signed biological priors, nonlinear
   correlation as production estimators (rebuild design §3.2).
2. Re-tuning of any of the three estimator candidates' selection-critical
   parameters on holdout cells (decision-log 2026-06-03 "Open
   questions").
3. New estimator candidates beyond `weighted_sparse`, `adaptive_threshold`,
   `pd_sparse`.
4. Algorithmic research aimed at outranking `sparcc_closed_form` on
   AUROC/AP. The trade-off framing is locked.
5. LaTeX, Pandoc, or any manuscript-rendering work. That belongs to the
   academic-pipeline (Stage 5) and runs only after the integration check
   in week 7 passes.
6. Reduction of `n_resamples` from 100. Lower `n_resamples` lowers
   `selection_probability` resolution and would mask, not close, the
   runtime gap.

## 12. Out-of-band reviewer note

The repository has signed up — in `README.md`, `docs/decision-log.md`,
and `docs/acceptance-gate-report.md` — to publish negative findings
plainly when the gates fail. The optimization push respects this
commitment. Every Primary that goes unmet at the 7-week cap will appear
in the paper's Acknowledged Limitations section with its number rather
than being elided.

If the integration check (`Verify A+B+D Primary Targets Met`) at week 7
discovers that two or more Primary targets remain unmet across the three
lines, the user is consulted before triggering the academic-pipeline.
The choice is between (a) extending the cap with explicit additional
budget, (b) writing the paper with the unmet targets as documented
limitations, or (c) aborting the writing stage and returning to research.
This decision belongs to the user, not to the orchestrator.
