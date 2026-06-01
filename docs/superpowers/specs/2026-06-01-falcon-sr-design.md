# Falcon-SR Design Specification

Date: 2026-06-01

Status: approved for implementation planning

## 1. Objective

Falcon-SR is a screen-refine algorithm for fast inference of latent
log-abundance Pearson correlations from compositional sequencing data.

The method targets the same estimand as SparCC and SparXCC:

```text
Corr(log absolute abundance_i, log absolute abundance_j)
```

The first release covers:

1. Single-domain compositional correlation inference.
2. Cross-domain correlation inference between two independently normalized
   compositions, such as bacteria and phages or bacteria and fungi.
3. Optional signed biological priors, such as phage-host evidence, implemented
   as a soft constraint that is disabled by default.

The primary operating regime is fewer than 1,000 samples and usually fewer than
5,000 features per domain.

## 2. Success Criteria

The default fast mode is allowed to be approximate. It optimizes recovery of
strong edges and the resulting network rather than element-wise reproduction of
an entire dense correlation matrix.

On the primary simulation grid, fast mode must target:

1. Candidate-screen recall of at least 0.99 relative to reference strong edges.
2. Final strong-edge overlap of at least 0.95 relative to SparCC for
   single-domain data and SparXCC for cross-domain data.
3. Final sign accuracy of at least 0.95 relative to the same references.
4. Accuracy against planted ground truth reported independently of agreement
   with the reference methods.
5. Measured wall-clock speedup and peak-memory reduction on the same hardware.

These are validation targets, not claims that may be made before experiments
pass.

## 3. Non-Goals

The first release does not:

1. Treat proportionality as a synonym for latent log-abundance correlation.
2. Claim causal interaction inference from correlations.
3. Treat precision-matrix methods as if they estimate the same quantity as
   marginal-correlation methods.
4. Use an unvalidated Fisher-z approximation as the default significance test
   for estimated latent correlations.
5. Require phage-host priors for the statistical estimator to work.
6. Optimize for feature counts far above 5,000 in the default implementation.
   Random projection or approximate-nearest-neighbor screening remains a
   later extension.

## 4. Terminology

| Term | Meaning |
|---|---|
| basis abundance | Unobserved absolute abundance before normalization |
| latent correlation | Pearson correlation between log basis abundances |
| screen | Fast generation of a sparse candidate-edge set |
| refine | Iterative correction restricted to candidates and affected nodes |
| strict mode | Dense reference-compatible path used for validation |
| fast mode | Sparse screen-refine path used by default |
| prior mode | Optional fast mode with finite-weight signed biological priors |

## 5. Statistical Foundation

### 5.1 Single-domain identity

For composition `x` with latent basis abundance `w`, define:

```text
t_ij = Var(log(x_i / x_j))
     = omega_i^2 + omega_j^2 - 2 * omega_i * omega_j * rho_ij
```

where `omega_i^2 = Var(log(w_i))` and `rho_ij` is the target latent
log-abundance correlation.

The initial SparCC-compatible estimate uses the sparse-average-correlation
approximation to recover basis variances from the variation matrix. The
variation matrix is computed through one covariance GEMM followed by
vectorized broadcasting:

```text
cov_log = centered_log_x.T @ centered_log_x / (n - 1)
t = diag(cov_log)[:, None] + diag(cov_log)[None, :] - 2 * cov_log
```

### 5.2 Cross-domain identity

For independently normalized compositions `X` and `Y`, the target is:

```text
rho_ik = Corr(log basis_X_i, log basis_Y_k)
```

Falcon-SR uses the SparXCC Case-C basis-variance and cross-correlation
identities for its dense initial score. Each domain receives its own
single-domain basis-variance estimate. The cross-domain score is computed by
matrix operations rather than Python-level pair loops.

### 5.3 Research hypothesis

Strong-edge recovery should not require repeated dense refinement over every
possible pair. A sparse candidate set generated from the dense base estimate
should retain almost all strong reference edges while allowing iterative
correction to operate only on candidates and their affected nodes.

This is the core research hypothesis. It must be accepted or rejected by the
candidate-recall experiments before the manuscript claims a new algorithm.

## 6. Architecture

### 6.1 Public API

```python
infer_single(
    counts,
    *,
    mode="fast",
    top_k=50,
    min_abs_score=None,
    zero_policy="multiplicative",
    calibration="permutation",
    seed=0,
) -> NetworkResult

infer_cross(
    counts_x,
    counts_y,
    *,
    mode="fast",
    top_k=50,
    min_abs_score=None,
    prior=None,
    prior_weight=0.0,
    zero_policy="multiplicative",
    calibration="permutation",
    seed=0,
) -> NetworkResult
```

`NetworkResult` contains:

```python
NetworkResult(
    edges=EdgeTable(...),
    diagnostics=Diagnostics(...),
    initial_matrix=None,
)
```

Strict mode may populate `initial_matrix`. Fast mode returns a sparse edge
table by default.

### 6.2 Module boundaries

The current monolithic package should be split by responsibility:

| File | Responsibility |
|---|---|
| `src/falcon/preprocessing.py` | validation, filtering, zero handling, log composition |
| `src/falcon/single.py` | single-domain dense base score and sparse refinement |
| `src/falcon/cross.py` | SparXCC-compatible dense base score and sparse refinement |
| `src/falcon/screen.py` | top-k union, thresholds, adaptive candidate growth |
| `src/falcon/calibration.py` | permutation calibration and reproducible seeds |
| `src/falcon/prior.py` | signed prior validation and finite-weight soft penalties |
| `src/falcon/types.py` | immutable result, edge, diagnostic, and prior data types |
| `src/falcon/__init__.py` | stable public exports |

Existing prototype functions may remain temporarily as compatibility wrappers
until their replacement paths are tested.

## 7. Data Flow

### 7.1 Shared preprocessing

1. Validate finite, non-negative, two-dimensional input.
2. Require paired sample rows for cross-domain inference.
3. Filter features by configurable prevalence and total-count thresholds.
4. Replace zeros according to an explicit `zero_policy`.
5. Normalize each sample to a composition.
6. Compute log compositions and record a preprocessing report.

Zero handling is a sensitivity parameter. It is not treated as a theorem.

### 7.2 Single-domain fast path

1. Compute the dense variation matrix with one GEMM.
2. Compute a dense SparCC-compatible base correlation score.
3. Build a candidate set with the union of:
   - per-node top-k absolute scores;
   - scores above an optional adaptive absolute threshold;
   - previously retained edges during adaptive growth.
4. Refine only candidate edges and affected node equations.
5. Grow the candidate budget if diagnostics show instability.
6. Calibrate retained edges by permutation.
7. Return a sparse edge table and diagnostics.

### 7.3 Cross-domain fast path

1. Compute separate basis-variance estimates for both domains.
2. Compute a dense SparXCC-compatible Case-C base score with matrix operations.
3. Build candidates using a bidirectional top-k union:
   - top-k `Y` partners for every `X` feature;
   - top-k `X` partners for every `Y` feature;
   - optional threshold candidates;
   - optional prior candidates.
4. Refine candidate edges while updating only affected row and column bias
   terms.
5. Grow the candidate budget if diagnostics show instability.
6. Calibrate retained edges by permutation.
7. Return the same sparse edge-table schema as the single-domain path.

## 8. Screening And Adaptive Growth

The default candidate budget starts at `top_k=50`, but this is a default for
evaluation, not a fixed scientific constant.

Fast mode runs a stability audit:

1. Infer candidates at `k`.
2. Infer candidates at `2k`.
3. Compare the refined top-edge sets and signs.
4. Accept the smaller budget if overlap and sign stability exceed configured
   thresholds.
5. Otherwise grow to `2k` and repeat until stable or until the candidate set is
   too dense.
6. Fall back to strict mode or report a diagnostic warning when sparse
   refinement no longer provides a meaningful reduction.

The diagnostic output includes:

```text
candidate_count
candidate_density
screen_budget
screen_growth_rounds
edge_overlap_across_budgets
sign_stability_across_budgets
fallback_reason
```

This prevents a fast result from silently degrading on dense or adversarial
correlation structures.

## 9. Sparse Refinement

### 9.1 Single domain

The first implementation should mirror SparCC's exclusion logic while changing
the execution scope:

1. Start from the dense base estimate.
2. Rank candidate edges by absolute correlation.
3. Exclude strong candidate pairs above the configured exclusion threshold.
4. Re-estimate only equations for nodes incident to excluded pairs.
5. Stop when no candidate crosses the threshold, the maximum exclusion count is
   reached, or the sparsity assumption is diagnostically violated.

The implementation must be checked against a strict small-matrix reference.

### 9.2 Cross domain

The cross-domain refinement starts from SparXCC Case C and restricts updates to
candidate edges and affected row-column bias terms. The base and iterative
SparXCC behaviors remain available separately because the SparXCC paper reports
regimes where iterative correction is not preferable.

Falcon-SR selects among:

```text
cross_base
cross_refined
cross_prior_refined
```

and reports which path produced the result. It does not silently assume that
iterative correction always improves estimates.

## 10. Optional Signed Biological Priors

Prior mode is optional and disabled when `prior_weight=0`.

A prior record contains:

```python
PriorEdge(
    source_feature="phage_a",
    target_feature="bacterium_b",
    expected_sign=-1,
    confidence=0.8,
    provenance="crispr_spacer",
)
```

Prior behavior:

1. Prior edges enter the candidate set even if the statistical screen would
   omit them.
2. During refinement, a finite-weight penalty encourages but does not force the
   expected sign.
3. A contradictory data signal may override the prior.
4. Edge output records the prior weight and whether data agreed with the prior.

The prior-centered penalty is:

```text
lambda_prior * confidence_e * (rho_e - sign_e * target_magnitude)^2
```

`target_magnitude` is deliberately weak and configurable. A prior provides a
soft direction, not an invented effect size.

Prior evaluation must include:

1. no-prior mode;
2. partial correct priors;
3. incomplete priors;
4. noisy priors;
5. wrong-sign priors.

## 11. Calibration

The default inference output separates ranking from calibration.

Ranking:

```text
screen score -> refined latent correlation -> retained edge order
```

Calibration:

```text
permutation null -> empirical edge or max-statistic threshold
```

The first release uses permutation thresholding because SparXCC explicitly
identifies difficulties with theoretical null tests in this setting.

To control cost:

1. Permutations reuse vectorized base-score primitives.
2. Fast mode calibrates candidate edges rather than every possible pair.
3. Sequential stopping may be added only after tests show that it preserves
   decisions relative to a fixed permutation budget.
4. The benchmark reports ranking metrics separately from calibrated power and
   false discovery rate.

## 12. Baseline Matrix

### 12.1 Same or closely aligned estimand

| Method | Year | Role |
|---|---:|---|
| SparCC | 2012 | original latent basis-correlation reference |
| FastSpar | 2019 | optimized SparCC implementation and permutation reference |
| CCLasso | 2015 | sparse latent basis-correlation estimator |
| REBACCA | 2015 | sparse latent basis-correlation estimator |
| COAT | 2019 | fast sparse basis-covariance thresholding baseline |
| SECOM linear | 2022 | sparse correlation estimator with sample and taxon bias modeling |
| fastCCLasso | 2024 | recent `O(p^2)`-per-iteration correlation-network baseline |
| SparXCC base and iterative | 2024 | cross-domain Case-C reference |
| Direct covariance estimation | 2024 | recent positive-definite latent covariance baseline |
| sparse basis covariance hard thresholding | 2024 | recent thresholding baseline |

### 12.2 Adjacent but different estimand

These methods belong in context tables but not in claims of correlation-matrix
equivalence:

| Method | Reason to separate |
|---|---|
| SPIEC-EASI | estimates conditional-dependence structure |
| CARE | estimates a sparse basis precision matrix |
| proportionality methods | estimate proportionality rather than latent Pearson correlation |
| mmvec | learns conditional co-occurrence representations |

### 12.3 Recent context and stress-test sources

These papers sharpen the evaluation design but are not direct same-estimand
baselines:

| Method or study | Year | Role |
|---|---:|---|
| Brunner et al. cross-kingdom reconstruction analysis | 2024 | motivates explicit cross-domain stress tests and warns against naive composition concatenation |
| LUPINE | 2025 | longitudinal partial-correlation direction, outside the first release |
| SpeSpeNet | 2025 | recent exploratory network tooling, useful context for filtering and zero-handling choices |

## 13. Simulation Plan

### 13.1 Single-domain primary grid

Core grid:

```text
n: 100, 250, 500, 1000
p: 100, 500, 1000, 2500, 5000
topology: random_sparse, hub, block
density: low, moderate
sign balance: balanced, positive_hub
zero regime: none, sampling_zeros, excess_zeros
sequencing depth: low, moderate, high
```

Stress scenarios:

1. Dense correlation matrices that violate the sparsity assumption.
2. Hub nodes with mostly same-sign edges.
3. Unequal basis variances.
4. Heavy-tailed latent log abundances.
5. Rare-feature enrichment.

### 13.2 Cross-domain primary grid

Core grid:

```text
n: 100, 250, 500, 1000
p, q: (100, 100), (500, 500), (1000, 1000), (500, 2500)
topology: sparse_bipartite, hub_host, block
mean cross-correlation: near_zero, shifted
zero regime: none, sampling_zeros, excess_zeros
prior coverage: 0, 0.1, 0.3, 0.5
prior correctness: 1.0, 0.9, 0.7
```

Stress scenarios:

1. Dense bipartite interactions.
2. Strongly unequal domain dimensions.
3. Prior sign errors.
4. Prior edges concentrated on hubs.
5. Batch effects and confounding covariates.

### 13.3 Metrics

Report:

```text
candidate recall
edge Jaccard overlap
Recall@K
precision@K
sign accuracy
AUROC
AUPRC
matrix RMSE in strict mode
power after calibration
false discovery rate after calibration
wall-clock time
peak resident memory
```

Every accuracy result is measured both against planted truth and against the
appropriate SparCC-family reference.

## 14. Public-Data Validation

Public data validate reproducibility, stability, runtime, and biological
plausibility. They do not provide ground-truth accuracy.

Initial public datasets:

1. The paired 16S and ITS root microbiome data used by SparXCC for
   bacteria-fungi cross-correlations.
2. The forehead-palm microbiome analysis used by SECOM to test related
   ecosystems.
3. A single-domain public microbiome dataset used by fastCCLasso or COAT when
   its processed input and metadata are reproducibly available.
4. A cross-kingdom environmental dataset from the Brunner et al. reconstruction
   study when its processed input and metadata are reproducibly available.

For each dataset, report:

```text
runtime
peak memory
edge overlap with reference method
sign agreement
subsample stability
parameter sensitivity
```

## 15. Test Strategy

Implementation follows test-driven development.

Unit tests:

1. Input validation rejects negative, non-finite, and mismatched sample rows.
2. Zero handling preserves finite log-ratios and row sums.
3. Vectorized variation matrices match pair-loop reference calculations.
4. Single-domain base scores match a small strict SparCC reference.
5. Cross-domain base scores match a small strict SparXCC reference.
6. Candidate unions are symmetric for single-domain inference.
7. Bidirectional candidate unions include both domain directions.
8. Candidate sets grow monotonically with `top_k`.
9. `prior_weight=0` leaves no-prior results unchanged.
10. Wrong-sign priors remain soft and can be overridden by strong data.
11. Permutation calibration is reproducible under a fixed seed.

Property tests:

1. Single-domain outputs are symmetric with unit diagonals in strict mode.
2. Correlations remain in `[-1, 1]`.
3. Cross-domain shapes are `(p, q)`.
4. Fast-mode edge tables contain no self edges or duplicates.
5. Adaptive growth either stabilizes or emits an explicit fallback reason.

Integration tests:

1. Small single-domain simulation achieves high overlap with strict SparCC.
2. Small cross-domain simulation achieves high overlap with strict SparXCC.
3. Prior mode improves recovery when supplied with partial correct priors.
4. Wrong priors do not silently dominate a strong observed signal.
5. Benchmark smoke run writes schema-valid CSV output.

## 16. Benchmark Rules

1. Use `uv` for the Python environment.
2. Run wall-clock comparisons on the same host, core allocation, and input
   matrix.
3. Distinguish measured runtime from extrapolated runtime.
4. Report whether a baseline uses dense output, sparse output, permutations,
   bootstrap resampling, or iterative refinement.
5. Compare ranking-only runtime separately from calibrated-network runtime.
6. Preserve raw replicate-level benchmark rows before aggregation.
7. Record package versions, CPU model, BLAS backend, thread count, random seed,
   and peak resident memory.

## 17. Figure Contract

Backend: Python only.

Core conclusion:

```text
Falcon-SR preserves strong latent log-abundance correlation edges from
SparCC-compatible estimators while reducing runtime through sparse refinement;
optional signed biological priors add an independent cross-domain gain.
```

Figure archetype: schematic-led composite.

Target: editable SVG and PDF plus 600-DPI TIFF.

Figure 1 panel map:

```text
a: single-domain screen-refine workflow
b: candidate recall and final overlap versus candidate budget
c: accuracy versus SparCC, FastSpar, fastCCLasso, COAT, and SECOM-linear
d: same-host wall-clock and peak memory versus feature count
```

Figure 2 panel map:

```text
a: cross-domain screen-refine workflow with optional prior branch
b: overlap, sign accuracy, AUROC, and AUPRC versus SparXCC
c: runtime versus p and q
d: prior ablation under increasing prior coverage and prior noise
```

Reviewer risks to expose visually:

1. Accuracy-speed trade-off as `top_k` changes.
2. Sparse-assumption failure on dense networks.
3. Difference between ranking and calibrated-network runtime.
4. Prior benefit and degradation under prior noise.

## 18. Literature Basis

Primary sources checked during design:

1. Friedman J, Alm EJ. Inferring correlation networks from genomic survey
   data. PLoS Computational Biology (2012).
   https://doi.org/10.1371/journal.pcbi.1002687
2. Watts SC et al. FastSpar: rapid and scalable correlation estimation for
   compositional data. Bioinformatics (2019).
   https://doi.org/10.1093/bioinformatics/bty734
3. Cao Y, Lin W, Li H. Large covariance estimation for compositional data via
   composition-adjusted thresholding. JASA (2019).
   https://doi.org/10.1080/01621459.2018.1442340
4. Lin H, Eggesbo M, Peddada SD. Linear and nonlinear correlation estimators
   unveil undescribed taxa interactions in microbiome data. Nature
   Communications (2022).
   https://doi.org/10.1038/s41467-022-32243-x
5. Zhang S, Fang H, Hu T. fastCCLasso: a fast and efficient algorithm for
   estimating correlation matrix from compositional data. Bioinformatics
   (2024).
   https://doi.org/10.1093/bioinformatics/btae314
6. Jensen IT et al. Compositionally aware estimation of cross-correlations for
   microbiome data. PLoS ONE (2024).
   https://doi.org/10.1371/journal.pone.0305032
7. Molstad AJ, Ekvall KO, Suder PM. Direct covariance matrix estimation with
   compositional data. Electronic Journal of Statistics (2024).
   https://doi.org/10.1214/24-EJS2222
8. Zhang S, Wang H, Lin W. CARE: Large precision matrix estimation for
   compositional data. JASA (2024).
   https://doi.org/10.1080/01621459.2024.2335586
9. Sparse basis covariance matrix estimation for high dimensional
   compositional data via hard thresholding. Statistics and Probability
   Letters (2024).
   https://doi.org/10.1016/j.spl.2024.110088
10. Brunner JD, Robinson AJ, Chain PSG. Combining compositional data sets
    introduces error in covariance network reconstruction. ISME Communications
    (2024).
    https://doi.org/10.1093/ismeco/ycae057
11. Kodikara S, Le Cao KA. Microbial network inference for longitudinal
    microbiome studies with LUPINE. Microbiome (2025).
    https://doi.org/10.1186/s40168-025-02041-w
12. SpeSpeNet: an interactive and user-friendly tool to create and explore
    microbial correlation networks. ISME Communications (2025).
    https://doi.org/10.1093/ismeco/ycaf036

## 19. Open Research Risks

1. Base-score top-k screening may miss strong edges after full iterative
   correction in hub-heavy or dense regimes.
2. Sparse refinement may save less wall-clock time than expected if the dense
   initial GEMM dominates at `p < 5000`.
3. FastSpar's optimized C++ implementation may remain faster for ranking-only
   workloads even if Falcon-SR reduces refinement work.
4. Candidate-only permutation calibration may require a selective-inference
   correction or a separate screening-calibration split to control false
   discovery rate.
5. Cross-domain soft priors can improve accuracy only when their provenance and
   uncertainty are represented honestly.

Each risk is part of the benchmark plan. A failed hypothesis should narrow the
paper claim rather than be hidden by presentation.
