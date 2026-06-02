# Single-Domain Compositional Network Estimator Rebuild Design

Date: 2026-06-02

Status: approved for implementation planning

## 1. Objective

Rebuild the repository around a statistically defensible single-domain
compositional network estimator. The production implementation remains
Python-only. It must earn its place through a frozen comparison protocol
rather than inherit the current Falcon-SR claims.

The first release succeeds only if one Python estimator simultaneously:

1. improves edge recovery against the strongest matched-estimand baseline;
2. provides an honest uncertainty output with validated interpretation;
3. improves runtime and peak memory on medium- and high-dimensional grids;
4. remains stable under subsampling on public microbiome data; and
5. can be reproduced from committed benchmark code and source-data tables.

If no estimator clears every gate, the repository must report the negative
result plainly. It must not publish an advantage claim.

## 2. Why the Current Architecture Is Replaced

The current Falcon-SR fast path computes a full dense SparCC-compatible base
matrix before candidate screening. The screen and sparse-refinement stages
therefore add work after the quadratic bottleneck instead of removing it.

The committed feasibility tables expose two failures:

1. On the single-domain hard cell `n=100, p=1000, top_k=10`, Falcon-SR fast
   has candidate recall about `0.075` and wall-clock about `0.120 s`, while
   the SparCC closed-form baseline takes about `0.018 s`.
2. On the cross-domain hard cell `n=100, p=q=500, top_k=10`, Falcon-SR fast
   has candidate recall about `0.756` and wall-clock about `0.097 s`, while
   the SparXCC base baseline takes about `0.004 s`.

This is an architectural mismatch, not a tuning defect. The rebuild does not
preserve the current screen-refine method as the primary estimator.

## 3. Scope

### 3.1 Included in phase 1

1. Single-domain count matrices only.
2. Python implementations of three estimator candidates.
3. A frozen simulation harness with stronger data-generating processes.
4. Python and R baseline adapters used only by benchmark code.
5. Stability selection and honest calibration diagnostics.
6. Public real-data stability evaluation.
7. A clean source-data layout for future figures and manuscript claims.
8. Removal of the stale Falcon-SR manuscript, generated figures, generated
   benchmark CSVs, ignored caches, and obsolete prototype documentation.

### 3.2 Explicitly deferred

1. Cross-domain inference.
2. Signed biological priors.
3. Nonlinear correlation as a production estimator.
4. Bayesian modeling.
5. A full manuscript rewrite.

Cross-domain inference and priors may return only after the single-domain
estimator clears every gate. The public package must not imply that deferred
capabilities are validated.

## 4. Candidate Estimators

The rebuild evaluates three Python-only candidate estimators behind one common
interface. Only candidates with a coherent estimand remain eligible for the
public API.

### 4.1 Weighted sparse covariance candidate

This candidate follows the statistical idea used by fastCCLasso: estimate a
sparse basis covariance matrix from the CLR covariance through penalized
weighted least squares. The Python implementation is a clean-room
implementation derived from the published method description, not copied from
the LGPL R reference code.

The optimization alternates between:

1. nuisance-offset updates that account for compositional closure; and
2. weighted soft-threshold updates for off-diagonal covariance entries.

The diagonal remains unpenalized. The covariance estimate is converted into a
correlation matrix only after convergence checks pass.

### 4.2 Adaptive threshold covariance candidate

This candidate follows the COAT family: estimate a composition-adjusted
covariance matrix, then apply entry-specific thresholding based on estimated
variability. The method is computationally simple and provides an important
speed floor.

The implementation supports hard and soft thresholding internally, but the
benchmark freezes one selected mode before the holdout grid is evaluated.

### 4.3 Positive-definite sparse covariance candidate

This candidate extends thresholded covariance estimation with a
positive-definite correction inspired by direct covariance estimation for
compositional data. The correction must preserve symmetry and bound the minimum
eigenvalue without silently changing the selected-edge support.

This path remains optional in the final public API. It is retained only if the
positive-definite correction improves numerical reliability without losing the
accuracy and efficiency gates.

## 5. Public Python Interface

The first public surface is intentionally small:

```python
from falcon import infer_network

result = infer_network(
    counts,
    estimator="weighted_sparse",
    zero_policy="multiplicative",
    selection="stability",
    n_resamples=100,
    seed=0,
)
```

The result schema is:

```python
NetworkResult(
    edges=EdgeTable(
        pairs=...,
        scores=...,
        selection_probability=...,
        pvalue_approx=...,
        qvalue_approx=...,
    ),
    diagnostics=EstimatorDiagnostics(
        estimator=...,
        lambda_value=...,
        converged=...,
        iterations=...,
        min_eigenvalue=...,
        calibration_method=...,
        uncertainty_interpretation=...,
        preprocess_report=...,
    ),
    correlation=...,
)
```

`selection_probability` is always the primary uncertainty output when
resampling is enabled. Approximate p-values and q-values are populated only for
a calibration procedure whose simulation FDR behavior has been measured and
whose interpretation is recorded in `uncertainty_interpretation`.

The legacy `infer_single`, `infer_cross`, and `PriorEdge` exports are removed
after compatibility tests prove that no retained benchmark or documentation
path still depends on them.

## 6. Data Flow

```text
raw counts
  -> validate finite non-negative matrix
  -> prevalence and total-count filter
  -> zero-handling policy
  -> row normalization and CLR transform
  -> estimator candidate
  -> sparse covariance and correlation
  -> stability selection by subsampling
  -> optional validated calibration
  -> edge table, diagnostics, and source-data rows
```

Zero handling is a sensitivity axis, not a hidden implementation detail.
`multiplicative`, `pseudocount`, and complete-case pairwise sensitivity runs are
reported separately. The default may be chosen only after the training grid is
evaluated.

## 7. Baseline Policy

### 7.1 Python baselines

1. SparCC closed-form and iterative reference paths.
2. Pearson correlation after CLR transformation.
3. COAT-style threshold covariance when it is not the selected production
   estimator.
4. Weighted sparse covariance candidates when they are not selected as the
   production estimator.

### 7.2 R baselines

R code runs only in benchmark adapters:

1. `fastCCLasso`
2. `COAT`
3. `SECOM`

The production package does not invoke R. R baseline output is normalized into
the same benchmark schema and labeled with its package or repository version.

### 7.3 Estimand labeling

Precision-matrix and nonlinear methods may be reported as adjacent-estimand
context, but they must not be used as matched-estimand evidence for a win.
Every benchmark row records `estimand_family`.

## 8. Simulation Protocol

The simulation grid is split into training and holdout configurations before
method tuning. Holdout rows are never used to select thresholds, lambda rules,
or the winning estimator.

Each run records:

```text
scenario
split
seed
n
p
density
zero_fraction
distribution
method
estimand_family
auroc
average_precision
recall_at_k
precision_at_k
fdr_at_target
wallclock_seconds
peak_bytes
converged
```

The frozen scenarios are:

1. sparse random graph;
2. hub graph;
3. block graph;
4. heavy-tailed latent abundance;
5. negative-binomial counts with zero inflation; and
6. sensitivity runs across low, medium, and high `n/p` ratios.

Training grids use small local cells. Holdout grids add larger cells and run
through a generated server script so the full experiment can be executed
outside the laptop without changing the schema.

## 9. Real-Data Validation

The first public-data validation uses the archived SECOM materials associated
with Lin, Eggesbo, and Peddada (2022), including the forehead and palm skin
microbiome illustration and the Norwegian Microbiome study illustration. The
paper archive is fixed at version `v1.0.0`:

```text
https://doi.org/10.5281/zenodo.6809029
```

The corresponding concept DOI is `10.5281/zenodo.6809028`.

The second public-data route is the NIH Human Microbiome Project umbrella
BioProject:

```text
PRJNA43021
```

with the 16S rRNA diversity subproject:

```text
PRJNA48489
```

The implementation must not commit raw third-party datasets. It commits:

1. download instructions and stable identifiers;
2. checksum records for downloaded archives;
3. a processing script;
4. processed source-data tables only when redistribution terms allow it; and
5. subsampling-stability summaries mapped to the public identifier.

Real-data evaluation reports stability and reproducibility. It does not claim
biological truth from the absence of ground-truth edges.

## 10. Figure Contract

Core conclusion:

> The selected Python estimator is retained only if it improves edge recovery,
> uncertainty behavior, and computational efficiency under a frozen fair
> comparison.

Figure archetype: `quantitative grid`.

Backend: Python only.

Planned figure evidence:

1. hero panel: holdout AUROC and Recall@K against matched-estimand baselines;
2. calibration panel: empirical FDR against nominal target;
3. efficiency panel: wall-clock and peak memory across `p`;
4. robustness panel: scenario and zero-policy sensitivity;
5. real-data panel: subsampling edge stability on public datasets.

Figures are regenerated only after the selected estimator clears the acceptance
gates. SVG, PDF, and TIFF exports are produced by Python with editable SVG/PDF
text and 600-DPI TIFF output.

## 11. Data Availability Package

The repository must contain:

1. `data/README.md` mapping each generated table to its benchmark command and
   figure panel;
2. `data/manifest.tsv` with file role, provenance, generator, and licence;
3. stable public-data identifiers and download instructions;
4. generated simulation source-data tables;
5. environment metadata sufficient to reproduce benchmark runs; and
6. a ready-to-adapt Data Availability statement that distinguishes generated
   simulation data, derived public-data summaries, third-party archives, and
   code.

No repository DOI, accession, licence, or archive URL may be invented.

## 12. Cleanup

The first implementation commit removes:

1. `manuscript/`
2. `data/falcon_sr_single_feasibility.csv`
3. `data/falcon_sr_cross_feasibility.csv`
4. ignored Python caches and pytest caches
5. obsolete Falcon-SR manuscript claims and old FastProp wording

The existing local `.worktrees/` directory and its branches are preserved.
They are historical recovery points. Old algorithm modules are removed only
after replacement imports, tests, and benchmark smoke paths are green.

## 13. Testing Strategy

Implementation follows test-driven development.

Required test groups:

1. preprocessing validation and zero-policy sensitivity;
2. weighted sparse optimizer convergence and symmetry;
3. adaptive threshold support recovery;
4. positive-definite correction eigenvalue floor;
5. stability-selection reproducibility under fixed seed;
6. uncertainty schema honesty;
7. benchmark schema validation;
8. R adapter skip behavior when R dependencies are absent;
9. public-data manifest validation; and
10. package API removal checks for obsolete exports.

Reference tests compare the clean-room Python candidates against small
hand-computed matrices and, where licensing permits runtime comparison,
against external R baseline output without vendoring R source code.

## 14. Acceptance Gates

The selected estimator must clear all gates on frozen holdout cells:

1. AUROC and Recall@K each exceed the strongest matched-estimand baseline on
   the primary sparse and zero-inflated scenarios.
2. Empirical FDR is reported at nominal targets `0.01`, `0.05`, and `0.10`.
   Approximate q-values are exposed only if the observed calibration is
   defensible across holdout scenarios.
3. Medium- and high-dimensional runtime and peak memory each improve against
   the strongest accurate baseline.
4. All selected-estimator runs converge or return an explicit non-convergence
   diagnostic.
5. Public real-data subsampling produces a stability report with dataset
   identifier, seed, resample count, and selection threshold.
6. Every numerical claim maps to a committed source-data row and generator
   command.

Failure to clear any gate blocks advantage claims. Negative results remain
valid outputs.

## 15. Implementation Phases

1. Remove stale generated outputs and obsolete claims.
2. Replace package types and preprocessing contracts.
3. Implement the adaptive threshold covariance candidate.
4. Implement the weighted sparse covariance candidate.
5. Implement the positive-definite correction.
6. Add stability selection and honest uncertainty fields.
7. Replace simulation and benchmark harnesses.
8. Add external R baseline adapters.
9. Add public-data manifest and processing entrypoints.
10. Run the local training grid, freeze choices, then generate the server
    holdout script.
11. Rebuild source-data tables and figures only after acceptance-gate review.

## 16. Definition of Done

Phase 1 is complete only when:

1. the package exposes the new single-domain API and no validated-looking
   cross-domain API;
2. all Python tests pass;
3. local benchmark smoke runs produce schema-valid rows;
4. the server holdout command is generated and documented;
5. R baseline adapters either run successfully or skip with explicit reasons;
6. public-data download and processing instructions use stable identifiers;
7. stale Falcon-SR data, manuscript files, and claims are removed;
8. data manifest and Data Availability draft exist; and
9. the repository states whether the estimator cleared or failed each gate.
