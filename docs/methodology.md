# Methodology — Single-Domain Compositional Network Estimator

This document describes how the rebuilt single-domain estimator works at the
level of its statistical claims. The full algorithm-level design lives in
[`superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md`](superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md).
Until the acceptance gates have been evaluated this repository does not assert
that the methodology beats published baselines.

## 1. Estimand

For a single-domain compositional dataset with latent basis abundances `w_i`,
the estimator targets the basis log-abundance Pearson correlation matrix:

```text
rho_ij = Corr(log w_i, log w_j)
```

This is the same target that SparCC, fastCCLasso, COAT, and SECOM-linear are
designed for. The estimator does not target proportionality and does not target
a precision matrix. Methods with a different estimand (SPIEC-EASI, CARE,
mmvec) are reported as adjacent-estimand context but are never used as
matched-estimand evidence for an advantage claim.

## 2. Pipeline

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

Zero handling is a sensitivity axis, not a hidden default. Three policies are
exposed: `multiplicative`, `pseudocount`, `complete_case`. The benchmark
records `zero_policy` per row.

## 3. Estimator candidates

Three Python-only estimators sit behind one common interface. All three are
clean-room implementations from the published method descriptions; no
reference R or LGPL code is consulted.

### 3.1 Adaptive threshold (COAT-style)

Compute the CLR sample covariance `S_clr`, estimate per-entry variance
`theta_ij = (1/n) sum_k (Z_ki Z_kj - sigma_ij)^2`, and apply entry-specific
thresholding:

```text
lambda_ij = c * sqrt(theta_ij * log p / n)
T_lambda(S_ij) = sign(S_ij) * max(|S_ij| - lambda_ij, 0)    (soft)
              or S_ij * I(|S_ij| >= lambda_ij)              (hard)
```

The diagonal is preserved untouched (it carries the basis variances which are
not identifiable from `S_clr` alone). The selected mode (hard vs. soft) is
frozen before the holdout grid is evaluated.

### 3.2 Weighted sparse covariance (fastCCLasso-style)

Alternates two cheap closed-form updates:

* nuisance-offset update `f` accounting for compositional closure
  (`Sigma_basis = S_clr + f1' + 1f'` plus a sparse perturbation), and
* weighted soft-threshold update on off-diagonal covariance entries with
  per-entry weight `w_ij = 1 / sqrt(theta_ij)`.

The diagonal remains unpenalized. Complexity is `O(p^2)` per iteration with
no eigendecomposition or linear solve.

### 3.3 Positive-definite sparse covariance

Applies the adaptive-threshold estimator first, then enforces a minimum
eigenvalue floor by diagonal loading. Diagonal loading is the only PD
correction we use because it preserves the off-diagonal selected support
exactly. The candidate is retained only if it improves numerical reliability
without losing accuracy or efficiency.

## 4. Stability-based uncertainty

The primary uncertainty output is `selection_probability`: the fraction of
subsamples in which an off-diagonal entry was non-zero. Subsamples are drawn
of size `n * subsample_fraction` without replacement. Under a fixed seed the
output is bit-reproducible.

`pvalue_approx` and `qvalue_approx` stay `None` unless a calibration
procedure whose simulation FDR behaviour has been measured fills them in. The
`uncertainty_interpretation` field on the diagnostics record names the regime
explicitly so downstream code never treats the values as a calibration-tight
test.

## 5. Frozen comparison protocol

The simulation grid (six scenarios) is split into training and holdout
configurations before any tuning. Holdout rows are never used to select
thresholds, lambda rules, or the winning estimator. The benchmark schema
records `estimand_family` so adjacent-estimand methods cannot be silently
promoted into match-evidence for an advantage claim.

Public-data evaluation reports subsampling stability mapped to the dataset
identifier — not biological truth.

## 6. References

The estimators draw on published method descriptions for:

1. Friedman J, Alm EJ. *Inferring correlation networks from genomic survey
   data.* PLoS Comput Biol (2012).
2. Cao Y, Lin W, Li H. *Large covariance estimation for compositional data
   via composition-adjusted thresholding.* JASA (2019).
3. Lin H, Eggesbo M, Peddada SD. *Linear and nonlinear correlation
   estimators unveil undescribed taxa interactions in microbiome data.*
   Nat Commun (2022). https://doi.org/10.1038/s41467-022-32243-x
4. Zhang S, Fang H, Hu T. *fastCCLasso.* Bioinformatics (2024).
   https://doi.org/10.1093/bioinformatics/btae314
5. Meinshausen N, Buhlmann P. *Stability selection.* J. R. Stat. Soc. B
   (2010).

The Python implementations in this repository do not vendor or copy code
from the references. Verification against external R baselines happens only
through the benchmark adapters at `benchmarks/r_adapters.py`, which call
already-installed R packages by name.
