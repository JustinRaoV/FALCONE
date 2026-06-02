# Falcon-SR Methodology

## 1. Estimand

Falcon-SR estimates the Pearson correlations of the latent (unobserved)
log-absolute-abundance vectors that compositional sequencing samples induce.
For a single domain with basis abundances $w_i$,

$$
\rho_{ij} \;=\; \mathrm{Corr}\bigl(\log w_i,\; \log w_j\bigr),
$$

and for two independently normalised compositions with basis abundances
$w^X, w^Y$,

$$
\rho_{ik} \;=\; \mathrm{Corr}\bigl(\log w_i^X,\; \log w_k^Y\bigr).
$$

This is the same estimand that SparCC (Friedman & Alm, 2012) and SparXCC
Case-C (Jensen et al., 2024) target. Falcon-SR is **not** a proportionality
estimator; the spec explicitly rejects equating proportionality $\rho_p$ with
latent log-abundance correlation (`docs/superpowers/specs/2026-06-01-falcon-sr-design.md`
§3 non-goals).

## 2. Single-Domain Algorithm

### 2.1 Variation matrix and base score

The Aitchison variation matrix

$$
t_{ij} \;=\; \mathrm{Var}\bigl(\log(x_i / x_j)\bigr)
       \;=\; \omega_i^2 + \omega_j^2 - 2\,\omega_i\,\omega_j\,\rho_{ij}
$$

is computed through one BLAS covariance GEMM plus vectorised broadcasting:

```
cov_log = centered_log_x.T @ centered_log_x / (n - 1)
t       = diag(cov_log)[:, None] + diag(cov_log)[None, :] - 2 * cov_log
```

The basis variances $\omega_i^2$ come from the SparCC sparse-average-
correlation closed form. The base correlation is

$$
\hat\rho_{ij} = \frac{\omega_i^2 + \omega_j^2 - t_{ij}}{2\,\omega_i\,\omega_j}.
$$

### 2.2 Sparse candidate union

`single_candidates` builds a symmetric top-k union of the base score: for
every feature $i$, retain the $k$ partners with largest $|\hat\rho_{ij}|$;
canonicalise to `i < j` and deduplicate. An optional `min_abs_score`
threshold adds additional candidates above a configured magnitude.

### 2.3 Sparse refinement

For each excluded pair $(i, j) \in E$, the basis-variance linear system is
solved on the sparse complement using a SciPy `LinearOperator + cg` matvec
that never materialises the dense $p \times p$ modifier. Candidate edge
scores are recomputed each round from the refreshed basis variances; the
loop exits when no candidate crosses `exclusion_threshold` or after
`max_exclusions` rounds.

### 2.4 Adaptive growth

`infer_single` runs the screen-refine pipeline at `top_k`, then again at
`2 * top_k`, and compares the resulting top-edge sets for both Jaccard
overlap and sign stability. If both meet `stability_threshold` the smaller
budget is accepted; otherwise growth continues until convergence or
`max_top_k` is reached, at which point `diagnostics.fallback_reason`
records the failure.

## 3. Cross-Domain Algorithm

### 3.1 SparXCC Case-C identity

Falcon-SR adopts the SparXCC Case-C double-centred identity for the cross
base score:

$$
\hat\rho_{ik} \;=\; \frac{\bigl(H_p\,\mathrm{cov}(\log x, \log y)\,H_q^\top\bigr)_{ik}}{\alpha_i\,\beta_k},
$$

where $H_p$ and $H_q$ are the centring matrices and $\alpha, \beta$ are the
per-domain SparCC basis standard deviations. This is the dense reference
recovered by `benchmarks.comparison_methods.sparxcc_base`; the Falcon-SR
implementation in `src/falcon/cross.py` agrees with it to $10^{-10}$ on
small inputs.

### 3.2 Bidirectional top-k candidate union

`cross_candidates` returns the union of (a) top-k Y partners per X feature,
(b) top-k X partners per Y feature, and (c) optional threshold candidates.

### 3.3 Edge-driven sparse refinement

The sparse refinement prunes one X row and one Y column from the centring
pool for every excluded candidate $(i, k)$:

```
S = { i in [p] : no (i, k) in E }
T = { k in [q] : no (i, k) in E }
centered = cov_xy - rowmean(S) - colmean(T) + grand(S × T)
rho      = centered / (alpha ⊗ beta)
```

When $|S| < 3$ or $|T| < 3$ the centring falls back to the full
$S = [p], T = [q]$ pool and `diagnostics.fallback_to_base_centering` is
set. This refinement geometry reduces to SparXCC base when $E = \varnothing$
and matches SparXCC iter when the candidate set covers all pairs.

### 3.4 Adaptive growth

Same machinery as the single-domain path, using the bipartite candidate
density and a cross-domain analogue of the strong-edge Jaccard / sign
stability comparison.

## 4. Optional Signed Biological Priors

A `PriorEdge` carries `(source_feature, target_feature, expected_sign,
confidence, provenance)`. With `prior_weight > 0`:

1. The prior pair is forced into the candidate set even when the
   statistical screen omits it.
2. After sparse refinement the candidate edge score is replaced by the
   analytic minimiser of
   $(\rho - \hat\rho_{\text{data}})^2 + \lambda\,\text{conf}\,(\rho - \text{sign}\cdot\text{target})^2$,
   which gives
   $\rho = (\hat\rho_{\text{data}} + \lambda\,\text{conf}\,\text{sign}\,\text{target})
            / (1 + \lambda\,\text{conf})$.

The prior does not influence the iterative exclusion choices inside
refinement, and `diagnostics.data_disagreed_with_prior_count` records how
many priors fought the data. With `prior_weight = 0` the prior pipeline is a
no-op: the score equals $\hat\rho_{\text{data}}$ exactly and prior pairs are
not injected.

## 5. Permutation Calibration

Calibration permutes columns of the log composition independently per
permutation (single domain) or shuffles Y rows relative to X (cross
domain), recomputes the closed-form base correlation, and records
$\max_{(i,j) \in \text{candidates}} |\hat\rho^{\text{perm}}_{ij}|$ to form
an empirical FWER null. Edge p-values are the standard add-one estimate

$$
\hat{p}_e \;=\; \frac{1 + \#\{r : \max_r \ge |\hat\rho_e^{\text{refined}}|\}}{1 + R},
$$

and q-values follow Benjamini-Hochberg. The result is labelled
`permutation_base_only` to flag that we skip rerunning sparse refinement per
permutation; this is an approximation, defended in execution-design §2.3 as
the only path that stays inside the feasibility wall-clock budget for
$p \le 1000$ and $R = 100$. Spec §19 risk 4 is therefore left explicit
rather than silently fixed.

## 6. Public API

```
infer_single(counts, *, mode="fast", top_k=50, max_top_k=None,
             min_abs_score=None, exclusion_threshold=0.1,
             max_exclusions=10, stability_threshold=0.95,
             zero_policy="multiplicative",
             calibration="permutation", n_permutations=100, seed=0)
            -> NetworkResult

infer_cross(counts_x, counts_y, *, mode="fast", top_k=50, max_top_k=None,
            min_abs_score=None, exclusion_threshold=0.1,
            max_exclusions=10, stability_threshold=0.95,
            zero_policy="multiplicative",
            prior=None, prior_weight=0.0, prior_target_magnitude=0.3,
            calibration="permutation", n_permutations=100, seed=0)
           -> NetworkResult
```

`mode="strict"` returns the dense reference correlation matrix; `mode="fast"`
returns a sparse edge table from the screen-refine path. `seed` controls
permutation reproducibility.

## 7. Reference Implementations Used as Validation Anchors

| Reference | File | Used for |
|---|---|---|
| SparCC closed-form | `benchmarks/comparison_methods.py::sparcc_py` | Single-domain base equivalence |
| SparXCC base | `benchmarks/comparison_methods.py::sparxcc_base` | Cross-domain base equivalence |
| SparXCC iter | `benchmarks/comparison_methods.py::sparxcc_iter` | Cross-domain ranking baseline |
| Pearson(CLR) | `benchmarks/comparison_methods.py::pearson_clr` | Sanity baseline |
| SPIEC-EASI glasso / MB | `benchmarks/comparison_methods.py::spieceasi_*` | Adjacent-estimand reference |

## 8. Feasibility Benchmarks

Two runners, two CSVs:

- `benchmarks/falcon_sr_single.py` → `data/falcon_sr_single_feasibility.csv`
- `benchmarks/falcon_sr_cross.py`  → `data/falcon_sr_cross_feasibility.csv`

See `docs/superpowers/specs/2026-06-02-falcon-sr-rewrite-execution-design.md`
§5 for cells and acceptance gates.

## 9. References

1. Friedman J, Alm EJ. Inferring correlation networks from genomic survey
   data. *PLoS Comput Biol* (2012).
2. Watts SC et al. FastSpar: rapid and scalable correlation estimation for
   compositional data. *Bioinformatics* (2019).
3. Jensen IT et al. Compositionally aware estimation of cross-correlations
   for microbiome data. *PLoS ONE* (2024).
4. Cao Y, Lin W, Li H. Large covariance estimation for compositional data
   via composition-adjusted thresholding (COAT). *JASA* (2019).
5. Lin H, Eggesbø M, Peddada SD. Linear and nonlinear correlation
   estimators (SECOM). *Nat Commun* (2022).
6. Brunner JD, Robinson AJ, Chain PSG. Combining compositional data sets
   introduces error in covariance network reconstruction.
   *ISME Commun* (2024).
7. Aitchison J. *The Statistical Analysis of Compositional Data.* Chapman
   and Hall (1986).
