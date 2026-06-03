# FALCON — Single-Domain Compositional Network Estimator

> **Status: rebuild in progress.** This repository was reset on 2026-06-02 to
> drop unvalidated screen-refine claims and rebuild around a statistically
> defensible single-domain estimator. The Python package compiles, the new
> public API is wired end-to-end, and tests are green — but **no acceptance
> gate has been evaluated yet**. See [`Acceptance Gates`](#acceptance-gates).

The goal of the first release is one Python estimator that simultaneously:

1. improves edge recovery against the strongest matched-estimand baseline;
2. provides an honest uncertainty output with validated interpretation;
3. improves runtime and peak memory on medium- and high-dimensional grids;
4. remains stable under subsampling on public microbiome data; and
5. can be reproduced from committed benchmark code and source-data tables.

If no estimator clears every gate, this repository is required to publish the
negative result plainly. It will not publish an advantage claim.

The full design is in
[`docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md`](docs/superpowers/specs/2026-06-02-single-domain-estimator-rebuild-design.md).

---

## Quickstart

```bash
uv sync
uv run pytest -q
```

```python
import numpy as np
from falcon import infer_network

rng = np.random.default_rng(0)
counts = rng.integers(1, 200, size=(200, 50))

result = infer_network(
    counts,
    estimator="weighted_sparse",     # or "adaptive_threshold", "pd_sparse"
    zero_policy="multiplicative",    # or "pseudocount", "complete_case"
    selection="stability",
    n_resamples=100,
    seed=0,
)

print(result.edges.pairs[:5], result.edges.scores[:5])
print("selection prob:", result.edges.selection_probability[:5])
print("estimator:", result.diagnostics.estimator)
print("converged:", result.diagnostics.converged)
print("min eigenvalue:", result.diagnostics.min_eigenvalue)
```

`NetworkResult` carries:

- `edges` — `EdgeTable(pairs, scores, selection_probability, pvalue_approx, qvalue_approx)`.
  `selection_probability` is the primary uncertainty output. `pvalue_approx`
  / `qvalue_approx` stay `None` until a calibration procedure whose
  simulation FDR behavior has been measured fills them in.
- `diagnostics` — `EstimatorDiagnostics(estimator, lambda_value, converged,
  iterations, min_eigenvalue, calibration_method, uncertainty_interpretation,
  preprocess_report, notes)`.
- `correlation` — full `(p, p)` correlation matrix the estimator produced.

---

## Estimator candidates

Three candidates share the public entrypoint. Each is a clean-room Python
implementation derived from the published method description, not copied from
any reference R code.

| Key | Role | Description |
|---|---|---|
| `weighted_sparse` | **production default** (frozen 2026-06-03 on training grid) | fastCCLasso-style weighted soft-thresholded covariance, alternating offset + soft-threshold updates. |
| `adaptive_threshold` | auxiliary | COAT-style composition-adjusted thresholding with hard or soft thresholding. |
| `pd_sparse` | auxiliary | Adaptive threshold + diagonal-loading PD correction that preserves selected support. Same edge ranking as `adaptive_threshold`; useful when the consumer needs a PD covariance. |

Three zero-handling policies are exposed as a sensitivity axis:

| Policy | When to use |
|---|---|
| `multiplicative` (default) | Low and moderate zero fraction (≤ 15 %). |
| `pseudocount` | Equivalent to `multiplicative` on training cells; available for backward-compatibility studies. |
| `complete_case` | **Re-run with this when zero fraction > 0.20.** Training showed +0.15 AUROC on `negative_binomial_zi` over `multiplicative`. |

The benchmark records `zero_policy` per row so the choice is never silent.
See `docs/decision-log.md` for the training-grid evidence behind these
defaults.

---

## Benchmark

The frozen benchmark schema is documented in design §8. A local run looks
like:

```bash
uv run python benchmarks/run_benchmark.py \
    --split training \
    --output data/bench_training.csv \
    --reps 1
```

The holdout grid is bigger and is wrapped in a generated server script:

```bash
bash benchmarks/run_server_holdout.sh
```

R baselines are invoked via subprocess adapters that **skip with an
explicit reason** when R or the upstream source repos are not present.
The production package never invokes R.

| Method | Source | Notes |
|---|---|---|
| `cclasso` | https://github.com/huayingfang/CCLasso (LGPL-2.1+) | Fang et al. (2015) v2.0; closest publicly-available analog to fastCCLasso (the 2024 paper does not publish a standalone R package). |
| `coat` | https://github.com/yuanpeicao/COAT (research code) | Cao, Lin, Li (2019) reference implementation, soft thresholding mode. |
| `secom` | inside ANCOMBC (Bioconductor) | Lin, Eggesbo, Peddada (2022); always reports skip until the operator runs the BiocManager install on the target host. |

To enable R baselines on a host that already has R:

```bash
mkdir -p ~/.falcon-r-baselines
git clone https://github.com/huayingfang/CCLasso.git ~/.falcon-r-baselines/CCLasso
git clone https://github.com/yuanpeicao/COAT.git    ~/.falcon-r-baselines/COAT
```

Override the location with `FALCON_R_BASELINE_DIR` if the default is not
suitable.

---

## Public-data validation

Public-data evaluation reports stability and reproducibility, not biological
truth. The repository commits download instructions and stable identifiers
only — never raw third-party archives.

| Dataset | Identifier |
|---|---|
| SECOM archive (Lin, Eggesbo, Peddada 2022) | https://doi.org/10.5281/zenodo.6809029 |
| NIH HMP umbrella BioProject | `PRJNA43021` |
| HMP 16S rRNA diversity subproject | `PRJNA48489` |

See `data/README.md` and `data/public/*.md` for the download and processing
recipes.

---

## Acceptance gates

The selected estimator must clear all six gates on frozen holdout cells
before the repository asserts an advantage. Until then this section
honestly reports `not yet evaluated`.

| # | Gate | Status |
|---|---|---|
| 1 | AUROC and Recall@K each exceed the strongest matched-estimand baseline on sparse and zero-inflated scenarios | not yet evaluated |
| 2 | Empirical FDR reported at nominal targets `0.01`, `0.05`, `0.10`; q-values exposed only if calibration is defensible across holdout scenarios | not yet evaluated |
| 3 | Medium- and high-dimensional runtime and peak memory each improve against the strongest accurate baseline | not yet evaluated (training shows weighted_sparse is ~27× faster than `cclasso` and ~750× slower than `sparcc_closed_form` at `p ≤ 100`; holdout must confirm the gap on larger `p`) |
| 4 | All selected-estimator runs converge or return an explicit non-convergence diagnostic | code path supported; full grid not yet run |
| 5 | Public real-data subsampling produces a stability report with dataset identifier, seed, resample count, and selection threshold | scripts scaffolded; report not yet produced |
| 6 | Every numerical claim maps to a committed source-data row and generator command | infrastructure in place (`data/manifest.tsv`); no claims yet |

---

## License

MIT. See `pyproject.toml`.
