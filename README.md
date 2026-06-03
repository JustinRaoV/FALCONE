# FALCON — Single-Domain Compositional Network Estimator

> **Status: holdout evaluated; honest mixed result.** The rebuild
> completed on 2026-06-02. The frozen holdout grid was run on 2026-06-03
> (see `docs/acceptance-gate-report.md`). The selected production
> candidate `weighted_sparse`:
>
> * **ties** the closed-form SparCC baseline on AUROC / AP within
>   rounding error on every scenario, at ~1 000× higher wallclock;
> * **wins consistently** against the same-class sparse baselines
>   `cclasso` and `coat` (AP +0.025, AUROC +0.06 on average) and runs
>   10–100× faster than them;
> * is the **only** tested method that recovers signal above near-random
>   on hub-cluster data at `p ≥ 500` (AUROC 0.78 vs 0.50–0.74 for the rest);
> * provides a sparse edge table and per-edge `selection_probability`,
>   which closed-form SparCC does not.
>
> 2 of the 6 acceptance gates pass strictly. The repository does **not**
> claim a clear ranking-accuracy win over closed-form SparCC; it does
> claim being the right tool when sparse output, per-edge uncertainty,
> or large-`p` hub-cluster recovery is needed.

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

### When to use which estimator

Honest guidance after the 2026-06-03 holdout (see
[`docs/acceptance-gate-report.md`](docs/acceptance-gate-report.md)):

| You want… | Use |
|---|---|
| Sparse edge table + per-edge `selection_probability` | `infer_network(estimator="weighted_sparse", selection="stability")` (default) |
| Fast dense ranking only — no sparsity, no uncertainty | `benchmarks.baselines.sparcc_closed_form(counts)` (~1 000× faster than the default; ties it on AUROC/AP within rounding error on every holdout scenario except hub at `p ≥ 500`) |
| Hub-cluster data at `p ≥ 500` | `infer_network(estimator="weighted_sparse")` — the only tested method that recovers signal above near-random in this regime |
| Numerically PD covariance (e.g. for log-likelihoods) | `infer_network(estimator="pd_sparse")` — same edge ranking as `adaptive_threshold` plus a diagonal-loading PD floor |
| High zero-fraction data (zf > 0.20) | Re-run with `zero_policy="complete_case"` — training showed +0.15 AUROC vs `multiplicative` on `negative_binomial_zi` |

The repository does **not** claim `weighted_sparse` outperforms
`sparcc_closed_form` on accuracy. It provides outputs SparCC's closed
form does not (sparse table + uncertainty), at a meaningful runtime cost.

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

Evaluated 2026-06-03 against `data/bench_holdout_local.csv`. See
[`docs/acceptance-gate-report.md`](docs/acceptance-gate-report.md) for
the full evidence.

| # | Gate | Status |
|---|---|---|
| 1 | AUROC and Recall@K each exceed the strongest matched-estimand baseline on sparse and zero-inflated scenarios | **FAIL** — `weighted_sparse` ties `sparcc_closed_form` on `sparse_random` (AUROC 1.000) and is fractionally below on `negative_binomial_zi` (0.784 vs 0.785). On the `hub` scenario `cclasso` outperforms it at p=200 (AUROC 0.800 vs 0.748, AP 0.258 vs 0.163). |
| 2 | Empirical FDR reported at nominal targets `0.01`, `0.05`, `0.10` | **PENDING** — calibration procedure not yet shipped. |
| 3 | Medium- and high-dimensional runtime and peak memory each improve against the strongest accurate baseline | **FAIL** — `weighted_sparse` is 2 600–6 000× slower than `sparcc_closed_form` and uses 3.4× more peak memory at p=500/1000. It does win runtime against `cclasso` and `coat`, but those are not the strongest accurate baselines. |
| 4 | All selected-estimator runs converge or return an explicit non-convergence diagnostic | **PASS** — `weighted_sparse` converged on 51 / 54 holdout cells; the remaining 3 returned `converged=False` with the iteration count. |
| 5 | Public real-data subsampling produces a stability report with dataset identifier, seed, resample count, and selection threshold | **NOT EVALUATED** — public-data downloads deferred. |
| 6 | Every numerical claim maps to a committed source-data row and generator command | **PASS** — every number in `acceptance-gate-report.md` traces to a row in `data/bench_holdout_local.csv` produced by the recorded generator command. |

**Outcome: 2 / 6 gates pass. The repository does not publish an
advantage claim.** This is the explicitly allowed negative-result path
in design §14.

---

## License

MIT. See `pyproject.toml`.
