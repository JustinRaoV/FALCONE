# Decision Log — Falcon-SR

## Active decisions

### 2026-06-02 — Replace proportionality framing with latent log-abundance correlation

**Decision.** The legacy FastProp / RandProp / CrossNet code, which
estimated the Aitchison proportionality metric $\rho_p$, has been removed.
Falcon-SR now estimates the latent log-abundance Pearson correlation
targeted by SparCC and SparXCC Case-C. This change is required by the
2026-06-01 design specification (§3 non-goals: "Treat proportionality as
a synonym for latent log-abundance correlation").

**Why.** Proportionality and latent correlation are different estimands.
Comparing $\rho_p$ to SparCC outputs implies an equivalence that does not
hold, and any speed claim built on that comparison would be incoherent. By
aligning the estimand we make head-to-head benchmarks meaningful and
honest.

**Where it shows up.** `src/falcon/__init__.py` now exports only
`infer_single`, `infer_cross`, `PriorEdge`. The legacy
`fastprop` / `randprop` / `crossnet` / `clr_transform` /
`multiplicative_replacement` / `extract_network` / `fastprop_pvalues`
symbols are gone, along with `benchmarks/run_on_server.py` and the legacy
`data/*.csv` outputs and the prior `manuscript/` directory.

### 2026-06-02 — Cross-domain refinement uses edge-driven feature pruning

**Decision.** When a candidate $(i, k)$ is excluded as a strong edge, the
entire X row $i$ and Y column $k$ are dropped from the centring pool for
the next round. This preserves the $H_p \otimes H_q^\top$ identity that
underlies SparXCC base and iter; the alternative (non-rectangular per-cell
exclusion) breaks the centring algebra.

**Why.** Edge-driven pruning gives two clean properties: (a) reduces to
SparXCC base when no edges are excluded; (b) is interchangeable with
SparXCC iter when the candidate set is full and the exclusion threshold
matches. Both make it possible to gate the implementation against the
published SparXCC reference.

### 2026-06-02 — Prior penalty is post-hoc analytic shrinkage

**Decision.** Signed priors do **not** enter the iterative exclusion
choices. After sparse refinement, candidate edges with a matching prior
record have their score replaced by the closed-form minimiser of
$(\rho - \hat\rho_{\text{data}})^2 + \lambda\,\text{conf}\,(\rho - \text{sign}\cdot\text{target})^2$,
which is
$\rho = (\hat\rho_{\text{data}} + \lambda\,\text{conf}\,\text{sign}\,\text{target})
        / (1 + \lambda\,\text{conf})$.

**Why.** Spec §10 demands a soft direction, not an invented effect size.
Letting priors steer refinement would risk hiding wrong-sign data
signals; the post-hoc formula keeps data sovereign at every iterative
step and only blends at the end. `prior_weight = 0` collapses the formula
to $\hat\rho_{\text{data}}$ identically, so existing call sites cannot
trigger prior side-effects by accident.

### 2026-06-02 — Permutation calibration permutes the base score only

**Decision.** `calibrate_single` / `calibrate_cross` recompute the
closed-form SparCC base correlation (or the SparXCC double-centred score)
per permutation, **not** the full sparse refine pipeline. The
`CalibrationResult.method` field is `permutation_base_only` so downstream
code never treats the result as a calibration-tight test.

**Why.** Full per-permutation refinement costs $R \cdot
\mathcal{O}(p^3)$ which puts $p = 1000, R = 100$ beyond the feasibility
wall-clock budget. The base-only approximation is conservative under most
conditions but is not guaranteed; spec §19 risk 4 stays explicit rather
than being silently fixed.

### 2026-06-02 — Calibration off by default in benchmarks for time comparison

**Decision.** `falcon_sr_fast` cells in
`benchmarks/falcon_sr_single.py` and `benchmarks/falcon_sr_cross.py` run
with `calibration="none"` so their wall-clock numbers are apples-to-apples
with SparCC / Pearson(CLR) / SparXCC base / SparXCC iter. A separate
`*_calibrated` method row exposes the calibration overhead explicitly.

**Why.** Combining calibration cost with ranking cost would muddy the
comparison and let either side cherry-pick. Separate rows let the same
CSV answer both "how fast is the ranking?" and "how expensive is the
calibration?".

### 2026-06-02 — Inline CLR / multiplicative-replacement helpers in `comparison_methods.py`

**Decision.** `benchmarks/comparison_methods.py` defines its own
`multiplicative_replacement` and `clr_transform` rather than importing
them from `falcon`.

**Why.** Baselines must be self-contained: they should not silently
shift if Falcon-SR changes a helper. This also unblocked the legacy-code
removal (those helpers used to live in the legacy module).

### 2026-06-02 — macOS Accelerate BLAS matmul warnings suppressed

**Decision.** `pyproject.toml` filters the three spurious "divide by
zero / overflow / invalid value encountered in matmul" RuntimeWarnings
emitted by Apple Accelerate + NumPy 2.x on otherwise finite inputs.
Benchmark runners install the same filter at import.

**Why.** The warnings are a known false positive on Apple Silicon;
they appear even for completely finite matrices and obscure real
diagnostic output. Real numerical issues remain visible through explicit
`np.isfinite` checks.

## Open questions

- **Selective-inference correction for candidate-only calibration.** Spec
  §19 risk 4. Not addressed in this rewrite. A follow-up plan must decide
  whether to ship sample-splitting or a selection-aware test before the
  manuscript claims power / FDR control.
- **Random-projection screening for $p > 5000$.** Spec §3 lists this as
  out of scope for the first release. Will need a separate plan.
- **Real-data validation.** Spec §14 lists candidate public datasets;
  this rewrite leaves them for the next plan since each dataset has its
  own preprocessing and licence pitfalls.

## Historical (superseded) entries

The pre-2026-06-02 entries from this file lived in the proportionality
framing and are no longer accurate; consult git history if you need to
recover them. The shape of the change is described above under "Replace
proportionality framing with latent log-abundance correlation".
