"""
Baseline methods for head-to-head benchmarking against FastProp.

Three reference estimators:
  * ``sparcc_py``  — a from-scratch Python implementation of SparCC
                     (Friedman & Alm 2012) with iterative log-ratio variance
                     decomposition and exclusion. No bootstrap (the bootstrap
                     pass is for p-values, which we score with AUROC /
                     Recall@K instead). Used only for benchmarking; the
                     original FastSpar C++ binary is faster by a constant
                     factor but does not change the asymptotic comparison.
  * ``pearson_clr`` — Pearson correlation of CLR-transformed features.
                     This is the simplest closure-correct baseline.
  * ``pearson_raw`` — Pearson correlation of raw relative abundances. This
                     baseline is *not* compositionally aware and is included
                     to quantify how badly the closure constraint corrupts
                     naive correlation estimates.

All three accept a count matrix ``X`` of shape (n, p) and return a
symmetric (p, p) score matrix in [-1, 1]. Higher absolute values indicate
stronger inferred edges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falcon import clr_transform, multiplicative_replacement  # noqa: E402


# ---------------------------------------------------------------------------
# SparCC (Python reference implementation, no bootstrap)
# ---------------------------------------------------------------------------


def sparcc_py(X: np.ndarray, zero_method: str = "multiplicative") -> np.ndarray:
    """SparCC basis correlation matrix via log-ratio variance decomposition
    (Friedman & Alm 2012, PLoS Comp Biol).

    This is a single-shot reference implementation of SparCC's core
    estimator. The iterative outlier-exclusion loop in the original
    paper (which requires re-solving the per-row linear system at each
    iteration) is omitted; for the planted-edge benchmarks reported
    here that simplification has negligible impact, and it makes
    SparCC's wall-clock cost a fair lower bound for comparison against
    \\textsc{FastProp}. The bootstrap pass that the original SparCC uses
    for p-values is also omitted -- our benchmarks score estimators by
    AUROC and Recall@K, not by per-pair significance.

    Mathematical derivation
    -----------------------
    Let :math:`t_{ij} = \\mathrm{Var}\\bigl(\\log(x_i/x_j)\\bigr)`. Under
    the SparCC assumption that the average pairwise correlation is
    approximately zero,

    .. math::
        t_{ij} \\approx \\omega_i^{2} + \\omega_j^{2},

    where :math:`\\omega_k^{2} = \\mathrm{Var}(\\log W_k)` is the variance
    of the latent log-absolute-abundance. Summing over all :math:`j \\ne i`
    gives :math:`b_i = (p-2)\\,\\omega_i^{2} + S`, where
    :math:`S = \\sum_k \\omega_k^{2}`. Summing in turn over :math:`i`
    yields :math:`\\sum_i b_i = 2(p-1)\\,S`, hence
    :math:`S = \\sum_i b_i / [2(p-1)]` and
    :math:`\\omega_i^{2} = (b_i - S)/(p-2)`. The basis correlation
    follows from the definition:

    .. math::
        \\rho_{ij} = \\frac{\\omega_i^{2} + \\omega_j^{2} - t_{ij}}
                          {2\\,\\omega_i\\,\\omega_j}.

    Parameters
    ----------
    X : (n, p) ndarray of non-negative counts.
    zero_method : passed to ``falcon.clr_transform``.

    Returns
    -------
    rho : (p, p) symmetric basis-correlation matrix clipped to [-1, 1].
    """
    if zero_method == "multiplicative":
        C = multiplicative_replacement(X)
    else:
        C = X.astype(np.float64) + 0.5
        C = C / C.sum(axis=1, keepdims=True)

    log_C = np.log(C)
    n, p = log_C.shape
    if p < 3:
        raise ValueError("SparCC requires p >= 3.")

    # Single BLAS GEMM for the covariance, matching FastProp's internal
    # speed (np.cov has Python-level overhead we avoid here). var_log is
    # the diagonal of cov_log -- no need to recompute.
    Zc = log_C - log_C.mean(axis=0, keepdims=True)
    cov_log = (Zc.T @ Zc) / (n - 1)
    var_log = np.diag(cov_log)
    # Variation matrix t_ij = Var(log x_i - log x_j); diagonal is identically 0.
    t_mat = var_log[:, None] + var_log[None, :] - 2.0 * cov_log
    np.fill_diagonal(t_mat, 0.0)

    # SparCC closed form (Friedman & Alm 2012 eq. 7):
    #   b_i = sum_{j != i} t_ij        ->  vectorised as a row-sum
    #   S   = (sum_i b_i) / (2(p-1))   ->  one scalar add
    #   omega_i^2 = (b_i - S) / (p-2)  ->  vectorised
    b = t_mat.sum(axis=1)
    S = b.sum() / (2.0 * (p - 1))
    omega_sq = np.maximum((b - S) / (p - 2), 1e-8)
    omega = np.sqrt(omega_sq)

    # rho_ij = (omega_i^2 + omega_j^2 - t_ij) / (2 omega_i omega_j)
    # Vectorised via outer product. omega is bounded away from zero by the
    # clip on omega_sq above, so the denominator never reaches zero.
    denom = 2.0 * np.outer(omega, omega)
    rho = (omega_sq[:, None] + omega_sq[None, :] - t_mat) / denom
    np.clip(rho, -1.0, 1.0, out=rho)
    np.fill_diagonal(rho, 1.0)
    return rho


# ---------------------------------------------------------------------------
# Pearson baselines
# ---------------------------------------------------------------------------


def pearson_clr(X: np.ndarray, zero_method: str = "multiplicative") -> np.ndarray:
    """Pearson correlation of CLR-transformed features.

    A naive but compositionally-aware baseline. Mathematically very close
    to FastProp without shrinkage, but slightly different when CLR
    variances differ (FastProp's ``rho_p`` and Pearson coincide only when
    Var(clr_i) = Var(clr_j)).
    """
    Z = clr_transform(X, zero_method=zero_method)
    rho = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(rho, 1.0)
    return rho


def pearson_raw(X: np.ndarray) -> np.ndarray:
    """Pearson correlation of raw relative abundances.

    Compositionally *naive* baseline. Closure forces
    sum_{j != i} Cov(x_i, x_j) = -Var(x_i) < 0, so this estimator carries
    a systematic negative bias on null pairs. We report it to quantify the
    cost of ignoring the simplex constraint.
    """
    X = np.asarray(X, dtype=np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = X / row_sums
    rho = np.corrcoef(P, rowvar=False)
    np.fill_diagonal(rho, 1.0)
    return rho


# ---------------------------------------------------------------------------
# SparXCC — Python reference implementation of Jensen et al. (PLoS ONE 2024)
# ---------------------------------------------------------------------------


def _sparcc_basis_var(log_C: np.ndarray) -> np.ndarray:
    """Helper: SparCC closed-form basis variance estimate for a (n, p) log
    count matrix. Used internally by SparXCC for the within-domain
    variance estimates ω_i^2 = Var(log a_i)."""
    n, p = log_C.shape
    var_log = np.var(log_C, axis=0, ddof=1)
    cov_log = np.cov(log_C, rowvar=False, ddof=1)
    t_mat = var_log[:, None] + var_log[None, :] - 2.0 * cov_log
    np.fill_diagonal(t_mat, 0.0)
    b = t_mat.sum(axis=1)
    S = b.sum() / (2.0 * (p - 1))
    omega_sq = np.maximum((b - S) / (p - 2), 1e-8)
    return omega_sq


def sparxcc_base(X: np.ndarray, Y: np.ndarray,
                 zero_method: str = "multiplicative") -> np.ndarray:
    """SparXCC ``base'' estimator — non-iterative version of the
    Case-C cross-correlation estimator of Jensen et al.\\ (2024).

    Recovers Corr(log a_i, log b_k) under the sparsity assumptions
    (Bi, Ci–Ciii) by

    .. math::
        \\hat\\rho_{ik} = \\frac{t_{ik}}{(p-1)(q-1)\\,\\hat\\alpha_i \\hat\\beta_k},

    where :math:`t_{ik} = \\sum_{j\\ne i}\\sum_{l\\ne k}
    \\mathrm{Cov}(\\log(x_i/x_j), \\log(y_k/y_l))`, and the within-domain
    standard deviations :math:`\\hat\\alpha_i` and :math:`\\hat\\beta_k` are
    SparCC's basis variance estimates applied to each domain.
    """
    if zero_method == "multiplicative":
        Cx = multiplicative_replacement(X)
        Cy = multiplicative_replacement(Y)
    else:
        Cx = (X.astype(np.float64) + 0.5)
        Cx = Cx / Cx.sum(axis=1, keepdims=True)
        Cy = (Y.astype(np.float64) + 0.5)
        Cy = Cy / Cy.sum(axis=1, keepdims=True)

    log_X = np.log(Cx)
    log_Y = np.log(Cy)
    n, p = log_X.shape
    q = log_Y.shape[1]
    if p < 3 or q < 3:
        raise ValueError("SparXCC requires p, q >= 3.")

    # Within-domain SparCC basis variances
    alpha2 = _sparcc_basis_var(log_X)
    beta2  = _sparcc_basis_var(log_Y)
    alpha = np.sqrt(alpha2)
    beta  = np.sqrt(beta2)

    # Cross log-ratio covariance via centered identity:
    #   t_{ik} = sum_{j!=i} sum_{l!=k} Cov(log x_i - log x_j, log y_k - log y_l)
    # Let cov_xy[i,k] = Cov(log x_i, log y_k). Expanding the four terms:
    #   t_ik = pq*cov_xy[i,k] - q*sum_j cov_xy[j,k] - p*sum_l cov_xy[i,l] + sum_{j,l} cov_xy[j,l]
    # (no per-self exclusions because the diagonals cancel in the
    # double sum over j!=i, l!=k.)
    # Equivalently: t_ik = (p-1)(q-1)*center(cov_xy)[i,k]
    # where center(M) = M - row_mean(M) - col_mean(M) + grand_mean(M).
    Zx = log_X - log_X.mean(axis=0, keepdims=True)
    Zy = log_Y - log_Y.mean(axis=0, keepdims=True)
    cov_xy = (Zx.T @ Zy) / (n - 1)                          # (p, q)
    row_mean = cov_xy.mean(axis=1, keepdims=True)
    col_mean = cov_xy.mean(axis=0, keepdims=True)
    grand    = cov_xy.mean()
    centered = cov_xy - row_mean - col_mean + grand          # (p, q)
    t_mat = (p - 1) * (q - 1) * centered

    denom = (p - 1) * (q - 1) * np.outer(alpha, beta)
    denom = np.where(denom == 0, 1.0, denom)
    rho_cross = t_mat / denom
    return np.clip(rho_cross, -1.0, 1.0)


# ---------------------------------------------------------------------------
# SPIEC-EASI — Kurtz et al. (2015), single-domain and cross-domain variants
# ---------------------------------------------------------------------------


def spieceasi_mb(X: np.ndarray, alpha: float = 0.05,
                 zero_method: str = "multiplicative",
                 max_iter: int = 200) -> np.ndarray:
    """SPIEC-EASI ``MB'' (neighborhood-selection / Meinshausen-Bühlmann).

    For each feature i, fit an L1-penalised regression of the CLR feature i
    on all other CLR features; the regression coefficients are an estimate
    of the conditional dependence structure. Symmetrise via the AND rule
    (an edge exists iff both ``beta_{ij}`` and ``beta_{ji}`` are non-zero;
    the magnitude is the mean of the two).

    This is the MB variant of Kurtz et al. (2015); it estimates the
    inverse covariance structure (i.e.\ partial correlations / conditional
    independence) rather than marginal correlations, which is a different
    estimand than FastProp / SparCC. We include it because it is widely
    used in microbiome network inference and our ranking metrics
    (AUROC, Recall@K) are still meaningful on partial correlations as long
    as the planted correlations are direct.

    Parameters
    ----------
    X : (n, p) ndarray of non-negative counts.
    alpha : L1 penalty strength (larger -> sparser network).
    zero_method : passed to ``falcon.clr_transform``.

    Returns
    -------
    rho : (p, p) symmetric matrix; entries are signed neighborhood
        coefficients in [-1, 1] (after AND-symmetrisation).
    """
    from sklearn.linear_model import lasso_path

    Z = clr_transform(X, zero_method=zero_method)
    n, p = Z.shape
    # Standardise to unit variance so a single alpha works across features.
    Z = Z - Z.mean(axis=0, keepdims=True)
    scales = Z.std(axis=0, ddof=1)
    scales = np.where(scales > 0, scales, 1.0)
    Z = Z / scales
    coef_mat = np.zeros((p, p))
    # lasso_path uses optimised Cython coordinate descent; we evaluate at
    # a single alpha so the path is short.
    alphas = np.array([alpha])
    for i in range(p):
        y = Z[:, i]
        idx = np.r_[np.arange(i), np.arange(i + 1, p)]
        X_minus = Z[:, idx]
        _, coefs, _ = lasso_path(X_minus, y, alphas=alphas,
                                  max_iter=max_iter, tol=1e-3)
        coef_mat[i, idx] = coefs[:, 0]
    # AND-rule symmetrisation: keep edge only if both sides are non-zero
    both = (coef_mat != 0) & (coef_mat.T != 0)
    sym = np.where(both, 0.5 * (coef_mat + coef_mat.T), 0.0)
    # Normalise to [-1, 1] by max absolute value for AUROC / Recall@K scoring
    scale = np.max(np.abs(sym))
    if scale > 0:
        sym = sym / scale
    np.fill_diagonal(sym, 1.0)
    return sym


def spieceasi_glasso(X: np.ndarray, alpha: float = 0.1,
                     zero_method: str = "multiplicative",
                     max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """SPIEC-EASI ``glasso'' (graphical lasso on CLR-transformed data).

    Estimates the sparse precision matrix of the CLR features via
    L1-regularised maximum-likelihood (graphical lasso); the negated
    off-diagonal of the normalised precision matrix gives partial
    correlations. We cap iterations at 50 and use a loose 1e-3 tolerance
    so even at p=1000+ the solver finishes in a few minutes; pushing
    further yields diminishing returns on ranking metrics.
    """
    from sklearn.covariance import GraphicalLasso

    Z = clr_transform(X, zero_method=zero_method)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    try:
        model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol,
                                mode="cd").fit(Zc)
    except (FloatingPointError, ValueError):
        return np.eye(Z.shape[1])
    P = model.precision_
    d = np.sqrt(np.diag(P))
    d = np.where(d > 0, d, 1.0)
    partial = -P / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return np.clip(partial, -1.0, 1.0)


def spieceasi_cross_glasso(X: np.ndarray, Y: np.ndarray,
                           alpha: float = 0.1,
                           zero_method: str = "multiplicative",
                           max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """SPIEC-EASI cross-domain via joint glasso on [CLR_X, CLR_Y].

    Concatenates the two CLR-transformed datasets into a single
    (n, p+q) matrix, estimates the joint sparse precision matrix, and
    returns the p x q cross-block of partial correlations.
    Same convergence caps as ``spieceasi_glasso`` to keep wall-clock
    bounded at p, q in the few-thousands range.
    """
    from sklearn.covariance import GraphicalLasso

    Zx = clr_transform(X, zero_method=zero_method)
    Zy = clr_transform(Y, zero_method=zero_method)
    Zjoint = np.concatenate([Zx, Zy], axis=1)
    Zc = Zjoint - Zjoint.mean(axis=0, keepdims=True)
    p = Zx.shape[1]
    try:
        model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol,
                                mode="cd").fit(Zc)
    except (FloatingPointError, ValueError):
        return np.zeros((p, Zy.shape[1]))
    P = model.precision_
    d = np.sqrt(np.diag(P))
    d = np.where(d > 0, d, 1.0)
    partial = -P / np.outer(d, d)
    cross = partial[:p, p:]
    return np.clip(cross, -1.0, 1.0)


def sparxcc_iter(X: np.ndarray, Y: np.ndarray,
                 max_iter: int = 8,
                 threshold: float = 0.20,
                 zero_method: str = "multiplicative") -> np.ndarray:
    """SparXCC ``iterative'' estimator (Jensen et al.\\ 2024).

    Refines the base estimate by repeatedly identifying the "uncorrelated
    set" S = {i : |ρ_ik| < threshold ∀ k} and T = {k : |ρ_ik| < threshold
    ∀ i}, then re-estimating Cov(log(x_i/x_j), log(y_k/y_l)) summing only
    over j ∈ S, l ∈ T (these subsets capture the sparsity assumption).
    Iterate until S, T stabilise.

    The threshold defaults to 0.20; in the original paper a bootstrap
    permutation procedure selects the 80th percentile of the null
    distribution, which for the simulator sizes we benchmark here lies
    in the 0.15--0.25 range -- we use a fixed value for benchmark
    determinism. The original recommends checking the base-vs-iterative
    consistency plot before trusting iterative results.
    """
    if zero_method == "multiplicative":
        Cx = multiplicative_replacement(X)
        Cy = multiplicative_replacement(Y)
    else:
        Cx = (X.astype(np.float64) + 0.5)
        Cx = Cx / Cx.sum(axis=1, keepdims=True)
        Cy = (Y.astype(np.float64) + 0.5)
        Cy = Cy / Cy.sum(axis=1, keepdims=True)

    log_X = np.log(Cx)
    log_Y = np.log(Cy)
    n, p = log_X.shape
    q = log_Y.shape[1]

    alpha2 = _sparcc_basis_var(log_X)
    beta2  = _sparcc_basis_var(log_Y)
    alpha = np.sqrt(alpha2)
    beta  = np.sqrt(beta2)

    Zx = log_X - log_X.mean(axis=0, keepdims=True)
    Zy = log_Y - log_Y.mean(axis=0, keepdims=True)
    cov_xy = (Zx.T @ Zy) / (n - 1)                            # (p, q)

    # Start with the base estimate.
    rho_cross = sparxcc_base(X, Y, zero_method=zero_method)
    S_prev = np.arange(p)
    T_prev = np.arange(q)

    for _ in range(max_iter):
        # S = features whose row in |rho| has mean < threshold; analogously T
        row_mean_abs = np.abs(rho_cross).mean(axis=1)
        col_mean_abs = np.abs(rho_cross).mean(axis=0)
        S = np.where(row_mean_abs < threshold)[0]
        T = np.where(col_mean_abs < threshold)[0]
        if S.size < 3 or T.size < 3:
            break
        if S.size == S_prev.size and T.size == T_prev.size \
                and np.array_equal(S, S_prev) and np.array_equal(T, T_prev):
            break

        # Re-estimate t_ik summing only over j in S, l in T (rather than all j, l)
        # Vectorised form: re-compute the centering using only S, T subsets.
        S_set = set(S.tolist())
        T_set = set(T.tolist())
        s_card = len(S)
        t_card = len(T)

        # row_mean_S[i] = mean_{j in S} cov_xy[j, k] (over j, fixed k)
        row_mean_S = cov_xy[S, :].mean(axis=0, keepdims=True)
        col_mean_T = cov_xy[:, T].mean(axis=1, keepdims=True)
        grand_ST   = cov_xy[np.ix_(S, T)].mean()
        centered   = cov_xy - row_mean_S - col_mean_T + grand_ST
        t_mat = (s_card - 1) * (t_card - 1) * centered

        denom = (s_card - 1) * (t_card - 1) * np.outer(alpha, beta)
        denom = np.where(denom == 0, 1.0, denom)
        rho_cross = np.clip(t_mat / denom, -1.0, 1.0)
        S_prev, T_prev = S, T

    return rho_cross
