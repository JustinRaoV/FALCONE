"""Adapters for R baseline correlation estimators.

Three baselines are exposed:

* ``cclasso`` — Fang et al. (2015) v2.0 reference implementation. Used as
  the closest publicly-available analog to fastCCLasso (Zhang, Fang, Hu
  2024); the 2024 paper does not publish a standalone R package, so the
  v2.0 implementation by the same author family is our matched baseline.
  Source: https://github.com/huayingfang/CCLasso (LGPL-2.1+).
* ``coat`` — Cao, Lin, Li (2019) reference implementation. Source:
  https://github.com/yuanpeicao/COAT (no SPDX licence in repo; treated as
  research code).
* ``secom`` — Lin, Eggesbo, Peddada (2022) "linear" estimator. Bundled
  inside the ``ANCOMBC`` Bioconductor package; this adapter currently
  always reports skip until the operator has run the BiocManager install
  on the target host (heavy dependency chain).

The production package never invokes R. These adapters are used only by
the benchmark runner. They do NOT vendor the R sources — the operator
must clone the upstream repos so the licence boundary stays clean. The
default location is ``~/.falcon-r-baselines/{CCLasso,COAT}``; override
with the ``FALCON_R_BASELINE_DIR`` environment variable.

Install on a host that has R:

    git clone https://github.com/huayingfang/CCLasso.git ~/.falcon-r-baselines/CCLasso
    git clone https://github.com/yuanpeicao/COAT.git    ~/.falcon-r-baselines/COAT

If any precondition fails the adapter returns an :class:`RAdapterSkip`
with an explicit reason. Skips are first-class values; the runner records
a "skipped" row instead of a fake numeric result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_BASELINES = ("cclasso", "coat", "secom")
DEFAULT_BASELINE_DIR = Path("~/.falcon-r-baselines").expanduser()


@dataclass(frozen=True)
class RAdapterSkip:
    method: str
    reason: str


@dataclass(frozen=True)
class RAdapterResult:
    method: str
    estimand_family: str
    correlation: np.ndarray
    package_version: str
    converged: bool
    iterations: int


def _baseline_dir() -> Path:
    override = os.environ.get("FALCON_R_BASELINE_DIR")
    return Path(override).expanduser() if override else DEFAULT_BASELINE_DIR


def _rscript_path() -> str | None:
    return shutil.which("Rscript")


def _resolve_script(method: str) -> Path | None:
    """Return the local R script path for the given method or None if missing."""
    base = _baseline_dir()
    paths = {
        "cclasso": base / "CCLasso" / "R" / "cclasso.R",
        "coat": base / "COAT" / "coat.R",
    }
    p = paths.get(method)
    return p if p is not None and p.is_file() else None


# Mapping from method -> (R body, estimand_family)
# The R body assumes an ``x`` matrix is in scope (composition with rows
# summing to ~1, n x p) and writes ``corr`` (a p x p correlation matrix).
_RUNNERS = {
    "cclasso": (
        # Reduced n_boot from default 20 -> 5 to keep wallclock viable on
        # holdout cells (p=1000 with n_boot=20 took ~5 min/cell). The
        # bootstrap only refines p-values; the cor_w estimate from one
        # CV-selected lambda is the same. We use this estimate for
        # ranking only, so the saving is safe for our acceptance gates.
        "fit <- cclasso(x, counts=FALSE, n_boot=5, k_cv=3)\n"
        "corr <- fit$cor_w\n",
        "latent_log_abundance_correlation",
    ),
    "coat": (
        "fit <- coat(x, soft=1)\n"
        "corr <- fit$corr\n",
        "latent_log_abundance_correlation",
    ),
}


def _multiplicative_replacement(composition: np.ndarray) -> np.ndarray:
    p = composition.shape[1]
    delta = 0.65 / (p * p)
    zero_mask = composition == 0
    zero_count = zero_mask.sum(axis=1, keepdims=True)
    scale = 1.0 - zero_count * delta
    return np.where(zero_mask, delta, composition * scale)


def _run_r(rscript: str, script_body: str, counts: np.ndarray) -> tuple[bool, str, np.ndarray | None]:
    """Run an R script body that produces ``corr``. Returns
    ``(success, message, corr_or_none)``."""
    composition = counts / counts.sum(axis=1, keepdims=True)
    composition = _multiplicative_replacement(composition)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "x.csv")
        out_path = os.path.join(tmp, "corr.csv")
        np.savetxt(in_path, composition, delimiter=",")
        script = (
            f"x <- as.matrix(read.csv({in_path!r}, header=FALSE))\n"
            f"{script_body}"
            f"write.table(corr, {out_path!r}, sep=',', row.names=FALSE, col.names=FALSE)\n"
        )
        script_path = os.path.join(tmp, "run.R")
        with open(script_path, "w") as fh:
            fh.write(script)
        # Override via FALCON_R_TIMEOUT (seconds); default 600s = 10 min.
        timeout_s = float(os.environ.get("FALCON_R_TIMEOUT", "600"))
        try:
            proc = subprocess.run(
                [rscript, "--no-save", script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return False, f"Rscript failed: {exc!r}", None
        if proc.returncode != 0:
            return False, f"Rscript returncode {proc.returncode}: {proc.stderr.strip()[:400]}", None
        try:
            corr = np.loadtxt(out_path, delimiter=",")
        except OSError as exc:
            return False, f"failed to read R output: {exc!r}", None
        return True, "ok", corr


def run_r_baseline(method: str, counts: np.ndarray) -> RAdapterSkip | RAdapterResult:
    if method not in R_BASELINES:
        raise ValueError(f"unknown R baseline {method!r}; valid: {R_BASELINES}")
    if method == "secom":
        return RAdapterSkip(
            method=method,
            reason="SECOM lives in ANCOMBC (Bioconductor); install via BiocManager on the target host first",
        )

    rscript = _rscript_path()
    if rscript is None:
        return RAdapterSkip(method=method, reason="Rscript not found on PATH")

    script_path = _resolve_script(method)
    if script_path is None:
        return RAdapterSkip(
            method=method,
            reason=(
                f"baseline R script for {method!r} not found under "
                f"{_baseline_dir()}; clone the upstream repo first "
                f"(see benchmarks/r_adapters.py docstring)"
            ),
        )

    body_template, family = _RUNNERS[method]
    body = f'source({str(script_path)!r})\n{body_template}'
    success, msg, corr = _run_r(rscript, body, counts)
    if not success:
        return RAdapterSkip(method=method, reason=msg)

    p = counts.shape[1]
    if corr.shape != (p, p):
        return RAdapterSkip(
            method=method,
            reason=f"R returned correlation of shape {corr.shape}, expected {(p, p)}",
        )
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return RAdapterResult(
        method=method,
        estimand_family=family,
        correlation=corr,
        package_version=str(script_path),
        converged=True,
        iterations=1,
    )


def iterate_r_baselines(counts: np.ndarray, methods=R_BASELINES):
    for method in methods:
        yield run_r_baseline(method, counts)
