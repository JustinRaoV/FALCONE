#!/usr/bin/env bash
# Frozen holdout-grid runner. The local laptop does NOT run this; it
# emits enough rows that wall-clock and peak-memory measurements need
# server-grade hardware.
#
# Each invocation appends to data/bench_holdout.csv. Resuming after an
# interruption is safe — duplicate rows are tolerated by analysis code
# because the schema includes (scenario, seed, n, p, method).
#
# Usage on a server with R + the three R baseline packages installed:
#
#   bash benchmarks/run_server_holdout.sh
#
# Without R, the R-baseline rows record skip reasons.

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="${OUTPUT:-data/bench_holdout.csv}"
N_RESAMPLES="${N_RESAMPLES:-100}"
REPS="${REPS:-3}"
METHODS="${METHODS:-falcon_weighted_sparse,falcon_adaptive_threshold,falcon_pd_sparse,sparcc_closed_form,pearson_clr,cclasso,coat,secom}"

uv run python benchmarks/run_benchmark.py \
    --split holdout \
    --output "$OUTPUT" \
    --reps "$REPS" \
    --n-resamples "$N_RESAMPLES" \
    --methods "$METHODS"
