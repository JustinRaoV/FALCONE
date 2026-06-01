#!/usr/bin/env bash
#
# Run every FALCON benchmark and produce a tar.gz to ship back.
#
# Single command on the server. Each task is launched with its own
# (n, p, rho) grid because the five tasks have different scaling
# characteristics. Tasks run sequentially; CSVs are rolling-saved so a
# killed job loses at most the in-flight cell.
#
# Usage (uv-managed environment, preferred):
#     uv sync                                  # one-time: create .venv + deps
#     uv run bash benchmarks/run_all.sh        # default 16 workers
#     uv run bash benchmarks/run_all.sh 32     # explicit worker count
#
# If your server fails the PyPI TLS handshake, use `uv sync --native-tls`.
#
# Or if uv is unavailable:
#     python -m venv .venv && source .venv/bin/activate
#     pip install -e .
#     bash benchmarks/run_all.sh 16
#
# Output: data/*.csv (rolling-saved as cells finish) and
#         falcon_results_<host>_<date>.tar.gz at repo root.

set -euo pipefail

WORKERS="${1:-16}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> FALCON benchmark suite on $(hostname)  (workers=$WORKERS)"
date

# Wipe stale CSVs so a fresh run produces a clean tar.gz.
rm -f data/*.csv

PY="${PYTHON:-python}"
RUNNER="benchmarks/run_on_server.py"

# Grids below target ~10 minutes total on a 16-core node, while
# covering biologically realistic regimes (samples <= 1000, features
# <= 10000) and ensuring every method that appears in a comparison
# panel ran on the SAME (n, p) cells -- no cherry-picking. SPIEC-EASI-
# glasso is O(p^3) per iteration and cannot run at p >= 2000 in any
# reasonable time, so method_comparison naturally caps at p=1000.
# Scalability uses FastProp/SparCC/RandProp (all O(np^2)) and goes
# all the way to p=10000.

echo "==> [1/5] scalability  (FastProp + SparCC + RandProp up to p=10000; ~2 min)"
"$PY" "$RUNNER" --task scalability --workers "$WORKERS" \
    --n 500 1000 \
    --p 500 1000 2000 5000 10000

echo "==> [2/5] detection  (FastProp power + AUROC + Recall@K; ~2 min)"
"$PY" "$RUNNER" --task detection --workers "$WORKERS" --reps 5 \
    --n 500 1000 \
    --p 500 1000 2000 \
    --rho 0.4 0.7

echo "==> [3/5] method_comparison  (ALL 6 methods on same cells; ~3 min)"
echo "    Fair-comparison grid: every cell runs every method to completion."
echo "    SPIEC-EASI-glasso is O(p^3); cap p=1000 keeps the cell budget reasonable."
"$PY" "$RUNNER" --task method_comparison --workers "$WORKERS" --reps 5 \
    --n 500 1000 \
    --p 500 1000 \
    --rho 0.7

echo "==> [4/5] cross_domain  (ALL 5 methods on same simulator; ~2 min)"
"$PY" "$RUNNER" --task cross_domain --workers "$WORKERS" --reps 10 \
    --cross-n 300 --cross-p 500 --cross-q 500 --cross-edges 50

echo "==> [5/5] fdr_control  (FastProp BH-FDR calibration; ~1 min)"
"$PY" "$RUNNER" --task fdr_control --workers "$WORKERS" --reps 10 \
    --fdr-n 1000 --fdr-p 500 \
    --alpha 0.01 0.05 0.10 0.20

OUT="falcon_results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
echo "==> Packing CSVs into $OUT"
tar czf "$OUT" data/
ls -lh "$OUT"
echo "==> DONE. Send $OUT back to the laptop and run:"
echo "    tar xzf $OUT && python manuscript/figures/generate_fig1.py && python manuscript/figures/generate_fig2.py"
