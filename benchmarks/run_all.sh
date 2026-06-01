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

echo "==> [1/5] scalability  (FastProp + RandProp wall-clock, up to p=10000, n=5000)"
"$PY" "$RUNNER" --task scalability --workers "$WORKERS" \
    --n 500 1000 2000 5000 \
    --p 500 1000 2000 5000 10000

echo "==> [2/5] detection  (power + AUROC + Recall@K across (n, p, rho))"
"$PY" "$RUNNER" --task detection --workers "$WORKERS" --reps 10 \
    --n 500 1000 2000 5000 \
    --p 500 1000 2000 5000 \
    --rho 0.4 0.7

echo "==> [3/5] method_comparison  (FastProp vs SparCC vs Pearson vs SPIEC-EASI x2)"
echo "    NOTE: SPIEC-EASI-glasso is O(p^3) per iteration; at p>=2000 each cell"
echo "          takes minutes. Cap at p=2000 by default to keep wall-clock bounded."
"$PY" "$RUNNER" --task method_comparison --workers "$WORKERS" --reps 5 \
    --n 500 1000 2000 \
    --p 500 1000 2000 \
    --rho 0.7

echo "==> [4/5] cross_domain  (CrossNet vs SparXCC base/iter vs SPIEC-EASI-cross vs naive CLR)"
"$PY" "$RUNNER" --task cross_domain --workers "$WORKERS" --reps 20 \
    --cross-n 300 --cross-p 500 --cross-q 500 --cross-edges 50

echo "==> [5/5] fdr_control  (analytic Fisher-z + BH-FDR calibration)"
"$PY" "$RUNNER" --task fdr_control --workers "$WORKERS" --reps 30 \
    --fdr-n 2000 --fdr-p 500 \
    --alpha 0.01 0.05 0.10 0.20

OUT="falcon_results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
echo "==> Packing CSVs into $OUT"
tar czf "$OUT" data/
ls -lh "$OUT"
echo "==> DONE. Send $OUT back to the laptop and run:"
echo "    tar xzf $OUT && python manuscript/figures/generate_fig1.py && python manuscript/figures/generate_fig2.py"
