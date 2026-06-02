#!/usr/bin/env bash
# Falcon-SR feasibility benchmark orchestrator.
# Runs both the single-domain and cross-domain grids end-to-end on the
# default feasibility cells, writing data/falcon_sr_*_feasibility.csv.
#
# Override the grids by passing flags directly to the underlying runners:
#   ./benchmarks/run_all.sh --single-args '--n 100 --p 100 --top-k 10 --reps 1'
#   ./benchmarks/run_all.sh --cross-args  '--n 100 --pq 100,100 --top-k 10 --reps 1'
#   ./benchmarks/run_all.sh --skip-cross           # single only
#   ./benchmarks/run_all.sh --skip-single          # cross only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

SINGLE_ARGS=""
CROSS_ARGS=""
SKIP_SINGLE=""
SKIP_CROSS=""

while (( "$#" )); do
    case "$1" in
        --single-args) SINGLE_ARGS="$2"; shift 2 ;;
        --cross-args)  CROSS_ARGS="$2";  shift 2 ;;
        --skip-single) SKIP_SINGLE=1;    shift ;;
        --skip-cross)  SKIP_CROSS=1;     shift ;;
        *) echo "Unknown flag $1"; exit 1 ;;
    esac
done

# Wipe stale feasibility CSVs so a fresh run produces clean output.
rm -f data/falcon_sr_single_feasibility.csv data/falcon_sr_cross_feasibility.csv

if [[ -z "${SKIP_SINGLE}" ]]; then
    echo "==> single-domain feasibility ${SINGLE_ARGS}"
    # shellcheck disable=SC2086
    uv run python benchmarks/falcon_sr_single.py ${SINGLE_ARGS}
fi

if [[ -z "${SKIP_CROSS}" ]]; then
    echo "==> cross-domain feasibility ${CROSS_ARGS}"
    # shellcheck disable=SC2086
    uv run python benchmarks/falcon_sr_cross.py ${CROSS_ARGS}
fi

echo "==> DONE"
ls -lh data/falcon_sr_*.csv 2>/dev/null || true
