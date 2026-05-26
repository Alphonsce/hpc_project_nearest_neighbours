#!/usr/bin/env bash
# HNSW recall-vs-speed sweep: vary ef (query beam width) at fixed thread count.
# Produces a CSV suitable for plot_hnsw_ef.py.
#
# Usage:
#     scripts/bench_hnsw_ef.sh
#     DIM=100 M=16 EF_VALS="10 20 50 100 200" THREADS=4 scripts/bench_hnsw_ef.sh
#     PLOT=1 scripts/bench_hnsw_ef.sh
#
# Env knobs:
#     DIM, N, QUERIES, TOPK          dataset + workload
#     M, EF_CONSTRUCTION             HNSW build parameters
#     EF_VALS                        space-separated ef values to sweep
#     THREADS                        OMP thread count (fixed for this sweep)
#     MODES                          space-separated modes (default: serial queries build)
#     SOURCE                         glove source (default twitter.27B)
#     PLOT=1                         call plot_hnsw_ef.py after the sweep

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
SOURCE=${SOURCE:-twitter.27B}
M=${M:-16}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF_VALS=${EF_VALS:-"10 20 30 50 75 100 150 200 300 500"}
THREADS=${THREADS:-4}
MODES=${MODES:-"serial queries build"}

source .venv/bin/activate 2>/dev/null || true

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp hnsw >/dev/null

mkdir -p results
CSV="results/hnsw_ef_sweep_M${M}_P${THREADS}.csv"
echo "mode,ef,threads,hnsw_ms,recall" > "$CSV"

printf "%-12s %5s %3s %12s %8s\n" mode ef P "hnsw(ms)" "recall"

for MODE in $MODES; do
    for EF in $EF_VALS; do
        out=$(OMP_NUM_THREADS=$THREADS OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=3 \
              ./cpp/hnsw "$BASE" \
                --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
                --topk "$TOPK" --queries "$QUERIES" --mode "$MODE" 2>/dev/null)

        hnsw=$(echo   "$out" | awk '/^hnsw/   {print $3}')
        recall=$(echo "$out" | awk '/^recall/ {print $3}')

        printf "%-12s %5s %3s %12.2f %8s\n" "$MODE" "$EF" "$THREADS" "$hnsw" "$recall"
        echo "$MODE,$EF,$THREADS,$hnsw,$recall" >> "$CSV"
    done
done

echo "wrote $CSV"

if [[ "${PLOT:-0}" == "1" ]]; then
    python scripts/plot_hnsw_ef.py --csv "$CSV"
fi
