#!/usr/bin/env bash
# Strong-scaling benchmark for HNSW across parallelization modes.
# Mirrors scripts/bench.sh for LSH.
#
# For each (mode, P): run ./cpp/hnsw, parse stdout, write one CSV row.
#
# Usage:
#     scripts/bench_hnsw.sh
#     DIM=100 M=16 EF=50 THREADS="1 2 4 8" scripts/bench_hnsw.sh
#     MODES="queries queries_dyn neighbors features build" scripts/bench_hnsw.sh
#     PLOT=1 scripts/bench_hnsw.sh          # also call plot_hnsw_scaling.py
#
# Env knobs:
#     DIM, N, QUERIES, TOPK          dataset + workload
#     M, EF_CONSTRUCTION, EF         HNSW parameters
#     THREADS                        space-separated thread counts
#     MODES                          space-separated mode names
#     SOURCE                         glove source (default twitter.27B)
#     PLOT=1                         produce PNG via plot_hnsw_scaling.py

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
SOURCE=${SOURCE:-twitter.27B}
M=${M:-16}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF=${EF:-50}
THREADS=${THREADS:-"1 2 4 8"}
MODES=${MODES:-"serial queries queries_dyn neighbors features build"}

source .venv/bin/activate 2>/dev/null || true

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp hnsw >/dev/null

mkdir -p results
CSV="results/hnsw_scaling_M${M}_ef${EF}.csv"
echo "mode,threads,brute_ms,hnsw_ms,speedup_brute,speedup_hnsw,recall" > "$CSV"

printf "%-12s %3s %12s %12s %9s %8s %8s\n" mode P "brute(ms)" "hnsw(ms)" "S_brute" "S_hnsw" "recall"

for MODE in $MODES; do
    base_b=""; base_h=""
    for P in $THREADS; do
        out=$(OMP_NUM_THREADS=$P OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=3 \
              ./cpp/hnsw "$BASE" \
                --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
                --topk "$TOPK" --queries "$QUERIES" --mode "$MODE" 2>/dev/null)

        brute=$(echo  "$out" | awk '/^brute/  {print $3}')
        hnsw=$(echo   "$out" | awk '/^hnsw/   {print $3}')
        recall=$(echo "$out" | awk '/^recall/ {print $3}')

        [[ -z "$base_b" ]] && { base_b=$brute; base_h=$hnsw; }

        sb=$(awk -v a="$base_b" -v b="$brute" 'BEGIN{printf "%.2f", a/b}')
        sh=$(awk -v a="$base_h" -v b="$hnsw"  'BEGIN{printf "%.2f", a/b}')

        printf "%-12s %3s %12.2f %12.2f %9s %8s %8s\n" \
               "$MODE" "$P" "$brute" "$hnsw" "$sb" "$sh" "$recall"
        echo "$MODE,$P,$brute,$hnsw,$sb,$sh,$recall" >> "$CSV"
    done
done

echo "wrote $CSV"

if [[ "${PLOT:-0}" == "1" ]]; then
    python scripts/plot_hnsw_scaling.py --csv "$CSV"
fi
