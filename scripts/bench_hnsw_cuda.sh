#!/usr/bin/env bash
# Compare CPU HNSW vs GPU HNSW vs GPU brute-force.
# Mirrors scripts/bench_cuda.sh for LSH.
#
# Writes results/hnsw_cuda_compare_M${M}_ef${EF}.csv with columns:
#   method, threads, brute_ms, hnsw_ms, recall, build_ms
#
# Usage:
#     scripts/bench_hnsw_cuda.sh
#     DIM=100 M=16 EF=50 QUERIES=500 scripts/bench_hnsw_cuda.sh
#
# Env knobs:
#     DIM, N, QUERIES, TOPK          dataset + workload
#     M, EF_CONSTRUCTION, EF         HNSW parameters
#     CPU_MODE                       CPU query mode (default: queries)
#     CPU_THREADS                    space-separated thread counts for CPU runs
#     SOURCE                         glove source (default twitter.27B)

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
CPU_MODE=${CPU_MODE:-queries}
CPU_THREADS=${CPU_THREADS:-"1 8"}

source .venv/bin/activate 2>/dev/null || true

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp hnsw hnsw_cuda >/dev/null

mkdir -p results
CSV="results/hnsw_cuda_compare_M${M}_ef${EF}.csv"
echo "method,threads,brute_ms,hnsw_ms,recall,build_ms" > "$CSV"

printf "%-20s %3s %12s %12s %10s %10s\n" method P "brute(ms)" "hnsw(ms)" "recall" "build(ms)"

# --- CPU HNSW runs ---
for P in $CPU_THREADS; do
    out=$(OMP_NUM_THREADS=$P ./cpp/hnsw "$BASE" \
            --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
            --topk "$TOPK" --queries "$QUERIES" --mode "$CPU_MODE" 2>/dev/null)
    brute=$(echo  "$out" | awk '/^brute/ {print $3}')
    hnsw=$(echo   "$out" | awk '/^hnsw/  {print $3}')
    recall=$(echo "$out" | awk '/^recall/{print $3}')
    printf "%-20s %3s %12.2f %12.2f %10s %10s\n" "cpu_$CPU_MODE" "$P" "$brute" "$hnsw" "$recall" "-"
    echo "cpu_$CPU_MODE,$P,$brute,$hnsw,$recall,-" >> "$CSV"
done

# --- GPU brute-force ---
gout=$(./cpp/hnsw_cuda "$BASE" \
         --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
         --topk "$TOPK" --queries "$QUERIES" --mode cuda_brute 2>/dev/null)
brute=$(echo  "$gout" | awk '/^brute/ {print $3}')
recall=$(echo "$gout" | awk '/^recall/{print $3}')
printf "%-20s %3s %12.2f %12s %10s %10s\n" "gpu_brute" "-" "$brute" "-" "$recall" "-"
echo "gpu_brute,-,$brute,-,$recall,-" >> "$CSV"

# --- GPU HNSW ---
gout=$(./cpp/hnsw_cuda "$BASE" \
         --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
         --topk "$TOPK" --queries "$QUERIES" --mode cuda_hnsw 2>/dev/null)
brute=$(echo  "$gout" | awk '/^brute/ {print $3}')
hnsw=$(echo   "$gout" | awk '/^hnsw/  {print $3}')
build=$(echo  "$gout" | awk '/^build/ {print $3}')
recall=$(echo "$gout" | awk '/^recall/{print $3}')
printf "%-20s %3s %12.2f %12.2f %10s %10s\n" "gpu_hnsw" "-" "$brute" "$hnsw" "$recall" "$build"
echo "gpu_hnsw,-,$brute,$hnsw,$recall,$build" >> "$CSV"

echo "wrote $CSV"
python scripts/plot_hnsw_cuda.py --csv "$CSV"
