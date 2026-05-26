#!/usr/bin/env bash
# Compare CPU vs GPU brute / LSH on the same dataset and (L, K).
# Emits results/cuda_compare_L${L}_K${K}.csv with columns:
#   method, threads, brute_ms, lsh_ms, recall, build_ms
#
# Usage:
#     scripts/bench_cuda.sh
#     DIM=100 N=1000000 L=32 K=12 QUERIES=500 TOPK=10 scripts/bench_cuda.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
SOURCE=${SOURCE:-twitter.27B}
L=${L:-32}
K=${K:-12}
CPU_MODE=${CPU_MODE:-queries}
CPU_THREADS=${CPU_THREADS:-"1 8"}

source .venv/bin/activate

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp >/dev/null

mkdir -p results
CSV="results/cuda_compare_L${L}_K${K}.csv"
echo "method,threads,brute_ms,lsh_ms,recall,build_ms" > "$CSV"

printf "%-14s %3s %12s %12s %10s %10s\n" method P "brute(ms)" "lsh(ms)" "recall" "build(ms)"

# --- CPU runs ---
for P in $CPU_THREADS; do
    out=$(OMP_NUM_THREADS=$P ./cpp/lsh "$BASE" \
            --L "$L" --K "$K" --topk "$TOPK" --queries "$QUERIES" --mode "$CPU_MODE" 2>/dev/null)
    brute=$(echo "$out"  | awk '/^brute/  {print $3}')
    lsh=$(echo "$out"    | awk '/^lsh/    {print $3}')
    recall=$(echo "$out" | awk '/^recall/ {print $3}')
    printf "%-14s %3s %12.2f %12.2f %10s %10s\n" "cpu_$CPU_MODE" "$P" "$brute" "$lsh" "$recall" "-"
    echo "cpu_$CPU_MODE,$P,$brute,$lsh,$recall,-" >> "$CSV"
done

# --- GPU brute ---
gout=$(./cpp/lsh_cuda "$BASE" --L "$L" --K "$K" --topk "$TOPK" --queries "$QUERIES" --mode cuda_brute 2>/dev/null)
brute=$(echo "$gout"  | awk '/^brute/  {print $3}')
recall=$(echo "$gout" | awk '/^recall/ {print $3}')
printf "%-14s %3s %12.2f %12s %10s %10s\n" "gpu_brute" "-" "$brute" "-" "$recall" "-"
echo "gpu_brute,-,$brute,-,$recall,-" >> "$CSV"

# --- GPU LSH ---
gout=$(./cpp/lsh_cuda "$BASE" --L "$L" --K "$K" --topk "$TOPK" --queries "$QUERIES" --mode cuda_lsh 2>/dev/null)
brute=$(echo "$gout"  | awk '/^brute/  {print $3}')
lsh=$(echo "$gout"    | awk '/^lsh/    {print $3}')
build=$(echo "$gout"  | awk '/^build/  {print $3}')
recall=$(echo "$gout" | awk '/^recall/ {print $3}')
printf "%-14s %3s %12.2f %12.2f %10s %10s\n" "gpu_lsh" "-" "$brute" "$lsh" "$recall" "$build"
echo "gpu_lsh,-,$brute,$lsh,$recall,$build" >> "$CSV"

echo "wrote $CSV"

python scripts/plot_cuda.py --csv "$CSV"
