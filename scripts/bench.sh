#!/usr/bin/env bash
# Strong-scaling benchmark: run ./cpp/lsh once per thread count, parse the
# "brute" / "lsh" / "recall" lines, write CSV + print table.
#
# Usage:
#     scripts/bench.sh
#     DIM=200 N=1000000 L=64 K=10 THREADS="1 2 4 8" scripts/bench.sh
#
# Env knobs:
#     DIM, N, QUERIES, TOPK     dataset + workload
#     L, K                       LSH parameters
#     THREADS                    space-separated thread counts (e.g. "1 2 4 8")
#     SOURCE                     glove source (default twitter.27B)

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
SOURCE=${SOURCE:-twitter.27B}
L=${L:-16}
K=${K:-12}
THREADS=${THREADS:-"1 2 4 8 12"}

source .venv/bin/activate

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp >/dev/null

CSV="data/scaling_L${L}_K${K}.csv"
echo "threads,brute_ms,lsh_ms,speedup_brute,speedup_lsh,recall" > "$CSV"

printf "%3s %12s %12s %9s %8s %8s\n" P "brute(ms)" "lsh(ms)" "S_brute" "S_lsh" "recall"
base_b=""; base_l=""
for P in $THREADS; do
    out=$(OMP_NUM_THREADS=$P ./cpp/lsh "$BASE" \
            --L "$L" --K "$K" --topk "$TOPK" --queries "$QUERIES" 2>/dev/null)
    brute=$(echo "$out"  | awk '/^brute/  {print $3}')
    lsh=$(echo "$out"    | awk '/^lsh/    {print $3}')
    recall=$(echo "$out" | awk '/^recall/ {print $3}')
    if [[ -z "$base_b" ]]; then base_b=$brute; base_l=$lsh; fi
    sb=$(awk -v a="$base_b" -v b="$brute" 'BEGIN{printf "%.2f", a/b}')
    sl=$(awk -v a="$base_l" -v b="$lsh"   'BEGIN{printf "%.2f", a/b}')
    printf "%3s %12.2f %12.2f %9s %8s %8s\n" "$P" "$brute" "$lsh" "$sb" "$sl" "$recall"
    echo "$P,$brute,$lsh,$sb,$sl,$recall" >> "$CSV"
done
echo "wrote $CSV"
