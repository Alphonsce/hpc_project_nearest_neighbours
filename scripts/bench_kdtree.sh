#!/usr/bin/env bash
# Strong-scaling benchmark for KD-tree across build and query parallelization modes.
#
# Usage:
#     scripts/bench_kdtree.sh
#     DIM=50 N=400000 THREADS="1 2 4 8 16" scripts/bench_kdtree.sh
#     BUILD_MODES="serial parallel-flat" QUERY_MODES="serial atomic-global" scripts/bench_kdtree.sh
#     PLOT=1 scripts/bench_kdtree.sh
#
# Env knobs:
#     DIM, N, ITERS, K       dataset + workload
#     THREADS                space-separated thread counts
#     BUILD_MODES            space-separated build mode names
#     QUERY_MODES            space-separated query mode names
#     SOURCE                 glove source (default twitter.27B for 25d, 6B otherwise)
#     PLOT=1                 produce PNG via scripts/plot_kdtree.py

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-25}
N=${N:-400000}
ITERS=${ITERS:-200}
K=${K:-10}
THREADS=${THREADS:-"1 2 4 8 16"}
BUILD_MODES=${BUILD_MODES:-"serial parallel-tasks parallel-flat"}
QUERY_MODES=${QUERY_MODES:-"serial local-heaps atomic-global"}

if [[ $DIM -eq 25 ]]; then
    SOURCE=${SOURCE:-twitter.27B}
else
    SOURCE=${SOURCE:-6B}
fi

source .venv/bin/activate

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

make -C cpp kd_tree >/dev/null

mkdir -p results

parse_brute()   { echo "$1" | awk '/^brute  :/ {print $3}'; }
parse_build()   { echo "$1" | awk '/^build  :/ {print $3}'; }
parse_query()   { echo "$1" | awk '/^query  :/ {print $3}'; }
parse_us()      { echo "$1" | awk '/^query  :/ {gsub(/[()]/,""); print $5}'; }
parse_nodes()   { echo "$1" | awk '/^query  :/ {print $7}' | cut -d= -f2; }
parse_speedup() { echo "$1" | awk '/^query  :/ {print $9}' | cut -dx -f2; }

# Build scaling: sweep build modes and threads, 1 query iteration
BUILD_CSV="results/kdtree_build_D${DIM}_N${N}.csv"
echo "build_mode,threads,build_ms,speedup" > "$BUILD_CSV"

echo "=== build scaling  D=${DIM} N=${N} ==="
printf "%-16s %3s %12s %9s\n" build_mode P "build(ms)" speedup
for BM in $BUILD_MODES; do
    base_b=""
    for P in $THREADS; do
        out=$(OMP_NUM_THREADS=$P \
              ./cpp/kd_tree "$BASE" --build-mode "$BM" --query-mode serial \
                            --k "$K" --iters 1 --threads "$P" 2>/dev/null)
        build_ms=$(parse_build "$out")
        [[ -z "$base_b" ]] && base_b=$build_ms
        sb=$(awk -v a="$base_b" -v b="$build_ms" 'BEGIN{printf "%.2f", a/b}')
        printf "%-16s %3s %12.3f %9s\n" "$BM" "$P" "$build_ms" "$sb"
        echo "$BM,$P,$build_ms,$sb" >> "$BUILD_CSV"
    done
done
echo "wrote $BUILD_CSV"

echo ""

# Query scaling: sweep query modes and threads, serial build
QUERY_CSV="results/kdtree_query_D${DIM}_N${N}.csv"
echo "query_mode,threads,brute_ms,query_ms,query_us,nodes_mean,speedup_threads,speedup_brute" > "$QUERY_CSV"

echo "=== query scaling  D=${DIM} N=${N} ==="
printf "%-16s %3s %12s %12s %12s %9s %9s\n" query_mode P "brute(ms)" "query(ms)" "us/query" "S_thread" "S_brute"
for QM in $QUERY_MODES; do
    base_q=""
    for P in $THREADS; do
        out=$(OMP_NUM_THREADS=$P \
              ./cpp/kd_tree "$BASE" --build-mode serial --query-mode "$QM" \
                            --k "$K" --iters "$ITERS" --threads "$P" 2>/dev/null)
        brute_ms=$(parse_brute   "$out")
        query_ms=$(parse_query   "$out")
        query_us=$(parse_us      "$out")
        nodes=$(parse_nodes      "$out")
        sb=$(parse_speedup       "$out")
        [[ -z "$base_q" ]] && base_q=$query_ms
        sq=$(awk -v a="$base_q" -v b="$query_ms" 'BEGIN{printf "%.2f", a/b}')
        printf "%-16s %3s %12.3f %12.3f %12.1f %9s %9s\n" "$QM" "$P" "$brute_ms" "$query_ms" "$query_us" "$sq" "$sb"
        echo "$QM,$P,$brute_ms,$query_ms,$query_us,$nodes,$sq,$sb" >> "$QUERY_CSV"
    done
done
echo "wrote $QUERY_CSV"

if [[ "${PLOT:-0}" == "1" ]]; then
    python scripts/plot_kdtree.py --build-csv "$BUILD_CSV" --query-csv "$QUERY_CSV"
fi
