#!/usr/bin/env bash
# Benchmark CUDA KD-tree vs GPU brute-force vs CPU KD-tree.
# Sweeps multiple dimensionalities to show curse-of-dimensionality on the GPU.
#
# Writes results/kdtree_cuda_D${DIM}_N${N}.csv per dimension with columns:
#   method, dim, brute_ms, kd_ms, recall, build_ms, speedup_vs_brute
#
# Usage:
#     scripts/bench_kdtree_cuda.sh
#     DIM="25 50 100" N=400000 K=10 QUERIES=200 scripts/bench_kdtree_cuda.sh
#     PLOT=1 scripts/bench_kdtree_cuda.sh
#
# Env knobs:
#     DIMS      space-separated list of dimensionalities (default: "25 50 100")
#     N         number of database points (default: 400000)
#     K         neighbours to retrieve (default: 10)
#     QUERIES   number of query vectors (default: 200)
#     CPU_THREADS  space-separated thread counts for CPU runs (default: "1 8 16")
#     PLOT=1    produce PNG via scripts/plot_kdtree_cuda.py

set -euo pipefail
cd "$(dirname "$0")/.."

DIMS=${DIMS:-"25 50 100"}
N=${N:-400000}
K=${K:-10}
QUERIES=${QUERIES:-200}
CPU_THREADS=${CPU_THREADS:-"1 8 16"}

source .venv/bin/activate

make -C cpp kd_tree kd_tree_cuda >/dev/null

mkdir -p results

# ---- parse helpers (match kd_tree_cuda output) ----
parse_brute()   { echo "$1" | awk '/^brute  :/ {print $3}'; }
parse_build()   { echo "$1" | awk '/^build  :/ {print $3}'; }
parse_query()   { echo "$1" | awk '/^query  :/ {print $3}'; }
parse_speedup() { echo "$1" | awk '/^query  :/ {print $NF}' | cut -dx -f2; }
parse_recall()  { echo "$1" | awk '/^recall@/  {print $3}'; }

# ---- parse helpers (match kd_tree CPU output) ----
cpu_parse_brute()   { echo "$1" | awk '/^brute  :/ {print $3}'; }
cpu_parse_build()   { echo "$1" | awk '/^build  :/ {print $3}'; }
cpu_parse_query()   { echo "$1" | awk '/^query  :/ {print $3}'; }
cpu_parse_speedup() { echo "$1" | awk '/^query  :/ {print $NF}' | cut -dx -f2; }

SUMMARY_CSV="results/kdtree_cuda_summary.csv"
echo "method,dim,n,brute_ms,kd_ms,recall,build_ms,speedup_vs_brute" > "$SUMMARY_CSV"

for DIM in $DIMS; do
    if [[ $DIM -eq 25 ]]; then
        SOURCE="twitter.27B"
    else
        SOURCE="6B"
    fi

    BASE="data/glove${DIM}_${N}"
    [[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
    [[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"

    CSV="results/kdtree_cuda_D${DIM}_N${N}.csv"
    echo "method,dim,n,brute_ms,kd_ms,recall,build_ms,speedup_vs_brute" > "$CSV"

    echo ""
    echo "=== D=${DIM}  N=${N}  k=${K}  queries=${QUERIES} ==="
    printf "%-22s %10s %10s %10s %10s %10s\n" method "brute(ms)" "kd(ms)" recall "build(ms)" "speedup"

    # ---- GPU: brute + KD-tree ----
    gout=$(./cpp/kd_tree_cuda "$BASE" --k "$K" --queries "$QUERIES" 2>/dev/null)
    gpu_brute=$(parse_brute   "$gout")
    gpu_kd=$(parse_query      "$gout")
    gpu_build=$(parse_build   "$gout")
    gpu_recall=$(parse_recall "$gout")
    gpu_sp=$(parse_speedup    "$gout")

    printf "%-22s %10.3f %10.3f %10.4f %10.3f %10s\n" \
        "gpu_brute"  "$gpu_brute" "-"      "1.0000"     "-"         "1.00"
    printf "%-22s %10.3f %10.3f %10.4f %10.3f %10s\n" \
        "gpu_kd"     "$gpu_brute" "$gpu_kd" "$gpu_recall" "$gpu_build" "${gpu_sp}x"

    echo "gpu_brute,$DIM,$N,$gpu_brute,-,1.0000,-,1.00"  >> "$CSV"
    echo "gpu_kd,$DIM,$N,$gpu_brute,$gpu_kd,$gpu_recall,$gpu_build,$gpu_sp" >> "$CSV"
    echo "gpu_brute,$DIM,$N,$gpu_brute,-,1.0000,-,1.00"  >> "$SUMMARY_CSV"
    echo "gpu_kd,$DIM,$N,$gpu_brute,$gpu_kd,$gpu_recall,$gpu_build,$gpu_sp" >> "$SUMMARY_CSV"

    # ---- CPU: kd_tree at various thread counts ----
    for P in $CPU_THREADS; do
        cout=$(OMP_NUM_THREADS=$P \
               ./cpp/kd_tree "$BASE" \
                   --build-mode serial --query-mode serial \
                   --k "$K" --iters 5 --threads "$P" 2>/dev/null)
        cpu_brute=$(cpu_parse_brute   "$cout")
        cpu_kd=$(cpu_parse_query      "$cout")
        cpu_build=$(cpu_parse_build   "$cout")
        cpu_sp=$(awk -v g="$gpu_brute" -v c="$cpu_kd" \
                     'BEGIN{printf "%.2f", g/c}')
        label="cpu_serial_P${P}"
        printf "%-22s %10.3f %10.3f %10s %10.3f %10s\n" \
            "$label" "$cpu_brute" "$cpu_kd" "-" "$cpu_build" "${cpu_sp}x"
        echo "$label,$DIM,$N,$cpu_brute,$cpu_kd,-,$cpu_build,$cpu_sp" >> "$CSV"
        echo "$label,$DIM,$N,$cpu_brute,$cpu_kd,-,$cpu_build,$cpu_sp" >> "$SUMMARY_CSV"
    done

    echo "wrote $CSV"
done

echo ""
echo "wrote $SUMMARY_CSV"

if [[ "${PLOT:-0}" == "1" ]]; then
    python scripts/plot_kdtree_cuda.py --csv "$SUMMARY_CSV"
fi
