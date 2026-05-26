#!/usr/bin/env bash
# Ablation: HNSW "queries" parallelism — CPU thread scaling vs GPU.
#
# Runs CPU HNSW in "queries" mode at 1, 2, 4, 8 threads, then GPU brute-force
# and GPU HNSW. The plot lets you directly see at which thread count the CPU
# catches up to (or is beaten by) the GPU.
#
# Usage:
#     bash scripts/ablate_cuda_queries.sh
#     DIM=100 N=1000000 QUERIES=500 scripts/ablate_cuda_queries.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
M=${M:-16}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF=${EF:-50}
CPU_THREADS=${CPU_THREADS:-"1 2 4 8"}
SOURCE=${SOURCE:-twitter.27B}
CPU_MODE=queries

source .venv/bin/activate 2>/dev/null || true

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"
make -C cpp hnsw hnsw_cuda >/dev/null

mkdir -p results
CSV="results/ablate_cuda_queries_M${M}_ef${EF}.csv"
echo "method,threads,brute_ms,hnsw_ms,speedup_vs_cpu1,recall" > "$CSV"

printf "%-22s %3s %12s %12s %18s %8s\n" \
       method P "brute(ms)" "hnsw(ms)" "speedup_vs_cpu1" "recall"

cpu1_hnsw=""
for P in $CPU_THREADS; do
    out=$(OMP_NUM_THREADS=$P \
          ./cpp/hnsw "$BASE" \
            --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
            --topk "$TOPK" --queries "$QUERIES" --mode "$CPU_MODE" 2>/dev/null)

    brute=$(echo  "$out" | awk '/^brute/  {print $3}')
    hnsw=$(echo   "$out" | awk '/^hnsw/   {print $3}')
    recall=$(echo "$out" | awk '/^recall/ {print $3}')

    [[ -z "$cpu1_hnsw" ]] && cpu1_hnsw=$hnsw
    spd=$(awk -v a="$cpu1_hnsw" -v b="$hnsw" 'BEGIN{printf "%.3f", a/b}')

    label="cpu_queries_P${P}"
    printf "%-22s %3s %12.2f %12.2f %18s %8s\n" "$label" "$P" "$brute" "$hnsw" "$spd" "$recall"
    echo "$label,$P,$brute,$hnsw,$spd,$recall" >> "$CSV"
done

# GPU brute-force
gout=$(./cpp/hnsw_cuda "$BASE" \
         --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
         --topk "$TOPK" --queries "$QUERIES" --mode cuda_brute 2>/dev/null)
gbrute=$(echo  "$gout" | awk '/^brute/  {print $3}')
grecall=$(echo "$gout" | awk '/^recall/ {print $3}')
spd=$(awk -v a="$cpu1_hnsw" -v b="$gbrute" 'BEGIN{printf "%.3f", a/b}')
printf "%-22s %3s %12.2f %12s %18s %8s\n" "gpu_brute" "-" "$gbrute" "-" "$spd" "$grecall"
echo "gpu_brute,-,$gbrute,-,$spd,$grecall" >> "$CSV"

# GPU HNSW
gout=$(./cpp/hnsw_cuda "$BASE" \
         --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
         --topk "$TOPK" --queries "$QUERIES" --mode cuda_hnsw 2>/dev/null)
gbrute=$(echo  "$gout" | awk '/^brute/  {print $3}')
ghnsw=$(echo   "$gout" | awk '/^hnsw/   {print $3}')
grecall=$(echo "$gout" | awk '/^recall/ {print $3}')
spd=$(awk -v a="$cpu1_hnsw" -v b="$ghnsw" 'BEGIN{printf "%.3f", a/b}')
printf "%-22s %3s %12.2f %12.2f %18s %8s\n" "gpu_hnsw" "-" "$gbrute" "$ghnsw" "$spd" "$grecall"
echo "gpu_hnsw,-,$gbrute,$ghnsw,$spd,$grecall" >> "$CSV"

echo "wrote $CSV"

# ---- inline plot ----
python3 - "$CSV" <<'PYEOF'
import sys, csv
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

labels, hnsw_ms, speedups, recalls, colors = [], [], [], [], []
for r in rows:
    m = r["method"]
    if m.startswith("cpu_"):
        p = r["threads"]
        labels.append(f"CPU queries\nP={p}")
        hnsw_ms.append(float(r["hnsw_ms"]))
        colors.append("tab:blue")
    elif m == "gpu_brute":
        labels.append("GPU\nbrute")
        hnsw_ms.append(float(r["brute_ms"]))
        colors.append("tab:orange")
    elif m == "gpu_hnsw":
        labels.append("GPU\nHNSW")
        hnsw_ms.append(float(r["hnsw_ms"]))
        colors.append("tab:red")
    speedups.append(float(r["speedup_vs_cpu1"]))
    recalls.append(float(r["recall"]))

x = np.arange(len(labels))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Wall time (log)
ax1.bar(x, hnsw_ms, color=colors)
ax1.set_yscale("log"); ax1.set_ylabel("Query time (ms, log)")
ax1.set_title("Wall time — CPU queries vs GPU")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
for i, v in enumerate(hnsw_ms):
    ax1.text(i, v * 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

# Speedup vs CPU P=1
ax2.bar(x, speedups, color=colors)
ax2.set_ylabel("Speedup vs CPU P=1"); ax2.set_title("Speedup vs CPU serial")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
for i, v in enumerate(speedups):
    ax2.text(i, v + 0.05, f"{v:.2f}×", ha="center", va="bottom", fontsize=8)

# Recall
ax3.bar(x, recalls, color=colors)
ax3.set_ylim(0, 1.1); ax3.set_ylabel("Recall@k"); ax3.set_title("Recall")
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
for i, v in enumerate(recalls):
    ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Ablation: 'queries' parallelism — CPU threads vs GPU", fontsize=13)
fig.tight_layout()
out = path.replace(".csv", ".png")
fig.savefig(out, dpi=140)
print(f"saved -> {out}")
PYEOF
