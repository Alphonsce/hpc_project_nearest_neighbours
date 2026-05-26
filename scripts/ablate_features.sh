#!/usr/bin/env bash
# Ablation: HNSW "features" parallelism — thread scaling (1, 2, 4, 8).
#
# Parallelises the innermost loop: the D-dimensional dot product inside every
# single distance call, using OMP parallel reduction. Shows that spawning a
# thread team for a D-float sum (D=50..300) is orders of magnitude slower than
# serial, making this the worst possible granularity for HNSW parallelism.
#
# Usage:
#     bash scripts/ablate_features.sh
#     DIM=100 N=1000000 QUERIES=100 scripts/ablate_features.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DIM=${DIM:-100}
N=${N:-1000000}
QUERIES=${QUERIES:-500}
TOPK=${TOPK:-10}
M=${M:-16}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF=${EF:-50}
THREADS=${THREADS:-"1 2 4 8"}
SOURCE=${SOURCE:-twitter.27B}
MODE=features

source .venv/bin/activate 2>/dev/null || true

BASE="data/glove${DIM}_${N}"
[[ -f "${BASE}_norm.npy" ]] || python scripts/prepare_glove.py --source "$SOURCE" --dim "$DIM" --n "$N"
[[ -f "${BASE}.f32"      ]] || python scripts/export_for_cpp.py --dim "$DIM" --n "$N"
make -C cpp hnsw >/dev/null

mkdir -p results
CSV="results/ablate_features_M${M}_ef${EF}.csv"
echo "threads,brute_ms,hnsw_ms,speedup_vs_serial,speedup_vs_brute,recall" > "$CSV"

printf "%-8s %12s %12s %18s %18s %8s\n" \
       threads "brute(ms)" "hnsw(ms)" "speedup_vs_serial" "speedup_vs_brute" "recall"

serial_hnsw=""
for P in $THREADS; do
    out=$(OMP_NUM_THREADS=$P OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=3 \
          ./cpp/hnsw "$BASE" \
            --M "$M" --ef_construction "$EF_CONSTRUCTION" --ef "$EF" \
            --topk "$TOPK" --queries "$QUERIES" --mode "$MODE" 2>/dev/null)

    brute=$(echo  "$out" | awk '/^brute/  {print $3}')
    hnsw=$(echo   "$out" | awk '/^hnsw/   {print $3}')
    recall=$(echo "$out" | awk '/^recall/ {print $3}')

    [[ -z "$serial_hnsw" ]] && serial_hnsw=$hnsw

    s_serial=$(awk -v a="$serial_hnsw" -v b="$hnsw"  'BEGIN{printf "%.3f", a/b}')
    s_brute=$(awk  -v a="$brute"       -v b="$hnsw"  'BEGIN{printf "%.3f", a/b}')

    printf "%-8s %12.2f %12.2f %18s %18s %8s\n" \
           "$P" "$brute" "$hnsw" "$s_serial" "$s_brute" "$recall"
    echo "$P,$brute,$hnsw,$s_serial,$s_brute,$recall" >> "$CSV"
done

echo "wrote $CSV"

# ---- inline plot ----
python3 - "$CSV" <<'PYEOF'
import sys, csv
import matplotlib.pyplot as plt

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
threads   = [int(r["threads"])             for r in rows]
hnsw_ms   = [float(r["hnsw_ms"])           for r in rows]
brute_ms  = [float(r["brute_ms"])          for r in rows]
s_serial  = [float(r["speedup_vs_serial"]) for r in rows]
s_brute   = [float(r["speedup_vs_brute"])  for r in rows]
recall    = [float(r["recall"])            for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 — wall time (log scale: features is much slower)
axes[0].plot(threads, hnsw_ms,  "o-", color="tab:purple", label="HNSW (features)")
axes[0].plot(threads, brute_ms, "s--", color="tab:gray",  label="Brute force")
axes[0].set_yscale("log")
axes[0].set_xlabel("Threads"); axes[0].set_ylabel("Wall time (ms, log)")
axes[0].set_title("Wall time vs Threads (log scale)"); axes[0].legend()
axes[0].set_xticks(threads)
for x, y in zip(threads, hnsw_ms):
    axes[0].annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

# Panel 2 — speedup (anti-speedup expected: degrades with more threads)
axes[1].plot(threads, s_serial, "o-", color="tab:orange", label="Speedup vs P=1")
axes[1].plot(threads, threads,  "k--", linewidth=1,       label="Ideal linear")
axes[1].axhline(1.0, color="tab:red", linewidth=0.8, linestyle=":", label="No gain")
axes[1].set_xlabel("Threads"); axes[1].set_ylabel("Speedup")
axes[1].set_title("Speedup vs Serial\n(fork overhead >> D-dim dot product)"); axes[1].legend()
axes[1].set_xticks(threads)
for x, y in zip(threads, s_serial):
    axes[1].annotate(f"{y:.3f}×", (x, y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

# Panel 3 — recall (should be identical across thread counts)
axes[2].plot(threads, recall, "^-", color="tab:green")
axes[2].set_xlabel("Threads"); axes[2].set_ylabel("Recall@k")
axes[2].set_ylim(0, 1.05); axes[2].set_title("Recall stability across thread counts")
axes[2].set_xticks(threads)
for x, y in zip(threads, recall):
    axes[2].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

fig.suptitle("HNSW Ablation — 'features' parallelism (inner dot-product reduction)", fontsize=13)
fig.tight_layout()
out = path.replace(".csv", ".png")
fig.savefig(out, dpi=140)
print(f"saved -> {out}")
PYEOF
