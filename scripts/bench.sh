#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/kd-tree"
STATS="$ROOT/stats"

K="${1:-10}"
ITERS="${2:-200}"
THREADS=(1 2 4 8 16)
DATASETS=(glove25_400000.npy glove50_400000.npy glove100_400000.npy)

if [[ ! -f "$BIN" ]]; then
    echo "binary not found: $BIN — run cmake --build build first" >&2
    exit 1
fi

for f in "${DATASETS[@]}"; do
    if [[ ! -f "$ROOT/data/$f" ]]; then
        echo "missing data/$f — run scripts/prepare_data.sh first" >&2
        exit 1
    fi
done

rm -f "$STATS"/kdtree_build.csv "$STATS"/kdtree_query.csv \
      "$STATS"/exhaustive_query.csv

for DATA in "${DATASETS[@]}"; do
    echo "=== $DATA ==="

    echo "  exhaustive"
    for t in "${THREADS[@]}"; do
        "$BIN" --data "$DATA" --algo exhaustive \
            --k "$K" --iters "$ITERS" --threads "$t" > /dev/null
        echo -n "."
    done
    echo ""

    for BM in serial parallel-tasks parallel-flat; do
        echo "  kdtree build=$BM"
        for t in "${THREADS[@]}"; do
            "$BIN" --data "$DATA" --algo kdtree \
                --build-mode "$BM" --query-mode serial \
                --k "$K" --iters 1 --threads "$t" > /dev/null
            echo -n "."
        done
        echo ""
    done

    echo "  kdtree query sweep (serial build)"
    for QM in serial local-heaps atomic-global; do
        echo "    query=$QM"
        for t in "${THREADS[@]}"; do
            "$BIN" --data "$DATA" --algo kdtree \
                --build-mode serial --query-mode "$QM" \
                --k "$K" --iters "$ITERS" --threads "$t" > /dev/null
            echo -n "."
        done
        echo ""
    done
done

echo ""
echo "done — results in stats/"
ls "$STATS"/*.csv | xargs -I{} basename {}
