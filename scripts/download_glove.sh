#!/usr/bin/env bash
# Download and prepare both GloVe datasets used by the benchmarks.
# Run this once on any machine with internet access.
#
# Usage:
#   bash scripts/download_glove.sh
#
# Env overrides:
#   SKIP_SMALL=1   skip the 6B-50d 20k subset
#   SKIP_LARGE=1   skip the twitter.27B-100d 1M subset

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv if present
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# --- small visualisable subset ---
if [[ "${SKIP_SMALL:-0}" != "1" ]]; then
    echo "==> GloVe 6B-50d  (20k rows)  ~862 MB zip download"
    python scripts/prepare_glove.py --source 6B --dim 50 --n 20000
    python scripts/export_for_cpp.py --dim 50 --n 20000
fi

# --- full benchmark dataset ---
if [[ "${SKIP_LARGE:-0}" != "1" ]]; then
    echo "==> GloVe twitter.27B-100d  (1M rows)  ~1.5 GB zip download"
    python scripts/prepare_glove.py --source twitter.27B --dim 100 --n 1000000
    python scripts/export_for_cpp.py --dim 100 --n 1000000
fi

echo ""
echo "All data ready in data/:"
ls -lh data/glove*.npy data/glove*.f32 data/glove*.shape 2>/dev/null || true
