"""Plot CPU vs GPU HNSW wall times from hnsw_cuda_compare_*.csv.
Mirrors scripts/plot_cuda.py for LSH.

Two panels:
    left:  bar chart of query wall time (ms) per method (log scale)
    right: bar chart of speedup vs CPU brute single-threaded

Input CSV columns: method, threads, brute_ms, hnsw_ms, recall, build_ms

Usage:
    python scripts/plot_hnsw_cuda.py --csv results/hnsw_cuda_compare_M16_ef50.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Build (label, time_ms, recall) entries.
    labels: list[str] = []
    times:  list[float] = []
    recalls: list[float] = []

    for r in rows:
        m = r["method"]; p = r["threads"]
        if m.startswith("cpu_"):
            mode_tag = m[4:]   # e.g. "queries"
            labels.append(f"CPU brute (P={p})")
            times.append(float(r["brute_ms"]))
            recalls.append(1.0)

            labels.append(f"CPU HNSW/{mode_tag} (P={p})")
            times.append(float(r["hnsw_ms"]))
            recalls.append(float(r["recall"]))

        elif m == "gpu_brute":
            labels.append("GPU brute")
            times.append(float(r["brute_ms"]))
            recalls.append(1.0)

        elif m == "gpu_hnsw":
            labels.append("GPU HNSW")
            times.append(float(r["hnsw_ms"]))
            recalls.append(float(r["recall"]))

    # Speedup relative to the first row (CPU brute P=1).
    baseline = times[0]
    speedups = [baseline / t for t in times]

    # Colour: brute=blue family, HNSW=orange/red family; GPU darker.
    colors = []
    for lab in labels:
        if "GPU" in lab:
            colors.append("tab:red" if "HNSW" in lab else "tab:orange")
        else:
            colors.append("tab:cyan" if "HNSW" in lab else "tab:blue")

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(labels))

    # Left: wall time.
    ax_t.bar(x, times, color=colors)
    ax_t.set_yscale("log")
    ax_t.set_ylabel("query wall time (ms)")
    ax_t.set_title("CPU vs GPU HNSW — query wall time (log)")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for i, (t, r) in enumerate(zip(times, recalls)):
        ax_t.text(i, t * 1.15, f"{t:.1f}ms\nrec={r:.2f}",
                  ha="center", va="bottom", fontsize=8)

    # Right: speedup.
    ax_s.bar(x, speedups, color=colors)
    ax_s.set_yscale("log")
    ax_s.set_ylabel(f"speedup vs {labels[0]}")
    ax_s.set_title("Speedup vs CPU brute P=1 (log)")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    for i, s in enumerate(speedups):
        ax_s.text(i, s * 1.15, f"×{s:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("HNSW: CPU (OpenMP) vs GPU (CUDA)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
