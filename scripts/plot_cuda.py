"""Plot CPU vs GPU brute / LSH wall times from cuda_compare_*.csv.

Two panels:
    left:  bar chart of query wall time (ms) per method (log scale)
    right: bar chart of speedup vs CPU brute single-threaded
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

    # Build per-method (label, time_ms, recall)
    labels, times, recalls = [], [], []
    for r in rows:
        m = r["method"]; p = r["threads"]
        if m.startswith("cpu_"):
            # brute (single col) + lsh (single col) per row
            labels.append(f"CPU brute (P={p})");  times.append(float(r["brute_ms"])); recalls.append(1.0)
            labels.append(f"CPU LSH (P={p})");    times.append(float(r["lsh_ms"]));   recalls.append(float(r["recall"]))
        elif m == "gpu_brute":
            labels.append("GPU brute");           times.append(float(r["brute_ms"])); recalls.append(1.0)
        elif m == "gpu_lsh":
            labels.append("GPU LSH");             times.append(float(r["lsh_ms"]));   recalls.append(float(r["recall"]))

    baseline = times[0]  # first row is CPU brute P=1
    speedups = [baseline / t for t in times]

    colors = []
    for lab in labels:
        if "GPU" in lab:    colors.append("tab:red" if "LSH" in lab else "tab:orange")
        else:               colors.append("tab:cyan" if "LSH" in lab else "tab:blue")

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))

    ax_t.bar(x, times, color=colors)
    ax_t.set_yscale("log")
    ax_t.set_ylabel("query wall time (ms)")
    ax_t.set_title("CPU vs GPU  —  query wall time (log)")
    ax_t.set_xticks(x); ax_t.set_xticklabels(labels, rotation=25, ha="right")
    for i, (t, r) in enumerate(zip(times, recalls)):
        ax_t.text(i, t, f"{t:.1f}ms\nrec={r:.2f}", ha="center", va="bottom", fontsize=9)

    ax_s.bar(x, speedups, color=colors)
    ax_s.set_yscale("log")
    ax_s.set_ylabel(f"speedup vs {labels[0]}")
    ax_s.set_title("Speedup vs CPU brute P=1 (log)")
    ax_s.set_xticks(x); ax_s.set_xticklabels(labels, rotation=25, ha="right")
    for i, s in enumerate(speedups):
        ax_s.text(i, s, f"x{s:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"LSH on GPU (CUDA) vs CPU (OpenMP)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
