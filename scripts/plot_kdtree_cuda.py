"""Plot CUDA KD-tree benchmark results.

Input CSV produced by scripts/bench_kdtree_cuda.sh (summary CSV).

Usage:
    python scripts/plot_kdtree_cuda.py \
        --csv results/kdtree_cuda_summary.csv \
        [--out results/kdtree_cuda.png]

Panels (2×2):
    [0,0]  Query time vs dimensionality — GPU brute, GPU KD, CPU 16T
    [0,1]  GPU KD speedup vs GPU brute across dims (curse of dimensionality)
    [1,0]  GPU KD-tree vs best CPU KD-tree speedup across dims
    [1,1]  GPU KD-tree recall@K across dims
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="summary CSV from bench_kdtree_cuda.sh")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    rows = read_csv(args.csv)

    # ---- group rows ----
    # keyed by (method, dim)
    by_method_dim: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["method"], int(r["dim"]))
        by_method_dim[key] = r

    dims = sorted({int(r["dim"]) for r in rows})

    def get_kd_ms(d: int) -> float | None:
        r = by_method_dim.get(("gpu_kd", d))
        return float(r["kd_ms"]) if r and r["kd_ms"] not in ("-", "") else None

    def get_brute_ms(d: int) -> float | None:
        r = by_method_dim.get(("gpu_brute", d))
        return float(r["brute_ms"]) if r and r["brute_ms"] not in ("-", "") else None

    def get_recall(d: int) -> float | None:
        r = by_method_dim.get(("gpu_kd", d))
        return float(r["recall"]) if r and r["recall"] not in ("-", "") else None

    def get_cpu_ms(p: int, d: int) -> float | None:
        r = by_method_dim.get((f"cpu_serial_P{p}", d))
        return float(r["kd_ms"]) if r and r["kd_ms"] not in ("-", "") else None

    # pick best CPU thread count that's actually present
    cpu_threads = sorted({
        int(k[0].replace("cpu_serial_P", ""))
        for k in by_method_dim
        if k[0].startswith("cpu_serial_P")
    })
    best_cpu_p = max(cpu_threads) if cpu_threads else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_time   = axes[0][0]
    ax_gpusp  = axes[0][1]
    ax_cpusp  = axes[1][0]
    ax_recall = axes[1][1]

    x = np.arange(len(dims))
    width = 0.25

    # ---- [0,0] query time bar chart ----
    gpu_brute_ms = [get_brute_ms(d) or 0 for d in dims]
    gpu_kd_ms    = [get_kd_ms(d)    or 0 for d in dims]
    cpu_best_ms  = [get_cpu_ms(best_cpu_p, d) or 0 for d in dims] if best_cpu_p else []

    ax_time.bar(x - width, gpu_brute_ms, width, label="GPU brute",          color="steelblue")
    ax_time.bar(x,         gpu_kd_ms,    width, label="GPU KD-tree",        color="darkorange")
    if cpu_best_ms:
        ax_time.bar(x + width, cpu_best_ms, width,
                    label=f"CPU KD-tree ({best_cpu_p}T)", color="seagreen")
    ax_time.set_xticks(x); ax_time.set_xticklabels([str(d) for d in dims])
    ax_time.set_xlabel("dimensions"); ax_time.set_ylabel("query time (ms)")
    ax_time.set_yscale("log"); ax_time.grid(True, which="both", alpha=0.3, axis="y")
    ax_time.legend(); ax_time.set_title("Query time vs dimensionality")

    # ---- [0,1] GPU KD speedup vs GPU brute ----
    gpu_speedups = []
    for d in dims:
        b = get_brute_ms(d); k = get_kd_ms(d)
        gpu_speedups.append(b / k if b and k else None)

    valid = [(d, s) for d, s in zip(dims, gpu_speedups) if s is not None]
    if valid:
        ds_v, sp_v = zip(*valid)
        ax_gpusp.plot(ds_v, sp_v, "o-", color="darkorange", label="GPU KD-tree")
        ax_gpusp.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="break-even")
    ax_gpusp.set_xlabel("dimensions"); ax_gpusp.set_ylabel("speedup vs GPU brute")
    ax_gpusp.set_xticks(dims); ax_gpusp.grid(True, alpha=0.3); ax_gpusp.legend()
    ax_gpusp.set_title("GPU KD-tree speedup vs GPU brute\n(curse of dimensionality)")

    # ---- [1,0] GPU KD vs best CPU KD speedup ----
    if best_cpu_p:
        gpu_vs_cpu = []
        for d in dims:
            cpu = get_cpu_ms(best_cpu_p, d); gpu = get_kd_ms(d)
            gpu_vs_cpu.append(cpu / gpu if cpu and gpu else None)
        valid2 = [(d, s) for d, s in zip(dims, gpu_vs_cpu) if s is not None]
        if valid2:
            ds2, sp2 = zip(*valid2)
            ax_cpusp.plot(ds2, sp2, "s-", color="purple",
                          label=f"GPU KD vs CPU KD ({best_cpu_p}T)")
            ax_cpusp.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="break-even")
        # also add lower thread counts for comparison
        for p in cpu_threads:
            if p == best_cpu_p:
                continue
            vals = []
            for d in dims:
                cpu = get_cpu_ms(p, d); gpu = get_kd_ms(d)
                vals.append(cpu / gpu if cpu and gpu else None)
            valid3 = [(d, s) for d, s in zip(dims, vals) if s is not None]
            if valid3:
                ds3, sp3 = zip(*valid3)
                ax_cpusp.plot(ds3, sp3, "--", alpha=0.6, label=f"GPU KD vs CPU KD ({p}T)")
    ax_cpusp.set_xlabel("dimensions"); ax_cpusp.set_ylabel("speedup (CPU / GPU time)")
    ax_cpusp.set_xticks(dims); ax_cpusp.grid(True, alpha=0.3); ax_cpusp.legend()
    ax_cpusp.set_title(f"GPU KD-tree vs CPU KD-tree")

    # ---- [1,1] recall@K ----
    recalls = [get_recall(d) for d in dims]
    valid4 = [(d, r) for d, r in zip(dims, recalls) if r is not None]
    if valid4:
        ds4, rc4 = zip(*valid4)
        ax_recall.plot(ds4, rc4, "D-", color="crimson", label="recall@K")
        ax_recall.set_ylim(0, 1.05)
    ax_recall.set_xlabel("dimensions"); ax_recall.set_ylabel("recall@K")
    ax_recall.set_xticks(dims); ax_recall.grid(True, alpha=0.3); ax_recall.legend()
    ax_recall.set_title("GPU KD-tree recall@K vs dimensionality")

    # ---- metadata from first row ----
    n_val = rows[0]["n"] if rows else "?"
    k_val = next(
        (str(int(float(r["kd_ms"]) > 0)) for r in rows if r["method"] == "gpu_kd"), "?"
    )
    fig.suptitle(f"CUDA KD-tree  N={n_val}", fontsize=13)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path(args.csv).with_name("kdtree_cuda.png")
    fig.savefig(out_path, dpi=140)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
