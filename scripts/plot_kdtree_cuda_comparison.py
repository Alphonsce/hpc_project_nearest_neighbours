"""Plot CUDA KD-tree against GPU brute force and CPU KD-tree."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("gpu_brute", "GPU brute", "tab:blue"),
    ("gpu_kd_tree", "GPU KD-tree", "tab:orange"),
    ("cpu_kd_tree", "CPU KD-tree, 8T", "tab:green"),
]


def fmt_ms(value: float) -> str:
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.1f}ms"


def read_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_dataset: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            by_dataset[row["dataset"]][row["method"]] = row
    return dict(by_dataset)


def add_log_headroom(ax: plt.Axes, values: list[float], factor: float = 3.0) -> None:
    ax.set_ylim(min(values) / 1.8, max(values) * factor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/kdtree_cuda_comparison_summary.csv")
    parser.add_argument("--out", default="results/kdtree_cuda_comparison.png")
    args = parser.parse_args()

    data = read_rows(Path(args.csv))
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.24
    offsets = np.linspace(-width, width, len(METHODS))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    all_times: list[float] = []
    for offset, (method, label, color) in zip(offsets, METHODS):
        values = [float(data[ds][method]["query_ms"]) for ds in datasets]
        all_times.extend(values)
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.18,
                fmt_ms(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylabel("query time (ms)")
    ax.set_title("KD-tree Query Time")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_times)

    ax = axes[1]
    speedup_methods = METHODS[1:]
    speedup_offsets = [-width / 2, width / 2]
    all_speedups: list[float] = []
    for offset, (method, label, color) in zip(speedup_offsets, speedup_methods):
        values = []
        for ds in datasets:
            brute = float(data[ds]["gpu_brute"]["query_ms"])
            query = float(data[ds][method]["query_ms"])
            values.append(brute / query)
        all_speedups.extend(values)
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.18,
                f"x{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs GPU brute")
    ax.set_title("Speedup vs GPU Brute")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_speedups + [1.0], factor=4.0)

    ax = axes[2]
    build_methods = METHODS[1:]
    all_builds: list[float] = []
    for offset, (method, label, color) in zip(speedup_offsets, build_methods):
        values = [float(data[ds][method]["build_ms"]) for ds in datasets]
        all_builds.extend(values)
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.18,
                fmt_ms(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylabel("build time (ms)")
    ax.set_title("KD-tree Build Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_builds)

    fig.suptitle("CUDA KD-tree Measurements", fontsize=14)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
