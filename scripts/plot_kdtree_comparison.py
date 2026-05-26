"""Plot KD-tree against brute force, LSH, and HNSW benchmark summaries."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("brute", "Brute force", "tab:blue"),
    ("lsh", "LSH", "tab:cyan"),
    ("hnsw", "HNSW", "tab:orange"),
    ("kd_tree", "KD-tree", "tab:green"),
]


def format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.1f}ms"


def add_log_headroom(ax: plt.Axes, values: list[float], factor: float = 3.0) -> None:
    ax.set_ylim(min(values) / 1.8, max(values) * factor)


def read_summary(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_dataset: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            by_dataset[row["dataset"]][row["method"]] = row
    return dict(by_dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/kdtree_lsh_hnsw_summary.csv")
    parser.add_argument("--out", default="results/kdtree_vs_lsh_hnsw_visualization.png")
    args = parser.parse_args()

    data = read_summary(Path(args.csv))
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    all_times: list[float] = []
    for offset, (method, label, color) in zip(offsets, METHODS):
        times = [float(data[ds][method]["search_ms"]) for ds in datasets]
        recalls = [float(data[ds][method]["recall"]) for ds in datasets]
        all_times.extend(times)
        bars = ax.bar(x + offset, times, width, label=label, color=color)
        for bar, t, rec in zip(bars, times, recalls):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                t * 1.18,
                f"{format_ms(t)}\nrec={rec:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylabel("query wall time (ms)")
    ax.set_title("CPU Query Time, 8 Threads")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_times)

    ax = axes[1]
    all_speedups: list[float] = []
    for offset, (method, label, color) in zip(offsets[1:], METHODS[1:]):
        speedups = []
        for ds in datasets:
            brute = float(data[ds]["brute"]["search_ms"])
            query = float(data[ds][method]["search_ms"])
            speedups.append(brute / query)
        all_speedups.extend(speedups)
        bars = ax.bar(x + offset, speedups, width, label=label, color=color)
        for bar, speedup in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                speedup * 1.18,
                f"x{speedup:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs brute force")
    ax.set_title("Speedup vs Brute Force")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_speedups + [1.0], factor=4.0)

    ax = axes[2]
    build_methods = METHODS[1:]
    build_offsets = np.linspace(-width, width, len(build_methods))
    all_builds: list[float] = []
    for offset, (method, label, color) in zip(build_offsets, build_methods):
        xs = []
        builds = []
        for idx, ds in enumerate(datasets):
            value = data[ds][method]["build_ms"]
            if value == "-":
                continue
            xs.append(x[idx] + offset)
            builds.append(float(value))
        if not builds:
            continue
        all_builds.extend(builds)
        bars = ax.bar(xs, builds, width, color=color, label=label)
        for bar, build in zip(bars, builds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                build * 1.18,
                format_ms(build),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_ylabel("build time (ms)")
    ax.set_title("Index Build Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    if all_builds:
        add_log_headroom(ax, all_builds)

    fig.suptitle("KD-tree vs LSH and HNSW", fontsize=14)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
