"""Create HNSW-only and HNSW-vs-LSH summary visualizations.

Inputs are the compact summary CSVs written after benchmark runs:
    results/hnsw_lsh_small_N20000_summary.csv
    results/hnsw_lsh_1M_100d_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dataset_label(notes: str) -> str:
    n_match = re.search(r"N=(\d+)", notes)
    d_match = re.search(r"DIM=(\d+)", notes)
    n_label = "N=?"
    if n_match:
        n = int(n_match.group(1))
        n_label = f"N={n // 1000}k" if n < 1_000_000 else f"N={n // 1_000_000}M"
    d_label = f"D={d_match.group(1)}" if d_match else "D=?"
    return f"{n_label}, {d_label}"


def load_summary(path: Path) -> tuple[str, dict[tuple[str, str], dict[str, str]]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    dataset_label = path.stem
    with path.open() as fh:
        for row in csv.DictReader(fh):
            method = row["method"]
            threads = row["threads"]
            rows[(method, threads)] = row
            if method in {"cpu_lsh", "cpu_hnsw"}:
                dataset_label = parse_dataset_label(row["notes"])
    return dataset_label, rows


def as_float(value: str) -> float | None:
    return None if value == "-" else float(value)


def format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.1f}ms"


def add_log_headroom(ax: plt.Axes, values: list[float], factor: float = 3.0) -> None:
    bottom = min(values) / 1.8
    top = max(values) * factor
    ax.set_ylim(bottom, top)


def plot_hnsw_only(data: list[tuple[str, dict[tuple[str, str], dict[str, str]]]], out: Path, ef_csv: Path | None) -> None:
    labels = [label for label, _ in data]
    cpu_times = [as_float(rows[("cpu_hnsw", "8")]["search_ms"]) for _, rows in data]
    gpu_times = [as_float(rows[("gpu_hnsw", "-")]["search_ms"]) for _, rows in data]
    recalls = [float(rows[("gpu_hnsw", "-")]["recall"]) for _, rows in data]
    gpu_build = [as_float(rows[("gpu_hnsw", "-")]["build_ms"]) for _, rows in data]

    fig, axes = plt.subplots(1, 3 if ef_csv and ef_csv.exists() else 2, figsize=(17, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    x = np.arange(len(labels))
    width = 0.34

    ax = axes[0]
    ax.bar(x - width / 2, cpu_times, width, label="CPU HNSW, 8 threads", color="tab:blue")
    ax.bar(x + width / 2, gpu_times, width, label="GPU HNSW", color="tab:orange")
    ax.set_yscale("log")
    ax.set_ylabel("query wall time (ms)")
    ax.set_title("HNSW Query Time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    for idx, (cpu, gpu, rec) in enumerate(zip(cpu_times, gpu_times, recalls)):
        ax.text(idx - width / 2, cpu * 1.18, format_ms(cpu), ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, gpu * 1.18, f"{format_ms(gpu)}\nrec={rec:.3f}", ha="center", va="bottom", fontsize=8)
    add_log_headroom(ax, [v for v in cpu_times + gpu_times if v is not None])

    ax = axes[1]
    ax.bar(x, gpu_build, width=0.45, color="tab:red")
    ax.set_yscale("log")
    ax.set_ylabel("index build time (ms)")
    ax.set_title("HNSW GPU/Index Build Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", which="both", alpha=0.25)
    for idx, build in enumerate(gpu_build):
        ax.text(idx, build * 1.18, format_ms(build), ha="center", va="bottom", fontsize=8)
    add_log_headroom(ax, [v for v in gpu_build if v is not None])

    if len(axes) == 3:
        ef_rows = []
        with ef_csv.open() as fh:
            for row in csv.DictReader(fh):
                ef_rows.append((int(row["ef"]), float(row["hnsw_ms"]), float(row["recall"])))
        ef_rows.sort()
        ax = axes[2]
        ax.plot([r[1] for r in ef_rows], [r[2] for r in ef_rows], "o-", color="tab:green")
        for ef, t, rec in ef_rows:
            ax.annotate(f"ef={ef}", (t, rec), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("query wall time (ms)")
        ax.set_ylabel("recall@10")
        ax.set_title("HNSW Recall-Speed Sweep, N=20k")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("HNSW Performance Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


def plot_hnsw_vs_lsh(data: list[tuple[str, dict[tuple[str, str], dict[str, str]]]], out: Path) -> None:
    labels = [label for label, _ in data]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    algos = [
        ("brute", "Brute force", "tab:blue"),
        ("lsh", "LSH", "tab:cyan"),
        ("hnsw", "HNSW", "tab:orange"),
    ]
    x = np.arange(len(labels))
    width = 0.25
    offsets = np.linspace(-width, width, len(algos))

    for ax, runtime, title in [
        (axes[0], "cpu", "CPU Query Time, 8 Threads"),
        (axes[1], "gpu", "GPU Query Time"),
    ]:
        all_times = []
        for offset, (algo, display, color) in zip(offsets, algos):
            times = []
            recalls = []
            for _, rows in data:
                key = (f"{runtime}_{algo}", "8" if runtime == "cpu" else "-")
                times.append(float(rows[key]["search_ms"]))
                recalls.append(float(rows[key]["recall"]))
            all_times.extend(times)
            bars = ax.bar(x + offset, times, width, label=display, color=color)
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
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", which="both", alpha=0.25)
        ax.legend()
        add_log_headroom(ax, all_times)

    ax = axes[2]
    build_algos = [("lsh", "LSH", "tab:cyan"), ("hnsw", "HNSW", "tab:orange")]
    build_offsets = [-width / 2, width / 2]
    all_builds = []
    for offset, (algo, display, color) in zip(build_offsets, build_algos):
        builds = [float(rows[(f"gpu_{algo}", "-")]["build_ms"]) for _, rows in data]
        all_builds.extend(builds)
        bars = ax.bar(x + offset, builds, width, label=display, color=color)
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
    ax.set_title("GPU/Index Build Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend()
    add_log_headroom(ax, all_builds)

    fig.suptitle("HNSW vs LSH Benchmark Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries",
        nargs="+",
        default=[
            "results/hnsw_lsh_small_N20000_summary.csv",
            "results/hnsw_lsh_1M_100d_summary.csv",
        ],
    )
    parser.add_argument("--ef-csv", default="results/hnsw_ef_sweep_M16_P8.csv")
    parser.add_argument("--hnsw-out", default="results/hnsw_separate_visualization.png")
    parser.add_argument("--vs-out", default="results/hnsw_vs_lsh_visualization.png")
    args = parser.parse_args()

    summaries = [load_summary(Path(path)) for path in args.summaries]
    plot_hnsw_only(summaries, Path(args.hnsw_out), Path(args.ef_csv))
    plot_hnsw_vs_lsh(summaries, Path(args.vs_out))


if __name__ == "__main__":
    main()
