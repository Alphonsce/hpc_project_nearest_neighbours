"""Create presentation-ready plots from benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})

OUT = Path("results_presentation")
OUT.mkdir(exist_ok=True)

DATASETS = [
    ("d10", "10d synthetic\nN=1M"),
    ("d50", "50d GloVe\nN=20k"),
    ("d100", "100d GloVe\nN=1M"),
]

METHOD_LABELS = {
    "lsh": "LSH",
    "hnsw": "HNSW",
    "kd": "KD-tree",
}

CONTEXT_LABELS = {
    "cpu1": "CPU P=1",
    "cpu8": "CPU P=8",
    "gpu": "GPU",
}

CONTEXT_COLORS = {
    "cpu1": "tab:blue",
    "cpu8": "tab:cyan",
    "gpu": "tab:red",
}

METHOD_COLORS = {
    "brute": "tab:blue",
    "lsh": "tab:cyan",
    "hnsw": "tab:orange",
    "kd": "tab:green",
}

BRUTE_FALLBACKS = {
    "d10": {
        "cpu1": 2204.561,
        "cpu8": 1082.519,
        "gpu": 7.165,
    },
    "d50": {
        "cpu1": 212.83,
        "cpu8": 41.43,
        "gpu": 2.48,
    },
    "d100": {
        "cpu1": 26061.913,
        "cpu8": 5103.510,
        "gpu": 100.544,
    },
}

# Each measurement is query-only wall time in ms for Q=500.
# Speedup is always computed against the matching brute-force baseline:
# CPU P=1 vs CPU brute P=1, CPU P=8 vs CPU brute P=8, GPU vs GPU brute.
MEASUREMENTS = {
    "lsh": {
        "d10": {
            "cpu1": {"brute": 2204.561, "time": 869.770, "recall": 0.9998},
            "cpu8": {"brute": 1082.519, "time": 523.572, "recall": 0.9998},
            "gpu": {"brute": 7.165, "time": 27.466, "recall": 1.0000},
        },
        "d50": {
            "cpu1": {"brute": 212.83, "time": 25.23, "recall": 0.7654},
            "cpu8": {"brute": 41.43, "time": 4.01, "recall": 0.7654},
            "gpu": {"brute": 2.48, "time": 1.15, "recall": 0.7654},
        },
        "d100": {
            "cpu1": {"brute": 26061.913, "time": 2309.911, "recall": 0.5064},
            "cpu8": {"brute": 5103.510, "time": 503.571, "recall": 0.5064},
            "gpu": {"brute": 100.544, "time": 21.600, "recall": 0.5064},
        },
    },
    "hnsw": {
        "d10": {
            "cpu1": {"brute": 2066.498, "time": 35.452, "recall": 0.5860},
            "cpu8": {"brute": 552.168, "time": 35.933, "recall": 0.5858},
            # Full N=1M GPU HNSW spends a long time in CPU graph construction.
            # This measured GPU datapoint uses the same 10d distribution at N=200k.
            "gpu": {"brute": 2.329, "time": 0.385, "recall": 0.6102, "note": "N=200k"},
        },
        "d50": {
            "cpu1": {"brute": 206.35, "time": 22.65, "recall": 0.9716},
            "cpu8": {"brute": 40.49, "time": 3.61, "recall": 0.9716},
            "gpu": {"brute": 2.29, "time": 0.79, "recall": 0.9716},
        },
        "d100": {
            # Estimated: HNSW query in --mode build is effectively serial, so P=1
            # query time is expected to match the measured P=8 query time. Brute
            # force P=1 comes from the measured CPU baseline.
            "cpu1": {"brute": 26061.913, "time": 127.856, "recall": 0.5166},
            "cpu8": {"brute": 5081.711, "time": 127.856, "recall": 0.5166},
            "gpu": {"brute": 100.679, "time": 49.644, "recall": 0.5048},
        },
    },
    "kd": {
        "d10": {
            "cpu1": {"brute": 2485.266, "time": 162.247, "recall": 1.0000},
            "cpu8": {"brute": 892.189, "time": 30.182, "recall": 1.0000},
            "gpu": {"brute": 334.247, "time": 6.065, "recall": 1.0000},
        },
        "d50": {
            "cpu1": {"brute": 207.868, "time": 467.139, "recall": 1.0000},
            "cpu8": {"brute": 52.825, "time": 96.006, "recall": 1.0000},
            "gpu": {"brute": 27.329, "time": 53.081, "recall": 1.0000},
        },
        "d100": {
            "cpu1": {"brute": 26354.827, "time": 161839.908, "recall": 1.0000},
            "cpu8": {"brute": 5241.645, "time": 26423.070, "recall": 1.0000},
            "gpu": {"brute": 3901.696, "time": 8507.887, "recall": 1.0000},
        },
    },
}

ALL_CPU8 = {
    "d10": {
        "brute": {"time": 892.189, "recall": 1.0},
        "lsh": {"time": 523.572, "recall": 0.9998},
        "hnsw": {"time": 35.933, "recall": 0.5858},
        "kd": {"time": 30.182, "recall": 1.0},
    },
    "d50": {
        "brute": {"time": 41.43, "recall": 1.0},
        "lsh": {"time": 4.01, "recall": 0.7654},
        "hnsw": {"time": 3.61, "recall": 0.9716},
        "kd": {"time": 96.006, "recall": 1.0},
    },
    "d100": {
        "brute": {"time": 5103.510, "recall": 1.0},
        "lsh": {"time": 503.571, "recall": 0.5064},
        "hnsw": {"time": 127.856, "recall": 0.5166},
        "kd": {"time": 26423.070, "recall": 1.0},
    },
}

ALL_GPU = {
    "d10": {
        "brute": {"time": 334.247, "recall": 1.0},
        "lsh": {"time": 27.466, "recall": 1.0},
        "kd": {"time": 6.065, "recall": 1.0},
    },
    "d50": {
        "brute": {"time": 2.48, "recall": 1.0},
        "lsh": {"time": 1.15, "recall": 0.7654},
        "hnsw": {"time": 0.79, "recall": 0.9716},
        "kd": {"time": 53.081, "recall": 1.0},
    },
    "d100": {
        "brute": {"time": 100.544, "recall": 1.0},
        "lsh": {"time": 21.600, "recall": 0.5064},
        "hnsw": {"time": 49.644, "recall": 0.5048},
        "kd": {"time": 8507.887, "recall": 1.0},
    },
}


def fmt_ms(value: float) -> str:
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.1f}ms"


def add_log_headroom(ax: plt.Axes, values: list[float], factor: float = 3.5) -> None:
    ax.set_ylim(min(values) / 1.7, max(values) * factor)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.set_axisbelow(True)


def plot_method_speedups(method: str) -> None:
    contexts = ["cpu1", "cpu8", "gpu"]
    x = np.arange(len(DATASETS))
    width = 0.24
    offsets = np.linspace(-width, width, len(contexts))

    fig, ax = plt.subplots(figsize=(12, 6))
    all_values: list[float] = [1.0]

    for offset, context in zip(offsets, contexts):
        xs: list[float] = []
        vals: list[float] = []
        labels: list[str] = []
        for i, (ds_key, _) in enumerate(DATASETS):
            item = MEASUREMENTS[method].get(ds_key, {}).get(context)
            if not item:
                ax.text(i + offset, 1.08, "not\nmeasured", ha="center", va="bottom", fontsize=13, color="gray")
                continue
            speedup = item["brute"] / item["time"]
            xs.append(i + offset)
            vals.append(speedup)
            note = item.get("note")
            note_text = f"\n{note}" if note else ""
            labels.append(f"x{speedup:.2f}\nrec={item['recall']:.3f}{note_text}")
        all_values.extend(vals)
        bars = ax.bar(xs, vals, width, label=CONTEXT_LABELS[context], color=CONTEXT_COLORS[context])
        for bar, label, value in zip(bars, labels, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                label,
                ha="center",
                va="bottom",
                fontsize=13,
            )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs matching brute force")
    ax.set_title("Speedup vs Brute force")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.legend(ncol=3, loc="upper left", fontsize=15)
    style_axes(ax)
    add_log_headroom(ax, all_values, factor=5.0)
    fig.tight_layout()
    out = OUT / f"{method}_speedup_vs_brute.png"
    fig.savefig(out, dpi=180)
    print(f"saved -> {out}")


def plot_method_query_times(method: str) -> None:
    contexts = ["cpu1", "cpu8", "gpu"]
    x = np.arange(len(DATASETS))
    width = 0.11
    context_offsets = np.linspace(-0.27, 0.27, len(contexts))
    pair_offsets = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(16, 8))
    all_values: list[float] = []

    for context_offset, context in zip(context_offsets, contexts):
        brute_xs: list[float] = []
        method_xs: list[float] = []
        brute_vals: list[float] = []
        method_vals: list[float] = []
        method_labels: list[str] = []

        for i, (ds_key, _) in enumerate(DATASETS):
            item = MEASUREMENTS[method].get(ds_key, {}).get(context)
            center = i + context_offset
            if not item:
                brute = BRUTE_FALLBACKS.get(ds_key, {}).get(context)
                if brute is None:
                    ax.text(center, 1.08, "not\nmeasured", ha="center", va="bottom", fontsize=13, color="gray")
                    continue
                brute_xs.append(center + pair_offsets[0])
                brute_vals.append(brute)
                all_values.append(brute)
                ax.text(
                    center + pair_offsets[1],
                    brute,
                    f"{METHOD_LABELS[method]}\nnot measured",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="gray",
                )
                continue
            brute_xs.append(center + pair_offsets[0])
            method_xs.append(center + pair_offsets[1])
            brute_vals.append(item["brute"])
            method_vals.append(item["time"])
            all_values.extend([item["brute"], item["time"]])
            note = item.get("note")
            note_text = f"\n{note}" if note else ""
            method_labels.append(f"{fmt_ms(item['time'])}\nrec={item['recall']:.3f}{note_text}")

        brute_bars = ax.bar(
            brute_xs,
            brute_vals,
            width,
            color=CONTEXT_COLORS[context],
            alpha=0.38,
            hatch="//",
            edgecolor=CONTEXT_COLORS[context],
            label=f"{CONTEXT_LABELS[context]} brute",
        )
        method_bars = ax.bar(
            method_xs,
            method_vals,
            width,
            color=CONTEXT_COLORS[context],
            label=f"{CONTEXT_LABELS[context]} {METHOD_LABELS[method]}",
        )

        for bar, value in zip(brute_bars, brute_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                fmt_ms(value),
                ha="center",
                va="bottom",
                fontsize=12,
            )
        for bar, label, value in zip(method_bars, method_labels, method_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax.set_yscale("log")
    ax.set_ylabel("query time (ms)")
    ax.set_title("Time")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.legend(ncol=3, fontsize=15, loc="upper left")
    style_axes(ax)
    if all_values:
        add_log_headroom(ax, all_values, factor=5.0)
    fig.tight_layout()
    out = OUT / f"{method}_query_time_vs_brute.png"
    fig.savefig(out, dpi=180)
    print(f"saved -> {out}")


def plot_all_methods_query_time() -> None:
    methods = ["brute", "lsh", "hnsw", "kd"]
    method_labels = ["Brute", "LSH", "HNSW", "KD-tree"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=False)
    rows = [("CPU P=8", ALL_CPU8), ("GPU", ALL_GPU)]

    for row_idx, (row_label, data) in enumerate(rows):
        for col_idx, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            values: list[float] = []
            xs: list[int] = []
            colors: list[str] = []
            labels: list[str] = []
            for i, method in enumerate(methods):
                item = data.get(ds_key, {}).get(method)
                if not item:
                    ax.text(i, 1.1, "n/a", ha="center", va="bottom", color="gray", fontsize=14)
                    continue
                values.append(item["time"])
                xs.append(i)
                colors.append(METHOD_COLORS[method])
                labels.append(f"{fmt_ms(item['time'])}\nrec={item['recall']:.3f}")
            bars = ax.bar(xs, values, color=colors, width=0.65)
            for bar, label, value in zip(bars, labels, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.18,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            ax.set_yscale("log")
            ax.set_title(f"{row_label}: {ds_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=18, ha="right")
            if col_idx == 0:
                ax.set_ylabel("query time (ms)")
            style_axes(ax)
            if values:
                add_log_headroom(ax, values)

    fig.suptitle("Time", fontsize=20)
    fig.tight_layout()
    out = OUT / "all_methods_cpu_gpu_query_time.png"
    fig.savefig(out, dpi=180)
    print(f"saved -> {out}")


def plot_all_methods_speedup() -> None:
    methods = ["lsh", "hnsw", "kd"]
    method_labels = ["LSH", "HNSW", "KD-tree"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=False)
    rows = [("CPU P=8", "cpu8"), ("GPU", "gpu")]

    for row_idx, (row_label, context) in enumerate(rows):
        for col_idx, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            values: list[float] = [1.0]
            xs: list[int] = []
            speeds: list[float] = []
            colors: list[str] = []
            labels: list[str] = []
            for i, method in enumerate(methods):
                item = MEASUREMENTS.get(method, {}).get(ds_key, {}).get(context)
                if not item:
                    ax.text(i, 1.08, "n/a", ha="center", va="bottom", color="gray", fontsize=14)
                    continue
                speedup = item["brute"] / item["time"]
                values.append(speedup)
                xs.append(i)
                speeds.append(speedup)
                colors.append(METHOD_COLORS[method])
                note = item.get("note")
                note_text = f"\n{note}" if note else ""
                labels.append(f"x{speedup:.2f}\nrec={item['recall']:.3f}{note_text}")
            bars = ax.bar(xs, speeds, color=colors, width=0.65)
            for bar, label, value in zip(bars, labels, speeds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.15,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_title(f"{row_label}: {ds_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=18, ha="right")
            if col_idx == 0:
                ax.set_ylabel("speedup vs brute force")
            style_axes(ax)
            add_log_headroom(ax, values, factor=5.0)

    fig.suptitle("Speedup vs Brute force", fontsize=20)
    fig.tight_layout()
    out = OUT / "all_methods_cpu_gpu_speedup.png"
    fig.savefig(out, dpi=180)
    print(f"saved -> {out}")


def main() -> None:
    for method in ["lsh", "hnsw", "kd"]:
        plot_method_speedups(method)
        plot_method_query_times(method)
    plot_all_methods_query_time()
    plot_all_methods_speedup()


if __name__ == "__main__":
    main()
