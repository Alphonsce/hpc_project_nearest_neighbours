"""Plot strong-scaling curves for HNSW modes.
Mirrors scripts/plot_scaling.py for LSH.

Input CSV columns: mode,threads,brute_ms,hnsw_ms,speedup_brute,speedup_hnsw,recall

Usage:
    python scripts/plot_hnsw_scaling.py --csv results/hnsw_scaling_M16_ef50.csv
    python scripts/plot_hnsw_scaling.py --csv results/hnsw_scaling_M16_ef50.csv --out my.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # rows_by_mode[mode] = [(threads, brute_ms, hnsw_ms), ...]
    rows_by_mode: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows_by_mode[r["mode"]].append((
                int(r["threads"]),
                float(r["brute_ms"]),
                float(r["hnsw_ms"]),
            ))
    for v in rows_by_mode.values():
        v.sort()

    all_ps = sorted({p for v in rows_by_mode.values() for p, *_ in v})

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(14, 5))

    # Brute-force is identical for all modes up to timing noise; use first mode.
    first_rows = next(iter(rows_by_mode.values()))
    brute_ps = [r[0] for r in first_rows]
    brute_ms = [r[1] for r in first_rows]
    brute_base = brute_ms[0]
    brute_speedups = [brute_base / t for t in brute_ms]

    # ---- left: speedup curves ----
    ax_s.plot(all_ps, all_ps, "k--", alpha=0.5, label="ideal")
    ax_s.plot(brute_ps, brute_speedups, "o-", label="brute force")
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]
        ts = [r[2] for r in rows]
        base = ts[0]
        ax_s.plot(ps, [base / t for t in ts], "o-", label=f"HNSW: {m}")
    ax_s.set_xlabel("OpenMP threads")
    ax_s.set_ylabel("speedup  T(1) / T(P)")
    ax_s.set_xticks(all_ps)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=11)
    ax_s.set_title("Speedup")

    # ---- right: wall-time curves (log scale) ----
    ax_t.plot(brute_ps, brute_ms, "o-", label="brute force")
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]
        ts = [r[2] for r in rows]
        ax_t.plot(ps, ts, "o-", label=f"HNSW: {m}")
    ax_t.set_xlabel("OpenMP threads")
    ax_t.set_ylabel("wall time (ms)")
    ax_t.set_xticks(all_ps)
    ax_t.set_yscale("log")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=11)
    ax_t.set_title("Wall time")

    fig.suptitle("HNSW Parallelism Methods (M=16, ef=50)", fontsize=14)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
