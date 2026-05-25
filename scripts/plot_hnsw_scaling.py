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

    # rows_by_mode[mode] = [(threads, speedup_hnsw, hnsw_ms, recall), ...]
    rows_by_mode: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows_by_mode[r["mode"]].append((
                int(r["threads"]),
                float(r["speedup_hnsw"]),
                float(r["hnsw_ms"]),
                float(r["recall"]),
            ))
    for v in rows_by_mode.values():
        v.sort()

    all_ps = sorted({p for v in rows_by_mode.values() for p, *_ in v})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_s, ax_t, ax_r = axes

    # ---- left: speedup curves ----
    ax_s.plot(all_ps, all_ps, "k--", alpha=0.5, label="ideal")
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]
        ss = [r[1] for r in rows]
        ax_s.plot(ps, ss, "o-", label=m)
    ax_s.set_xlabel("OpenMP threads")
    ax_s.set_ylabel("speedup  T(1) / T(P)")
    ax_s.set_xticks(all_ps)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=11)
    ax_s.set_title("HNSW strong scaling — speedup by mode")

    # ---- middle: wall-time curves (log scale) ----
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]
        ts = [r[2] for r in rows]
        ax_t.plot(ps, ts, "o-", label=m)
    ax_t.set_xlabel("OpenMP threads")
    ax_t.set_ylabel("HNSW wall time (ms)")
    ax_t.set_xticks(all_ps)
    ax_t.set_yscale("log")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=11)
    ax_t.set_title("HNSW wall time by mode (log)")

    # ---- right: recall@k per mode (should stay roughly constant) ----
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]
        rs = [r[3] for r in rows]
        ax_r.plot(ps, rs, "o-", label=m)
    ax_r.set_xlabel("OpenMP threads")
    ax_r.set_ylabel("recall@k")
    ax_r.set_xticks(all_ps)
    ax_r.set_ylim(0, 1.05)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=11)
    ax_r.set_title("Recall@k by mode (should be stable)")

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
