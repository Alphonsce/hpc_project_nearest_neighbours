"""Plot the HNSW recall-speed Pareto curve over ef values.

Two panels:
    left:  recall@k vs ef parameter (quality vs beam width)
    right: recall@k vs hnsw_ms (quality vs query speed) — the Pareto frontier

Input CSV columns: mode,ef,threads,hnsw_ms,recall

Usage:
    python scripts/plot_hnsw_ef.py --csv results/hnsw_ef_sweep_M16_P4.csv
    python scripts/plot_hnsw_ef.py --csv results/hnsw_ef_sweep_M16_P4.csv --out my.png
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

    # rows_by_mode[mode] = [(ef, hnsw_ms, recall), ...]
    rows_by_mode: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows_by_mode[r["mode"]].append((
                int(r["ef"]),
                float(r["hnsw_ms"]),
                float(r["recall"]),
            ))
    for v in rows_by_mode.values():
        v.sort()  # sort by ef ascending

    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- left: recall vs ef ----
    for m, rows in rows_by_mode.items():
        ef_vals  = [r[0] for r in rows]
        recalls  = [r[2] for r in rows]
        ax_q.plot(ef_vals, recalls, "o-", label=m)
        for ef, rec in zip(ef_vals, recalls):
            ax_q.annotate(f"{ef}", (ef, rec),
                          textcoords="offset points", xytext=(0, 5),
                          ha="center", fontsize=7, color="gray")
    ax_q.set_xlabel("ef  (query beam width)")
    ax_q.set_ylabel("recall@k")
    ax_q.set_xscale("log")
    ax_q.set_ylim(0, 1.05)
    ax_q.grid(True, which="both", alpha=0.3)
    ax_q.legend(fontsize=11)
    ax_q.set_title("HNSW recall vs ef")

    # ---- right: Pareto recall-speed frontier ----
    for m, rows in rows_by_mode.items():
        times   = [r[1] for r in rows]
        recalls = [r[2] for r in rows]
        ef_vals = [r[0] for r in rows]
        ax_p.plot(times, recalls, "o-", label=m)
        for t, rec, ef in zip(times, recalls, ef_vals):
            ax_p.annotate(f"ef={ef}", (t, rec),
                          textcoords="offset points", xytext=(4, 0),
                          fontsize=7, color="gray")
    ax_p.set_xlabel("HNSW query wall time (ms)")
    ax_p.set_ylabel("recall@k")
    ax_p.set_xscale("log")
    ax_p.set_ylim(0, 1.05)
    ax_p.grid(True, which="both", alpha=0.3)
    ax_p.legend(fontsize=11)
    ax_p.set_title("HNSW recall-speed Pareto frontier")

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
