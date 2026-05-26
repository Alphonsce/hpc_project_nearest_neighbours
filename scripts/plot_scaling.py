"""Plot strong-scaling curves per mode from a scaling CSV.

Input CSV columns: mode,threads,brute_ms,lsh_ms,speedup_brute,speedup_lsh,recall

Usage:
    python scripts/plot_scaling.py --csv data/scaling_modes_L32_K12.csv
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

    rows_by_mode: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows_by_mode[r["mode"]].append(
                (int(r["threads"]), float(r["speedup_lsh"]), float(r["lsh_ms"]))
            )
    for v in rows_by_mode.values():
        v.sort()

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(13, 5))

    all_ps = sorted({p for v in rows_by_mode.values() for p, _, _ in v})
    ax_s.plot(all_ps, all_ps, "k--", alpha=0.5, label="ideal")
    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]; ss = [r[1] for r in rows]
        ax_s.plot(ps, ss, "o-", label=m)
    ax_s.set_xlabel("OpenMP threads"); ax_s.set_ylabel("speedup  T(1) / T(P)")
    ax_s.set_xticks(all_ps); ax_s.grid(True, alpha=0.3); ax_s.legend(fontsize=16)
    ax_s.set_title("LSH strong scaling by mode")

    for m, rows in rows_by_mode.items():
        ps = [r[0] for r in rows]; ts = [r[2] for r in rows]
        ax_t.plot(ps, ts, "o-", label=m)
    ax_t.set_xlabel("OpenMP threads"); ax_t.set_ylabel("LSH wall time (ms)")
    ax_t.set_xticks(all_ps); ax_t.set_yscale("log")
    ax_t.grid(True, which="both", alpha=0.3); ax_t.legend(fontsize=16)
    ax_t.set_title("LSH wall time by mode (log)")

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
