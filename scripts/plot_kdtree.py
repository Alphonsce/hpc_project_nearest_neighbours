"""Plot KD-tree build and query scaling results.

Input CSVs produced by scripts/bench_kdtree.sh.

Usage:
    python scripts/plot_kdtree.py --build-csv results/kdtree_build_D25_N400000.csv \
                                  --query-csv results/kdtree_query_D25_N400000.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-csv", required=True)
    ap.add_argument("--query-csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    build_rows = read_csv(args.build_csv)
    query_rows = read_csv(args.query_csv)

    # ---- group by mode ----
    build_by_mode: dict[str, list] = defaultdict(list)
    for r in build_rows:
        build_by_mode[r["build_mode"]].append(
            (int(r["threads"]), float(r["build_ms"]), float(r["speedup"]))
        )
    for v in build_by_mode.values():
        v.sort()

    query_by_mode: dict[str, list] = defaultdict(list)
    for r in query_rows:
        query_by_mode[r["query_mode"]].append(
            (int(r["threads"]), float(r["query_ms"]),
             float(r["speedup_threads"]), float(r["brute_ms"]))
        )
    for v in query_by_mode.values():
        v.sort()

    all_build_p = sorted({r[0] for v in build_by_mode.values() for r in v})
    all_query_p = sorted({r[0] for v in query_by_mode.values() for r in v})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_bs  = axes[0][0]
    ax_bt  = axes[0][1]
    ax_qs  = axes[0][2]
    ax_qt  = axes[1][0]
    ax_brc = axes[1][1]
    ax_emp = axes[1][2]
    # rearrange: row0 = build speedup + time + query speedup
    #            row1 = query time + brute comparison + (empty)

    # ---- build speedup ----
    ax_bs.plot(all_build_p, all_build_p, "k--", alpha=0.4, label="ideal")
    for m, rows in build_by_mode.items():
        ps = [r[0] for r in rows]; ss = [r[2] for r in rows]
        ax_bs.plot(ps, ss, "o-", label=m)
    ax_bs.set_xlabel("threads"); ax_bs.set_ylabel("speedup  T(1) / T(P)")
    ax_bs.set_xticks(all_build_p); ax_bs.grid(True, alpha=0.3); ax_bs.legend()
    ax_bs.set_title("Build scaling")

    # ---- build time ----
    for m, rows in build_by_mode.items():
        ps = [r[0] for r in rows]; ts = [r[1] for r in rows]
        ax_bt.plot(ps, ts, "o-", label=m)
    ax_bt.set_xlabel("threads"); ax_bt.set_ylabel("build time (ms)")
    ax_bt.set_xticks(all_build_p); ax_bt.grid(True, alpha=0.3); ax_bt.legend()
    ax_bt.set_title("Build time")

    # ---- query speedup ----
    ax_qs.plot(all_query_p, all_query_p, "k--", alpha=0.4, label="ideal")
    for m, rows in query_by_mode.items():
        ps = [r[0] for r in rows]; ss = [r[2] for r in rows]
        ax_qs.plot(ps, ss, "o-", label=m)
    ax_qs.set_xlabel("threads"); ax_qs.set_ylabel("speedup  T(1) / T(P)")
    ax_qs.set_xticks(all_query_p); ax_qs.grid(True, alpha=0.3); ax_qs.legend()
    ax_qs.set_title("Query scaling")

    # ---- query time (log) ----
    for m, rows in query_by_mode.items():
        ps = [r[0] for r in rows]; ts = [r[1] for r in rows]
        ax_qt.plot(ps, ts, "o-", label=m)
    ax_qt.set_xlabel("threads"); ax_qt.set_ylabel("batch query time (ms)")
    ax_qt.set_xticks(all_query_p); ax_qt.set_yscale("log")
    ax_qt.grid(True, which="both", alpha=0.3); ax_qt.legend()
    ax_qt.set_title("Query time (log)")

    # ---- kd-tree vs brute-force (serial query mode only, all thread counts) ----
    serial_rows = sorted(query_by_mode.get("serial", []))
    if serial_rows:
        ps       = [r[0] for r in serial_rows]
        kd_ms    = [r[1] for r in serial_rows]
        brute_ms = [r[3] for r in serial_rows]
        ax_brc.plot(ps, brute_ms, "s--", color="gray", label="brute-force")
        ax_brc.plot(ps, kd_ms,   "o-",  label="kd-tree (serial)")
        ax_brc.set_xlabel("threads"); ax_brc.set_ylabel("batch query time (ms)")
        ax_brc.set_xticks(ps); ax_brc.set_yscale("log")
        ax_brc.grid(True, which="both", alpha=0.3); ax_brc.legend()
        ax_brc.set_title("KD-tree vs brute-force")

    # ---- nodes visited ----
    nodes_by_mode: dict[str, list] = defaultdict(list)
    for r in query_rows:
        nodes_by_mode[r["query_mode"]].append(
            (int(r["threads"]), float(r["nodes_mean"]))
        )
    for v in nodes_by_mode.values():
        v.sort()
    for m, rows in nodes_by_mode.items():
        ps = [r[0] for r in rows]; ns = [r[1] for r in rows]
        ax_emp.plot(ps, ns, "o-", label=m)
    ax_emp.set_xlabel("threads"); ax_emp.set_ylabel("mean nodes visited")
    ax_emp.set_xticks(all_query_p); ax_emp.grid(True, alpha=0.3); ax_emp.legend()
    ax_emp.set_title("Nodes visited per query")

    stem = Path(args.build_csv).stem.replace("kdtree_build_", "")
    fig.suptitle(f"KD-tree  {stem}", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.build_csv).with_name(
        f"kdtree_{stem}.png")
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
