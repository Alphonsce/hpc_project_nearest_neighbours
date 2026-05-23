"""Strong-scaling benchmark: time(1 thread) / time(P threads) for brute and LSH.

Runs ./cpp/lsh with OMP_NUM_THREADS = 1, 2, 4, ..., parses the brute and lsh
total times from stdout, then plots and saves a CSV.

Usage:
    python scripts/scaling.py --base data/glove100_1000000 --L 32 --K 12 --queries 500
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BRUTE_RE = re.compile(r"^brute\s*:\s*([0-9.]+)\s*ms", re.M)
LSH_RE   = re.compile(r"^lsh\s*:\s*([0-9.]+)\s*ms",   re.M)
RECALL_RE= re.compile(r"^recall@\d+\s*:\s*([0-9.]+)", re.M)


def run(threads: int, args) -> tuple[float, float, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [str(ROOT / "cpp" / "lsh"), args.base,
           "--L", str(args.L), "--K", str(args.K),
           "--topk", str(args.topk), "--queries", str(args.queries),
           "--seed", str(args.seed)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    brute = float(BRUTE_RE.search(out).group(1))
    lsh   = float(LSH_RE.search(out).group(1))
    rec   = float(RECALL_RE.search(out).group(1))
    return brute, lsh, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/glove100_1000000")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    rows = []
    print(f"{'P':>3} {'brute(ms)':>12} {'lsh(ms)':>12} {'S_brute':>8} {'S_lsh':>8} {'recall':>8}")
    base_b = base_l = None
    for p in threads:
        b, l, r = run(p, args)
        if base_b is None:
            base_b, base_l = b, l
        sb, sl = base_b / b, base_l / l
        rows.append((p, b, l, sb, sl, r))
        print(f"{p:>3} {b:>12.2f} {l:>12.2f} {sb:>8.2f} {sl:>8.2f} {r:>8.4f}")

    out_png = Path(args.out) if args.out else ROOT / "data" / f"scaling_L{args.L}_K{args.K}.png"
    csv_path = out_png.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threads", "brute_ms", "lsh_ms", "speedup_brute", "speedup_lsh", "recall"])
        w.writerows(rows)

    print(f"saved -> {csv_path}")
    if args.no_plot:
        return

    ps = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, ps, "k--", label="ideal", alpha=0.6)
    ax.plot(ps, [r[3] for r in rows], "o-", label="brute-force")
    ax.plot(ps, [r[4] for r in rows], "s-", label=f"LSH (L={args.L}, K={args.K})")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("strong-scaling speedup  T(1) / T(P)")
    ax.set_title(f"strong scaling  {Path(args.base).name}  queries={args.queries} topk={args.topk}")
    ax.set_xticks(ps); ax.set_yticks(ps)
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
