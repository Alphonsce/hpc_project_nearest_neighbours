"""Generate the synthetic 10d clustered dataset used in KD-tree benchmarks.

The generated data is deterministic and intentionally not checked into git.

Usage:
    python scripts/prepare_synthetic10.py
    python scripts/prepare_synthetic10.py --n 200000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    if args.dim != 10:
        raise SystemExit("This helper is intended for the 10d synthetic benchmark; keep --dim 10.")

    DATA.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"synthetic10_{args.n}"
    base = DATA / prefix

    rng = np.random.default_rng(args.seed)
    centers = rng.normal(size=(args.clusters, args.dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = rng.integers(0, args.clusters, size=args.n)
    x = centers[labels] + args.noise * rng.normal(size=(args.n, args.dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = np.ascontiguousarray(x.astype(np.float32))

    np.save(str(base) + ".npy", x)
    np.save(str(base) + "_norm.npy", x)
    x.tofile(str(base) + ".f32")
    Path(str(base) + ".shape").write_text(f"{args.n} {args.dim}\n")
    Path(str(base) + "_meta.json").write_text(json.dumps({
        "dim": args.dim,
        "n": args.n,
        "source": "synthetic clustered unit vectors",
        "seed": args.seed,
        "clusters": args.clusters,
        "cluster_noise": args.noise,
    }, indent=2) + "\n")

    print(f"wrote {base}  shape=({args.n}, {args.dim})")


if __name__ == "__main__":
    main()
