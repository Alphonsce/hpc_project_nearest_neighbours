"""Dump the prepared GloVe subset as raw binaries the C++ code can mmap/read.

    data/glove50_<N>.f32   row-major float32, L2-normalised, shape (N, 50)
    data/glove50_<N>.shape text "N D"

Usage:
    python scripts/export_for_cpp.py --n 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    tag = f"glove{args.dim}_{args.n}"
    vecs = np.load(DATA / f"{tag}_norm.npy")
    assert vecs.dtype == np.float32 and vecs.ndim == 2

    out = DATA / f"{tag}.f32"
    vecs.tofile(out)
    (DATA / f"{tag}.shape").write_text(f"{vecs.shape[0]} {vecs.shape[1]}\n")
    print(f"wrote {out}  ({vecs.nbytes/1e6:.1f} MB)  shape={vecs.shape}")


if __name__ == "__main__":
    main()
