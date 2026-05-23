"""Find and visualise k nearest neighbours of a query word in the GloVe subset.

Brute-force cosine kNN (used as a ground-truth baseline for the future GPU
methods). Prints the neighbours and saves a 2-D PCA scatter highlighting them.

Usage:
    python scripts/visualize_neighbors.py --word king --k 10
    python scripts/visualize_neighbors.py --word king --k 10 --n 20000 --out fig.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(n: int):
    vecs = np.load(DATA / f"glove50_{n}_norm.npy")
    words = (DATA / f"glove50_{n}_words.txt").read_text(encoding="utf-8").splitlines()
    return words, vecs


def knn_cosine(vecs_norm: np.ndarray, q_idx: int, k: int) -> np.ndarray:
    sims = vecs_norm @ vecs_norm[q_idx]
    sims[q_idx] = -np.inf
    return np.argpartition(-sims, k)[:k][np.argsort(-sims[np.argpartition(-sims, k)[:k]])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=None, help="output PNG path (default: data/nn_<word>.png)")
    args = ap.parse_args()

    words, vecs = load(args.n)
    w2i = {w: i for i, w in enumerate(words)}
    if args.word not in w2i:
        raise SystemExit(f"'{args.word}' not in top-{args.n} GloVe vocab")
    qi = w2i[args.word]
    nn = knn_cosine(vecs, qi, args.k)

    print(f"query: {args.word}")
    for rank, j in enumerate(nn, 1):
        sim = float(vecs[qi] @ vecs[j])
        print(f"  {rank:2d}. {words[j]:<20s} cos={sim:.4f}")

    sample = np.random.default_rng(0).choice(len(words), size=min(2000, len(words)), replace=False)
    idx = np.unique(np.concatenate([sample, nn, [qi]]))
    pts = PCA(n_components=2, random_state=0).fit_transform(vecs[idx])
    pos = {i: pts[k] for k, i in enumerate(idx)}

    fig, ax = plt.subplots(figsize=(9, 7))
    bg = np.array([pos[i] for i in idx if i != qi and i not in set(nn)])
    ax.scatter(bg[:, 0], bg[:, 1], s=4, c="#cccccc", alpha=0.5, label="background")
    nn_pts = np.array([pos[i] for i in nn])
    ax.scatter(nn_pts[:, 0], nn_pts[:, 1], s=40, c="tab:blue", label=f"{args.k}-NN")
    ax.scatter([pos[qi][0]], [pos[qi][1]], s=120, c="tab:red", marker="*", label="query")

    for i in [qi, *nn]:
        ax.annotate(words[i], pos[i], fontsize=9, xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"GloVe-50d  PCA of {args.k}-NN around '{args.word}'  (vocab={args.n})")
    ax.legend(loc="best")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()

    out = Path(args.out) if args.out else DATA / f"nn_{args.word}.png"
    fig.savefig(out, dpi=140)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
