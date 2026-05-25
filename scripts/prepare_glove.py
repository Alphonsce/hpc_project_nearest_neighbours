"""Prepare a GloVe subset for nearest-neighbour experiments.

Sources:
    6B          400k vocab,   dims {50, 100, 200, 300}    (Wikipedia + Gigaword)
    twitter.27B 1.2M vocab,   dims {25, 50, 100, 200}     (Twitter)

Outputs (in data/):
    glove<dim>_<N>.npy           float32 (N, dim)  raw
    glove<dim>_<N>_norm.npy      float32 (N, dim)  L2-normalised
    glove<dim>_<N>_words.txt     one token per line
    glove<dim>_<N>_meta.json     {dim, n, source}

Usage:
    python scripts/prepare_glove.py --source twitter.27B --dim 100 --n 1000000
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "6B":          ("https://nlp.stanford.edu/data/glove.6B.zip",         {50, 100, 200, 300}),
    "twitter.27B": ("https://nlp.stanford.edu/data/glove.twitter.27B.zip", {25, 50, 100, 200}),
}


def fetch_raw(source: str, dim: int) -> Path:
    url, dims = SOURCES[source]
    if dim not in dims:
        raise SystemExit(f"source {source} has no dim {dim}; available: {sorted(dims)}")
    member = f"glove.{source}.{dim}d.txt"
    out = DATA / member
    if out.exists():
        return out
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / f"glove.{source}.zip"
    if not zip_path.exists():
        print(f"downloading {url} -> {zip_path} ...", file=sys.stderr)
        tmp = zip_path.with_suffix(".zip.part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk); bar.update(len(chunk))
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, DATA)
    return out


def load_subset(path: Path, n: int, dim: int):
    words: list[str] = []
    vecs = np.empty((n, dim), dtype=np.float32)
    kept = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if kept >= n:
                break
            parts = line.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                continue  # skip malformed lines (a few exist in twitter.27B)
            words.append(parts[0])
            vecs[kept] = np.asarray(parts[1:], dtype=np.float32)
            kept += 1
    if kept < n:
        print(f"warning: only {kept} usable rows in source, requested {n}", file=sys.stderr)
        vecs = vecs[:kept]
    return words, vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="6B", choices=list(SOURCES))
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    raw = fetch_raw(args.source, args.dim)
    words, vecs = load_subset(raw, args.n, args.dim)
    n_eff = len(words)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_norm = (vecs / norms).astype(np.float32)

    DATA.mkdir(parents=True, exist_ok=True)
    tag = f"glove{args.dim}_{n_eff}"
    np.save(DATA / f"{tag}.npy", vecs)
    np.save(DATA / f"{tag}_norm.npy", vecs_norm)
    (DATA / f"{tag}_words.txt").write_text("\n".join(words), encoding="utf-8")
    (DATA / f"{tag}_meta.json").write_text(
        json.dumps({"dim": args.dim, "n": n_eff, "source": args.source}, indent=2)
    )
    print(f"wrote data/{tag}.npy  shape={vecs.shape}  source={args.source}")


if __name__ == "__main__":
    main()