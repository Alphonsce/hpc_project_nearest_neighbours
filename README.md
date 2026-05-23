# hpc_project_nearest_neighbours

Speeding up nearest-neighbour search on GPU. The first piece in place is a small,
visualisable benchmark dataset built from GloVe-50d.

## Dataset prep

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install numpy scikit-learn matplotlib tqdm requests

# downloads glove.6B.zip (~822 MB) on first run, then keeps the top-N frequent tokens
python scripts/prepare_glove.py --n 20000
```

Produces in `data/`:

| file | shape / type | notes |
|---|---|---|
| `glove50_<N>.npy` | float32 (N, 50) | raw GloVe vectors |
| `glove50_<N>_norm.npy` | float32 (N, 50) | L2-normalised — use for cosine kNN |
| `glove50_<N>_words.txt` | N lines | row-aligned tokens |
| `glove50_<N>_meta.json` | json | `{dim, n, source}` |

## Visualise neighbours

```bash
python scripts/visualize_neighbors.py --word king --k 10
# prints cosine kNN, writes data/nn_king.png (PCA-2D, query + neighbours highlighted)
```

The brute-force cosine search in `visualize_neighbors.py` is the ground-truth
baseline for future GPU implementations (KD/Ball-tree, HNSW, LSH, …).
