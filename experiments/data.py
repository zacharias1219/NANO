"""Download, verify (SHA-256) and load MNIST and Fashion-MNIST as float64 arrays.

Usage: ``python -m experiments.data`` downloads both datasets into ``data/``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPEC = json.loads((Path(__file__).with_name("datasets.json")).read_text())
SEEDS = json.loads((Path(__file__).with_name("seeds.json")).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str) -> Path:
    spec, folder = SPEC[name], DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    for fname, digest in spec["files"].items():
        path = folder / fname
        if not path.exists() or _sha256(path) != digest:
            print(f"downloading {name}/{fname}", file=sys.stderr)
            urllib.request.urlretrieve(spec["base_url"] + fname, path)
        if _sha256(path) != digest:
            raise RuntimeError(f"checksum mismatch for {path}: dataset changed upstream or download corrupted")
    return folder


def _idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(name: str = "mnist", split: str = "train", n: int | None = None, seed: int | None = None,
         flat: bool = True, dtype=np.float64):
    """Images scaled to [0, 1] then standardised with the training-set mean/std.

    ``n`` draws a fixed random subset (seeded by ``seeds.json: data_subset_seed`` unless ``seed`` is given).
    Returns (X, y) with X of shape (n, 784) if ``flat`` else (n, 1, 28, 28).
    """
    folder = download(name)
    prefix = "train" if split == "train" else "t10k"
    X = _idx(folder / f"{prefix}-images-idx3-ubyte.gz").astype(np.float64) / 255.0
    y = _idx(folder / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64)
    if n is not None and n < len(y):
        rng = np.random.default_rng(SEEDS["data_subset_seed"] if seed is None else seed)
        idx = np.sort(rng.choice(len(y), size=n, replace=False))
        X, y = X[idx], y[idx]
    mean, std = (0.1307, 0.3081) if name == "mnist" else (0.2860, 0.3530)  # train-set statistics
    X = ((X - mean) / std).astype(dtype)
    return (X.reshape(len(y), -1) if flat else X.reshape(len(y), 1, 28, 28)), y


if __name__ == "__main__":
    for name in SPEC:
        if not name.startswith("_"):
            print(download(name))
