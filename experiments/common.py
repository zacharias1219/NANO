"""Shared helpers for experiments: threading discipline, quick mode, batching, figures.

Import this module before NumPy in every experiment so BLAS stays single-threaded
(parity and timing are single-threaded by design; sweeps parallelise over processes).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"
RAW = ROOT / "results" / "raw"
QUICK = os.environ.get("NANO_QUICK", "0") == "1"
JOBS = int(os.environ.get("NANO_JOBS", os.cpu_count() or 1))
SEEDS = json.loads((Path(__file__).with_name("seeds.json")).read_text())

PALETTE = {"float64": "#2B6CB0", "float32": "#C05621", "nano": "#2B6CB0", "torch": "#C05621",
           "relu": "#2F855A", "tanh": "#B7791F", "sigmoid": "#9B2C2C", "grey": "#718096"}


def batch_schedule(n: int, batch: int, steps: int, seed: int) -> np.ndarray:
    """(steps, batch) index array: epochs of seeded permutations, identical for Nano and PyTorch."""
    rng = np.random.default_rng(seed)
    per_epoch = n // batch
    epochs = -(-steps // per_epoch)
    idx = np.concatenate([rng.permutation(n)[: per_epoch * batch] for _ in range(epochs)])
    return idx.reshape(-1, batch)[:steps]


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    })
    return pyplot


def save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    import matplotlib.pyplot as pyplot

    pyplot.close(fig)
    return f"figures/{name}"


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
