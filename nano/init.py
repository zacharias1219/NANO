"""Weight initialisation schemes (F5): zeros, small normal, Xavier, He.

All schemes draw from an explicit ``np.random.Generator`` so every run is
reproducible from a seed. Fan-in and fan-out follow PyTorch: for a conv weight
(out, in, kh, kw) the receptive field kh*kw multiplies both. Derivation of the
variance-preserving scales: report/derivations/DERIVATIONS.md section D9.
"""
from __future__ import annotations

import numpy as np

_rng = np.random.default_rng(0)


def manual_seed(seed: int) -> None:
    """Reseed the default generator used when a layer is built without ``rng``."""
    global _rng
    _rng = np.random.default_rng(seed)


def default_rng() -> np.random.Generator:
    return _rng


def fans(shape: tuple) -> tuple[int, int]:
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


def zeros(shape, rng=None):
    return np.zeros(shape)


def small_normal(shape, rng=None, std: float = 0.01):
    """N(0, 0.01^2): the classic 'small random numbers' init that vanishes with depth."""
    return (rng or _rng).normal(0.0, std, size=shape)


def xavier(shape, rng=None):
    """Glorot normal: Var = 2 / (fan_in + fan_out), preserves variance for tanh-like units. [D9.2]"""
    fi, fo = fans(shape)
    return (rng or _rng).normal(0.0, np.sqrt(2.0 / (fi + fo)), size=shape)


def he(shape, rng=None):
    """Kaiming normal: Var = 2 / fan_in, preserves forward variance through ReLU. [D9.3]"""
    fi, _ = fans(shape)
    return (rng or _rng).normal(0.0, np.sqrt(2.0 / fi), size=shape)


SCHEMES = {"zeros": zeros, "small": small_normal, "small_normal": small_normal, "xavier": xavier, "he": he}


def get(scheme):
    return scheme if callable(scheme) else SCHEMES[scheme]
