"""Nano: a reverse-mode autodiff engine, MLP and CNN stack in pure NumPy.

Verified against central finite differences and PyTorch (float64) under the
precision contract in PRECISION.md. PyTorch is never imported here.
"""
from . import conv, gradcheck, init, nn, ops, optim, probes, sched
from .conv import conv2d, conv2d_naive, maxpool2d
from .init import manual_seed
from .ops import cross_entropy_composed, logsumexp, relu, sigmoid, softmax_cross_entropy, tanh
from .tensor import OPS, Op, Tensor, no_grad, op, set_default_dtype, unbroadcast

__version__ = "1.0.0"

__all__ = [
    "OPS", "Op", "Tensor", "conv", "conv2d", "conv2d_naive", "cross_entropy_composed", "gradcheck",
    "init", "logsumexp", "manual_seed", "maxpool2d", "nn", "no_grad", "op", "ops", "optim", "probes",
    "relu", "sched", "set_default_dtype", "sigmoid", "softmax_cross_entropy", "tanh", "unbroadcast",
]
