"""Differentiable primitives (F2) and the fused softmax cross entropy (F3).

Each op is a forward computation plus its vector-Jacobian product. Inputs and
the returned gradients are plain ndarrays; ``unbroadcast`` (tensor.py) maps each
gradient back to its input's shape, so elementwise ops return full-shape grads.
The tag ``[Dx.y]`` points at the derivation in report/derivations/DERIVATIONS.md.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor, op


def _axes(axis, ndim: int) -> tuple:
    if axis is None:
        return tuple(range(ndim))
    axis = (axis,) if np.isscalar(axis) else tuple(axis)
    return tuple(a % ndim for a in axis)


def _expand(g: np.ndarray, shape: tuple, axis, keepdims: bool) -> np.ndarray:
    """Re-insert reduced axes so ``g`` broadcasts against the input [D3.6]."""
    if not keepdims:
        g = np.expand_dims(g, _axes(axis, len(shape)))
    return np.broadcast_to(g, shape)


# --- elementwise arithmetic [D3.1-D3.5] ---------------------------------------
@op
def add(a, b):
    """z = a + b; dz/da = dz/db = 1. [D3.1]"""
    return a + b, lambda g: (g, g)


@op
def sub(a, b):
    """z = a - b; dz/da = 1, dz/db = -1. [D3.1]"""
    return a - b, lambda g: (g, -g)


@op
def mul(a, b):
    """z = a * b; dz/da = b, dz/db = a. [D3.2]"""
    return a * b, lambda g: (g * b, g * a)


@op
def div(a, b):
    """z = a / b; dz/da = 1/b, dz/db = -a/b^2. [D3.3]"""
    return a / b, lambda g: (g / b, -g * a / (b * b))


@op
def pow(a, p: float):  # noqa: A001 - mirrors the operator name
    """z = a^p for a constant exponent p; dz/da = p a^(p-1). [D3.4]"""
    return a**p, lambda g: (g * p * a ** (p - 1),)


@op
def neg(a):
    """z = -a; dz/da = -1. [D3.1]"""
    return -a, lambda g: (-g,)


@op
def exp(a):
    """z = e^a; dz/da = z. [D3.5]"""
    out = np.exp(a)
    return out, lambda g: (g * out,)


@op
def log(a):
    """z = ln a; dz/da = 1/a. [D3.5]"""
    return np.log(a), lambda g: (g / a,)


# --- activations [D4] ---------------------------------------------------------
@op
def relu(a):
    """z = max(a, 0); dz/da = 1[a > 0] (subgradient 0 at the kink). [D4.1]"""
    mask = a > 0
    return np.maximum(a, 0), lambda g: (g * mask,)


@op
def tanh(a):
    """z = tanh a; dz/da = 1 - z^2. [D4.2]"""
    out = np.tanh(a)
    return out, lambda g: (g * (1 - out * out),)


@op
def sigmoid(a):
    """z = 1/(1+e^-a), computed as exp(-softplus(-a)); dz/da = z(1-z). [D4.3]"""
    out = np.exp(-np.logaddexp(0.0, -a))
    return out, lambda g: (g * out * (1 - out),)


# --- linear algebra and shape ops [D5] ----------------------------------------
@op
def matmul(a, b):
    """Z = A B (batched); dA = G B^T, dB = A^T G, batch axes unbroadcast. [D5.1]"""
    t = lambda m: np.swapaxes(m, -1, -2)  # noqa: E731
    return a @ b, lambda g: (g @ t(b), t(a) @ g)


@op
def reshape(a, shape: tuple):
    """Reshape is a relabelling of elements, so its VJP reshapes back. [D5.2]"""
    return a.reshape(shape), lambda g: (g.reshape(a.shape),)


@op
def transpose(a, axes: tuple | None = None):
    """Permute axes; the VJP applies the inverse permutation. [D5.3]"""
    inv = None if axes is None else tuple(np.argsort(axes))
    return a.transpose(axes), lambda g: (g.transpose(inv),)


@op
def getitem(a, idx):
    """Gather; the VJP scatter-adds, so repeated indices accumulate. [D5.4]"""
    def vjp(g):
        out = np.zeros_like(a)
        np.add.at(out, idx, g)
        return (out,)
    return a[idx], vjp


# --- reductions [D3.6, D6] ----------------------------------------------------
@op
def sum(a, axis=None, keepdims: bool = False):  # noqa: A001
    """z = sum over axis; each input gets the upstream gradient. [D3.6]"""
    return a.sum(axis=axis, keepdims=keepdims), lambda g: (_expand(g, a.shape, axis, keepdims),)


@op
def mean(a, axis=None, keepdims: bool = False):
    """z = sum / n; each input gets g / n. [D3.6]"""
    n = int(np.prod([a.shape[i] for i in _axes(axis, a.ndim)]))
    return a.mean(axis=axis, keepdims=keepdims), lambda g: (_expand(g, a.shape, axis, keepdims) / n,)


@op
def max(a, axis=None, keepdims: bool = False):  # noqa: A001
    """z = max over axis; ties share the gradient equally (torch.amax). [D6.1]"""
    m = a.max(axis=axis, keepdims=True)
    mask = (a == m).astype(a.dtype)
    share = mask / mask.sum(axis=axis, keepdims=True)
    out = m if keepdims else np.squeeze(m, axis=_axes(axis, a.ndim))
    return out, lambda g: (share * _expand(g, a.shape, axis, keepdims),)


@op
def logsumexp(a, axis=-1, keepdims: bool = False):
    """z = m + log sum e^(a-m), m = max a; dz/da = softmax(a). [D6.2]"""
    m = a.max(axis=axis, keepdims=True)
    e = np.exp(a - m)
    s = e.sum(axis=axis, keepdims=True)
    out = m + np.log(s)
    out = out if keepdims else np.squeeze(out, axis=_axes(axis, a.ndim))
    return out, lambda g: (_expand(g, a.shape, axis, keepdims) * (e / s),)


# --- loss [D7] -----------------------------------------------------------------
@op
def softmax_cross_entropy(logits, y):
    """Mean CE of integer targets y, fused and max-shifted. dL/dz = (p - onehot)/N. [D7.1]"""
    n = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    p = np.exp(z - lse[:, None])

    def vjp(g):
        d = p.copy()
        d[np.arange(n), y] -= 1.0
        return (g * d / n,)

    return np.mean(lse - z[np.arange(n), y]), vjp


def cross_entropy_composed(logits: Tensor, y) -> Tensor:
    """softmax -> log -> NLL built from primitives, with NO max shift.

    Kept on purpose for the overflow failure demo [D7.2]: exp overflows to inf
    once a logit exceeds ~709 (float64) or ~88 (float32).
    """
    e = logits.exp()
    p = e / e.sum(axis=1, keepdims=True)
    return -p[np.arange(len(y)), np.asarray(y)].log().mean()


# --- Tensor methods and operators ----------------------------------------------
def _install() -> None:
    T = Tensor
    T.__add__ = lambda a, b: add(a, b)
    T.__radd__ = lambda a, b: add(b, a)
    T.__sub__ = lambda a, b: sub(a, b)
    T.__rsub__ = lambda a, b: sub(b, a)
    T.__mul__ = lambda a, b: mul(a, b)
    T.__rmul__ = lambda a, b: mul(b, a)
    T.__truediv__ = lambda a, b: div(a, b)
    T.__rtruediv__ = lambda a, b: div(b, a)
    T.__pow__ = lambda a, p: pow(a, p=p)
    T.__neg__ = lambda a: neg(a)
    T.__matmul__ = lambda a, b: matmul(a, b)
    T.__rmatmul__ = lambda a, b: matmul(b, a)
    T.__getitem__ = lambda a, idx: getitem(a, idx=idx.data if isinstance(idx, T) else idx)
    T.sum = lambda a, axis=None, keepdims=False: sum(a, axis=axis, keepdims=keepdims)
    T.mean = lambda a, axis=None, keepdims=False: mean(a, axis=axis, keepdims=keepdims)
    T.max = lambda a, axis=None, keepdims=False: max(a, axis=axis, keepdims=keepdims)
    T.logsumexp = lambda a, axis=-1, keepdims=False: logsumexp(a, axis=axis, keepdims=keepdims)
    for f in (exp, log, relu, tanh, sigmoid):
        setattr(T, f.name, lambda a, f=f: f(a))
    T.reshape = lambda a, *shape: reshape(a, shape=shape[0] if len(shape) == 1 and not np.isscalar(shape[0]) else shape)
    T.transpose = lambda a, *axes: transpose(a, axes=(axes[0] if len(axes) == 1 else axes) or None)
    T.T = property(lambda a: transpose(a, axes=None))
    T.flatten = lambda a: reshape(a, shape=(a.shape[0], -1))


_install()
