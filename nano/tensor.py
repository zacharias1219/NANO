"""Tensor, the dynamic tape, and reverse-mode backward.

Every differentiable primitive is an `Op`: a pure function from NumPy arrays to
``(output, vjp)`` where ``vjp(g)`` returns one gradient per input (or ``None``).
Calling an Op on Tensors runs the forward pass and records a node on the tape;
``Tensor.backward`` walks the tape in reverse topological order and applies the
chain rule. Broadcasting is undone in exactly one place, ``unbroadcast``.

Derivations: report/derivations/DERIVATIONS.md, sections D1 (reverse mode on a
DAG) and D2 (unbroadcast).
"""
from __future__ import annotations

import itertools
from contextlib import contextmanager

import numpy as np

DEFAULT_DTYPE = np.float64
_GRAD_ENABLED = True
_SCOPE: list[str] = []  # module path stack, filled by nn.Module.__call__
_SEQ = itertools.count()  # creation order of tape nodes, i.e. forward execution order


def set_default_dtype(dtype) -> None:
    """Set the dtype used when a Tensor is built from non-float data."""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = np.dtype(dtype).type


@contextmanager
def no_grad():
    """Run a block without recording a tape (evaluation, optimizer updates)."""
    global _GRAD_ENABLED
    prev, _GRAD_ENABLED = _GRAD_ENABLED, False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, the adjoint of NumPy broadcasting (D2).

    Broadcasting copies an input along new leading axes and along axes of size
    one, so the adjoint sums the upstream gradient over exactly those axes.
    """
    shape = tuple(shape)
    if grad.shape == shape:
        return grad
    lead = grad.ndim - len(shape)
    if lead > 0:
        grad = grad.sum(axis=tuple(range(lead)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    if grad.shape != shape:
        raise ValueError(f"unbroadcast: cannot reduce {grad.shape} to {shape}")
    return grad


class Op:
    """A differentiable primitive. ``fn(*arrays, **kw) -> (out, vjp)``."""

    def __init__(self, fn):
        self.fn, self.name, self.__doc__ = fn, fn.__name__, fn.__doc__

    def __call__(self, *args, **kw) -> Tensor:
        dtype = next((a.dtype for a in args if isinstance(a, Tensor)), DEFAULT_DTYPE)
        ts = tuple(a if isinstance(a, Tensor) else Tensor(a, dtype=dtype) for a in args)
        out, vjp = self.fn(*(t.data for t in ts), **kw)
        need = _GRAD_ENABLED and any(t.requires_grad for t in ts)
        node = Tensor(out, requires_grad=need)
        if need:
            node._ctx = (self, ts, kw, vjp)
            node.scope, node.seq = "/".join(_SCOPE), next(_SEQ)
        return node

    def __repr__(self) -> str:
        return f"Op({self.name})"


OPS: dict[str, Op] = {}


def op(fn) -> Op:
    """Register ``fn`` as a differentiable op. See README "Adding an op"."""
    OPS[fn.__name__] = Op(fn)
    return OPS[fn.__name__]


class Tensor:
    """An ndarray plus the tape bookkeeping needed for reverse-mode autodiff."""

    __array_ufunc__ = None  # make `ndarray + Tensor` defer to Tensor.__radd__

    def __init__(self, data, requires_grad: bool = False, dtype=None, name: str | None = None):
        if dtype is None:
            arr = np.asarray(data)
            dtype = arr.dtype if arr.dtype.kind == "f" else DEFAULT_DTYPE
        self.data = np.asarray(data, dtype=dtype)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self.name, self.scope, self.seq = name, "", -1
        self._ctx = None  # (op, parents, kwargs, vjp) for non-leaf nodes
        self._retain = False

    # -- basic properties -------------------------------------------------
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    dtype = property(lambda self: self.data.dtype)
    size = property(lambda self: self.data.size)
    is_leaf = property(lambda self: self._ctx is None)

    def item(self) -> float:
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def retain_grad(self) -> Tensor:
        """Keep ``.grad`` on this non-leaf node after backward (used by probes)."""
        self._retain = True
        return self

    def __repr__(self) -> str:
        op = f", op={self._ctx[0].name}" if self._ctx else ""
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad}{op})"

    def __len__(self) -> int:
        return len(self.data)

    # -- the tape ---------------------------------------------------------
    def topo(self) -> list[Tensor]:
        """Nodes reachable from self, parents before children (iterative DFS)."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            if node._ctx is not None:
                stack.extend((p, False) for p in node._ctx[1] if id(p) not in seen)
        return order

    def backward(self, grad=None, retain_graph: bool = False) -> None:
        """Accumulate d(self)/d(leaf) into ``leaf.grad`` for every leaf (D1).

        ``grad`` defaults to ones, i.e. the gradient of ``self.sum()``.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        seed = np.ones_like(self.data) if grad is None else np.asarray(grad, self.dtype)
        grads = {id(self): seed}
        for node in reversed(self.topo()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._ctx is None or node._retain:
                node.grad = np.array(g) if node.grad is None else node.grad + g
            if node._ctx is None:
                continue
            _, parents, _, vjp = node._ctx
            for p, pg in zip(parents, vjp(g)):
                if pg is None or not p.requires_grad:
                    continue
                pg = unbroadcast(np.asarray(pg, dtype=p.dtype), p.shape)
                grads[id(p)] = grads[id(p)] + pg if id(p) in grads else pg
            if not retain_graph:
                node._ctx = None
