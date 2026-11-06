"""One table of op test cases shared by the gradcheck tests (A1), PyTorch parity tests (A2) and E1.

Each case gives input shapes, an input sampler that stays away from kinks and
ties (the documented blind spots of finite differences), the Nano function and
the equivalent PyTorch function. ``COVERS`` maps every registered op to a case,
so adding an op without a case fails ``tests/test_gradcheck.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import nano
from nano import Tensor


@dataclass
class Case:
    name: str
    shapes: list
    nano_fn: object
    torch_fn: object
    sampler: str = "normal"  # normal | positive | away_from_zero | distinct
    grad_mask: list = field(default_factory=list)  # which inputs require grad (default: all)

    def inputs(self, rng: np.random.Generator, dtype=np.float64) -> list[np.ndarray]:
        out = []
        for shape in self.shapes:
            if self.sampler == "positive":
                x = rng.uniform(0.5, 2.0, size=shape)
            elif self.sampler == "away_from_zero":
                x = rng.standard_normal(shape)
                x = np.sign(x) * (np.abs(x) + 0.1)
            elif self.sampler == "distinct":  # a permutation of well-separated values: no ties
                x = rng.permutation(np.prod(shape)).reshape(shape) * 0.1 + rng.uniform(0, 0.01, shape)
            else:
                x = rng.standard_normal(shape)
            out.append(x.astype(dtype))
        return out


def _torch():
    import torch
    import torch.nn.functional as F  # noqa: N812

    return torch, F


Y = np.array([3, 0, 4, 4, 1])  # fixed targets for the 5-row CE case
IDX = (np.array([0, 2, 2, 4]), slice(1, 4))


def _cases() -> list[Case]:
    t, F = _torch()  # noqa: N806
    return [
        Case("add", [(5, 4), (4,)], lambda a, b: a + b, lambda a, b: a + b),
        Case("add_bcast", [(2, 1, 4), (3, 1)], lambda a, b: a + b, lambda a, b: a + b),
        Case("sub", [(3, 4), (3, 1)], lambda a, b: a - b, lambda a, b: a - b),
        Case("mul", [(2, 3, 4), (3, 4)], lambda a, b: a * b, lambda a, b: a * b),
        Case("div", [(3, 4), (1, 4)], lambda a, b: a / b, lambda a, b: a / b, sampler="positive"),
        Case("pow", [(3, 4)], lambda a: a**2.5, lambda a: a**2.5, sampler="positive"),
        Case("pow_int", [(3, 4)], lambda a: a**3, lambda a: a**3),
        Case("neg", [(3, 4)], lambda a: -a, lambda a: -a),
        Case("exp", [(3, 4)], lambda a: a.exp(), lambda a: a.exp()),
        Case("log", [(3, 4)], lambda a: a.log(), lambda a: a.log(), sampler="positive"),
        Case("relu", [(4, 5)], lambda a: a.relu(), lambda a: a.relu(), sampler="away_from_zero"),
        Case("tanh", [(4, 5)], lambda a: a.tanh(), lambda a: a.tanh()),
        Case("sigmoid", [(4, 5)], lambda a: a.sigmoid(), lambda a: a.sigmoid()),
        Case("matmul", [(4, 3), (3, 5)], lambda a, b: a @ b, lambda a, b: a @ b),
        Case("matmul_batched", [(2, 4, 3), (3, 5)], lambda a, b: a @ b, lambda a, b: a @ b),
        Case("reshape", [(3, 4)], lambda a: a.reshape(2, 6), lambda a: a.reshape(2, 6)),
        Case("transpose", [(2, 3, 4)], lambda a: a.transpose(2, 0, 1), lambda a: a.permute(2, 0, 1)),
        Case("transpose_T", [(3, 4)], lambda a: a.T, lambda a: a.T),
        Case("getitem", [(5, 6)], lambda a: a[IDX], lambda a: a[t.as_tensor(IDX[0]), IDX[1]]),
        Case("sum", [(3, 4, 2)], lambda a: a.sum(axis=1), lambda a: a.sum(dim=1)),
        Case("sum_all", [(3, 4)], lambda a: a.sum(), lambda a: a.sum()),
        Case("mean", [(3, 4)], lambda a: a.mean(axis=0, keepdims=True), lambda a: a.mean(dim=0, keepdim=True)),
        Case("max", [(4, 5)], lambda a: a.max(axis=1), lambda a: t.amax(a, dim=1), sampler="distinct"),
        Case("max_all", [(3, 4)], lambda a: a.max(), lambda a: t.amax(a), sampler="distinct"),
        Case("logsumexp", [(4, 5)], lambda a: a.logsumexp(axis=1), lambda a: t.logsumexp(a, dim=1)),
        Case("softmax_cross_entropy", [(5, 6)], lambda z: nano.softmax_cross_entropy(z, y=Y),
             lambda z: F.cross_entropy(z, t.as_tensor(Y))),
        Case("conv2d", [(2, 3, 7, 7), (4, 3, 3, 3), (4,)],
             lambda x, w, b: nano.conv2d(x, w, b, stride=2, padding=1),
             lambda x, w, b: F.conv2d(x, w, b, stride=2, padding=1)),
        Case("conv2d_s1", [(2, 2, 6, 6), (3, 2, 3, 3), (3,)],
             lambda x, w, b: nano.conv2d(x, w, b), lambda x, w, b: F.conv2d(x, w, b)),
        Case("conv2d_naive", [(1, 2, 5, 5), (3, 2, 3, 3), (3,)],
             lambda x, w, b: nano.conv2d_naive(x, w, b, stride=1, padding=1),
             lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1)),
        Case("maxpool2d", [(2, 3, 6, 6)], lambda x: nano.maxpool2d(x, k=2), lambda x: F.max_pool2d(x, 2),
             sampler="distinct"),
        Case("maxpool2d_overlap", [(1, 2, 7, 7)], lambda x: nano.maxpool2d(x, k=3, stride=2),
             lambda x: F.max_pool2d(x, 3, 2), sampler="distinct"),
    ]


CASES: list[Case] = []


def cases() -> list[Case]:
    if not CASES:
        CASES.extend(_cases())
    return CASES


# every registered op must be exercised by at least one case
COVERS = {
    "add": "add", "sub": "sub", "mul": "mul", "div": "div", "pow": "pow", "neg": "neg", "exp": "exp",
    "log": "log", "relu": "relu", "tanh": "tanh", "sigmoid": "sigmoid", "matmul": "matmul",
    "reshape": "reshape", "transpose": "transpose", "getitem": "getitem", "sum": "sum", "mean": "mean",
    "max": "max", "logsumexp": "logsumexp", "softmax_cross_entropy": "softmax_cross_entropy",
    "conv2d": "conv2d", "conv2d_naive": "conv2d_naive", "maxpool2d": "maxpool2d",
}


def run_nano(case: Case, xs: list[np.ndarray], u: np.ndarray | None = None):
    """Forward and backward of <u, f(xs)> in Nano. Returns (out, grads)."""
    ts = [Tensor(x.copy(), requires_grad=True) for x in xs]
    out = case.nano_fn(*ts)
    u = np.ones(out.shape) if u is None else u
    (out * Tensor(u.astype(out.dtype))).sum().backward()
    return out.data, [t.grad if t.grad is not None else np.zeros_like(t.data) for t in ts]


def run_torch(case: Case, xs: list[np.ndarray], u: np.ndarray | None = None):
    torch, _ = _torch()
    ts = [torch.tensor(x.copy(), requires_grad=True) for x in xs]
    out = case.torch_fn(*ts)
    u = np.ones(tuple(out.shape)) if u is None else u
    (out * torch.tensor(u.astype(xs[0].dtype))).sum().backward()
    return out.detach().numpy(), [t.grad.numpy() if t.grad is not None else np.zeros_like(x) for t, x in zip(ts, xs)]
