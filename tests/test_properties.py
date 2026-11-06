"""Rung 4: property tests over random shapes and random broadcasting patterns."""
import numpy as np
import pytest
import torch

from nano import Tensor, unbroadcast
from nano.gradcheck import gradcheck, rel_error


def random_broadcast_pair(rng):
    """Two shapes that broadcast together: drop leading axes and set random axes to 1."""
    ndim = rng.integers(1, 5)
    full = tuple(int(s) for s in rng.integers(1, 5, size=ndim))
    def squash(shape):
        shape = list(shape[rng.integers(0, len(shape) + 1):]) or [1]
        return tuple(1 if rng.random() < 0.4 else s for s in shape)
    return squash(full), squash(full)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("opname", ["add", "sub", "mul", "div"])
def test_broadcasting_matches_torch_and_fd(seed, opname):
    rng = np.random.default_rng(seed)
    sa, sb = random_broadcast_pair(rng)
    a, b = rng.standard_normal(sa), rng.uniform(0.5, 2.0, sb)
    f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
         "mul": lambda x, y: x * y, "div": lambda x, y: x / y}[opname]
    ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
    out = f(ta, tb)
    u = rng.standard_normal(out.shape)
    (out * Tensor(u)).sum().backward()
    pa, pb = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
    (f(pa, pb) * torch.tensor(u)).sum().backward()
    assert ta.grad.shape == sa and tb.grad.shape == sb
    assert rel_error(ta.grad, pa.grad.numpy()) < 1e-12 and rel_error(tb.grad, pb.grad.numpy()) < 1e-12
    assert max(gradcheck(f, Tensor(a, requires_grad=True), Tensor(b, requires_grad=True), seed=seed)) < 1e-7


@pytest.mark.parametrize("seed", range(30))
def test_add_gradient_sums_to_upstream(seed):
    """Invariant: for z = a + b, sum(dL/db) = sum(dL/dz) whatever the broadcast pattern."""
    rng = np.random.default_rng(seed)
    sa, sb = random_broadcast_pair(rng)
    a, b = Tensor(rng.standard_normal(sa), requires_grad=True), Tensor(rng.standard_normal(sb), requires_grad=True)
    out = a + b
    g = rng.standard_normal(out.shape)
    out.backward(g)
    assert np.isclose(b.grad.sum(), g.sum()) and np.isclose(a.grad.sum(), g.sum())


@pytest.mark.parametrize("seed", range(30))
def test_unbroadcast_is_adjoint_of_broadcast(seed):
    """<broadcast(x), g> == <x, unbroadcast(g)> for every shape pair."""
    rng = np.random.default_rng(seed)
    small, _ = random_broadcast_pair(rng)
    big = tuple(int(s) for s in rng.integers(1, 4, size=rng.integers(0, 3))) + tuple(
        int(rng.integers(2, 4)) if s == 1 else s for s in small)
    x, g = rng.standard_normal(small), rng.standard_normal(big)
    lhs = np.sum(np.broadcast_to(x, big) * g)
    assert np.isclose(lhs, np.sum(x * unbroadcast(g, small)))


@pytest.mark.parametrize("seed", range(20))
def test_reductions_random_axes(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in rng.integers(1, 5, size=rng.integers(1, 4)))
    axis = int(rng.integers(0, len(shape)))
    keep = bool(rng.integers(0, 2))
    for name in ["sum", "mean", "logsumexp"]:
        f = lambda x: getattr(x, name)(axis=axis, keepdims=keep)
        assert max(gradcheck(f, Tensor(rng.standard_normal(shape), requires_grad=True))) < 1e-7
