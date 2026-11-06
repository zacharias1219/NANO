"""Rung 1-2 engine semantics: accumulation, shared nodes, deep graphs, no_grad, dtype."""
import numpy as np
import pytest

import nano
from nano import Tensor, no_grad


def test_reused_node_accumulates():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_diamond_graph():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    a, b = x.exp(), x * 2
    (a * b).sum().backward()
    np.testing.assert_allclose(x.grad, np.exp(x.data) * 2 * x.data + 2 * np.exp(x.data))


def test_grad_accumulates_across_backward_calls_like_pytorch():
    x = Tensor(np.ones(3), requires_grad=True)
    (x * 2).sum().backward()
    (x * 3).sum().backward()
    np.testing.assert_allclose(x.grad, 5.0)


def test_deep_chain_has_no_recursion_limit():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_no_grad_records_nothing():
    x = Tensor(np.ones(2), requires_grad=True)
    with no_grad():
        y = x * 2
    assert not y.requires_grad and y.is_leaf


def test_retain_grad_on_intermediate():
    x = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    h = (x * 3).retain_grad()
    (h * h).sum().backward()
    np.testing.assert_allclose(h.grad, 2 * h.data)


def test_float32_stays_float32():
    x = Tensor(np.ones((2, 2), np.float32), requires_grad=True)
    y = (x * 2.0 + 1.0).tanh().sum()
    y.backward()
    assert y.dtype == np.float32 and x.grad.dtype == np.float32


def test_numpy_on_the_left_defers_to_tensor():
    x = Tensor(np.ones(2), requires_grad=True)
    y = np.array([2.0, 3.0]) * x
    assert isinstance(y, Tensor)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 3.0])


def test_op_decorator_adds_a_differentiable_op_in_a_few_lines():
    """F10: a new op is a forward and a backward."""
    @nano.op
    def softplus(a):
        """z = log(1 + e^a); dz/da = sigmoid(a)."""
        return np.logaddexp(0.0, a), lambda g: (g / (1 + np.exp(-a)),)

    x = Tensor(np.linspace(-3, 3, 7), requires_grad=True)
    assert max(nano.gradcheck.gradcheck(softplus, x)) < 1e-7
    del nano.OPS["softplus"]


def test_backward_requires_grad():
    with pytest.raises(RuntimeError):
        Tensor(1.0).backward()
