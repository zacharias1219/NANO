"""A6: conv2d via im2col equals the naive-loop reference, max abs diff < 1e-12 (forward and backward)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from nano import Tensor, conv2d, conv2d_naive, maxpool2d
from nano.conv import col2im, im2col

A6_TOL = 1e-12
GEOMETRIES = [  # (N, C, H, W, OC, k, stride, pad)
    (2, 3, 7, 7, 4, 3, 1, 0), (2, 3, 7, 7, 4, 3, 2, 1), (1, 1, 5, 6, 2, 2, 1, 0),
    (3, 2, 8, 8, 3, 5, 1, 2), (1, 4, 9, 9, 2, 3, 3, 1), (2, 1, 6, 6, 1, 1, 1, 0),
]


def run(fn, x, w, b, stride, pad, g):
    tx, tw, tb = (Tensor(a.copy(), requires_grad=True) for a in (x, w, b))
    out = fn(tx, tw, tb, stride=stride, padding=pad)
    out.backward(g)
    return out.data, tx.grad, tw.grad, tb.grad


@pytest.mark.parametrize("geom", GEOMETRIES, ids=str)
def test_im2col_matches_naive_loop(geom):
    n, c, h, w_, oc, k, s, p = geom
    rng = np.random.default_rng(sum(geom))
    x, w, b = rng.standard_normal((n, c, h, w_)), rng.standard_normal((oc, c, k, k)), rng.standard_normal(oc)
    oh, ow = (h + 2 * p - k) // s + 1, (w_ + 2 * p - k) // s + 1
    g = rng.standard_normal((n, oc, oh, ow))
    fast, slow = run(conv2d, x, w, b, s, p, g), run(conv2d_naive, x, w, b, s, p, g)
    for a, r in zip(fast, slow):
        assert np.max(np.abs(a - r)) < A6_TOL


@pytest.mark.parametrize("geom", GEOMETRIES, ids=str)
def test_conv_matches_torch(geom):
    n, c, h, w_, oc, k, s, p = geom
    rng = np.random.default_rng(1 + sum(geom))
    x, w, b = rng.standard_normal((n, c, h, w_)), rng.standard_normal((oc, c, k, k)), rng.standard_normal(oc)
    tx, tw, tb = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    out = F.conv2d(tx, tw, tb, stride=s, padding=p)
    g = rng.standard_normal(tuple(out.shape))
    out.backward(torch.tensor(g))
    ours = run(conv2d, x, w, b, s, p, g)
    for a, r in zip(ours, (out.detach(), tx.grad, tw.grad, tb.grad)):
        assert np.max(np.abs(a - r.numpy())) < A6_TOL


@pytest.mark.parametrize("seed", range(5))
def test_col2im_is_adjoint_of_im2col(seed):
    """<im2col(x), C> == <x, col2im(C)>: the identity that makes the conv backward correct."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 3, 7, 6))
    cols = im2col(x, 3, 2, stride=2, pad=1)
    c = rng.standard_normal(cols.shape)
    assert np.isclose(np.sum(cols * c), np.sum(x * col2im(c, x.shape, 3, 2, stride=2, pad=1)))


@pytest.mark.parametrize("k,stride", [(2, None), (3, 2), (2, 1)])
def test_maxpool_matches_torch(k, stride):
    rng = np.random.default_rng(k)
    x = rng.standard_normal((2, 3, 8, 8))
    tx = torch.tensor(x, requires_grad=True)
    out = F.max_pool2d(tx, k, stride)
    g = rng.standard_normal(tuple(out.shape))
    out.backward(torch.tensor(g))
    nx = Tensor(x, requires_grad=True)
    o = maxpool2d(nx, k=k, stride=stride)
    o.backward(g)
    assert np.max(np.abs(o.data - out.detach().numpy())) < A6_TOL
    assert np.max(np.abs(nx.grad - tx.grad.numpy())) < A6_TOL
