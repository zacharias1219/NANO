"""2-D convolution via im2col, a naive-loop oracle, and max pooling (F7).

Layout is NCHW, weights are (out_channels, in_channels, kh, kw) as in PyTorch.
im2col turns every receptive field into a column so the convolution becomes one
batched matmul; its VJP is the same matmul transposed followed by col2im, the
adjoint of im2col, which scatter-adds columns back onto the padded image.
Derivations: report/derivations/DERIVATIONS.md section D8.
"""
from __future__ import annotations

import numpy as np

from .tensor import op


def _out(size: int, k: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - k) // stride + 1


def im2col(x, kh, kw, stride=1, pad=0):
    """(N,C,H,W) -> (N, C*kh*kw, OH*OW): one column per output pixel. [D8.1]"""
    n, c, h, w = x.shape
    oh, ow = _out(h, kh, stride, pad), _out(w, kw, stride, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride]
    return cols.reshape(n, c * kh * kw, oh * ow)


def col2im(cols, x_shape, kh, kw, stride=1, pad=0):
    """Adjoint of im2col: scatter-add columns back to (N,C,H,W). [D8.2]"""
    n, c, h, w = x_shape
    oh, ow = _out(h, kh, stride, pad), _out(w, kw, stride, pad)
    cols = cols.reshape(n, c, kh, kw, oh, ow)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride] += cols[:, :, i, j]
    return xp[:, :, pad : pad + h, pad : pad + w]


@op
def conv2d(x, w, b, stride: int = 1, padding: int = 0):
    """Y = W_mat @ im2col(X) + b; dW = G cols^T, dX = col2im(W_mat^T G), db = sum G. [D8.3]"""
    n, (oc, _, kh, kw) = x.shape[0], w.shape
    oh, ow = _out(x.shape[2], kh, stride, padding), _out(x.shape[3], kw, stride, padding)
    cols = im2col(x, kh, kw, stride, padding)
    wm = w.reshape(oc, -1)
    out = (wm @ cols + b[:, None]).reshape(n, oc, oh, ow)

    def vjp(g):
        g = g.reshape(n, oc, oh * ow)
        dw = (g @ cols.transpose(0, 2, 1)).sum(axis=0).reshape(w.shape)
        dx = col2im(wm.T @ g, x.shape, kh, kw, stride, padding)
        return dx, dw, g.sum(axis=(0, 2))

    return out, vjp


@op
def conv2d_naive(x, w, b, stride: int = 1, padding: int = 0):
    """Direct loops over (sample, out channel, row, col): the independent oracle for conv2d. [D8.3]"""
    n, (oc, _, kh, kw) = x.shape[0], w.shape
    oh, ow = _out(x.shape[2], kh, stride, padding), _out(x.shape[3], kw, stride, padding)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((n, oc, oh, ow), dtype=x.dtype)
    win = lambda i, j: (slice(i * stride, i * stride + kh), slice(j * stride, j * stride + kw))  # noqa: E731
    for s, o, i, j in np.ndindex(n, oc, oh, ow):
        out[s, o, i, j] = np.sum(xp[(s, slice(None), *win(i, j))] * w[o]) + b[o]

    def vjp(g):
        dxp, dw, db = np.zeros_like(xp), np.zeros_like(w), np.zeros_like(b)
        for s, o, i, j in np.ndindex(n, oc, oh, ow):
            dw[o] += g[s, o, i, j] * xp[(s, slice(None), *win(i, j))]
            dxp[(s, slice(None), *win(i, j))] += g[s, o, i, j] * w[o]
            db[o] += g[s, o, i, j]
        return dxp[:, :, padding : padding + x.shape[2], padding : padding + x.shape[3]], dw, db

    return out, vjp


@op
def maxpool2d(x, k: int = 2, stride: int | None = None):
    """Max over each k x k window; the VJP routes g to the first argmax. [D8.4]"""
    stride = stride or k
    n, c, h, w = x.shape
    oh, ow = _out(h, k, stride, 0), _out(w, k, stride, 0)
    cols = im2col(x.reshape(n * c, 1, h, w), k, k, stride).reshape(n * c, k * k, oh * ow)
    arg = cols.argmax(axis=1)
    out = np.take_along_axis(cols, arg[:, None], axis=1).reshape(n, c, oh, ow)

    def vjp(g):
        dcols = np.zeros_like(cols)
        np.put_along_axis(dcols, arg[:, None], g.reshape(n * c, 1, oh * ow), axis=1)
        return (col2im(dcols, (n * c, 1, h, w), k, k, stride).reshape(x.shape),)

    return out, vjp
