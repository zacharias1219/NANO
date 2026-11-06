"""Finite-difference gradient checking and the layerwise first-bad-op report (F8).

Two tools:

* ``gradcheck(fn, *inputs)`` compares the full reverse-mode gradient of a scalar
  projection ``<u, fn(inputs)>`` with central differences (acceptance A1).
* ``layerwise_report(loss)`` walks the tape of a real forward pass in execution
  order and checks every node's *local* vector-Jacobian product in isolation,
  re-running only that op on perturbed inputs. Because each op is checked on
  its own, the first failing row names the broken op even when every gradient
  downstream of it is also wrong (acceptance A7).

Error metric (the precision contract, PRECISION.md): the scale-aware max error
``max|a - n| / max(max|a|, max|n|, tiny)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tensor import Tensor, unbroadcast

H_DEFAULT = 1e-6  # central-difference step; justified by the E6 error curve
KINKED = {"relu", "max", "maxpool2d"}  # FD is invalid at kinks and ties: check these at a jittered point


def rel_error(a, n) -> float:
    a, n = np.asarray(a, np.float64), np.asarray(n, np.float64)
    if a.shape != n.shape:
        return float("inf")
    scale = max(np.max(np.abs(a), initial=0.0), np.max(np.abs(n), initial=0.0))
    return 0.0 if scale == 0 else float(np.max(np.abs(a - n)) / scale)


def numerical_grad(f, x: np.ndarray, h: float = H_DEFAULT, idx=None) -> np.ndarray:
    """Central difference (f(x+h e_i) - f(x-h e_i)) / 2h for each flat index in ``idx``."""
    if not x.flags.c_contiguous:
        raise ValueError("numerical_grad perturbs x in place and needs a C-contiguous array")
    flat = x.reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size) if idx is None else idx:
        old = flat[i]
        flat[i] = old + h
        fp = f()
        flat[i] = old - h
        fm = f()
        flat[i] = old
        grad[i] = (fp - fm) / (2 * h)
    return grad.reshape(x.shape)


def gradcheck(fn, *inputs: Tensor, h: float = H_DEFAULT, seed: int = 0) -> list[float]:
    """Per-input relative error of the analytic gradient of <u, fn(*inputs)> against FD."""
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    u = np.random.default_rng(seed).standard_normal(out.shape)
    (out * Tensor(u, dtype=out.dtype)).sum().backward()
    errs = []
    for t in inputs:
        if not t.requires_grad:
            continue
        analytic = np.zeros_like(t.data) if t.grad is None else t.grad
        f = lambda: float(np.sum(u * fn(*inputs).data))  # noqa: E731
        errs.append(rel_error(analytic, numerical_grad(f, t.data, h)))
    return errs


@dataclass
class NodeCheck:
    index: int
    op: str
    scope: str
    shape: tuple
    error: float
    ok: bool
    note: str = ""


def check_node(node: Tensor, h: float = H_DEFAULT, max_elems: int = 64, rng=None) -> tuple[float, str]:
    """Check one tape node's local VJP against FD of <g, op(inputs)> (inputs cast to float64)."""
    rng = rng or np.random.default_rng(0)
    op, parents, kw, _ = node._ctx
    xs = [np.array(p.data, dtype=np.float64, order="C") for p in parents]
    if op.name in KINKED:  # move off exact ties (e.g. maxpool over ReLU zeros), see PRECISION.md
        xs = [x + 1e-3 * (np.std(x) or 1.0) * rng.standard_normal(x.shape) for x in xs]
    g = rng.standard_normal(node.shape)
    try:
        _, vjp = op.fn(*xs, **kw)
        analytic = vjp(g)
    except Exception as e:  # a crashing backward is also a located bug
        return float("inf"), f"backward raised {type(e).__name__}: {e}"
    worst = 0.0
    for k, p in enumerate(parents):
        if not p.requires_grad or analytic[k] is None:
            continue
        try:
            a = unbroadcast(np.asarray(analytic[k], np.float64), xs[k].shape)
        except ValueError as e:
            return float("inf"), str(e)
        idx = rng.choice(xs[k].size, size=min(max_elems, xs[k].size), replace=False)
        f = lambda: float(np.sum(g * op.fn(*xs, **kw)[0]))  # noqa: E731
        num = numerical_grad(f, xs[k], h, idx).reshape(-1)[idx]
        worst = max(worst, rel_error(a.reshape(-1)[idx], num))
    return worst, ""


def layerwise_report(loss: Tensor, tol: float = 1e-6, h: float = H_DEFAULT, max_elems: int = 64, seed: int = 0):
    """Check every op node of ``loss``'s tape in forward order; return (rows, first_bad_row)."""
    rng = np.random.default_rng(seed)
    nodes = sorted((n for n in loss.topo() if n._ctx is not None), key=lambda n: n.seq)
    rows = []
    for i, n in enumerate(nodes):
        err, note = check_node(n, h, max_elems, rng)
        rows.append(NodeCheck(i, n._ctx[0].name, n.scope, n.shape, err, err < tol, note))
    first_bad = next((r for r in rows if not r.ok), None)
    return rows, first_bad


def format_report(rows, first_bad) -> str:
    lines = [f"{'#':>3}  {'op':<22}{'layer':<28}{'shape':<18}{'rel err':>10}  status"]
    for r in rows:
        mark = "ok" if r.ok else "FAIL" + (" <- first bad op" if r is first_bad else "")
        lines.append(f"{r.index:>3}  {r.op:<22}{r.scope:<28}{str(r.shape):<18}{r.error:>10.2e}  {mark}")
    verdict = f"first bad op: {first_bad.op} in {first_bad.scope or '<top>'}" if first_bad else "all ops pass"
    return "\n".join([*lines, verdict])
