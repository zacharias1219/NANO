"""E4 / A7: inject five silent bugs, one at a time, and check the layerwise checker names the op.

Each bug keeps every shape correct, so training still "runs" and the loss still goes
down for a while: the classic way a broken backward survives. For each bug we show
(1) the model-level symptom (whole-model gradcheck fails, but does not say where) and
(2) the layerwise report, which must name the injected op as the first bad op.
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import contextlib
import time

import numpy as np

import evalkit
import nano
from experiments.data import load
from nano import OPS, Tensor, nn, softmax_cross_entropy
from nano.gradcheck import format_report, layerwise_report, numerical_grad, rel_error


@contextlib.contextmanager
def patched_op(name: str, fn):
    old = OPS[name].fn
    OPS[name].fn = fn
    try:
        yield
    finally:
        OPS[name].fn = old


@contextlib.contextmanager
def patched_attr(module, attr: str, value):
    old = getattr(module, attr)
    setattr(module, attr, value)
    try:
        yield
    finally:
        setattr(module, attr, old)


# --------------------------------------------------------------------------------- the bugs
def bug_add_broadcast():
    """add: reduces the broadcast bias gradient with a mean instead of a sum."""
    def add(a, b):
        def vjp(g):
            gb = g.mean(axis=tuple(range(g.ndim - b.ndim))) if b.ndim < g.ndim else g
            return g, gb
        return a + b, vjp
    return patched_op("add", add)


def bug_transpose_reshape():
    """transpose: backward uses reshape instead of the inverse transpose (the 'missing transpose')."""
    def transpose(a, axes=None):
        return a.transpose(axes), lambda g: (g.reshape(a.shape),)
    return patched_op("transpose", transpose)


def bug_col2im_off_by_one():
    """conv2d: its col2im step scatters every column one pixel to the right (off-by-one window index)."""
    good = nano.conv.conv2d.fn

    def conv2d(x, w, b, stride=1, padding=0):
        out, vjp = good(x, w, b, stride=stride, padding=padding)

        def bad_vjp(g):
            dx, dw, db = vjp(g)
            return np.roll(dx, 1, axis=-1), dw, db
        return out, bad_vjp
    return patched_op("conv2d", conv2d)


def bug_shared_col2im():
    """The same off-by-one in the shared col2im helper, which maxpool2d's backward also uses."""
    good = nano.conv.col2im

    def col2im(cols, x_shape, kh, kw, stride=1, pad=0):
        return np.roll(good(cols, x_shape, kh, kw, stride, pad), 1, axis=-1)
    return patched_attr(nano.conv, "col2im", col2im)


def bug_tanh_derivative():
    """tanh: uses 1 - y instead of 1 - y^2."""
    def tanh(a):
        out = np.tanh(a)
        return out, lambda g: (g * (1 - out),)
    return patched_op("tanh", tanh)


def bug_ce_missing_mean():
    """softmax_cross_entropy: forward is a mean but the backward forgets to divide by N."""
    def softmax_cross_entropy(logits, y):
        n = logits.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        p = np.exp(z - lse[:, None])

        def vjp(g):
            d = p.copy()
            d[np.arange(n), y] -= 1.0
            return (g * d,)
        return np.mean(lse - z[np.arange(n), y]), vjp
    return patched_op("softmax_cross_entropy", softmax_cross_entropy)


BUGS = [  # (id, description, expected op, injector)
    ("B1", "wrong broadcast sum in add (mean instead of sum)", "add", bug_add_broadcast),
    ("B2", "missing transpose in transpose backward (reshape)", "transpose", bug_transpose_reshape),
    ("B3", "off-by-one in col2im", "conv2d", bug_col2im_off_by_one),
    ("B4", "tanh derivative 1 - y instead of 1 - y^2", "tanh", bug_tanh_derivative),
    ("B5", "cross entropy backward missing 1/N", "softmax_cross_entropy", bug_ce_missing_mean),
]
# Informational, not counted in A7: a bug in a helper shared by two ops surfaces at the first op that
# exercises it with a gradient-carrying input. The first conv's input (the images) needs no gradient,
# so its dx path never runs and the checker correctly points at maxpool2d.
EXTRA = [("B3-shared", "off-by-one in the shared col2im helper", "maxpool2d", bug_shared_col2im)]


def model_and_batch():
    rng = np.random.default_rng(0)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(4, 8, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                          nn.Linear(8 * 7 * 7, 32, init="xavier", rng=rng), nn.Tanh(),
                          nn.Linear(32, 10, init="xavier", rng=rng))
    X, y = load("mnist", "train", n=4, flat=False, seed=3)
    return model, Tensor(X), y


def whole_model_check(model, x, y, n_params: int = 24) -> float:
    """Model-level symptom: compare param grads with FD of the loss. Says *that*, not *where*."""
    model.zero_grad()
    softmax_cross_entropy(model(x), y=y).backward()
    rng, worst = np.random.default_rng(1), 0.0
    for p in model.parameters():
        idx = rng.choice(p.size, size=min(n_params, p.size), replace=False)
        f = lambda: softmax_cross_entropy(model(x), y=y).item()  # noqa: E731
        num = numerical_grad(f, p.data, idx=idx).reshape(-1)[idx]
        worst = max(worst, rel_error(p.grad.reshape(-1)[idx], num))
    return worst


def run_case(injector=None):
    model, x, y = model_and_batch()
    with injector() if injector else contextlib.nullcontext():
        symptom = whole_model_check(model, x, y)
        loss = softmax_cross_entropy(model(x), y=y)
        rows, first_bad = layerwise_report(loss)
    return symptom, rows, first_bad


def main():
    t0 = time.perf_counter()
    symptom, rows, first_bad = run_case()
    print("control (no bug):\n" + format_report(rows, first_bad))
    results = [{"bug": "control", "description": "no bug injected", "expected": None,
                "named": first_bad.op if first_bad else None, "located": first_bad is None,
                "whole_model_rel_err": symptom, "first_bad_scope": None}]
    reports = {"control": format_report(rows, first_bad)}
    for bid, desc, expected, injector in BUGS + EXTRA:
        symptom, rows, first_bad = run_case(injector)
        named = first_bad.op if first_bad else None
        print(f"\n{bid}: {desc}\n  whole-model rel err {symptom:.2e}; checker names: {named} (expected {expected})")
        print(format_report(rows, first_bad))
        reports[bid] = format_report(rows, first_bad)
        results.append({"bug": bid, "description": desc, "expected": expected, "named": named,
                        "located": named == expected, "whole_model_rel_err": symptom,
                        "first_bad_scope": first_bad.scope if first_bad else None,
                        "first_bad_error": first_bad.error if first_bad else None})
    located = sum(r["located"] for r in results[1:1 + len(BUGS)])
    print(f"\nA7: located {located} of {len(BUGS)} injected bugs; control clean: {results[0]['located']}")
    evalkit.write_fragment("E4", {
        "title": "E4 bug-injection demo: the layerwise checker names the broken op",
        "config": {"model": "conv3x3(4)-relu-pool-conv3x3(8)-relu-pool-linear(32)-tanh-linear(10)",
                   "batch": "4 MNIST training images", "fd_step": nano.gradcheck.H_DEFAULT, "tolerance": 1e-6},
        "metrics": {"bugs_located": evalkit.metric(located, n=len(BUGS), unit="bugs"),
                    "control_clean": evalkit.metric(float(results[0]["located"]))},
        "tables": {"bugs": results, "reports": reports},
        "notes": ["B3-shared is informational and not counted in A7: see the comment on EXTRA in bughunt.py."],
        "runtime_s": time.perf_counter() - t0,
    }, common.ROOT / "results")


if __name__ == "__main__":
    main()
