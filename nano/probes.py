"""Probes (F9): per-layer gradient norms and activation statistics.

``ActivationProbe`` attaches forward hooks to activation modules and records,
per layer, the mean and std of the output, the fraction of dead units (ReLU
units that output zero for every example in the batch) and the fraction of
saturated units (|tanh| > 0.99, sigmoid outside [0.01, 0.99]).
``grad_norms`` reads the Frobenius norm of every weight gradient after
backward; ``grad_norm_ratio`` is the E2 predictor r = log10(first / last).
"""
from __future__ import annotations

import numpy as np

from .nn import Conv2d, Linear, Module, ReLU, Sigmoid, Tanh


def weight_layers(model: Module) -> list[tuple[str, Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (Linear, Conv2d))]


def grad_norms(model: Module) -> list[float]:
    """||dL/dW||_F for each Linear / Conv2d layer, input side first."""
    return [float(np.linalg.norm(m.weight.grad)) if m.weight.grad is not None else 0.0 for _, m in weight_layers(model)]


def grad_norm_ratio(model: Module) -> float:
    """r = log10(||dL/dW_first|| / ||dL/dW_last||); r << 0 means vanishing toward the input."""
    norms = grad_norms(model)
    with np.errstate(divide="ignore"):
        return float(np.log10(norms[0]) - np.log10(norms[-1]))


def _stats(module: Module, out: np.ndarray) -> dict:
    flat = out.reshape(out.shape[0], -1)
    s = {"mean": float(flat.mean()), "std": float(flat.std()), "dead": 0.0, "saturated": 0.0}
    if isinstance(module, ReLU):
        s["dead"] = float(np.mean(np.all(flat <= 0, axis=0)))
    elif isinstance(module, Tanh):
        s["saturated"] = float(np.mean(np.abs(flat) > 0.99))
    elif isinstance(module, Sigmoid):
        s["saturated"] = float(np.mean((flat < 0.01) | (flat > 0.99)))
    s["finite"] = bool(np.all(np.isfinite(flat)))
    return s


class ActivationProbe:
    """Records activation statistics of every activation module on each forward pass.

    >>> probe = ActivationProbe(model); model(x); probe.records[-1]  # list of per-layer dicts
    """

    def __init__(self, model: Module, keep: int = 1):
        self.records: list[list[dict]] = []
        self._current: list[dict] = []
        self.keep = keep
        self._removers = []
        acts = [m for _, m in model.named_modules() if isinstance(m, (ReLU, Tanh, Sigmoid))]
        for m in acts:
            self._removers.append(m.register_forward_hook(self._hook))
        self._last = acts[-1] if acts else None

    def _hook(self, module, x, out) -> None:
        self._current.append({"layer": module._name, **_stats(module, out.data)})
        if module is self._last:
            self.records = (self.records + [self._current])[-self.keep :]
            self._current = []

    def remove(self) -> None:
        for r in self._removers:
            r()

    def summary(self) -> dict:
        """Last record reduced to model-level numbers."""
        layers = self.records[-1] if self.records else []
        return {
            "max_dead": max((s["dead"] for s in layers), default=0.0),
            "max_saturated": max((s["saturated"] for s in layers), default=0.0),
            "finite": all(s["finite"] for s in layers),
            "std": [s["std"] for s in layers],
        }
