"""Optimizers (F6): SGD with momentum / Nesterov, Adam, AdamW.

Update rules follow torch.optim exactly (same order of operations where it
matters for float64 parity, A5). Updates run on raw ndarrays outside the tape.
Derivations and the stability bound for SGD: DERIVATIONS.md section D10.
"""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr
        self.state: list[dict] = [{} for _ in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        for p, st in zip(self.params, self.state):
            if p.grad is not None:
                p.data = self._update(p.data, p.grad, st)

    def _update(self, w: np.ndarray, g: np.ndarray, st: dict) -> np.ndarray:
        raise NotImplementedError


class SGD(Optimizer):
    """v <- mu v + g;  w <- w - lr * (g + mu v if nesterov else v).  [D10.1, D10.2]"""

    def __init__(self, params, lr=0.1, momentum=0.0, nesterov=False, weight_decay=0.0, dampening=0.0):
        super().__init__(params, lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")
        self.momentum, self.nesterov, self.wd, self.dampening = momentum, nesterov, weight_decay, dampening

    def _update(self, w, g, st):
        if self.wd:
            g = g + self.wd * w
        if self.momentum:
            if "v" not in st:
                st["v"] = g.copy()
            else:
                st["v"] = self.momentum * st["v"] + (1 - self.dampening) * g
            g = g + self.momentum * st["v"] if self.nesterov else st["v"]
        return w - self.lr * g


class Adam(Optimizer):
    """Bias-corrected first and second moment estimates (Kingma and Ba, 2015). [D10.3]

    ``weight_decay`` is the coupled L2 form (added to the gradient), as in torch.optim.Adam.
    """

    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        (self.b1, self.b2), self.eps, self.wd = betas, eps, weight_decay

    def _update(self, w, g, st):
        if self.wd and self.decoupled:
            w = w * (1 - self.lr * self.wd)
        elif self.wd:
            g = g + self.wd * w
        m = st["m"] = self.b1 * st.get("m", 0.0) + (1 - self.b1) * g
        v = st["v"] = self.b2 * st.get("v", 0.0) + (1 - self.b2) * g * g
        bc1, bc2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        return w - (self.lr / bc1) * m / (np.sqrt(v) / np.sqrt(bc2) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay: w <- w (1 - lr wd), then the Adam step (Loshchilov, Hutter 2019). [D10.4]"""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr, betas, eps, weight_decay)


OPTIMIZERS = {"sgd": SGD, "adam": Adam, "adamw": AdamW}
