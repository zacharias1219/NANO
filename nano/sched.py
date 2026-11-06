"""Learning-rate schedules (F6): constant, step, cosine, linear warmup.

A schedule is a pure function of the step index ``t`` (0-based, the number of
optimizer steps already taken), so it can be tabulated, plotted and compared
with torch.optim.lr_scheduler without running an optimizer.
"""
from __future__ import annotations

import math


class Schedule:
    def __call__(self, t: int) -> float:
        raise NotImplementedError


class Constant(Schedule):
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, t):
        return self.lr


class Step(Schedule):
    """lr * gamma^floor(t / step_size), like torch StepLR."""

    def __init__(self, lr: float, step_size: int, gamma: float = 0.1):
        self.lr, self.step_size, self.gamma = lr, step_size, gamma

    def __call__(self, t):
        return self.lr * self.gamma ** (t // self.step_size)


class Cosine(Schedule):
    """eta_min + (lr - eta_min)(1 + cos(pi t / T)) / 2, like torch CosineAnnealingLR for t <= T."""

    def __init__(self, lr: float, t_max: int, eta_min: float = 0.0):
        self.lr, self.t_max, self.eta_min = lr, t_max, eta_min

    def __call__(self, t):
        t = min(t, self.t_max)
        return self.eta_min + (self.lr - self.eta_min) * (1 + math.cos(math.pi * t / self.t_max)) / 2


class Warmup(Schedule):
    """Linear warmup from ``start_factor * after(0)`` to ``after(0)`` over ``steps``, then ``after(t - steps)``.

    With ``after = Constant(lr)`` this equals torch LinearLR(start_factor, total_iters=steps).
    """

    def __init__(self, after: Schedule, steps: int, start_factor: float = 0.1):
        self.after, self.steps, self.start = after, steps, start_factor

    def __call__(self, t):
        if t >= self.steps:
            return self.after(t - self.steps)
        return self.after(0) * (self.start + (1 - self.start) * t / self.steps)


class LRScheduler:
    """Drives ``optimizer.lr`` from a schedule; call ``step()`` after each optimizer step."""

    def __init__(self, optimizer, schedule: Schedule):
        self.opt, self.schedule, self.t = optimizer, schedule, 0
        self.opt.lr = schedule(0)

    def step(self) -> None:
        self.t += 1
        self.opt.lr = self.schedule(self.t)


def make(name: str, lr: float, total: int, warmup: int = 0) -> Schedule:
    """Factory used by experiments: 'constant' | 'step' | 'cosine' | 'warmup_cosine'."""
    if name == "constant":
        return Constant(lr)
    if name == "step":
        return Step(lr, step_size=max(1, total // 3), gamma=0.1)
    if name == "cosine":
        return Cosine(lr, total)
    if name == "warmup_cosine":
        return Warmup(Cosine(lr, total - warmup), warmup)
    raise ValueError(f"unknown schedule {name!r}")
