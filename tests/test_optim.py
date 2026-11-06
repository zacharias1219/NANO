"""A5: optimizers match torch.optim, per-step parameter difference < 1e-12 (float64).

Both sides receive identical gradients (a fixed function of the current parameters),
so the test isolates the update rule from autodiff.
"""
import numpy as np
import pytest
import torch

from nano import Tensor, optim, sched

A5_TOL = 1e-12
STEPS = 50


def grad_fn(w: np.ndarray, t: int) -> np.ndarray:
    """A deterministic, non-trivial gradient field: grad of a quartic plus a step-dependent term."""
    return 4 * w**3 - 2 * w + 0.1 * np.sin(t + w)


CONFIGS = [
    ("sgd", dict(lr=0.05)),
    ("sgd_wd", dict(lr=0.05, weight_decay=0.01)),
    ("momentum", dict(lr=0.05, momentum=0.9)),
    ("momentum_dampening", dict(lr=0.05, momentum=0.9, dampening=0.1)),
    ("nesterov", dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-3)),
    ("adam", dict(lr=0.01)),
    ("adam_l2", dict(lr=0.01, weight_decay=0.01, betas=(0.8, 0.99))),
    ("adamw", dict(lr=0.01, weight_decay=0.05)),
]


def make(name, params, kw, torch_side):
    base = name.split("_")[0]
    if base in ("sgd", "momentum", "nesterov"):
        return (torch.optim.SGD if torch_side else optim.SGD)(params, **kw)
    pair = {"adam": (torch.optim.Adam, optim.Adam), "adamw": (torch.optim.AdamW, optim.AdamW)}[base]
    return pair[0 if torch_side else 1](params, **kw)


def optimizer_max_diff(name, kw) -> float:
    """Worst per-step |w_nano - w_torch| over STEPS steps (also reported by experiments/collect.py)."""
    rng = np.random.default_rng(0)
    w0 = [rng.standard_normal((4, 3)), rng.standard_normal(3)]
    nano_p = [Tensor(w.copy(), requires_grad=True) for w in w0]
    torch_p = [torch.tensor(w.copy(), requires_grad=True) for w in w0]
    o_n, o_t = make(name, nano_p, kw, False), make(name, torch_p, kw, True)
    worst = 0.0
    for t in range(STEPS):
        for pn, pt in zip(nano_p, torch_p):
            g = grad_fn(pn.data, t)
            pn.grad, pt.grad = g, torch.tensor(grad_fn(pt.detach().numpy(), t))
        o_n.step()
        o_t.step()
        worst = max(worst, *(np.max(np.abs(pn.data - pt.detach().numpy())) for pn, pt in zip(nano_p, torch_p)))
    return float(worst)


@pytest.mark.parametrize("name,kw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_optimizer_matches_torch(name, kw):
    worst = optimizer_max_diff(name, kw)
    assert worst < A5_TOL, f"{name}: max per-step diff {worst:.3e}"


def torch_schedule(kind, steps):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    s = {
        "step": lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5),
        "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.001),
        "warmup": lambda: torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10),
    }[kind]()
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    return np.array(out)


@pytest.mark.parametrize("kind,ours", [
    ("step", sched.Step(0.1, 7, 0.5)),
    ("cosine", sched.Cosine(0.1, 40, 0.001)),
    ("warmup", sched.Warmup(sched.Constant(0.1), 10, start_factor=0.1)),
])
def test_schedules_match_torch(kind, ours):
    expected = torch_schedule(kind, 40)
    np.testing.assert_allclose([ours(t) for t in range(40)], expected, rtol=0, atol=1e-12)


def test_lr_scheduler_drives_optimizer():
    p = Tensor(np.zeros(1), requires_grad=True)
    opt = optim.SGD([p], lr=1.0)
    s = sched.LRScheduler(opt, sched.Step(1.0, 2, 0.5))
    lrs = []
    for _ in range(5):
        lrs.append(opt.lr)
        s.step()
    assert lrs == [1.0, 1.0, 0.5, 0.5, 0.25]
