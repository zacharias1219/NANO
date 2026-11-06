"""E1: parity against PyTorch per op, per step and per run, in float64 and float32.

Produces the numbers for acceptance criteria A1 (per-op FD error), A2 (per-op PyTorch
error), A3 (one training step) and A4 (a 300-step run), the float32 drift plot, and
the M4 CNN results on MNIST and Fashion-MNIST.
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import time

import numpy as np
import torch

import evalkit
from experiments.data import load
from experiments.op_cases import cases, run_nano, run_torch
from experiments.torch_mirror import mirror, torch_params
from nano import Tensor, nn, no_grad, optim, softmax_cross_entropy
from nano.gradcheck import gradcheck, rel_error

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
SEED = common.SEEDS["parity_seed"]
RUN_STEPS = 60 if common.QUICK else 300


def op_table(n_seeds: int = 5) -> list[dict]:
    rows = []
    for case in cases():
        fd, out64, grad64, out32, grad32 = [], [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            xs = case.inputs(rng)
            fd.append(max(gradcheck(case.nano_fn, *[Tensor(x.copy(), requires_grad=True) for x in xs], seed=s)))
            u = rng.standard_normal(run_nano(case, xs)[0].shape)
            on, gn = run_nano(case, xs, u)
            ot, gt = run_torch(case, xs, u)
            out64.append(rel_error(on, ot))
            grad64.append(max(rel_error(a, b) for a, b in zip(gn, gt)))
            xs32 = [x.astype(np.float32) for x in xs]
            on, gn = run_nano(case, xs32, u.astype(np.float32))
            ot, gt = run_torch(case, xs32, u.astype(np.float32))
            out32.append(rel_error(on, ot))
            grad32.append(max(rel_error(a, b) for a, b in zip(gn, gt)))
        rows.append({"case": case.name, "fd_rel_err": max(fd), "torch64_out": max(out64), "torch64_grad": max(grad64),
                     "torch32_out": max(out32), "torch32_grad": max(grad32), "n_seeds": n_seeds})
        print(f"  {case.name:24s} FD {max(fd):.1e}  f64 {max(grad64):.1e}  f32 {max(grad32):.1e}", flush=True)
    return rows


# ------------------------------------------------------------------------------ models
def make_mlp(rng):
    return nn.Sequential(nn.Linear(784, 128, init="he", rng=rng), nn.ReLU(), nn.Linear(128, 64, init="xavier", rng=rng),
                         nn.Tanh(), nn.Linear(64, 10, init="xavier", rng=rng))


def make_cnn(rng):
    """Two conv layers (PRD open question resolved: 2 x conv3x3 + maxpool, batch 64)."""
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(16 * 7 * 7, 10, init="xavier", rng=rng))


SETUPS = {  # name: (builder, dataset, flat, batch, lr)
    "mlp_mnist": (make_mlp, "mnist", True, 64, 0.05),
    "cnn_mnist": (make_cnn, "mnist", False, 64, 0.01),  # 0.05 killed every ReLU (docs/DEVIATIONS.md D-07)
    "cnn_fashion": (make_cnn, "fashion", False, 64, 0.01),
}


def run_pair(setup: str, dtype, steps: int, n_train: int = 10_000):
    """Train Nano and PyTorch side by side from identical weights and batches."""
    build, ds, flat, batch, lr = SETUPS[setup]
    X, y = load(ds, "train", n=n_train, flat=flat)
    X = X.astype(dtype)
    model = build(np.random.default_rng(SEED)).astype(dtype)
    tmodel = mirror(model, dtype=torch.float64 if dtype == np.float64 else torch.float32)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    topt = torch.optim.SGD(tmodel.parameters(), lr=lr, momentum=0.9)
    order = common.batch_schedule(len(y), batch, steps, SEED)
    ln, lt, first_step = [], [], {}
    for t, idx in enumerate(order):
        loss = softmax_cross_entropy(model(Tensor(X[idx])), y=y[idx])
        opt.zero_grad()
        loss.backward()
        tloss = torch.nn.functional.cross_entropy(tmodel(torch.from_numpy(X[idx])), torch.from_numpy(y[idx]))
        topt.zero_grad()
        tloss.backward()
        if t == 0:
            tgrads = [p.grad.numpy() for p in tmodel.parameters()]
            first_step["grad_abs"] = max(float(np.max(np.abs(p.grad - g))) for p, g in zip(model.parameters(), tgrads))
        opt.step()
        topt.step()
        if t == 0:
            first_step["param_abs"] = max(float(np.max(np.abs(p.data - q)))
                                          for p, q in zip(model.parameters(), torch_params(tmodel)))
            first_step["loss_abs"] = abs(loss.item() - tloss.item())
        ln.append(loss.item())
        lt.append(tloss.item())
    final_param = max(float(np.max(np.abs(p.data - q))) for p, q in zip(model.parameters(), torch_params(tmodel)))
    return np.array(ln), np.array(lt), first_step, final_param, model


def accuracy(model, ds, flat) -> float:
    X, y = load(ds, "test", flat=flat)
    with no_grad():
        preds = np.concatenate([model(Tensor(X[i:i + 1000])).data.argmax(1) for i in range(0, len(y), 1000)])
    return float(np.mean(preds == y))


def main():
    t0 = time.perf_counter()
    print("E1 per-op parity table")
    ops = op_table(2 if common.QUICK else 5)
    runs, curves = {}, {}
    for setup in SETUPS:
        for dname, dt in (("float64", np.float64), ("float32", np.float32)):
            ln, lt, first, final_param, model = run_pair(setup, dt, RUN_STEPS)
            diff = np.abs(ln - lt)
            runs[f"{setup}_{dname}"] = {
                "steps": RUN_STEPS, "one_step_param_abs": first["param_abs"], "one_step_grad_abs": first["grad_abs"],
                "one_step_loss_abs": first["loss_abs"], "run_loss_abs_max": float(diff.max()),
                "run_final_param_abs": final_param, "loss_first": float(ln[0]), "loss_last": float(ln[-1]),
                "test_acc_after_run": accuracy(model, SETUPS[setup][1], SETUPS[setup][2]) if dname == "float64" else None,
            }
            curves[f"{setup}_{dname}"] = (ln, lt)
            print(f"  {setup} {dname}: step param {first['param_abs']:.2e}  run loss max {diff.max():.2e}", flush=True)

    # float32 vs float64 in Nano itself: the drift a precision claim has to live with
    for setup in SETUPS:
        runs[f"{setup}_f32_vs_f64"] = {"run_loss_abs_max": float(np.max(np.abs(curves[f"{setup}_float32"][0]
                                                                                 - curves[f"{setup}_float64"][0])))}

    figures = [drift_figure(curves), op_figure(ops)]
    f64 = [k for k in runs if k.endswith("_float64")]
    metrics = {
        "A1_max_fd_rel_err": evalkit.metric(max(r["fd_rel_err"] for r in ops), n=len(ops), method="max over ops x seeds"),
        "A2_max_torch_rel_err_f64": evalkit.metric(max(max(r["torch64_out"], r["torch64_grad"]) for r in ops), n=len(ops)),
        "A3_one_step_param_abs_f64": evalkit.metric(max(runs[k]["one_step_param_abs"] for k in f64), n=len(f64)),
        "A4_run_loss_abs_max_f64": evalkit.metric(max(runs[k]["run_loss_abs_max"] for k in f64), n=len(f64)),
        "float32_run_loss_abs_max": evalkit.metric(max(runs[k]["run_loss_abs_max"] for k in runs if k.endswith("_float32"))),
        "max_torch_rel_err_f32": evalkit.metric(max(max(r["torch32_out"], r["torch32_grad"]) for r in ops), n=len(ops)),
    }
    evalkit.write_fragment("E1", {
        "title": "E1 parity vs PyTorch: per op, per step, per run; float64 and float32",
        "config": {"run_steps": RUN_STEPS, "optimizer": "SGD momentum 0.9 (lr per setup below)", "seed": SEED, "torch_threads": 1,
                   "setups": {k: {"dataset": v[1], "batch": v[3], "lr": v[4]} for k, v in SETUPS.items()}},
        "metrics": metrics, "tables": {"ops": ops, "runs": runs}, "figures": figures,
        "runtime_s": time.perf_counter() - t0,
        "notes": ["Error metric: max|a-b| / max(max|a|, max|b|) per tensor (PRECISION.md).",
                  "float32 parity is expected to drift and is reported, not hidden."],
    }, common.ROOT / "results")


def drift_figure(curves) -> str:
    common.RAW.mkdir(parents=True, exist_ok=True)
    np.savez(common.RAW / "e1_curves.npz", **{f"{k}__{i}": v[i] for k, v in curves.items() for i in (0, 1)})
    plt = common.plt()
    fig, axes = plt.subplots(1, len(SETUPS), figsize=(13, 3.8), sharey=True)
    series = lambda setup: [  # noqa: E731
        ("float64: Nano vs PyTorch", common.PALETTE["float64"], "-",
         np.abs(np.subtract(*curves[f"{setup}_float64"]))),
        ("float32: Nano vs PyTorch", common.PALETTE["float32"], "-",
         np.abs(np.subtract(*curves[f"{setup}_float32"]))),
        ("Nano float32 vs Nano float64", common.PALETTE["grey"], "--",
         np.abs(curves[f"{setup}_float32"][0] - curves[f"{setup}_float64"][0]))]
    for ax, setup in zip(axes, SETUPS):
        for label, color, ls, d in series(setup):
            t = np.arange(len(d))
            nz = d > 0
            ax.scatter(t[nz], d[nz], s=3, color=color, alpha=0.25, linewidths=0)
            ax.plot(t, np.maximum.accumulate(np.maximum(d, 1e-18)), color=color, lw=2, ls=ls, label=label)
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1)
        ax.axhline(1e-8, color="k", lw=0.8, ls=":")
        ax.text(len(d) * 0.02, 2e-8, "A4 bound 1e-8", fontsize=8)
        ax.set_title(setup.replace("_", " "))
        ax.set_xlabel("training step")
    axes[0].set_ylabel("|loss difference|: worst so far (line), per step (dots)")
    axes[1].legend(fontsize=8, loc="center right")
    fig.suptitle("E1 run parity: float64 stays at rounding level, float32 drifts by orders of magnitude", y=1.03)
    return common.save(fig, "e1_float32_drift.png")


def op_figure(ops) -> str:
    plt = common.plt()
    names = [r["case"] for r in ops]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for key, col, lab in [("fd_rel_err", "#718096", "finite difference (float64)"),
                          ("torch64_grad", common.PALETTE["float64"], "PyTorch, float64"),
                          ("torch32_grad", common.PALETTE["float32"], "PyTorch, float32")]:
        ax.scatter(np.maximum([r[key] for r in ops], 1e-18), y, color=col, label=lab, s=22)
    for x, lab in [(1e-7, "A1"), (1e-10, "A2")]:
        ax.axvline(x, color="k", lw=0.8, ls=":")
        ax.text(x, len(names) - 0.3, lab, fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("max relative gradient error (exact zero shown at 1e-18)")
    ax.set_title("Gradient error per op against two oracles")
    ax.legend(fontsize=8, loc="lower right")
    return common.save(fig, "e1_op_errors.png")


def replot() -> None:
    """Redraw the drift figure from results/raw/e1_curves.npz without retraining."""
    z = np.load(common.RAW / "e1_curves.npz")
    keys = {k.rsplit("__", 1)[0] for k in z.files}
    drift_figure({k: (z[f"{k}__0"], z[f"{k}__1"]) for k in keys})


if __name__ == "__main__":
    import sys

    replot() if "--replot" in sys.argv else main()
