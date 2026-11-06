"""E6: finite-difference error vs step size h, float64 and float32 (the U-curve).

Central differences have truncation error ~ h^2 |f'''| / 6 and rounding error ~ eps |f| / h,
so the total error is minimised near h* ~ (3 eps)^(1/3): about 6e-6 in float64 and 5e-3
in float32. This experiment measures the curve per op and justifies the gradcheck step
H_DEFAULT = 1e-6 used for A1.
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import time

import numpy as np

import evalkit
from experiments.op_cases import cases, run_nano
from nano import Tensor
from nano.gradcheck import H_DEFAULT, numerical_grad, rel_error

OPS = ["exp", "tanh", "sigmoid", "log", "matmul", "logsumexp", "softmax_cross_entropy", "conv2d"]
GRID = {"float64": np.logspace(-1, -13, 25), "float32": np.logspace(-0.5, -7, 14)}
EPS = {"float64": np.finfo(np.float64).eps, "float32": np.finfo(np.float32).eps}


def curve(case, dtype_name: str, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    xs = case.inputs(rng)
    out, _ = run_nano(case, xs)
    u = rng.standard_normal(out.shape)
    _, grads = run_nano(case, xs, u)  # float64 analytic gradient = reference
    dt = np.float64 if dtype_name == "float64" else np.float32
    errs = []
    for h in GRID[dtype_name]:
        x32 = [x.astype(dt) for x in xs]
        f = lambda: float(np.sum(u.astype(dt) * case.nano_fn(*[Tensor(x) for x in x32]).data))  # noqa: E731
        num = numerical_grad(f, x32[0], h=dt(h))
        errs.append(rel_error(grads[0], num))
    return errs


def main():
    t0 = time.perf_counter()
    table = {c.name: {} for c in cases() if c.name in OPS}
    for c in cases():
        if c.name not in OPS:
            continue
        for dname in GRID:
            errs = curve(c, dname)
            best = int(np.argmin(errs))
            table[c.name][dname] = {"h": GRID[dname].tolist(), "rel_err": errs,
                                    "best_h": float(GRID[dname][best]), "best_err": float(errs[best])}
            print(f"  {c.name:22s} {dname}: best h {GRID[dname][best]:.1e}  err {errs[best]:.1e}", flush=True)
    best64 = np.array([table[o]["float64"]["best_h"] for o in table])
    best32 = np.array([table[o]["float32"]["best_h"] for o in table])
    at_default = [float(np.interp(np.log10(H_DEFAULT), np.log10(table[o]["float64"]["h"][::-1]),
                                  table[o]["float64"]["rel_err"][::-1])) for o in table]
    fig = figure(table)
    evalkit.write_fragment("E6", {
        "title": "E6 finite-difference error vs step size h",
        "config": {"ops": OPS, "grid": {k: v.tolist() for k, v in GRID.items()}, "reference": "float64 analytic VJP"},
        "metrics": {
            "best_h_float64_geomean": evalkit.metric(float(10 ** np.mean(np.log10(best64))), n=len(best64)),
            "best_h_float32_geomean": evalkit.metric(float(10 ** np.mean(np.log10(best32))), n=len(best32)),
            "theory_h_float64": evalkit.metric(float((3 * EPS["float64"]) ** (1 / 3)), method="(3 eps)^(1/3)"),
            "theory_h_float32": evalkit.metric(float((3 * EPS["float32"]) ** (1 / 3)), method="(3 eps)^(1/3)"),
            "max_err_at_default_h": evalkit.metric(max(at_default), n=len(at_default), method=f"h = {H_DEFAULT:g}"),
        },
        "tables": {"curves": table}, "figures": [fig], "runtime_s": time.perf_counter() - t0,
    }, common.ROOT / "results")


def figure(table) -> str:
    plt = common.plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, dname in zip(axes, GRID):
        for k, (name, d) in enumerate(table.items()):
            ax.loglog(d[dname]["h"], np.maximum(d[dname]["rel_err"], 1e-17), marker=".", lw=1.2, color=cmap(k),
                      label=name)
            ax.scatter([d[dname]["best_h"]], [max(d[dname]["best_err"], 1e-17)], color=cmap(k), s=50,
                       edgecolor="k", zorder=5)
        hstar = (3 * EPS[dname]) ** (1 / 3)
        ax.axvline(hstar, color="k", ls="--", lw=0.9)
        ax.text(hstar * 1.3, 3e-1, "h* = (3 eps)^(1/3)", fontsize=8)
        if dname == "float64":
            ax.axvline(H_DEFAULT, color=common.PALETTE["float64"], ls=":", lw=1.2)
            ax.text(H_DEFAULT / 30, 3e-15, "gradcheck h = 1e-6", fontsize=8, color=common.PALETTE["float64"])
            ax.axhline(1e-7, color="k", ls=":", lw=0.8)
            ax.text(1e-12, 1.5e-7, "A1 bound", fontsize=8)
        ax.invert_xaxis()
        ax.set_xlabel("step size h (decreasing ->)")
        ax.set_title(f"{dname}: truncation (left) vs rounding (right)")
    axes[0].set_ylabel("relative error vs analytic gradient")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("E6: the finite-difference U-curve; dots mark the best h per op", y=1.02)
    return common.save(fig, "e6_fd_error.png")


if __name__ == "__main__":
    main()
