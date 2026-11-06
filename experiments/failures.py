"""The failure gallery: every way the engine and the networks go wrong, measured.

F1 float32 drift is produced by E1 (parity.py) and F6 (every failing E2 cell) by E2.
This script produces:
  F2 softmax overflow without max subtraction
  F3 dead ReLUs at high learning rate
  F4 SGD divergence past the stable learning rate 2 / lambda_max (the quadratic bound)
  F5 finite-difference failures at ReLU kinks and max-pool ties
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import time

import numpy as np

import evalkit
from experiments.data import load
from nano import Tensor, cross_entropy_composed, maxpool2d, nn, optim, probes, softmax_cross_entropy
from nano.gradcheck import H_DEFAULT, gradcheck


# ------------------------------------------------------------------------- F2 softmax overflow
def softmax_overflow() -> dict:
    rng = np.random.default_rng(0)
    z0, y = rng.standard_normal((8, 10)), rng.integers(0, 10, 8)
    scales = np.logspace(0, 4, 41)
    out = {}
    for dname, dt in (("float64", np.float64), ("float32", np.float32)):
        fused, composed = [], []
        with np.errstate(all="ignore"):
            for s in scales:
                z = (s * z0).astype(dt)
                a = Tensor(z, requires_grad=True)
                lf = softmax_cross_entropy(a, y=y)
                lf.backward()
                fused.append(lf.item() if np.all(np.isfinite(a.grad)) else np.nan)
                b = Tensor(z, requires_grad=True)
                lc = cross_entropy_composed(b, y)
                lc.backward()
                composed.append(lc.item() if np.isfinite(lc.item()) and np.all(np.isfinite(b.grad)) else np.nan)
        bad = [float(s * np.abs(z0).max()) for s, c in zip(scales, composed) if not np.isfinite(c)]
        out[dname] = {"scale": scales.tolist(), "fused": fused, "composed": composed,
                      "composed_breaks_at_max_logit": min(bad) if bad else None,
                      "fused_all_finite": bool(np.all(np.isfinite(fused)))}
    plt = common.plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for dname in out:
        ax.loglog(scales * np.abs(z0).max(), out[dname]["fused"], color=common.PALETTE[dname], lw=2,
                  label=f"fused, max-shifted ({dname})")
        ax.loglog(scales * np.abs(z0).max(), out[dname]["composed"], color=common.PALETTE[dname], ls="--",
                  marker="x", ms=4, label=f"softmax -> log -> NLL ({dname})")
    for x, lab in ((np.log(np.finfo(np.float32).max), "e^z overflows float32 (z ~ 88.7)"),
                   (np.log(np.finfo(np.float64).max), "float64 (z ~ 709.8)")):
        ax.axvline(x, color="k", ls=":", lw=0.8)
        ax.text(x * 1.05, 3, lab, rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel("largest |logit| in the batch")
    ax.set_ylabel("cross-entropy loss (missing = inf/nan)")
    ax.set_title("F2: the composed softmax overflows; the fused one never does")
    ax.legend(fontsize=7)
    return {**out, "figure": common.save(fig, "f2_softmax_overflow.png")}


# --------------------------------------------------------------------------- F3 dead ReLUs
def dead_relus(seeds=(0, 1, 2)) -> dict:
    X, y = load("mnist", "train", n=5000)
    lrs = [0.01, 0.1, 0.5, 1.0]
    steps = 100 if common.QUICK else 400
    res = {}
    for lr in lrs:
        traces, finals, losses = [], [], []
        for seed in seeds:
            model = nn.mlp(784, 128, 4, 10, act="relu", init="he", rng=np.random.default_rng(seed))
            probe = probes.ActivationProbe(model)
            opt = optim.SGD(model.parameters(), lr=lr)
            order = common.batch_schedule(len(y), 128, steps, seed)
            trace, loss_v = [], np.nan
            with np.errstate(all="ignore"):
                for t, idx in enumerate(order):
                    if t % 20 == 0:
                        model(Tensor(X[:512]))
                        trace.append(probe.summary()["max_dead"])
                    loss = softmax_cross_entropy(model(Tensor(X[idx])), y=y[idx])
                    loss_v = loss.item()
                    if not np.isfinite(loss_v):
                        break
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model(Tensor(X[:512]))
            trace.append(probe.summary()["max_dead"])
            traces.append(trace)
            finals.append(trace[-1])
            losses.append(loss_v)
        n = max(len(t) for t in traces)
        res[f"{lr:g}"] = {"dead_final": evalkit.bootstrap_ci(finals, n_boot=2000), "final_batch_loss": losses,
                          "trace_mean": np.nanmean([t + [np.nan] * (n - len(t)) for t in traces], axis=0).tolist()}
    plt = common.plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("magma")
    for k, lr in enumerate(lrs):
        tr = res[f"{lr:g}"]["trace_mean"]
        ax.plot(np.arange(len(tr)) * 20, tr, color=cmap(0.15 + 0.7 * k / len(lrs)), lw=2, label=f"lr = {lr:g}")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("fraction of dead units (worst layer)")
    ax.set_title("F3: high learning rates kill ReLUs (He init, 4 x 128, mean of 3 seeds)")
    ax.legend()
    return {"lrs": lrs, "steps": steps, "results": res, "figure": common.save(fig, "f3_dead_relus.png")}


# -------------------------------------------------------------------- F4 SGD divergence bound
def sgd_divergence() -> dict:
    """Least squares L(w) = ||Xw - y||^2 / 2N has Hessian H = X^T X / N; GD diverges iff lr > 2 / lambda_max."""
    rng = np.random.default_rng(0)
    n, d = 200, 20
    X = rng.standard_normal((n, d)) * np.linspace(0.2, 3.0, d)  # anisotropic: lambda_max >> lambda_min
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)
    lam = np.linalg.eigvalsh(X.T @ X / n)
    lr_star = 2 / lam.max()

    def run(lr, steps=300):
        w = Tensor(np.zeros(d), requires_grad=True)
        hist = []
        with np.errstate(all="ignore"):
            for _ in range(steps):
                r = Tensor(X) @ w.reshape(d, 1) - Tensor(y.reshape(n, 1))
                loss = (r * r).mean() * 0.5
                hist.append(loss.item())
                w.grad = None
                loss.backward()
                w.data = w.data - lr * w.grad
        return hist

    factors = [0.5, 0.9, 0.99, 1.01, 1.1]
    curves = {f"{c:g}": run(c * lr_star) for c in factors}
    lo, hi = 0.5 * lr_star, 1.5 * lr_star  # bisection for the empirical stability edge
    for _ in range(40):
        mid = (lo + hi) / 2
        h = run(mid, 2000)
        lo, hi = (mid, hi) if np.isfinite(h[-1]) and h[-1] <= h[0] * 10 else (lo, mid)
    plt = common.plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("coolwarm")
    for k, c in enumerate(factors):
        ax.semilogy(np.minimum(curves[f"{c:g}"], 1e30), color=cmap(k / (len(factors) - 1)), lw=1.6,
                    label=f"lr = {c:g} x 2/lambda_max")
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel("loss")
    ax.set_title("F4: the quadratic bound lr < 2 / lambda_max is exactly the edge")
    ax.legend(fontsize=8)
    return {"lambda_max": float(lam.max()), "lambda_min": float(lam.min()), "lr_bound": float(lr_star),
            "lr_empirical_edge": float(lo), "relative_gap": float(abs(lo - lr_star) / lr_star),
            "final_loss": {k: v[-1] for k, v in curves.items()}, "figure": common.save(fig, "f4_sgd_divergence.png")}


# ---------------------------------------------------------------- F5 FD at kinks and ties
def fd_blind_spots() -> dict:
    dists = np.logspace(-10, -3, 29)
    relu_err, pool_err = [], []
    for dist in dists:
        x = Tensor(np.array([dist, -dist, 1.0]), requires_grad=True)
        relu_err.append(gradcheck(lambda t: t.relu(), x)[0])
        w = Tensor(np.array([[[[1.0, 1.0 + dist], [0.3, -0.2]]]]), requires_grad=True)  # near-tie in one window
        pool_err.append(gradcheck(lambda t: maxpool2d(t, k=2), w)[0])
    plt = common.plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(dists, np.maximum(relu_err, 1e-17), marker=".", label="ReLU at distance d from the kink")
    ax.loglog(dists, np.maximum(pool_err, 1e-17), marker=".", label="max-pool with two values d apart")
    ax.axvline(H_DEFAULT, color="k", ls="--", lw=0.8)
    ax.text(H_DEFAULT * 1.3, 1e-12, "h = 1e-6", fontsize=8)
    ax.set_xlabel("distance d to the kink / tie")
    ax.set_ylabel("gradcheck relative error")
    ax.set_title("F5: finite differences fail within h of a kink or tie")
    ax.legend(fontsize=8)
    return {"distance": dists.tolist(), "relu_err": relu_err, "maxpool_err": pool_err,
            "figure": common.save(fig, "f5_fd_kinks.png")}


def main():
    t0 = time.perf_counter()
    f2 = softmax_overflow()
    print("F2", {k: v["composed_breaks_at_max_logit"] for k, v in f2.items() if k != "figure"})
    f4 = sgd_divergence()
    print("F4 bound", f4["lr_bound"], "empirical", f4["lr_empirical_edge"], "gap", f4["relative_gap"])
    f5 = fd_blind_spots()
    f3 = dead_relus()
    print("F3", {k: v["dead_final"]["value"] for k, v in f3["results"].items()})
    metrics = {
        "f2_composed_breaks_float32": evalkit.metric(f2["float32"]["composed_breaks_at_max_logit"], unit="max |logit|"),
        "f2_composed_breaks_float64": evalkit.metric(f2["float64"]["composed_breaks_at_max_logit"], unit="max |logit|"),
        "f4_bound_relative_gap": evalkit.metric(f4["relative_gap"], method="bisection, 40 halvings, 2000 steps"),
    }
    for lr, r in f3["results"].items():
        metrics[f"f3_dead_fraction_lr{lr}"] = evalkit.metric(r["dead_final"])
    evalkit.write_fragment("FAILURES", {
        "title": "Failure gallery: overflow, dead ReLUs, the SGD stability bound, FD blind spots",
        "metrics": metrics,
        "tables": {"F2": {k: v for k, v in f2.items() if k != "figure"}, "F3": f3, "F4": f4, "F5": f5},
        "figures": [f2["figure"], f3["figure"], f4["figure"], f5["figure"]],
        "runtime_s": time.perf_counter() - t0,
        "notes": ["F1 (float32 drift) is figure e1_float32_drift.png; F6 (failing E2 cells) is in E2."],
    }, common.ROOT / "results")


if __name__ == "__main__":
    main()
