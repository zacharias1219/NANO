"""E2: pre-registered trainability study (docs/PREREGISTRATION_E2.md).

    python -m experiments.trainability            # full sweep (resumable) + analysis
    python -m experiments.trainability --pilot    # pilot on seed 99 (not a study seed)
    python -m experiments.trainability --analyze  # analysis only, from cached runs

Every run is cached as results/raw/e2/<key>.json, so the sweep can be stopped and resumed.
"""
from __future__ import annotations

from experiments import common  # noqa: I001  (sets single-threaded BLAS before numpy loads)

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import evalkit
from evalkit.stats import roc_curve, wilson_interval
from experiments.data import load
from nano import Tensor, nn, no_grad, optim, probes, softmax_cross_entropy

CONFIG = json.loads(Path(__file__).with_name("e2_config.json").read_text())
CHANCE = math.log(10)
TRAIN_THRESHOLD = 0.5 * CHANCE
RAW = common.RAW / "e2"
_DATA: dict = {}


def data():
    if not _DATA:
        _DATA["train"] = load("mnist", "train", n=CONFIG["n_train"])
        _DATA["test"] = load("mnist", "test", n=1000)
    return _DATA["train"], _DATA["test"]


def full_loss(model, X, y, chunk: int = 1000) -> float:
    with no_grad(), np.errstate(all="ignore"):
        total = sum(softmax_cross_entropy(model(Tensor(X[i:i + chunk])), y=y[i:i + chunk]).item() * len(y[i:i + chunk])
                    for i in range(0, len(y), chunk))
    return total / len(y)


def accuracy(model, X, y) -> float:
    with no_grad(), np.errstate(all="ignore"):
        return float(np.mean(model(Tensor(X)).data.argmax(axis=1) == y))


def key(act, init, depth, seed, lr) -> str:
    return f"{act}_{init}_d{depth}_s{seed}_lr{lr:g}"


def run_one(job: tuple, steps: int, cache: Path) -> dict:
    act, init, depth, seed, lr = job
    path = cache / f"{key(*job)}.json"
    if path.exists():
        return json.loads(path.read_text())
    (X, y), (Xt, yt) = data()
    t0 = time.perf_counter()
    model = nn.mlp(784, CONFIG["width"], depth, 10, act=act, init=init, rng=np.random.default_rng(seed))
    probe = probes.ActivationProbe(model)

    # step-0 predictor: mean of r over 4 fixed batches; per-layer norms for the heatmaps and the machine
    fixed = np.random.default_rng(CONFIG["predictor_batch_seed"]).permutation(len(y))
    rs, norms = [], []
    with np.errstate(all="ignore"):
        for b in range(CONFIG["r_batches"]):
            idx = fixed[b * CONFIG["batch"]:(b + 1) * CONFIG["batch"]]
            model.zero_grad()
            softmax_cross_entropy(model(Tensor(X[idx])), y=y[idx]).backward()
            norms.append(probes.grad_norms(model))
            rs.append(probes.grad_norm_ratio(model))
        model(Tensor(X[fixed[:512]]))
    step0 = probe.summary()
    model.zero_grad()

    opt = optim.SGD(model.parameters(), lr=lr, momentum=CONFIG["momentum"])
    order = common.batch_schedule(len(y), CONFIG["batch"], steps, seed)
    curve, diverged, initial = [], False, full_loss(model, X, y)
    with np.errstate(all="ignore"):
        for t, idx in enumerate(order):
            loss = softmax_cross_entropy(model(Tensor(X[idx])), y=y[idx])
            if not np.isfinite(loss.item()):
                diverged = True
                break
            if t % CONFIG["log_every"] == 0:
                curve.append([t, loss.item()])
            opt.zero_grad()
            loss.backward()
            opt.step()
    final = math.inf if diverged else full_loss(model, X, y)
    diverged = diverged or not np.isfinite(final)
    final = math.inf if diverged else final
    model(Tensor(X[fixed[:512]]))
    end = probe.summary()
    rec = {
        "act": act, "init": init, "depth": depth, "seed": seed, "lr": lr, "steps": steps,
        "r": float(np.mean(rs)), "r_batches": rs, "grad_norms": np.mean(norms, axis=0).tolist(),
        "initial_loss": initial, "final_loss": final, "diverged": diverged,
        "trains": bool(final <= TRAIN_THRESHOLD), "test_acc": None if diverged else accuracy(model, Xt, yt),
        "dead_end": end["max_dead"], "saturated_step0": step0["max_saturated"],
        "act_std_step0": step0["std"], "curve": curve, "seconds": time.perf_counter() - t0,
    }
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evalkit.results._clean(rec)))
    return rec


def _run(args):
    return run_one(*args)


def sweep(jobs: list[tuple], steps: int, cache: Path) -> list[dict]:
    jobs = sorted(jobs, key=lambda j: -j[2])  # deepest first for load balance
    t0, out = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=common.JOBS) as ex:
        for i, rec in enumerate(ex.map(_run, [(j, steps, cache) for j in jobs], chunksize=1)):
            out.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} runs, {time.perf_counter() - t0:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------------------------- analysis
def load_runs(cache: Path) -> list[dict]:
    runs = []
    for p in sorted(cache.glob("*.json")):
        r = json.loads(p.read_text())
        r["final_loss"] = float(r["final_loss"]) if not isinstance(r["final_loss"], str) else float(r["final_loss"])
        runs.append(r)
    return runs


def per_seed(runs: list[dict]) -> list[dict]:
    """Collapse the LR grid: a (cell, seed) trains if any LR trains; its loss is the min over LRs."""
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["act"], r["init"], r["depth"], r["seed"]), []).append(r)
    out = []
    for (act, init, depth, seed), rs in sorted(groups.items()):
        best = min(rs, key=lambda r: r["final_loss"])
        out.append({"act": act, "init": init, "depth": depth, "seed": seed, "r": best["r"],
                    "final_loss": best["final_loss"], "trains": any(r["trains"] for r in rs),
                    "best_lr": best["lr"], "any_diverged": any(r["diverged"] for r in rs),
                    "all_diverged": all(r["diverged"] for r in rs), "dead_end": best["dead_end"],
                    "saturated_step0": best["saturated_step0"], "grad_norms": best["grad_norms"],
                    "test_acc": best["test_acc"], "n_lrs": len(rs), "act_std_step0": best["act_std_step0"]})
    return out


def diagnose(s: dict) -> str:
    """Pre-registered order of precedence for labelling a failing run."""
    if s["all_diverged"]:
        return "overflow"
    if s["act"] == "relu" and s["dead_end"] > 0.5:
        return "dead ReLUs"
    if s["act"] != "relu" and s["saturated_step0"] > 0.5:
        return "saturation"
    if s["r"] < -2:
        return "vanishing gradient"
    if s["r"] > 2:
        return "exploding gradient"
    return "slow / unexplained"


def fit_threshold(score: np.ndarray, fail: np.ndarray) -> float:
    cand = np.unique(score)
    mids = np.concatenate([[cand[0] - 1], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1]])
    def bacc(t):
        pred = score > t
        tpr = pred[fail].mean() if fail.any() else 0.0
        tnr = (~pred[~fail]).mean() if (~fail).any() else 0.0
        return (tpr + tnr) / 2
    return float(max(mids, key=bacc))


def balanced_accuracy(pred, fail) -> float:
    pred, fail = np.asarray(pred, bool), np.asarray(fail, bool)
    if fail.all() or (~fail).all():
        return float(np.mean(pred == fail))
    return float((pred[fail].mean() + (~pred[~fail]).mean()) / 2)


def analyze(cache: Path, pilot: bool = False) -> dict:
    runs = load_runs(cache)
    seeds = per_seed(runs)
    C = CONFIG
    absr = np.array([abs(s["r"]) for s in seeds])
    fail = np.array([not s["trains"] for s in seeds])
    loss = np.array([s["final_loss"] for s in seeds])
    seed_ids = np.array([s["seed"] for s in seeds])
    metrics, notes = {}, []

    auc = evalkit.bootstrap_stat(evalkit.auc, absr, fail, seed=0)
    spear = evalkit.bootstrap_stat(evalkit.spearman, absr, np.where(np.isfinite(loss), loss, 1e300), seed=1)
    metrics["auc_absr_predicts_failure"] = auc
    metrics["spearman_absr_vs_final_loss"] = spear
    metrics["fraction_runs_train"] = evalkit.metric(float(np.mean(~fail)), ci=wilson_interval(int((~fail).sum()), len(fail)),
                                                    n=len(fail), method="Wilson 95%")

    # exploratory predictors X1, X2 (registered after the pilot, before the sweep; not part of H1)
    with np.errstate(all="ignore"):
        g1 = np.array([-np.log10(max(s["grad_norms"][0], 1e-300)) for s in seeds])
        q = np.array([abs(np.log10(max(s["act_std_step0"][-1], 1e-300) / max(s["act_std_step0"][0], 1e-300)))
                      for s in seeds])
    metrics["exploratory_X1_auc_neg_log_first_grad"] = {**evalkit.bootstrap_stat(evalkit.auc, g1, fail, seed=4),
                                                         "exploratory": True}
    metrics["exploratory_X2_auc_forward_attenuation"] = {**evalkit.bootstrap_stat(evalkit.auc, q, fail, seed=5),
                                                          "exploratory": True}

    fit_mask, test_mask = np.isin(seed_ids, C["fit_seeds"]), np.isin(seed_ids, C["test_seeds"])
    tau = fit_threshold(absr[fit_mask], fail[fit_mask]) if fit_mask.any() else float("nan")
    held = {}
    if test_mask.any():
        pred = absr[test_mask] > tau
        held["accuracy"] = evalkit.bootstrap_stat(lambda p, f: np.mean(p == f), pred, fail[test_mask], seed=2)
        held["balanced_accuracy"] = evalkit.bootstrap_stat(balanced_accuracy, pred, fail[test_mask], seed=3)
        metrics["h1_heldout_accuracy"] = held["accuracy"]
        metrics["h1_heldout_balanced_accuracy"] = held["balanced_accuracy"]
    metrics["h1_threshold_tau"] = evalkit.metric(tau, n=int(fit_mask.sum()), method="fitted on seeds 0-2")

    lo = auc["ci"][0]
    if not np.isfinite(auc["value"]) or (np.isfinite(lo) and lo <= 0.5):
        verdict = "null"
    elif lo >= 0.75 and held and held["balanced_accuracy"]["value"] >= 0.80:
        verdict = "confirmed"
    else:
        verdict = "partially confirmed"

    # per-cell table
    cells = {}
    for s in seeds:
        c = cells.setdefault(f"{s['act']}|{s['init']}|{s['depth']}", {"act": s["act"], "init": s["init"], "depth": s["depth"],
                                                                      "seeds": [], "lrs": []})
        c["seeds"].append(s)
        c["lrs"].append(s["best_lr"])
    table, failures = [], []
    for c in cells.values():
        k = sum(s["trains"] for s in c["seeds"])
        n = len(c["seeds"])
        row = {
            "act": c["act"], "init": c["init"], "depth": c["depth"], "n_seeds": n, "n_train": k,
            "frac_train": k / n, "wilson95": wilson_interval(k, n),
            "best_lr": Counter(c["lrs"]).most_common(1)[0][0],
            "r_mean": float(np.mean([s["r"] for s in c["seeds"]])),
            "final_loss_median": float(np.median([s["final_loss"] for s in c["seeds"]])),
            "test_acc_median": float(np.median([s["test_acc"] or 0.0 for s in c["seeds"]])),
            "grad_norms_mean": np.exp(np.mean(np.log(np.maximum([s["grad_norms"] for s in c["seeds"]], 1e-300)),
                                              axis=0)).tolist(),
        }
        if k < n:
            labels = Counter(diagnose(s) for s in c["seeds"] if not s["trains"])
            row["failure_causes"] = dict(labels)
            failures.append({"cell": f"{c['act']} / {c['init']} / depth {c['depth']}", "failed_seeds": n - k,
                             "causes": dict(labels), "r_mean": row["r_mean"],
                             "final_loss_median": row["final_loss_median"]})
        table.append(row)
    table.sort(key=lambda r: (C["acts"].index(r["act"]), C["inits"].index(r["init"]), r["depth"]))

    cause_totals = Counter()
    for s in seeds:
        if not s["trains"]:
            cause_totals[diagnose(s)] += 1
    figures = [] if pilot else make_figures(table, seeds, auc, tau)
    if pilot:
        notes.append("PILOT on seed 99: not part of the study; used only to fix the step budget and LR grid.")
    return {
        "title": "E2 pilot (not part of the study)" if pilot else "E2 trainability study (pre-registered)",
        "config": {**C, "train_threshold_loss": TRAIN_THRESHOLD, "chance_loss": CHANCE,
                   "preregistration": "docs/PREREGISTRATION_E2.md"},
        "metrics": metrics, "verdict": verdict,
        "tables": {"cells": table, "failures": failures, "failure_cause_totals": dict(cause_totals),
                   "runs": [{k: s[k] for k in ("act", "init", "depth", "seed", "r", "final_loss", "trains", "best_lr")}
                            for s in seeds]},
        "figures": figures, "notes": notes, "n_runs": len(runs),
        "runtime_s": float(sum(r["seconds"] for r in runs)),
    }


def make_figures(table, seeds, auc, tau) -> list[str]:
    plt = common.plt()
    C, figs = CONFIG, []
    depths, acts, inits = C["depths"], C["acts"], C["inits"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, act in zip(axes, acts):
        grid = np.array([[next(r["frac_train"] for r in table if r["act"] == act and r["init"] == i and r["depth"] == d)
                          for d in depths] for i in inits])
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for (a, b), v in np.ndenumerate(grid):
            ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(depths)), depths)
        ax.set_yticks(range(len(inits)), inits)
        ax.set_xlabel("hidden layers")
        ax.set_title(act)
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="fraction of seeds that train", shrink=0.85)
    fig.suptitle("E2 phase diagram: where each (activation, init) stops training", y=1.04)
    figs.append(common.save(fig, "e2_phase_diagram.png"))

    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharex=False, sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, act in enumerate(acts):
        for j, init in enumerate(inits):
            ax = axes[i, j]
            for k, d in enumerate(depths):
                row = next(r for r in table if r["act"] == act and r["init"] == init and r["depth"] == d)
                g = np.log10(np.maximum(row["grad_norms_mean"], 1e-300))
                ax.plot(np.arange(1, len(g) + 1) / len(g), np.clip(g, -30, 10), color=cmap(k / (len(depths) - 1)),
                        label=f"{d}", lw=1.4)
            ax.set_title(f"{act} / {init}", fontsize=10)
            if i == 2:
                ax.set_xlabel("layer position (input = 0, output = 1)")
            if j == 0:
                ax.set_ylabel("log10 ||dL/dW|| at step 0")
    axes[0, 2].legend(title="depth", fontsize=8, ncol=2)
    fig.suptitle("E2 per-layer gradient norms at initialisation (geometric mean over seeds)")
    fig.tight_layout()
    figs.append(common.save(fig, "e2_grad_norms.png"))

    absr = np.array([abs(s["r"]) for s in seeds])
    fail = np.array([not s["trains"] for s in seeds])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if fail.any() and (~fail).any():
        fpr, tpr, _ = roc_curve(absr, fail)
        ax1.plot(fpr, tpr, color=common.PALETTE["float64"], lw=2)
    ax1.plot([0, 1], [0, 1], ls="--", color=common.PALETTE["grey"])
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate (failure detected)")
    ax1.set_title(f"|r| predicts failure: AUC {auc['value']:.3f} [{auc['ci'][0]:.3f}, {auc['ci'][1]:.3f}]")
    for act in acts:
        sel = [s for s in seeds if s["act"] == act]
        ax2.scatter([s["r"] for s in sel], [min(s["final_loss"], 10) for s in sel], s=14, alpha=0.75,
                    color=common.PALETTE[act], label=act)
    ax2.axhline(TRAIN_THRESHOLD, color="k", lw=0.8, ls=":")
    ax2.axvline(tau, color="k", lw=0.8, ls="--")
    ax2.axvline(-tau, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("r = log10(||dW_first|| / ||dW_last||) at step 0")
    ax2.set_ylabel("final training loss (diverged clipped at 10)")
    ax2.set_title(f"dashed: fitted |r| threshold {tau:.2f}; dotted: 'trains' line")
    ax2.legend()
    fig.tight_layout()
    figs.append(common.save(fig, "e2_predictor.png"))
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    C = CONFIG
    if args.pilot:
        P = C["pilot"]
        cache = RAW.parent / "e2_pilot"
        jobs = [(a, i, d, P["seed"], lr) for a in C["acts"] for i in C["inits"] for d in P["depths"] for lr in P["lr_grid"]]
        if not args.analyze:
            sweep(jobs, P["steps"], cache)
        evalkit.write_fragment("E2_pilot", analyze(cache, pilot=True), common.ROOT / "results")
        return
    depths, seeds, steps = C["depths"], C["seeds"], C["steps"]
    if common.QUICK:
        depths, seeds, steps = [2, 8], C["seeds"][:2], 60
        C = {**C, "depths": depths, "fit_seeds": [0], "test_seeds": [1]}
        globals()["CONFIG"] = C
    cache = RAW / ("quick" if common.QUICK else "")
    jobs = [(a, i, d, s, lr) for a in C["acts"] for i in C["inits"] for d in depths for s in seeds for lr in C["lr_grid"]]
    if not args.analyze:
        sweep(jobs, steps, cache)
    out = analyze(cache)
    if common.QUICK:
        out["notes"].append("QUICK mode: reduced grid, NOT the published study.")
    print(json.dumps({k: v for k, v in out["metrics"].items()}, indent=1, default=str)[:2000])
    print("verdict:", out["verdict"])
    evalkit.write_fragment("E2", out, common.ROOT / "results")


if __name__ == "__main__":
    main()
