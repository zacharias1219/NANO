"""E3: optimizer x init x schedule comparison on an MNIST MLP, 5 seeds, bootstrap CIs.

Claims are made only for differences whose paired-bootstrap 95% CI excludes zero
(pairs share the seed and every other factor). Each optimizer uses one standard
learning rate fixed in advance (not tuned per cell); see CONFIG.
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import evalkit
from experiments.data import load
from nano import Tensor, nn, no_grad, optim, sched, softmax_cross_entropy

CONFIG = {
    "model": "784-128-128-10 ReLU MLP", "n_train": 10_000, "n_test": 2_000, "batch": 128,
    "steps": 1000, "seeds": [0, 1, 2, 3, 4], "target_loss": 0.3, "ema": 0.9, "warmup_steps": 50,
    "optimizers": {
        "sgd": {"cls": "SGD", "lr": 0.1},
        "momentum": {"cls": "SGD", "lr": 0.01, "momentum": 0.9},
        "nesterov": {"cls": "SGD", "lr": 0.01, "momentum": 0.9, "nesterov": True},
        "adam": {"cls": "Adam", "lr": 1e-3},
        "adamw": {"cls": "AdamW", "lr": 1e-3, "weight_decay": 1e-2},
    },
    "inits": ["small", "xavier", "he"],
    "schedules": ["constant", "cosine", "warmup_cosine"],
}
if common.QUICK:
    CONFIG.update(steps=100, seeds=[0, 1], n_train=2000)
RAW = common.RAW / ("e3_quick" if common.QUICK else "e3")
_DATA: dict = {}


def run(job) -> dict:
    opt_name, init, schedule, seed = job
    path = RAW / f"{opt_name}_{init}_{schedule}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    if not _DATA:
        _DATA["tr"] = load("mnist", "train", n=CONFIG["n_train"])
        _DATA["te"] = load("mnist", "test", n=CONFIG["n_test"])
    (X, y), (Xt, yt) = _DATA["tr"], _DATA["te"]
    rng = np.random.default_rng(seed)
    model = nn.Sequential(nn.Linear(784, 128, init=init, rng=rng), nn.ReLU(),
                          nn.Linear(128, 128, init=init, rng=rng), nn.ReLU(), nn.Linear(128, 10, init=init, rng=rng))
    spec = dict(CONFIG["optimizers"][opt_name])
    opt = getattr(optim, spec.pop("cls"))(model.parameters(), **spec)
    s = sched.LRScheduler(opt, sched.make(schedule, spec["lr"], CONFIG["steps"], CONFIG["warmup_steps"]))
    ema, hit, curve = None, None, []
    for t, idx in enumerate(common.batch_schedule(len(y), CONFIG["batch"], CONFIG["steps"], seed)):
        loss = softmax_cross_entropy(model(Tensor(X[idx])), y=y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        s.step()
        ema = loss.item() if ema is None else CONFIG["ema"] * ema + (1 - CONFIG["ema"]) * loss.item()
        if hit is None and ema <= CONFIG["target_loss"]:
            hit = t + 1
        if t % 50 == 0:
            curve.append([t, ema])
    with no_grad():
        acc = float(np.mean(model(Tensor(Xt)).data.argmax(1) == yt))
    rec = {"optimizer": opt_name, "init": init, "schedule": schedule, "seed": seed, "test_acc": acc,
           "steps_to_target": hit, "final_ema_loss": ema, "curve": curve}
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def main():
    t0 = time.perf_counter()
    C = CONFIG
    jobs = list(itertools.product(C["optimizers"], C["inits"], C["schedules"], C["seeds"]))
    with ProcessPoolExecutor(max_workers=common.JOBS) as ex:
        recs = list(ex.map(run, jobs, chunksize=2))
    key = lambda r: (r["optimizer"], r["init"], r["schedule"])  # noqa: E731
    by_cfg: dict = {}
    for r in recs:
        by_cfg.setdefault(key(r), []).append(r)
    table = []
    budget = C["steps"] + 1  # runs that never hit the target are scored as budget + 1 (censored)
    for (o, i, s), rs in sorted(by_cfg.items()):
        rs.sort(key=lambda r: r["seed"])
        acc = [r["test_acc"] for r in rs]
        stt = [r["steps_to_target"] or budget for r in rs]
        table.append({"optimizer": o, "init": i, "schedule": s, "test_acc": evalkit.bootstrap_ci(acc, n_boot=5000),
                      "steps_to_target": evalkit.stats.median_iqr(stt), "n_censored": sum(r["steps_to_target"] is None
                                                                                          for r in rs)})
    # paired factor contrasts: level vs reference, pairing on seed and all other factors
    contrasts = []
    for factor, ref in (("optimizer", "sgd"), ("init", "he"), ("schedule", "constant")):
        levels = C[{"optimizer": "optimizers", "init": "inits", "schedule": "schedules"}[factor]]
        for lev in levels:
            if lev == ref:
                continue
            a, b, seeds_of = [], [], []
            for r in recs:
                if r[factor] != lev:
                    continue
                twin = next(q for q in recs if q[factor] == ref and q["seed"] == r["seed"]
                            and all(q[f] == r[f] for f in ("optimizer", "init", "schedule") if f != factor))
                a.append(r["test_acc"])
                b.append(twin["test_acc"])
                seeds_of.append(r["seed"])
            d = evalkit.paired_bootstrap_diff(a, b, n_boot=5000)
            # pairs that share a seed are not independent: also bootstrap the per-seed mean difference (n = 5)
            diff, sid = np.array(a) - np.array(b), np.array(seeds_of)
            per_seed = [float(diff[sid == s].mean()) for s in sorted(set(seeds_of))]
            dc = evalkit.bootstrap_ci(per_seed, n_boot=5000)
            sep = lambda m: bool(m["ci"][0] > 0 or m["ci"][1] < 0)  # noqa: E731
            contrasts.append({"factor": factor, "level": lev, "reference": ref, "acc_diff": d,
                              "acc_diff_seed_cluster": dc, "per_seed_diff": per_seed,
                              "p_sign_flip_seed_level": evalkit.paired_permutation_test(per_seed, [0.0] * len(per_seed)),
                              "separated": sep(d) and sep(dc)})
    best = max(table, key=lambda r: r["test_acc"]["value"])
    figs = [figure(table)]
    evalkit.write_fragment("E3", {
        "title": "E3 optimizer x init x schedule comparison (MNIST MLP)",
        "config": C, "tables": {"configs": table, "contrasts": contrasts},
        "metrics": {"best_config_test_acc": evalkit.metric(best["test_acc"], config=f"{best['optimizer']} / "
                                                                                    f"{best['init']} / {best['schedule']}"),
                    **{f"diff_{c['factor']}_{c['level']}_vs_{c['reference']}": evalkit.metric(c["acc_diff"])
                       for c in contrasts}},
        "figures": figs, "runtime_s": time.perf_counter() - t0,
        "notes": ["Test accuracy on 2,000 held-out MNIST test images after the fixed step budget.",
                  "Steps-to-target: first step where the EMA(0.9) training loss <= 0.3; censored runs count as "
                  "budget + 1.", "One standard LR per optimizer, not tuned per cell.",
                  "A contrast is 'separated' only if both the pair-level paired-bootstrap CI and the seed-clustered CI "
                  "(bootstrap over the 5 per-seed mean differences) exclude zero. With 5 seeds the exact sign-flip test "
                  "cannot go below p = 0.0625; it is reported for transparency."],
    }, common.ROOT / "results")


def figure(table) -> str:
    plt = common.plt()
    C = CONFIG
    fig, axes = plt.subplots(1, len(C["schedules"]), figsize=(13, 4), sharey=True)
    opts, inits = list(C["optimizers"]), C["inits"]
    width = 0.8 / len(inits)
    colors = {"small": "#9B2C2C", "xavier": "#B7791F", "he": "#2F855A"}
    for ax, s in zip(axes, C["schedules"]):
        for k, i in enumerate(inits):
            rows = [next(r for r in table if r["optimizer"] == o and r["init"] == i and r["schedule"] == s) for o in opts]
            v = np.array([r["test_acc"]["value"] for r in rows])
            lo = v - np.array([r["test_acc"]["ci"][0] for r in rows])
            hi = np.array([r["test_acc"]["ci"][1] for r in rows]) - v
            ax.bar(np.arange(len(opts)) + k * width, v, width, yerr=[lo, hi], color=colors[i], label=i, capsize=2)
        ax.set_xticks(np.arange(len(opts)) + width, opts, rotation=20)
        ax.set_title(s)
        ax.set_ylim(0.8, 1.0)
    axes[0].set_ylabel("test accuracy (95% bootstrap CI, 5 seeds)")
    axes[0].legend(title="init")
    fig.suptitle("E3: optimizer x init x schedule", y=1.02)
    return common.save(fig, "e3_optim_grid.png")


if __name__ == "__main__":
    main()
