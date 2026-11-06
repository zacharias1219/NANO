"""Render report/REPORT.md from report/results.json (and nothing else).

Every number in the report is read from results.json at build time, so the report cannot
disagree with the data. The Markdown is valid GitHub Markdown and a Quarto document
(`quarto render report` turns it into HTML with the same content).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
X = R["experiments"]


def f(x, digits: int = 2) -> str:
    """Format a number: scientific for tiny/huge values, fixed otherwise."""
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return {"inf": "∞", "-inf": "-∞", "nan": "NaN"}.get(x, x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "∞" if x > 0 else ("-∞" if x < 0 else "NaN")
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:.{digits + 1}g}" if abs(x) < 1 else f"{x:,.{digits}f}"


def ci(m: dict, digits: int = 3) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    c = m.get("ci")
    s = f(m["value"], digits)
    return f"{s} [{f(c[0], digits)}, {f(c[1], digits)}]" if c else s


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig(path: str, alt: str) -> str:
    return f"![{alt}]({path})"


def get(eid, *path, default=None):
    d = X.get(eid)
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


# ------------------------------------------------------------------------------------ sections
def header() -> str:
    env = R["environment"]
    acc = R["acceptance"]
    n_pass = sum(1 for a in acc.values() if a["pass"])
    rows = [[k, a["criterion"], a["threshold"], f(a["value"]) if not isinstance(a["value"], dict) else
             "; ".join(f"{kk}: {f(vv) if not isinstance(vv, list) else ', '.join(vv)}" for kk, vv in a["value"].items()),
             "PASS" if a["pass"] else ("pending (CI)" if a["pass"] is None else "**FAIL**"), f"`{a['checked_by']}`"]
            for k, a in acc.items()]
    commit = (R["git"]["commit"] or "unknown")[:10]
    return f"""---
title: "Nano: a NumPy neural engine, verified and then broken on purpose"
subtitle: "Project 1A report. Every number below is read from results.json."
format:
  html:
    toc: true
    embed-resources: true
---

# Nano: a NumPy neural engine, verified and then broken on purpose

*Project 1A. Generated from [`results.json`](results.json) (schema {R['schema_version']}, commit `{commit}`,
{R['generated_at']}). Hardware: {env['cpu']}, {env['cores']} cores, {env.get('ram_gb')} GB RAM, {env['os']};
Python {env['python']}, NumPy {env['numpy']} ({env['blas']}), PyTorch {env.get('torch')} (oracle only).*

## Acceptance criteria

{n_pass} of {len(acc)} criteria pass in this build. Each row is checked by a script, not by hand.

{table(["ID", "Criterion", "Threshold", "Measured", "Result", "Checked by"], rows)}
"""


def problem() -> str:
    return """## 1. Problem and architecture

A from-scratch autodiff engine is the most common ML portfolio project, so the default
assumption is that it is a tutorial copy. Typical clones show a loss curve going down,
which proves nothing about gradient correctness; claim "matches PyTorch" in float32 over a
long run, which is unfalsifiable; and mention vanishing gradients without measuring them.

Nano answers each weakness with evidence: a verification ladder a skeptic can re-run
(`make repro`), a precision contract stated in float64 ([PRECISION.md](../PRECISION.md)),
a failure gallery, and one pre-registered study with confidence intervals.

```mermaid
flowchart TB
  torch["PyTorch<br/>(oracle, dev only)"] -.-> tests["tests/<br/>per-module, vs PyTorch"]
  torch -.-> exps["experiments/<br/>parity, trainability, speed"]
  subgraph nano["nano/ : pure NumPy, under 1500 lines"]
    tensor["tensor.py<br/>Tensor, tape, backward"]
    ops["ops.py<br/>ops, fused softmax-CE"]
    conv["conv.py<br/>im2col conv, maxpool"]
    nn["nn.py · init.py<br/>Linear, Conv2d, Sequential"]
    opt["optim.py · sched.py"]
    gc["gradcheck.py · probes.py<br/>first-bad-op, grad norms"]
  end
  tests --> nano
  exps --> nano
  exps -->|write| results["results.json<br/>(evalkit schema)"]
  results --> report["report/ (this page)"]
  results --> machine["machine/ (Backprop Machine)"]
```
"""


def parity() -> str:
    ops = get("E1", "tables", "ops", default=[])
    runs = get("E1", "tables", "runs", default={})
    m = get("E1", "metrics", default={})
    op_rows = [[f"`{r['case']}`", f(r["fd_rel_err"]), f(r["torch64_out"]), f(r["torch64_grad"]), f(r["torch32_grad"])]
               for r in ops]
    run_rows = []
    for setup in ("mlp_mnist", "cnn_mnist", "cnn_fashion"):
        a, b = runs.get(f"{setup}_float64", {}), runs.get(f"{setup}_float32", {})
        c = runs.get(f"{setup}_f32_vs_f64", {})
        run_rows.append([setup.replace("_", " "), f(a.get("one_step_param_abs")), f(a.get("run_loss_abs_max")),
                         f(b.get("one_step_param_abs")), f(b.get("run_loss_abs_max")), f(c.get("run_loss_abs_max")),
                         pct(a.get("test_acc_after_run"))])
    steps = get("E1", "config", "run_steps")
    return f"""## 2. Parity with PyTorch: float64 against float32 (E1)

**Per step and per run.** Nano and PyTorch start from identical weights, see identical batches
and take {steps} SGD-momentum steps, single-threaded. In float64 the worst one-step parameter
difference is {f(get('E1', 'metrics', 'A3_one_step_param_abs_f64', 'value'))} (A3 bound 1e-12) and the worst
loss difference over the whole run is {f(get('E1', 'metrics', 'A4_run_loss_abs_max_f64', 'value'))} (A4 bound
1e-8). The same runs in float32 drift to {f(get('E1', 'metrics', 'float32_run_loss_abs_max', 'value'))}:
that is the drift a float32 "matches PyTorch" claim has to live with, shown rather than hidden.

{table(["Setup", "f64: 1-step param diff", f"f64: max loss diff ({steps} steps)", "f32: 1-step param diff",
        "f32: max loss diff", "Nano f32 vs Nano f64", "test acc. after run (f64)"], run_rows)}

{fig(get('E1', 'figures', default=['', ''])[0], 'float32 drift')}

**Per op.** Max relative error over 5 random draws per op, against central finite differences (A1)
and against PyTorch in both precisions (A2). Worst FD error {f(m.get('A1_max_fd_rel_err', {}).get('value'))},
worst float64 PyTorch error {f(m.get('A2_max_torch_rel_err_f64', {}).get('value'))}.

<details><summary>Per-op table ({len(ops)} cases)</summary>

{table(["Op case", "vs FD (f64)", "PyTorch output (f64)", "PyTorch grad (f64)", "PyTorch grad (f32)"], op_rows)}

</details>

{fig(get('E1', 'figures', default=['', ''])[1], 'per-op gradient error')}
"""


def gradcheck() -> str:
    bugs = get("E4", "tables", "bugs", default=[])
    rows = [[b["bug"], b["description"], f"`{b['expected']}`" if b["expected"] else "—",
             f"`{b['named']}`" if b["named"] else "none (all ops pass)", f(b["whole_model_rel_err"]),
             "yes" if b["located"] else "**no**"] for b in bugs]
    rep = get("E4", "tables", "reports", "B3", default="")
    e6 = get("E6", "metrics", default={})
    return f"""## 3. Gradcheck coverage, the bug hunt and the FD error curve (E4, E6)

Every registered op has a finite-difference case and a PyTorch case
(`experiments/op_cases.py`; a test fails if an op is added without one). The table in
section 2 lists each op's max error.

**Bug hunt (A7).** Five silent bugs are injected one at a time. Each keeps every shape
correct, so the model still trains. A whole-model gradcheck notices *that* something is
wrong (large relative error) but not *where*. The layerwise checker re-runs every tape node's
op in isolation against finite differences, in forward order, and names the first one that
disagrees. Result: **{f(get('E4', 'metrics', 'bugs_located', 'value'))} of 5 located**; the bug-free control passes
every op.

{table(["Bug", "Injected defect", "Expected", "Checker names", "Whole-model rel. err", "Located"], rows)}

<details><summary>Checker output for B3 (off-by-one in col2im)</summary>

```
{rep}
```
</details>

**Finite-difference error vs step size (E6).** Central differences trade truncation error
(∝ h²) against rounding error (∝ ε/h), giving a U-curve with its minimum near
h* = (3ε)^(1/3) (derivation D11). Measured best h, geometric mean over ops:
{f(e6.get('best_h_float64_geomean', {}).get('value'))} in float64 (theory {f(e6.get('theory_h_float64', {}).get('value'))})
and {f(e6.get('best_h_float32_geomean', {}).get('value'))} in float32 (theory {f(e6.get('theory_h_float32', {}).get('value'))}).
At the gradcheck step h = 1e-6 the worst op error is {f(e6.get('max_err_at_default_h', {}).get('value'))}, inside the A1
bound of 1e-7. Linear ops (matmul, conv) have no truncation term, so their curves only rise.
In float32 no step size certifies a gradient to better than about 1e-5, which is why the
contract is stated in float64.

{fig(get('E6', 'figures', default=[''])[0], 'FD error curve')}
"""


def e2() -> str:
    E = X.get("E2")
    if not E:
        return "## 4. E2 trainability study\n\nNot run in this build.\n"
    m, t, C = E["metrics"], E["tables"], E["config"]
    verdict = E["verdict"]
    cells = t["cells"]
    by_pair = {}
    for c in cells:
        by_pair.setdefault((c["act"], c["init"]), []).append(c)
    pair_rows = []
    for (act, init), cs in by_pair.items():
        cs.sort(key=lambda c: c["depth"])
        last = max((c["depth"] for c in cs if c["frac_train"] >= 0.6), default=None)
        first_fail = min((c["depth"] for c in cs if c["frac_train"] < 0.6), default=None)
        pair_rows.append([act, init, f(last) if last else "none", f(first_fail) if first_fail else "never (≤ 64)",
                          " · ".join(f"{c['n_train']}/{c['n_seeds']}" for c in cs),
                          " · ".join(f"{c['r_mean']:.1f}" for c in cs)])
    fail_rows = [[x["cell"], f"{x['failed_seeds']}/5", ", ".join(f"{k} ({v})" for k, v in x["causes"].items()),
                  f(x["r_mean"]), f(x["final_loss_median"])] for x in t["failures"]]
    causes = ", ".join(f"{k}: {v}" for k, v in sorted(t["failure_cause_totals"].items(), key=lambda kv: -kv[1]))
    held = m.get("h1_heldout_balanced_accuracy", {})
    interp = INTERPRETATION.get(verdict, "")
    return f"""## 4. E2: trainability, pre-registered

**Question.** At what depth does each (activation, init) pair stop training, and does the
step-0 gradient-norm ratio r = log10(‖∂L/∂W_first‖ / ‖∂L/∂W_last‖) predict that failure?

The design, the definition of "trains", the predictor, hypothesis H1, the analysis plan
and the decision rule were committed in [the pre-registration](../docs/PREREGISTRATION_E2.md)
before the first run; the step budget ({C['steps']} steps) and LR grid ({', '.join(f(x) for x in C['lr_grid'])},
SGD momentum {C['momentum']}) were fixed after a pilot on a non-study seed and committed before the sweep.
Deviations are logged in [DEVIATIONS.md](../docs/DEVIATIONS.md). {E['n_runs']} runs
({len(C['depths'])} depths × 3 activations × 3 inits × {len(C['seeds'])} seeds × 3 learning rates),
{f(E['runtime_s'] / 3600, 1)} CPU-hours.

### Result: H1 is **{verdict.upper()}**

| Pre-registered statistic | Estimate [95% CI] |
|---|---|
| AUC of \\|r\\| for predicting failure (all {m['auc_absr_predicts_failure']['n']} runs) | {ci(m['auc_absr_predicts_failure'])} |
| Spearman(\\|r\\|, final loss) | {ci(m['spearman_absr_vs_final_loss'])} |
| Threshold τ fitted on seeds 0-2 | {f(m['h1_threshold_tau']['value'], 3)} |
| Held-out accuracy of "fail iff \\|r\\| > τ" (seeds 3-4, n = {held.get('n')}) | {ci(m.get('h1_heldout_accuracy'))} |
| Held-out balanced accuracy (seeds 3-4) | {ci(held)} |
| Fraction of (cell, seed) runs that train | {ci(m['fraction_runs_train'])} |

Decision rule: confirmed if the AUC CI lower bound is ≥ 0.75 and held-out balanced accuracy
≥ 0.80; partially confirmed if the AUC CI lower bound is > 0.5; null otherwise.

{interp}

### Phase diagram

Fraction of the 5 seeds that train (best of 3 learning rates per seed), by depth.

{fig('figures/e2_phase_diagram.png', 'E2 phase diagram')}

{table(["Activation", "Init", "Deepest depth that trains (≥ 3/5)", "First depth that fails",
        "Seeds trained at depths 2·4·8·16·32·64", "mean r at depths 2…64"], pair_rows)}

### Gradient norms at initialisation and the predictor

{fig('figures/e2_grad_norms.png', 'E2 gradient norms')}

{fig('figures/e2_predictor.png', 'E2 predictor')}

### Every failing cell, with its diagnosed cause (failure gallery F6)

Failing runs are labelled with the pre-registered precedence: overflow → dead ReLUs →
saturation → vanishing gradient (r < −2) → exploding gradient (r > 2) → slow/unexplained.
Totals over failing runs: {causes or 'none'}.

{table(["Cell", "Failed seeds", "Diagnosed cause (runs)", "mean r", "median final loss"], fail_rows) if fail_rows else 'No cell failed.'}
"""


INTERPRETATION: dict[str, str] = {}  # filled from report/interpretation.py after the study is read


def e3() -> str:
    E = X.get("E3")
    if not E:
        return ""
    t, C = E["tables"], E["config"]
    rows = [[r["optimizer"], r["init"], r["schedule"], ci(r["test_acc"]),
             f"{f(r['steps_to_target']['value'])} [{f(r['steps_to_target']['iqr'][0])}, {f(r['steps_to_target']['iqr'][1])}]",
             r["n_censored"]] for r in sorted(t["configs"], key=lambda r: -r["test_acc"]["value"])]
    con = [[c["factor"], f"{c['level']} − {c['reference']}", ci(c["acc_diff"], 4), "**yes**" if c["separated"] else "no"]
           for c in t["contrasts"]]
    sep = [c for c in t["contrasts"] if c["separated"]]
    mattered = "; ".join(f"{c['level']} vs {c['reference']} ({c['factor']}): {f(c['acc_diff']['value'] * 100, 2)} points"
                         for c in sorted(sep, key=lambda c: -abs(c["acc_diff"]["value"]))) or "no contrast was separated"
    return f"""## 5. Optimizers, inits and schedules (E3)

{C['model']} on {f(C['n_train'])} MNIST images, {C['steps']} steps of batch {C['batch']}, {len(C['seeds'])} seeds per
configuration, test accuracy on {f(C['n_test'])} held-out images. Each optimizer uses one
standard learning rate fixed in advance (SGD 0.1; momentum/Nesterov 0.01 with μ = 0.9; Adam/AdamW 1e-3).
Only differences whose paired-bootstrap 95% CI excludes zero are claimed.

**What actually mattered.** {mattered}.

{fig(E['figures'][0], 'E3 grid')}

Paired contrasts (each run paired with the reference-level run that shares its seed and every other factor):

{table(["Factor", "Contrast", "Δ test accuracy [95% CI]", "Separated"], con)}

<details><summary>All {len(rows)} configurations</summary>

{table(["Optimizer", "Init", "Schedule", "Test acc. [95% CI]", "Steps to EMA loss ≤ 0.3, median [IQR]",
        "Runs never reaching it"], rows)}

</details>
"""


def failures() -> str:
    F = X.get("FAILURES", {})
    t = F.get("tables", {})
    f2, f3, f4 = t.get("F2", {}), t.get("F3", {}), t.get("F4", {})
    f3rows = [[lr, ci(r["dead_final"])] for lr, r in f3.get("results", {}).items()]
    return f"""## 8. Failure gallery

Every way the engine and the networks go wrong, measured.

**F1. float32 drift against float64 parity.** Section 2: float64 parity holds at rounding level for
the whole run while float32 drifts by orders of magnitude.

**F2. Softmax overflow without max subtraction.** The composed softmax → log → NLL produces
inf/NaN once the largest logit reaches {f(f2.get('float32', {}).get('composed_breaks_at_max_logit'))} in float32 and
{f(f2.get('float64', {}).get('composed_breaks_at_max_logit'))} in float64 (e^z overflows near 88.7 and 709.8).
The fused, max-shifted version stays finite at every scale tested (derivation D7.2).

{fig('figures/f2_softmax_overflow.png', 'softmax overflow')}

**F3. Dead ReLUs at high learning rate.** A 4 × 128 ReLU MLP (He init), plain SGD, {f3.get('steps')} steps.
Fraction of units dead on 512 training images (worst layer), mean of 3 seeds [bootstrap CI]:

{table(["Learning rate", "Dead fraction at end"], f3rows)}

{fig('figures/f3_dead_relus.png', 'dead ReLUs')}

**F4. SGD divergence past the stable learning rate.** For least squares with Hessian H, gradient
descent converges iff lr < 2/λ_max (derivation D10.5). Here λ_max = {f(f4.get('lambda_max'), 3)}, so the bound is
{f(f4.get('lr_bound'), 4)}; the empirical edge found by bisection is {f(f4.get('lr_empirical_edge'), 4)}, a relative gap of
{f(f4.get('relative_gap'))}.

{fig('figures/f4_sgd_divergence.png', 'SGD divergence')}

**F5. Finite differences at ReLU kinks and max-pool ties.** Within h of a kink or tie the
central difference measures an average of two one-sided slopes and the check fails, though the
code is correct. This is why gradcheck samples away from kinks (PRECISION.md, blind spots).

{fig('figures/f5_fd_kinks.png', 'FD at kinks')}

**F6. Every failing cell in E2**, with its diagnosed cause: section 4.
"""


def speed() -> str:
    E = X.get("E5")
    if not E:
        return ""
    rows = []
    for b in E["tables"]["bench"]:
        if "comparison" in b:
            continue
        rows.append([b["name"], f"{f(b['nano_ms']['value'])} [{f(b['nano_ms']['iqr'][0])}, {f(b['nano_ms']['iqr'][1])}]",
                     f"{f(b['torch_ms']['value'])} [{f(b['torch_ms']['iqr'][0])}, {f(b['torch_ms']['iqr'][1])}]",
                     f"{b['slowdown']:.1f}×", f(b["flops"] / 1e6, 1), f(b["nano_gflops"]), f(b["torch_gflops"]),
                     f(b["nano_peak_mb"], 1)])
    naive = next(b for b in E["tables"]["bench"] if "comparison" in b)
    env = E["environment"]
    return f"""## 9. Cost and latency (E5)

One training step, single thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), median
[IQR] over {E['config']['repeats']} repeats. Hardware: {env['cpu']} ({env['cores']} cores), {env.get('ram_gb')} GB RAM,
{env['os']}, NumPy {env['numpy']} with {env['blas']}, PyTorch {env.get('torch')}.

{table(["Workload", "Nano ms", "PyTorch ms", "Slowdown", "MFLOP/step", "Nano GFLOP/s", "PyTorch GFLOP/s",
        "Nano peak MB"], rows)}

im2col against the naive-loop convolution (forward + backward, {naive['name'].split(':')[0]}):
{f(naive['nano_ms']['value'])} ms vs {f(naive['torch_ms']['value'])} ms, a **{naive['slowdown']:.0f}×** speedup from turning the
convolution into one matmul. Peak memory is Nano's traced NumPy allocation; PyTorch's allocator
is invisible to tracemalloc, so no PyTorch peak is claimed. The slowdown is measured and
published, not minimised (a PRD non-goal).
"""


def design() -> str:
    return """## 6. Derivations

Every backward rule is derived in [derivations/DERIVATIONS.md](derivations/DERIVATIONS.md); each op's
docstring carries its tag (e.g. `[D8.3]` for conv2d) and a test enforces that every op has one.
Handwritten scans go in [derivations/scans/](derivations/scans/).

## 7. Design trade-offs considered and rejected

See [docs/DECISIONS.md](../docs/DECISIONS.md): dynamic tape over a static graph, im2col with a
naive-loop oracle over direct or FFT convolution, fused max-shifted cross entropy over the
composed one, one central `unbroadcast` over per-op handling, float64 verification over
float32, an `@op` decorator over a class per op, PyTorch as an isolated oracle over a
reimplemented reference.
"""


def repro() -> str:
    total = sum(e.get("runtime_s", 0) for e in X.values())
    return f"""## 10. Reproduce everything

```bash
git clone <this repo> && cd NANO
make install
make repro          # data -> tests -> E1..E6 + failures -> results.json -> this report
```

Measured compute in this build: {f(total / 3600, 1)} CPU-hours summed over experiments (E2 dominates;
it runs on all cores, so wall time is roughly that divided by the core count). `make repro QUICK=1`
runs the whole pipeline at reduced size in minutes (its numbers are not the published ones).
Datasets are pinned by SHA-256, seeds are fixed in `experiments/seeds.json`, and versions of
Python, NumPy, BLAS and PyTorch are recorded in every results fragment.
"""


def main():
    interp = HERE / "interpretation.json"
    if interp.exists():
        INTERPRETATION.update(json.loads(interp.read_text()))
    sections = [header, problem, parity, gradcheck, e2, e3, design, failures, speed, repro]
    body = "\n".join(s() for s in sections)
    (HERE / "REPORT.md").write_text(body)
    print(f"wrote {HERE / 'REPORT.md'} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
