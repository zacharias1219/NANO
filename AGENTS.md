# AGENTS.md: working on Nano

Guidance for AI coding agents (and humans) changing this repository. Read this before
editing anything. `CLAUDE.md` imports this file.

## What this project is

Nano is Project 1A of a portfolio: a reverse-mode autodiff engine with an MLP and CNN
stack in **pure NumPy**, verified against finite differences and PyTorch under a stated
precision contract, then used for one **pre-registered** research study (E2). The
deliverable is evidence, not features. The full spec is the PRD; the acceptance
criteria A1-A10 are the definition of done.

## Layout

```
nano/          the library (pure NumPy, < 1500 lines, never imports torch)
  tensor.py      Tensor, Op, @op, unbroadcast, backward
  ops.py         elementwise, matmul, reductions, activations, fused softmax-CE
  conv.py        im2col/col2im, conv2d, conv2d_naive (oracle), maxpool2d
  nn.py          Module, Linear, Conv2d, MaxPool2d, Flatten, activations, Sequential, mlp()
  init.py        zeros, small normal, Xavier, He
  optim.py       SGD (momentum, Nesterov), Adam, AdamW
  sched.py       constant, step, cosine, warmup
  gradcheck.py   FD gradcheck, layerwise first-bad-op report
  probes.py      per-layer grad norms, activation statistics
evalkit/       shared infra: bootstrap CIs, AUC/Spearman, results.json schema + writer
tests/         pytest; PyTorch is the oracle here
experiments/   E1 parity, E2 trainability, E3 optim grid, E4 bughunt, E5 speed, E6 fd_error, failures
results/       one JSON fragment per experiment (committed); results/raw/ is scratch (ignored)
report/        results.json (merged), REPORT.md (generated), figures/, derivations/,
               build_page.py -> site/ (published results page; generated, not committed)
docs/          PREREGISTRATION_E2.md, DEVIATIONS.md, DECISIONS.md
machine/       v1.1 Backprop Machine (static page; reads report/results.json only)
```

## Commands

```bash
make install      # pip install -e ".[dev]"  (PyTorch CPU is a dev dependency)
make test-fast    # what CI runs per commit
make test         # full suite
make lint loc     # ruff; A9 line budget
make parity       # E1 (A3, A4)
make e4           # bug hunt (A7)
make repro        # everything from a clean clone (hours: E2 is the long pole)
make repro QUICK=1  # minutes; numbers are NOT the published ones
make page         # build the results website in report/site/
```

Always run `make lint loc test-fast` before committing.

## Invariants you must not break

1. **`nano/` never imports torch** (or anything except NumPy and the standard library).
   Enforced by `tests/test_isolation.py`.
2. **`nano/` stays under 1500 physical lines** (`make loc`, A9). If a change pushes it over,
   simplify; do not move library code elsewhere to dodge the count.
3. **Every op is < 30 lines and its docstring carries a `[Dx.y]` derivation tag** that
   exists in `report/derivations/DERIVATIONS.md`. Write the derivation first.
4. **Every registered op has a case in `experiments/op_cases.py`** (`COVERS` map). That one
   table feeds A1, A2 and E1; a new op without a case fails the tests.
5. **Ops are pure functions of ndarrays returning `(out, vjp)`.** Broadcasting is undone
   only by `unbroadcast` in `Tensor.backward`. Do not add per-op broadcast handling.
6. **Tolerances are the precision contract** (`PRECISION.md`). Never loosen a threshold
   to make a test pass. A failing parity test is a bug until proven otherwise.
7. **Never skip, xfail or delete a test to get green.**
8. **Numbers come from `report/results.json`.** The report and the Backprop Machine may
   only display values present in that file. Do not hand-type results into docs.
9. **E2 is pre-registered.** `docs/PREREGISTRATION_E2.md` and `experiments/e2_config.json`
   are frozen. Any change after the freeze commit must be logged in
   `docs/DEVIATIONS.md` with the reason and date, and the null outcome is as publishable
   as a positive one. Never re-run E2 with new settings and replace the published result.
10. **Determinism.** Fixed seeds (`experiments/seeds.json`), single-threaded BLAS for parity
    and timing, identical batch order for Nano and PyTorch.
11. **Non-goals** (PRD section 3): no GPU, BatchNorm, dropout, attention, RNNs, plugin
    systems or speed work. Propose these in an issue, not a PR.

## Adding an op (the whole recipe)

```python
@op
def softplus(a):
    """z = log(1 + e^a); dz/da = sigmoid(a). [D4.4]"""
    return np.logaddexp(0.0, a), lambda g: (g / (1 + np.exp(-a)),)
```

Then: derivation section D4.4, a `Case` in `experiments/op_cases.py` plus its `COVERS`
entry, and run `make test`.

## Debugging a gradient

Use the layerwise checker; it names the first op whose local VJP disagrees with finite
differences:

```python
from nano.gradcheck import layerwise_report, format_report
rows, first_bad = layerwise_report(loss)
print(format_report(rows, first_bad))
```

## Conventions

* Python >= 3.10, ruff (line length 120). Type hints on public functions.
* Experiments import `experiments.common` first (it pins BLAS threads before NumPy loads),
  write `results/<ID>.json` via `evalkit.write_fragment`, save figures via `common.save`.
* Commit messages: imperative subject; reference the milestone (M0-M8) or criterion (A1-A10).
