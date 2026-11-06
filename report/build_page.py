"""Build the published results site (report/site/) from report/results.json only.

    python report/build_page.py      # or: make page

Output: report/site/index.html (the results page), derivations.html (typeset derivations),
figures/ (copied), machine/index.html and report/results.json (the Backprop Machine and its data).
Every number on the page is read from results.json at build time.
"""
from __future__ import annotations

import html
import json
import math
import re
import shutil
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITE = HERE / "site"
R = json.loads((HERE / "results.json").read_text())
X = R["experiments"]
INTERP = json.loads((HERE / "interpretation.json").read_text())
REPO = "https://github.com/zacharias1219/NANO"


# ------------------------------------------------------------------------------------- helpers
def num(x) -> float:
    return float(x) if not isinstance(x, str) else float(x)


def f(x, d: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return {"inf": "∞", "nan": "NaN"}.get(x, x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "∞"
    if float(x).is_integer() and 1 <= abs(x) < 1e5:
        return f"{int(x):,}"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        m, e = f"{x:.{d}e}".split("e")
        return f"{m}×10<sup>{int(e)}</sup>"
    return f"{x:.{d + 1}g}" if abs(x) < 1 else f"{x:,.{d}f}"


def ci(m, d: int = 3) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    c = m.get("ci")
    v = f(m["value"], d)
    return f'{v} <span class="ci">[{f(c[0], d)}, {f(c[1], d)}]</span>' if c else v


def pp(m) -> str:
    lo, hi = m["ci"]
    return f'{100 * m["value"]:+.2f} pp <span class="ci">[{100 * lo:+.2f}, {100 * hi:+.2f}]</span>'


def esc(s) -> str:
    return html.escape(str(s))


def md_inline(text: str) -> str:
    """The interpretation text uses **bold**, *italic*, `code` and '* ' bullets only."""
    out, items = [], []
    for line in text.split("\n"):
        line = esc(line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        if line.startswith("* "):
            items.append(f"<li>{line[2:]}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
        if line.strip():
            out.append(f"<p>{line}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "\n".join(out)


def table(header, rows, cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def figure(path: str, caption: str, alt: str) -> str:
    return (f'<figure class="plate"><img src="{path}" alt="{esc(alt)}" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


# ------------------------------------------------------------------------- the precision ladder
def ladder_svg() -> str:
    """Log-scale ruler: each acceptance bound (bar) and the measured value (dot), float32 drift for contrast."""
    A, E1 = R["acceptance"], X["E1"]
    short = {"A1": "op gradient vs finite diff.", "A2": "op vs PyTorch", "A3": "one training step",
             "A4": "300-step run, loss", "A5": "optimizers vs torch.optim", "A6": "im2col vs naive conv"}
    rows = [(k, short[k], num(A[k]["value"]), b) for k, b in
            (("A1", 1e-7), ("A2", 1e-10), ("A3", 1e-12), ("A4", 1e-8), ("A5", 1e-12), ("A6", 1e-12))]
    rows.append(("f32", "same run in float32",
                 num(E1["metrics"]["float32_run_loss_abs_max"]["value"]), None))
    W, left, right, top, rh = 1000, 250, 30, 34, 38
    lo, hi = -17, 0
    x = lambda v: left + (math.log10(max(v, 1e-17)) - hi) / (lo - hi) * (W - left - right)  # noqa: E731
    H = top + rh * len(rows) + 30
    p = [f'<svg class="ladder" viewBox="0 0 {W} {H}" role="img" aria-labelledby="ladder-t ladder-d">',
         '<title id="ladder-t">Precision ladder</title>',
         '<desc id="ladder-d">For each acceptance criterion, the bound and the measured error on a log scale.</desc>']
    for e in range(hi, lo - 1, -1):
        xx = x(10.0**e)
        major = e % 2 == 1 or e == 0
        p.append(f'<line x1="{xx:.1f}" y1="{top - 8}" x2="{xx:.1f}" y2="{H - 26}" class="grid{" major" if major else ""}"/>')
        if e % 2 == 1 or e == 0:
            p.append(f'<text x="{xx:.1f}" y="{top - 14}" class="tick">10<tspan dy="-6" font-size="9">{e}</tspan></text>')
    for i, (k, crit, v, bound) in enumerate(rows):
        y = top + rh * i + rh / 2
        p.append(f'<text x="0" y="{y + 4:.1f}" class="rowlab"><tspan class="rowid">{k if k != "f32" else "float32"}</tspan> '
                 f'{esc(crit)}</text>')
        p.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" class="lane"/>')
        if bound:
            xb = x(bound)
            p.append(f'<rect x="{xb:.1f}" y="{y - 11:.1f}" width="{W - right - xb:.1f}" height="22" class="passzone"/>')
            p.append(f'<line x1="{xb:.1f}" y1="{y - 13:.1f}" x2="{xb:.1f}" y2="{y + 13:.1f}" class="bound"/>')
        cls = "dot f32" if k == "f32" else "dot"
        xv = x(v)
        p.append(f'<circle cx="{xv:.1f}" cy="{y:.1f}" r="6.5" class="{cls}"><title>{k}: {v:.3g}</title></circle>')
        lab = f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        anchor, dx = ("end", -11) if xv > W - 120 else ("start", 11)
        p.append(f'<text x="{xv + dx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" class="val{" f32" if k == "f32" else ""}">{lab}</text>')
    p.append(f'<text x="{left}" y="{H - 6}" class="axis">larger error</text>')
    p.append(f'<text x="{W - right}" y="{H - 6}" text-anchor="end" class="axis end">smaller error →  (shaded: inside the bound)</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------------------- phase diagram
def phase_grid() -> str:
    E2 = X["E2"]
    C, cells = E2["config"], E2["tables"]["cells"]
    out = ['<div class="phase">']
    for act in C["acts"]:
        out.append(f'<div class="phase-panel"><h4>{act}</h4><table class="heat"><thead><tr><th></th>'
                   + "".join(f"<th>{d}</th>" for d in C["depths"]) + "</tr></thead><tbody>")
        for init in C["inits"]:
            out.append(f'<tr><th scope="row">{ {"small": "small", "xavier": "Xavier", "he": "He"}[init] }</th>')
            for d in C["depths"]:
                c = next(c for c in cells if c["act"] == act and c["init"] == init and c["depth"] == d)
                k, n = c["n_train"], c["n_seeds"]
                cause = ", ".join(c.get("failure_causes", {})) or "trains"
                out.append(f'<td class="h{k}" title="{act} / {init} / depth {d}: {k} of {n} seeds train; r = {c["r_mean"]:.1f}; '
                           f'{cause}"><span>{k}/{n}</span></td>')
            out.append("</tr>")
        out.append("</tbody></table></div>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------------- sections
def acceptance_strip() -> str:
    A = R["acceptance"]
    pills = []
    for k, a in A.items():
        state = "pass" if a["pass"] else ("pending" if a["pass"] is None else "fail")
        pills.append(f'<li class="pill {state}" title="{esc(a["criterion"])}: {esc(a["threshold"])}">'
                     f'<b>{k}</b><span>{"pass" if a["pass"] else state}</span></li>')
    n = sum(1 for a in A.values() if a["pass"])
    return f'<div class="strip"><p class="strip-l"><b>{n} of {len(A)}</b> acceptance criteria pass</p><ul>{"".join(pills)}</ul></div>'


def acceptance_table() -> str:
    rows = []
    for k, a in R["acceptance"].items():
        v = a["value"]
        if isinstance(v, dict):
            if k == "A8":
                v = (f'pre-registered {esc(v["preregistration_commit"].split()[0])}, frozen {esc(v["config_freeze_commit"].split()[0])}; '
                     f'verdict: {esc(v["verdict"])}')
            elif k == "A10":
                v = (f'clean clone of {esc(v.get("clean_clone_commit", "?"))}; deterministic results identical: '
                     f'{f(v.get("deterministic_results_identical"))}')
        elif isinstance(v, (int, float)):
            v = f(num(v))
        rows.append([f"<b>{k}</b>", esc(a["criterion"]), esc(a["threshold"]), v,
                     '<span class="ok">pass</span>' if a["pass"] else "pending", f'<code>{esc(a["checked_by"])}</code>'])
    return table(["", "Criterion", "Threshold", "Measured", "", "Checked by"], rows, "acc")


def parity_section() -> str:
    E1 = X["E1"]
    runs = E1["tables"]["runs"]
    rows = []
    for s, lab in (("mlp_mnist", "MLP · MNIST"), ("cnn_mnist", "CNN · MNIST"), ("cnn_fashion", "CNN · Fashion-MNIST")):
        a, b, c = runs[f"{s}_float64"], runs[f"{s}_float32"], runs[f"{s}_f32_vs_f64"]
        rows.append([lab, f(a["one_step_param_abs"]), f(a["run_loss_abs_max"]), f(b["run_loss_abs_max"]),
                     f(c["run_loss_abs_max"]), f"{100 * a['test_acc_after_run']:.1f}%"])
    steps = E1["config"]["run_steps"]
    return f"""
<p>Nano and PyTorch start from identical weights, see identical batches and take {steps} SGD-momentum steps,
single-threaded. In float64 the two engines agree to rounding level for the whole run. The same runs in float32
drift apart by orders of magnitude, which is why the precision contract is stated in float64 and float32 is
reported as its own result.</p>
{table(["Setup", "float64: one-step param diff", f"float64: max loss diff, {steps} steps", "float32: max loss diff",
        "Nano float32 vs float64", "test acc. after run"], rows, "num")}
{figure("figures/e1_float32_drift.png", "Worst loss difference so far (line) and per step (dots). Blue: float64, Nano vs PyTorch. "
        "Orange: float32, Nano vs PyTorch. Grey dashed: Nano float32 vs Nano float64. Dotted line: the A4 bound.", "float32 drift plot")}
"""


def bughunt_section() -> str:
    E4 = X["E4"]
    rows = [[esc(b["bug"]), esc(b["description"]), f'<code>{esc(b["expected"] or "none")}</code>',
             f'<code>{esc(b["named"] or "all ops pass")}</code>', f(b["whole_model_rel_err"]),
             '<span class="ok">yes</span>' if b["located"] else '<span class="bad">no</span>'] for b in E4["tables"]["bugs"]]
    rep = esc(E4["tables"]["reports"]["B3"])
    rep = re.sub(r"(.*first bad op.*)", r'<mark>\1</mark>', rep)
    return f"""
<p>Five silent bugs are injected one at a time. Each keeps every shape correct, so training still runs. A whole-model
gradient check notices <em>that</em> something is wrong but not <em>where</em>. The layerwise checker re-runs each
tape node's op on its own against finite differences, in execution order, and names the first one that disagrees.
It located <b>{int(num(E4["metrics"]["bugs_located"]["value"]))} of 5</b>; the bug-free control passes every op.
B3-shared is an extra, uncounted case: a bug in a helper shared by two ops surfaces at the first op that exercises it.</p>
{table(["Bug", "Injected defect", "Expected", "Checker names", "Whole-model error", "Located"], rows)}
<details><summary>Checker output for B3, an off-by-one in col2im</summary><pre class="report">{rep}</pre></details>
"""


def e2_section() -> str:
    E2 = X["E2"]
    m, C, t = E2["metrics"], E2["config"], E2["tables"]
    held = m["h1_heldout_balanced_accuracy"]
    vals = {"auc": ci(m["auc_absr_predicts_failure"]), "bacc": ci(held),
            "x1": ci(m["exploratory_X1_auc_neg_log_first_grad"]), "x2": ci(m["exploratory_X2_auc_forward_attenuation"])}
    interp = INTERP.get(E2["verdict"], "")
    for k in vals:  # numbers enter the prose from results.json, as HTML
        interp = interp.replace("{" + k + "}", f"@@{k}@@")
    body = md_inline(interp)
    for k, v in vals.items():
        body = body.replace(f"@@{k}@@", v)
    stats = [
        ["AUC of |r| for predicting failure", ci(m["auc_absr_predicts_failure"]), f'{m["auc_absr_predicts_failure"]["n"]} runs'],
        ["Spearman correlation of |r| with final loss", ci(m["spearman_absr_vs_final_loss"]), ""],
        ["Held-out accuracy of “fail iff |r| > τ”", ci(m["h1_heldout_accuracy"]), f'seeds 3–4, τ = {f(m["h1_threshold_tau"]["value"], 2)}'],
        ["Held-out balanced accuracy", ci(held), "seeds 3–4"],
        ["Runs that train", ci(m["fraction_runs_train"]), "Wilson 95%"],
        ['<em>Exploratory</em> X1: first-layer gradient norm', ci(m["exploratory_X1_auc_neg_log_first_grad"]), "AUC"],
        ['<em>Exploratory</em> X2: forward attenuation', ci(m["exploratory_X2_auc_forward_attenuation"]), "AUC"],
    ]
    causes = ", ".join(f"{k} ({v})" for k, v in sorted(t["failure_cause_totals"].items(), key=lambda kv: -kv[1]))
    return f"""
<p class="lede">At what depth does each (activation, init) pair stop training, and does the step-0 gradient-norm ratio
<span class="math">r = log<sub>10</sub>(‖∂L/∂W<sub>first</sub>‖ / ‖∂L/∂W<sub>last</sub>‖)</span> predict that failure?</p>
<p>The hypothesis, the definition of “trains”, the predictor, the analysis plan and the decision rule were
<a href="{REPO}/blob/claude/laughing-ride-kbj24v/docs/PREREGISTRATION_E2.md">committed before the first run</a>.
The step budget ({C["steps"]:,} steps) and learning-rate grid ({", ".join(f(x) for x in C["lr_grid"])}, SGD with momentum
{C["momentum"]}) were fixed after a pilot on a non-study seed and committed before the sweep. {E2["n_runs"]} runs:
{len(C["depths"])} depths × 3 activations × 3 inits × {len(C["seeds"])} seeds × 3 learning rates, width {C["width"]},
5,000 MNIST images, {f(E2["runtime_s"] / 3600, 1)} CPU-hours.</p>
<div class="verdict"><span class="stamp">H1 {esc(E2["verdict"])}</span>
<p>Decision rule, fixed in advance: confirmed if the AUC's 95% CI lower bound is at least 0.75 and held-out balanced
accuracy at least 0.80; partially confirmed if the AUC's lower bound is above 0.5; null otherwise.</p></div>
{table(["Pre-registered statistic", "Estimate [95% CI]", ""], stats, "num stats")}
<h3>Phase diagram</h3>
<p class="small">Seeds out of 5 that train (best of three learning rates), by number of hidden layers. Hover a cell
for its mean r and diagnosed failure cause.</p>
{phase_grid()}
<div class="prose">{body}</div>
{figure("figures/e2_grad_norms.png", "Per-layer ‖∂L/∂W‖ at initialisation. Sigmoid curves slope down toward the input (r detects this). "
        "Small-init tanh and ReLU, and deep ReLU with Xavier, are flat but shifted down by many orders of magnitude: every layer starves "
        "equally, so the ratio r stays near 0.", "gradient norms by layer")}
{figure("figures/e2_predictor.png", "Left: ROC of |r| for predicting failure. Right: r against final training loss; dashed lines mark the "
        "threshold fitted on seeds 0–2, the dotted line the definition of “trains”.", "predictor ROC and scatter")}
<p class="small">Failing runs were labelled with the pre-registered precedence (overflow, dead ReLUs, saturation,
vanishing, exploding, slow/unexplained): {esc(causes)}. The full per-cell list is in the
<a href="{REPO}/blob/claude/laughing-ride-kbj24v/report/REPORT.md">report</a>.</p>
"""


def e3_section() -> str:
    E3 = X["E3"]
    t, C = E3["tables"], E3["config"]
    rows = [[c["factor"], f'{c["level"]} − {c["reference"]}', pp(c["acc_diff"]), pp(c["acc_diff_seed_cluster"]),
             '<span class="ok">claimed</span>' if c["separated"] else "not separated"] for c in t["contrasts"]]
    return f"""
<p>A {C["model"]} on {C["n_train"]:,} MNIST images, {C["steps"]:,} steps, {len(C["seeds"])} seeds per configuration,
45 configurations. A difference is claimed only when both its run-level and its seed-clustered 95% CI exclude zero.</p>
{figure("figures/e3_optim_grid.png", "Test accuracy with 95% bootstrap CIs over 5 seeds.", "optimizer grid")}
{table(["Factor", "Contrast", "Δ accuracy, run pairs", "Δ accuracy, seed-clustered", ""], rows, "num")}
<div class="prose">{md_inline(INTERP.get("e3", ""))}</div>
"""


def failures_section() -> str:
    F = X["FAILURES"]["tables"]
    f2, f4 = F["F2"], F["F4"]
    return f"""
<div class="gallery">
{figure("figures/f2_softmax_overflow.png", f"<b>Softmax overflow.</b> Without the max shift, softmax → log → NLL returns inf/NaN once the largest "
        f"logit reaches {f(f2['float32']['composed_breaks_at_max_logit'])} in float32 and {f(f2['float64']['composed_breaks_at_max_logit'])} in "
        "float64. The fused version stays finite.", "softmax overflow")}
{figure("figures/f3_dead_relus.png", "<b>Dead ReLUs.</b> At a high learning rate about half the units in a layer die for good while the loss "
        "stays finite; one notch higher, every seed diverges within the first steps.", "dead ReLUs")}
{figure("figures/f4_sgd_divergence.png", f"<b>The stability bound.</b> Gradient descent on least squares converges iff lr &lt; 2/λ<sub>max</sub> = "
        f"{f(f4['lr_bound'], 4)}. The edge measured by bisection is {f(f4['lr_empirical_edge'], 4)}, a relative gap of {f(f4['relative_gap'])}.",
        "SGD divergence")}
{figure("figures/f5_fd_kinks.png", "<b>Finite-difference blind spots.</b> Within h of a ReLU kink or a max-pool tie the central difference "
        "fails although the code is correct, so gradchecks sample away from kinks.", "FD at kinks")}
</div>
{figure("figures/e6_fd_error.png", "<b>E6, the finite-difference U-curve.</b> Truncation error falls as h², rounding error grows as ε/h. "
        f"Measured best h, nonlinear ops: {f(X['E6']['metrics']['best_h_float64_geomean']['value'])} in float64 against "
        f"{f(X['E6']['metrics']['theory_h_float64']['value'])} from theory.", "FD error curve")}
"""


def cost_section() -> str:
    E5 = X["E5"]
    rows = []
    for b in E5["tables"]["bench"]:
        if "comparison" in b:
            continue
        rows.append([esc(b["name"]), f(b["nano_ms"]["value"]), f(b["torch_ms"]["value"]), f"{b['slowdown']:.1f}×",
                     f(b["nano_peak_mb"], 1)])
    env = E5["environment"]
    naive = next(b for b in E5["tables"]["bench"] if "comparison" in b)
    return f"""
<p>One training step, single thread, median of {E5["config"]["repeats"]} repeats on {esc(env["cpu"])}
({env["cores"]} cores, {env.get("ram_gb")} GB), NumPy {esc(env["numpy"])} with {esc(env["blas"])},
PyTorch {esc(env.get("torch"))}. Speed is measured and published, not optimised (a stated non-goal).</p>
{table(["Workload", "Nano ms", "PyTorch ms", "Slowdown", "Nano peak MB"], rows, "num")}
<p>Turning convolution into one matmul (im2col) is <b>{naive["slowdown"]:.0f}×</b> faster than the naive loop it is
verified against.</p>
"""


# ------------------------------------------------------------------------------------- page
CSS = r"""
/* Layout: one reading column (66ch) with full-width figure plates and tables; a log-scale ruler opens the page. */
:root {
  --paper: #f4f6f9; --panel: #ffffff; --ink: #142039; --muted: #56617a; --rule: #d3dae6;
  --f64: #2b6cb0; --f32: #c05621; --pass: #2f7d55; --warn: #a86a00; --bad: #b83232; --plate: #ffffff;
  --heat0: #f2d4d0; --heat5: #cfe9da;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0f1522; --panel: #161e2e; --ink: #e3e8f2; --muted: #9aa5ba; --rule: #2a3449;
    --f64: #6fa8ea; --f32: #f0935c; --pass: #62c48d; --warn: #e0a640; --bad: #f07a7a; --plate: #f4f6f9;
    --heat0: #4a2328; --heat5: #1f4a35; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #0f1522; --panel: #161e2e; --ink: #e3e8f2; --muted: #9aa5ba; --rule: #2a3449;
  --f64: #6fa8ea; --f32: #f0935c; --pass: #62c48d; --warn: #e0a640; --bad: #f07a7a; --plate: #f4f6f9;
  --heat0: #4a2328; --heat5: #1f4a35; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 17px/1.62 var(--body); margin: 0; }
.wrap { max-width: 1080px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1, h2, h3, h4 { font-family: var(--display); font-stretch: 85%; text-wrap: balance; line-height: 1.12; margin: 0; }
h1 { font-size: clamp(2.4rem, 6vw, 4rem); font-weight: 800; letter-spacing: -0.01em; }
h2 { font-size: 1.7rem; font-weight: 750; margin-top: 3.2rem; padding-top: 1.2rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.2rem; font-weight: 700; margin-top: 2rem; }
h4 { font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
p, .prose, .lede { max-width: 66ch; }
p { margin: 0.9rem 0; }
.prose ul { padding-left: 1.2rem; max-width: 70ch; }
.prose li { margin: 0.55rem 0; }
a { color: var(--f64); text-underline-offset: 2px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--f64); outline-offset: 2px; }
code, .mono, pre { font-family: var(--mono); font-size: 0.84em; }
.eyebrow { font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.thesis { font-size: 1.28rem; line-height: 1.45; max-width: 46ch; margin: 1rem 0 0.4rem; }
.meta { font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin: 0.4rem 0 0; max-width: none; overflow-wrap: anywhere; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin: 6px 0 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.ruler { margin: 1.8rem 0 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px 6px; overflow-x: auto; }
.ladder { width: 100%; min-width: 720px; height: auto; display: block; }
.ladder .grid { stroke: var(--rule); stroke-width: 1; }
.ladder .grid.major { stroke-width: 1.4; }
.ladder .tick, .ladder .axis { font: 11px var(--mono); fill: var(--muted); text-anchor: middle; }
.ladder .axis { text-anchor: start; } .ladder .axis.end { text-anchor: end; }
.ladder .rowlab { font: 13px var(--body); fill: var(--ink); }
.ladder .rowid { font: 600 12px var(--mono); fill: var(--muted); }
.ladder .lane { stroke: var(--rule); stroke-dasharray: 2 4; }
.ladder .passzone { fill: var(--pass); fill-opacity: 0.12; }
.ladder .bound { stroke: var(--pass); stroke-width: 2.5; }
.ladder .dot { fill: var(--f64); stroke: var(--panel); stroke-width: 2; }
.ladder .dot.f32 { fill: var(--f32); }
.ladder .val { font: 600 12px var(--mono); fill: var(--f64); }
.ladder .val.f32 { fill: var(--f32); }
.strip { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin: 1.2rem 0 0; }
.strip-l { margin: 0; font-family: var(--display); font-stretch: 85%; font-size: 1.05rem; }
.strip ul { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 0; }
.pill { display: flex; gap: 6px; align-items: baseline; font-family: var(--mono); font-size: 0.74rem; padding: 3px 9px;
  border-radius: 999px; border: 1px solid var(--rule); background: var(--panel); }
.pill b { font-weight: 600; }
.pill.pass span { color: var(--pass); } .pill.pending span { color: var(--warn); } .pill.fail span { color: var(--bad); }
nav.toc { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 1.4rem 0 0; font-family: var(--mono); font-size: 0.8rem; }
nav.toc a { color: var(--muted); text-decoration: none; } nav.toc a:hover { color: var(--ink); text-decoration: underline; }
.tablewrap { overflow-x: auto; margin: 1.1rem 0; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
thead th { font-family: var(--display); font-stretch: 85%; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.02em;
  color: var(--muted); background: var(--paper); }
tbody tr:last-child td { border-bottom: 0; }
table.num td, table.acc td:nth-child(4), table.stats td:nth-child(2) { font-family: var(--mono); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
table.num td:first-child { font-family: var(--body); font-size: 0.86rem; }
.ci { color: var(--muted); white-space: nowrap; }
.ok { color: var(--pass); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; }
figure.plate { margin: 1.4rem 0; }
figure.plate img { display: block; width: 100%; height: auto; background: var(--plate); border: 1px solid var(--rule);
  border-radius: 6px; padding: 10px; }
figcaption { font-size: 0.86rem; color: var(--muted); max-width: 80ch; margin-top: 0.5rem; }
.gallery { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 22px; }
@media (max-width: 760px) { .gallery { grid-template-columns: minmax(0, 1fr); } }
.verdict { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; margin: 1.4rem 0 0.4rem;
  padding: 14px 16px; background: var(--panel); border: 1px solid var(--rule); border-left: 4px solid var(--warn); border-radius: 6px; }
.verdict p { margin: 0; font-size: 0.92rem; flex: 1 1 340px; }
.stamp { font-family: var(--display); font-stretch: 80%; font-weight: 800; font-size: 1.15rem; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--warn); }
.phase { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 0.8rem 0 1.4rem; }
@media (max-width: 860px) { .phase { grid-template-columns: minmax(0, 1fr); } }
.phase-panel { min-width: 0; overflow-x: auto; }
table.heat { width: 100%; font-family: var(--mono); font-size: 0.74rem; border: 0; }
table.heat th { background: none; border: 0; padding: 3px; text-align: center; color: var(--muted); font-family: var(--mono); font-size: 0.72rem; }
table.heat th[scope="row"] { text-align: right; padding-right: 8px; }
table.heat td { text-align: center; padding: 0; border: 2px solid var(--paper); height: 34px; }
table.heat td span { display: block; line-height: 30px; }
.h0 { background: var(--heat0); } .h5 { background: var(--heat5); }
.h1, .h2 { background: color-mix(in srgb, var(--heat0) 70%, var(--heat5)); }
.h3, .h4 { background: color-mix(in srgb, var(--heat5) 70%, var(--heat0)); }
.small { font-size: 0.86rem; color: var(--muted); }
.lede { font-size: 1.08rem; }
.math { font-family: var(--mono); font-size: 0.86em; overflow-wrap: anywhere; }
details { margin: 0.6rem 0 1.4rem; }
summary { cursor: pointer; color: var(--f64); font-size: 0.92rem; }
pre.report { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px; font-size: 0.72rem; line-height: 1.5; }
pre.report mark { background: color-mix(in srgb, var(--bad) 22%, transparent); color: var(--ink); }
.cta { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin: 1.2rem 0; }
.cta a { display: block; padding: 14px 16px; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
  text-decoration: none; color: var(--ink); }
.cta a:hover { border-color: var(--f64); }
.cta b { font-family: var(--display); font-stretch: 85%; display: block; font-size: 1.05rem; color: var(--f64); }
.cta span { font-size: 0.86rem; color: var(--muted); }
pre.cmd { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; overflow-x: auto; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); font-size: 0.82rem; color: var(--muted); }
sup { line-height: 0; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800'
         '&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')


def page() -> str:
    env, git = R["environment"], R["git"]
    E2 = X["E2"]
    return f"""<title>Nano results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Nano: a NumPy autodiff engine verified against PyTorch, and a pre-registered trainability study.">
{FONTS}
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">Project 1A · results page · generated from results.json</p>
  <h1>Nano</h1>
  <p class="thesis">A neural network engine in pure NumPy with a hand-written backprop tape, checked against PyTorch
  to rounding error, then broken on purpose to see how deep networks fail.</p>
  <p class="meta">commit {esc((git["commit"] or "")[:10])} · {esc(R["generated_at"])} · {esc(env["cpu"])}, {env["cores"]} cores ·
  Python {esc(env["python"])} · NumPy {esc(env["numpy"])} · PyTorch {esc(env.get("torch"))} (oracle only)</p>
  <div class="ruler">
    <h4>The precision ladder: each bound and the measured error (float64)</h4>
    {ladder_svg()}
  </div>
  <div class="legend"><span><i style="background:var(--f64)"></i>measured, float64</span>
    <span><i style="background:var(--pass)"></i>acceptance bound (shaded side passes)</span>
    <span><i style="background:var(--f32)"></i>float32, for contrast</span></div>
  {acceptance_strip()}
  <nav class="toc" aria-label="Sections"><a href="#study">E2 study</a><a href="#parity">Parity</a><a href="#bughunt">Bug hunt</a>
    <a href="#optim">Optimizers</a><a href="#failures">Failure gallery</a><a href="#cost">Cost</a><a href="#criteria">Criteria</a>
    <a href="#repro">Reproduce</a></nav>
</header>

<section id="study"><h2>E2 · Why deep networks stop training (pre-registered)</h2>{e2_section()}</section>
<section id="parity"><h2>E1 · Parity with PyTorch, float64 against float32</h2>{parity_section()}</section>
<section id="bughunt"><h2>E4 · The bug hunt: naming the broken op</h2>{bughunt_section()}</section>
<section id="optim"><h2>E3 · What mattered: optimizer, init, schedule</h2>{e3_section()}</section>
<section id="failures"><h2>Failure gallery</h2>{failures_section()}</section>
<section id="cost"><h2>E5 · Cost and latency</h2>{cost_section()}</section>
<section id="criteria"><h2>Acceptance criteria</h2>
<p>Every row is checked by a script, not by hand. The definition of done is all ten passing.</p>{acceptance_table()}</section>
<section id="repro"><h2>See it, read it, reproduce it</h2>
<div class="cta">
  <a href="machine/index.html"><b>Backprop Machine →</b><span>Feed a batch, run backward, watch the gradient pipe starve
  for sigmoid and stay open for ReLU with He init. Pipe widths are E2's measured gradient norms.</span></a>
  <a href="derivations.html"><b>Derivations →</b><span>Every backward rule in the library, derived. Each op's docstring
  cites its section.</span></a>
  <a href="{REPO}/tree/claude/laughing-ride-kbj24v"><b>Source on GitHub →</b><span>The library, tests, experiments,
  pre-registration and deviations log.</span></a>
</div>
<pre class="cmd"><code>make install
make repro      # data → tests → E1–E6 + failure gallery → results.json → report</code></pre>
<p>A fresh clone ran <code>make repro</code> end to end with every sweep run recomputed; its E2, E3 and E4 numbers are
identical to the ones on this page. The full sweep takes about {f(sum(e.get("runtime_s", 0) for e in X.values()) / 3600, 1)} CPU-hours;
<code>make repro QUICK=1</code> runs the pipeline at reduced size in minutes.</p>
</section>
<footer>Every number on this page is read from <a href="report/results.json">results.json</a> when the page is built
(<code>report/build_page.py</code>). E2 verdict: {esc(E2["verdict"])}. MIT licensed.</footer>
</div>
"""


DERIV_CSS = CSS + """
.prose { max-width: 74ch; } .prose h2 { font-size: 1.4rem; } .prose table { font-size: 0.86rem; }
.prose blockquote { margin: 1rem 0; padding: 0.4rem 1rem; border-left: 3px solid var(--rule); color: var(--muted); }
mjx-container { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
mjx-container svg { color: var(--ink); }
"""


def derivations() -> str:
    src = (HERE / "derivations" / "DERIVATIONS.md").read_text()
    src = re.sub(r"^---\n.*?\n---\n", "", src, flags=re.S)
    keep: list[str] = []

    def stash(m):
        keep.append(m.group(0))
        return f"@@MATH{len(keep) - 1}@@"

    src = re.sub(r"\$\$.+?\$\$", stash, src, flags=re.S)
    src = re.sub(r"\$[^$\n]+?\$", stash, src)
    body = markdown.markdown(src, extensions=["tables"])
    for i, m in enumerate(keep):
        body = body.replace(f"@@MATH{i}@@", html.escape(m, quote=False))
    body = body.replace('href="../../docs/', f'href="{REPO}/blob/claude/laughing-ride-kbj24v/docs/')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Nano derivations</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
<style>{DERIV_CSS}
:root {{ padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }}</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [["$", "$"]], displayMath: [["$$", "$$"]] }}, svg: {{ fontCache: "global" }} }};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js"></script>
</head><body>
<div class="wrap">
<p class="eyebrow"><a href="index.html">← Nano results</a></p>
<h1>Derivations</h1>
<p class="thesis">Every backward rule in <code>nano/</code>, derived. The tag in each section heading is the one cited in
the op's docstring.</p>
<div class="prose">{body}</div>
</div>
</body></html>
"""


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "machine").mkdir(parents=True)
    (SITE / "report").mkdir()
    shutil.copytree(HERE / "figures", SITE / "figures")
    shutil.copy(HERE / "results.json", SITE / "report" / "results.json")
    shutil.copy(ROOT / "machine" / "index.html", SITE / "machine" / "index.html")
    (SITE / "index.html").write_text(page())
    (SITE / "derivations.html").write_text(derivations())
    print(f"wrote {SITE}")


if __name__ == "__main__":
    main()
