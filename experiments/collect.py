"""Merge results/*.json into report/results.json and evaluate acceptance criteria A1-A10.

    python -m experiments.collect           # write report/results.json
    python -m experiments.collect --check   # also exit 1 if any criterion fails
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import argparse
import json
import os
import subprocess
import sys

import evalkit

ROOT = common.ROOT
EXPECTED = ["E1", "E2", "E3", "E4", "E5", "E6", "FAILURES"]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def num(x) -> float:
    return float(x) if not isinstance(x, str) else float(x.replace("inf", "inf"))


def frag(frags, eid, *path):
    d = frags.get(eid)
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def criterion(text, threshold, value, ok, by) -> dict:
    return {"criterion": text, "threshold": threshold, "value": value, "pass": ok, "checked_by": by}


def acceptance(frags: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from tests.test_conv import GEOMETRIES, im2col_vs_naive
    from tests.test_optim import CONFIGS, optimizer_max_diff

    A = {}
    m = lambda e, k: frag(frags, e, "metrics", k, "value")  # noqa: E731
    v = m("E1", "A1_max_fd_rel_err")
    A["A1"] = criterion("Per-op gradient vs central finite difference (float64)", "max relative error < 1e-7",
                        v, None if v is None else num(v) < 1e-7, "tests/test_gradcheck.py; E1 table")
    v = m("E1", "A2_max_torch_rel_err_f64")
    A["A2"] = criterion("Per-op output and gradient vs PyTorch (float64)", "max relative error < 1e-10",
                        v, None if v is None else num(v) < 1e-10, "tests/test_parity_ops.py; E1 table")
    v = m("E1", "A3_one_step_param_abs_f64")
    A["A3"] = criterion("One training step vs PyTorch, MLP and CNN (float64)", "max abs diff < 1e-12",
                        v, None if v is None else num(v) < 1e-12, "experiments/parity.py")
    v = m("E1", "A4_run_loss_abs_max_f64")
    A["A4"] = criterion(f"{frag(frags, 'E1', 'config', 'run_steps')}-step run vs PyTorch (float64)",
                        "loss diff < 1e-8 at every step", v, None if v is None else num(v) < 1e-8,
                        "experiments/parity.py")
    a5 = max(optimizer_max_diff(n, kw) for n, kw in CONFIGS)
    A["A5"] = criterion("SGD, momentum, Nesterov, Adam, AdamW vs torch.optim", "per-step param diff < 1e-12",
                        a5, a5 < 1e-12, "tests/test_optim.py")
    a6 = max(im2col_vs_naive(g) for g in GEOMETRIES)
    A["A6"] = criterion("conv2d im2col vs naive-loop reference (forward + backward)", "max abs diff < 1e-12",
                        a6, a6 < 1e-12, "tests/test_conv.py")
    v = m("E4", "bugs_located")
    n = frag(frags, "E4", "metrics", "bugs_located", "n")
    A["A7"] = criterion("Bug-injection demo", "checker names the injected op in 5 of 5 bugs",
                        None if v is None else f"{int(num(v))} of {n}", None if v is None else int(num(v)) == 5 == n,
                        "experiments/bughunt.py")
    prereg = git("log", "--diff-filter=A", "--format=%h %cI", "--", "docs/PREREGISTRATION_E2.md").splitlines()
    freeze = git("log", "-1", "--format=%h %cI", "--", "experiments/e2_config.json")
    cfg_frozen = "PILOT" not in (ROOT / "experiments/e2_config.json").read_text()
    e2 = frags.get("E2", {})
    headline = ["auc_absr_predicts_failure", "spearman_absr_vs_final_loss", "h1_heldout_accuracy",
                "h1_heldout_balanced_accuracy", "fraction_runs_train"]
    with_ci = all(e2.get("metrics", {}).get(k, {}).get("ci") for k in headline)
    quick = any("QUICK" in s for s in e2.get("notes", []))
    A["A8"] = criterion("E2 study pre-registered", "hypothesis, metric and threshold committed before first run; "
                        "CIs on every headline number",
                        {"preregistration_commit": prereg[-1] if prereg else None, "config_freeze_commit": freeze,
                         "config_frozen": cfg_frozen, "headline_numbers_with_ci": with_ci, "verdict": e2.get("verdict")},
                        bool(prereg and cfg_frozen and with_ci and not quick), "repo history plus report")
    loc = sum(len(p.read_text().splitlines()) for p in (ROOT / "nano").glob("*.py"))
    A["A9"] = criterion("Library size", "under 1500 lines in nano/", loc, loc < 1500, "make loc")
    present = [e for e in EXPECTED if e in frags]
    repro = os.environ.get("NANO_REPRO") == "1"
    clean = os.environ.get("NANO_RAW_PREEXISTING") == "0"  # no cached sweep runs when make repro started
    ok = len(present) == len(EXPECTED) and repro
    rec_path = ROOT / "results" / "REPRO.json"  # evidence of a separate clean-clone run (scripts/verify_repro.py)
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    verified = bool(rec.get("make_repro_succeeded") and rec.get("clone_A10", {}).get("from_clean_clone"))
    value = {"fragments_present": present, "ran_under_make_repro": repro, "from_clean_clone": clean}
    if rec:
        value.update(clean_clone_commit=rec["clone_commit"][:10], clean_clone_succeeded=rec["make_repro_succeeded"],
                     deterministic_results_identical=rec["deterministic_results_identical"])
    A["A10"] = criterion("Reproduction", "make repro succeeds from a clean clone and regenerates results.json", value,
                         (True if (clean or verified) else None) if ok else False,
                         "CI job .github/workflows/repro.yml; scripts/verify_repro.py")
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    frags = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")}
    acc = acceptance(frags)
    evalkit.collect(ROOT / "results", ROOT / "report" / "results.json", acceptance=acc)
    for k, c in acc.items():
        print(f"{k:4s} {'PASS' if c['pass'] else ('n/a ' if c['pass'] is None else 'FAIL')}  {c['criterion']}: {c['value']}")
    if args.check and not all(c["pass"] for c in acc.values() if c["pass"] is not None):
        sys.exit(1)


if __name__ == "__main__":
    main()
