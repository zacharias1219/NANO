"""Record a clean-clone `make repro` and compare its numbers with the committed results (A10 evidence).

    python scripts/verify_repro.py <path to the clone that ran make repro> [repro.log]

Writes results/REPRO.json. Deterministic experiments (E2, E3, E4) must match the committed
fragments exactly; timing (E5) is expected to differ and is not compared.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ["E2", "E3", "E4"]


def metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {k: v.get("value") for k, v in d.get("metrics", {}).items()}
    if "verdict" in d:
        out["verdict"] = d["verdict"]
    return out


def main():
    clone = Path(sys.argv[1]).resolve()
    log = Path(sys.argv[2]) if len(sys.argv) > 2 else clone / "repro.log"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()
    acc = json.loads((clone / "report/results.json").read_text())["acceptance"]
    text = log.read_text() if log.exists() else ""
    compared = {}
    for e in DETERMINISTIC:
        a, b = metrics(clone / "results" / f"{e}.json"), metrics(ROOT / "results" / f"{e}.json")
        diffs = {k: [a.get(k), b.get(k)] for k in set(a) | set(b) if a.get(k) != b.get(k)}
        compared[e] = {"identical": not diffs, "differences": diffs}
    doc = {
        "id": "REPRO", "clone_commit": commit,
        "make_repro_succeeded": "make repro: done" in text,
        "clone_acceptance": {k: v["pass"] for k, v in acc.items()},
        "clone_A10": acc["A10"]["value"],
        "deterministic_results_identical": all(c["identical"] for c in compared.values()),
        "compared": compared,
    }
    (ROOT / "results" / "REPRO.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "compared"}, indent=1))
    print({e: c["identical"] for e, c in compared.items()})


if __name__ == "__main__":
    main()
