"""The results.json contract: environment capture, metric records, fragments and merging.

Every experiment writes ``results/<ID>.json`` (a *fragment*). ``collect`` merges the
fragments into ``report/results.json`` and validates it against
``evalkit/results.schema.json``. The report and the Backprop Machine read only the
merged file, so a number that is not in it cannot appear anywhere.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).with_name("results.schema.json")
SCHEMA_VERSION = "1.0"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _blas() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:  # numpy < 1.25 has no dict mode
        return "unknown"


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_info(root: Path | None = None) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
    except OSError:
        return {"commit": None, "dirty": None}


def env_info() -> dict:
    """Hardware and software versions recorded in every results.json (PRD section 13)."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas(),
        "torch": _version("torch"),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "argv": " ".join(sys.argv),
    }


def metric(value, ci=None, n=None, unit=None, method=None, **extra) -> dict:
    """A headline number. Every headline number in the report should carry a CI."""
    if isinstance(value, dict):  # output of evalkit.stats
        return {**value, **({"unit": unit} if unit else {}), **extra}
    out = {"value": None if value is None else float(value), "ci": ci, "n": n, "unit": unit, "method": method}
    return {**out, **extra}


def _clean(obj):
    """Make numpy scalars/arrays and non-finite floats JSON-safe (NaN/inf -> string)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else ("nan" if np.isnan(f) else ("inf" if f > 0 else "-inf"))
    return obj


def write_fragment(exp_id: str, payload: dict, out_dir: str | Path = "results") -> Path:
    """Write ``results/<exp_id>.json`` with environment and git provenance attached."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"id": exp_id, "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "environment": env_info(), "git": git_info(), **payload}
    path = out_dir / f"{exp_id}.json"
    path.write_text(json.dumps(_clean(doc), indent=1, sort_keys=False) + "\n")
    return path


def collect(fragment_dir="results", out_path="report/results.json", acceptance: dict | None = None,
            project: str = "1A-nano") -> dict:
    """Merge every fragment into one validated results.json."""
    frags = {}
    for p in sorted(Path(fragment_dir).glob("*.json")):
        d = json.loads(p.read_text())
        if "title" not in d:  # bookkeeping records (e.g. REPRO.json) are not experiments
            continue
        frags[d["id"]] = d
    doc = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "environment": env_info(),
        "acceptance": acceptance or {},
        "experiments": frags,
    }
    doc = _clean(doc)
    validate(doc)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def validate(doc: dict) -> None:
    """Validate against the JSON schema (uses ``jsonschema`` when installed, else a structural check)."""
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        import jsonschema
    except ImportError:
        for key in schema["required"]:
            if key not in doc:
                raise ValueError(f"results.json is missing required key {key!r}") from None
        return
    jsonschema.validate(doc, schema)
