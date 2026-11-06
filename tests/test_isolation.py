"""Non-functional requirements: nano/ never imports torch (isolation) and stays under 1500 lines (A9)."""
import ast
import subprocess
import sys
from pathlib import Path

NANO = Path(__file__).resolve().parents[1] / "nano"
A9_LIMIT = 1500


def test_nano_source_never_imports_torch():
    for path in NANO.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
            mod = getattr(node, "module", None) or ""
            assert not any(n.split(".")[0] == "torch" for n in names) and not mod.startswith("torch"), path


def test_importing_nano_does_not_load_torch():
    code = "import sys, nano; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=NANO.parent)


def test_library_size_under_limit():
    total = sum(len(p.read_text().splitlines()) for p in NANO.glob("*.py"))
    assert total < A9_LIMIT, f"nano/ has {total} lines"


def test_every_op_has_a_derivation_reference_and_is_short():
    import inspect

    from nano import OPS

    for name, op in OPS.items():
        src = inspect.getsource(op.fn)
        assert "[D" in (op.__doc__ or ""), f"{name} lacks a derivation reference"
        assert len(src.splitlines()) < 30, f"{name} is {len(src.splitlines())} lines"
