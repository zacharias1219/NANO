"""A7: the layerwise checker names the injected op for each of the five silent bugs."""
import pytest

from experiments.bughunt import BUGS, EXTRA, run_case

pytestmark = pytest.mark.slow  # needs MNIST (make data); runs in the nightly suite


def test_control_is_clean():
    _, _, first_bad = run_case()
    assert first_bad is None


@pytest.mark.parametrize("bug", BUGS + EXTRA, ids=[b[0] for b in BUGS + EXTRA])
def test_checker_names_injected_op(bug):
    _, desc, expected, injector = bug
    symptom, _, first_bad = run_case(injector)
    assert symptom > 1e-3, f"{desc}: whole-model check should see a symptom"
    assert first_bad is not None and first_bad.op == expected, f"{desc}: named {first_bad and first_bad.op}"


def test_injection_is_undone():
    from nano import OPS

    before = {k: v.fn for k, v in OPS.items()}
    run_case(BUGS[0][3])
    assert {k: v.fn for k, v in OPS.items()} == before
