"""A1: every op's gradient matches central finite differences, max relative error < 1e-7 (float64)."""
import numpy as np
import pytest

from experiments.op_cases import COVERS, cases
from nano import OPS, Tensor
from nano.gradcheck import gradcheck, layerwise_report, rel_error

A1_TOL = 1e-7


def test_every_registered_op_has_a_case():
    names = {c.name for c in cases()}
    missing = [op for op in OPS if COVERS.get(op) not in names]
    assert not missing, f"ops without a gradcheck case: {missing}"


@pytest.mark.parametrize("case", cases(), ids=lambda c: c.name)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_op_matches_finite_differences(case, seed):
    xs = case.inputs(np.random.default_rng(seed))
    ts = [Tensor(x, requires_grad=True) for x in xs]
    errs = gradcheck(case.nano_fn, *ts, seed=seed)
    assert max(errs) < A1_TOL, f"{case.name}: {errs}"


def test_composite_expression():
    rng = np.random.default_rng(7)
    a, b = Tensor(rng.standard_normal((3, 4)), requires_grad=True), Tensor(rng.standard_normal(4), requires_grad=True)
    f = lambda a, b: ((a * b).tanh() @ (a.T / (b.reshape(4, 1) ** 2 + 1))).logsumexp(axis=1).sigmoid()
    assert max(gradcheck(f, a, b)) < A1_TOL


def test_layerwise_report_passes_on_correct_model():
    from nano import nn, softmax_cross_entropy

    rng = np.random.default_rng(0)
    m = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                      nn.Linear(18, 8, rng=rng), nn.Sigmoid(), nn.Linear(8, 3, rng=rng))
    loss = softmax_cross_entropy(m(Tensor(rng.standard_normal((4, 1, 6, 6)))), y=np.array([0, 1, 2, 1]))
    rows, first_bad = layerwise_report(loss)
    assert first_bad is None, rows
    assert [r.op for r in rows][:3] == ["conv2d", "relu", "maxpool2d"]  # forward execution order


def test_blind_spot_relu_kink_is_real():
    """Documented blind spot: at x = 0 the central difference sees slope 1/2, the subgradient is 0."""
    x = Tensor(np.zeros(3), requires_grad=True)
    assert rel_error(np.zeros(3), np.full(3, 0.5)) == 1.0
    assert gradcheck(lambda x: x.relu(), x)[0] == pytest.approx(1.0)


def test_blind_spot_maxpool_tie_is_real():
    """Documented blind spot: a 2x2 window of equal values; FD splits the gradient, max routes it to one."""
    x = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
    from nano import maxpool2d

    assert gradcheck(lambda x: maxpool2d(x, k=2), x)[0] > 0.5
