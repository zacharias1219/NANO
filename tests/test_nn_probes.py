"""Modules, init schemes and probes (F5, F9)."""
import numpy as np
import pytest

from nano import Tensor, init, nn, probes, softmax_cross_entropy


def test_parameters_are_collected_in_layer_order():
    m = nn.mlp(5, 4, 2, 3, act="tanh", rng=np.random.default_rng(0))
    shapes = [p.shape for p in m.parameters()]
    assert shapes == [(4, 5), (4,), (4, 4), (4,), (3, 4), (3,)]
    assert [n for n, _ in m.named_parameters()][0] == "Sequential/0:Linear.weight"


@pytest.mark.parametrize("scheme,expected_std", [
    ("he", np.sqrt(2 / 400)), ("xavier", np.sqrt(2 / 600)), ("small", 0.01)])
def test_init_scales(scheme, expected_std):
    w = init.get(scheme)((200, 400), np.random.default_rng(0))
    assert w.std() == pytest.approx(expected_std, rel=0.02)


def test_conv_fans_include_receptive_field():
    assert init.fans((8, 3, 5, 5)) == (75, 200)


def test_forward_hook_and_scope():
    m = nn.Sequential(nn.Linear(3, 2, rng=np.random.default_rng(0)), nn.ReLU())
    seen = []
    remove = m[1].register_forward_hook(lambda mod, x, out: seen.append(out.shape))
    out = m(Tensor(np.ones((4, 3)), requires_grad=True))
    remove()
    m(Tensor(np.ones((4, 3))))
    assert seen == [(4, 2)]
    assert out._ctx[1][0].scope == "Sequential/0:Linear"


def test_grad_norm_ratio_detects_vanishing_sigmoid_stack():
    rng = np.random.default_rng(0)
    x, y = Tensor(rng.standard_normal((32, 20))), rng.integers(0, 4, 32)
    ratios = {}
    for act, scheme in [("sigmoid", "small"), ("relu", "he")]:
        m = nn.mlp(20, 32, 8, 4, act=act, init=scheme, rng=np.random.default_rng(1))
        softmax_cross_entropy(m(x), y=y).backward()
        ratios[act] = probes.grad_norm_ratio(m)
    assert ratios["sigmoid"] < -5 < ratios["relu"]


def test_activation_probe_reports_dead_units():
    m = nn.Sequential(nn.Linear(4, 6, rng=np.random.default_rng(0)), nn.ReLU())
    m[0].bias.data[:] = [-100, -100, 0, 0, 0, 0]  # two units can never fire
    probe = probes.ActivationProbe(m)
    m(Tensor(np.random.default_rng(1).standard_normal((50, 4))))
    assert probe.summary()["max_dead"] == pytest.approx(2 / 6)
