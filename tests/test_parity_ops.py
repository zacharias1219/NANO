"""A2: every op's output and gradient match PyTorch in float64, max relative error < 1e-10."""
import numpy as np
import pytest

from experiments.op_cases import cases, run_nano, run_torch
from nano.gradcheck import rel_error

A2_TOL = 1e-10


@pytest.mark.parametrize("case", cases(), ids=lambda c: c.name)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_op_matches_pytorch(case, seed):
    rng = np.random.default_rng(100 + seed)
    xs = case.inputs(rng)
    out_n, grads_n = run_nano(case, xs)
    u = rng.standard_normal(out_n.shape)
    out_n, grads_n = run_nano(case, xs, u)
    out_t, grads_t = run_torch(case, xs, u)
    assert rel_error(out_n, out_t) < A2_TOL
    for gn, gt in zip(grads_n, grads_t):
        assert rel_error(gn, gt) < A2_TOL


@pytest.mark.parametrize("case", [c for c in cases() if c.name in ("matmul", "tanh", "conv2d", "logsumexp")],
                         ids=lambda c: c.name)
def test_float32_is_supported_and_close(case):
    """float32 runs end to end and stays dtype-stable; its error is studied in E1, not hidden."""
    xs = case.inputs(np.random.default_rng(0), dtype=np.float32)
    out_n, grads_n = run_nano(case, xs)
    out_t, grads_t = run_torch(case, xs)
    assert out_n.dtype == np.float32 and all(g.dtype == np.float32 for g in grads_n)
    assert rel_error(out_n, out_t) < 1e-5
