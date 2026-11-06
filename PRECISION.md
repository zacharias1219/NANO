# The precision contract

"Matches PyTorch" is only a claim if it says *how closely, in what precision, and over
what horizon*. Nano's claim is tiered, stated in float64, and checked automatically
(`make test` on every commit, `make parity` nightly). Every threshold below is an
acceptance criterion in the PRD; the measured value lives in `report/results.json`.

| Tier | What is compared | Precision | Threshold | Checked by |
|---|---|---|---|---|
| A1 | each op's gradient vs central finite differences (h = 1e-6) | float64 | max rel. error < 1e-7 | `tests/test_gradcheck.py` |
| A2 | each op's output and gradient vs PyTorch | float64 | max rel. error < 1e-10 | `tests/test_parity_ops.py` |
| A3 | parameters after one training step vs PyTorch (MLP, CNN) | float64 | max abs. diff < 1e-12 | `experiments/parity.py` |
| A4 | training loss at every step of a 300-step run vs PyTorch | float64 | abs. diff < 1e-8 | `experiments/parity.py` |
| A5 | parameters after each optimizer step vs `torch.optim` | float64 | max abs. diff < 1e-12 | `tests/test_optim.py` |
| A6 | im2col conv vs the naive-loop conv, forward and backward | float64 | max abs. diff < 1e-12 | `tests/test_conv.py` |

## Error metric

For two tensors `a` (Nano) and `b` (reference) the relative error is

    rel_error(a, b) = max_i |a_i - b_i| / max(max_i |a_i|, max_i |b_i|)

(zero if both are zero). It is scale-aware but does not blow up on individual
near-zero entries, which elementwise `|a_i - b_i| / |b_i|` would do for gradients that
legitimately pass through zero. Absolute criteria (A3-A6) are used where magnitudes are
O(1) by construction.

## Why float64, and what float32 does

Two correct implementations that sum in different orders differ by about one unit in
the last place per operation. In float64 that is 1e-16 and stays near rounding level for
hundreds of steps (E1 measures ~1e-14). In float32 it is 1e-7 per operation and compounds
through training, so a "matches to 1e-6 over a long run" claim in float32 is either false
or unfalsifiable. Nano therefore:

* states the contract in float64;
* **supports** float32 end to end (dtype is preserved through every op and optimizer);
* **publishes** the float32 drift against PyTorch-float32 and against Nano-float64 as its
  own result (E1, `report/figures/e1_float32_drift.png`).

## Known blind spots (stated up front)

* **Kinks and ties.** Finite differences are invalid within `h` of a ReLU kink or a
  max/max-pool tie (failure gallery F5). Gradcheck inputs for `relu`, `max` and
  `maxpool2d` are sampled away from kinks and ties; the layerwise checker evaluates those
  three ops at a jittered point. The blind spot itself is tested
  (`test_blind_spot_relu_kink_is_real`, `test_blind_spot_maxpool_tie_is_real`).
* **Tie-breaking conventions.** `max` splits the gradient equally among ties (like
  `torch.amax`); `maxpool2d` routes it to the first maximum (like `F.max_pool2d`).
* **PyTorch is assumed correct.** It is the oracle; the FD checks (A1) and the naive conv
  (A6) are the two checks that do not depend on it.
* **Determinism.** All parity runs are single-threaded (`OMP_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`) with identical
  batch order.

## Choosing h

The gradcheck step `h = 1e-6` is taken from the E6 error curve: theory puts the optimum
of the central-difference error at `(3 eps)^(1/3) ~ 6e-6` in float64 with minimum error
`~ eps^(2/3) ~ 4e-11` (DERIVATIONS.md D11), comfortably inside the A1 bound of 1e-7.
