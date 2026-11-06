"""E5: cost and latency vs PyTorch CPU, single thread: MLP step, CNN step, im2col vs naive conv.

Reports median and IQR over repeated runs, Nano's peak allocation (tracemalloc sees
NumPy buffers), analytic FLOPs and the slowdown factor. Run on an otherwise idle machine.
"""
from __future__ import annotations

from experiments import common  # noqa: I001

import time
import tracemalloc

import numpy as np
import torch

import evalkit
from evalkit.stats import median_iqr
from experiments.parity import make_cnn
from experiments.torch_mirror import mirror
from nano import Tensor, conv2d, conv2d_naive, nn, optim, softmax_cross_entropy

torch.set_num_threads(1)
REPEATS = 5 if common.QUICK else 30


def mlp_flops(sizes, batch):  # forward 2*B*in*out per layer; backward ~2x forward
    return 3 * sum(2 * batch * a * b for a, b in zip(sizes[:-1], sizes[1:]))


def cnn_flops(batch):
    c1 = 2 * batch * 28 * 28 * 8 * 1 * 9
    c2 = 2 * batch * 14 * 14 * 16 * 8 * 9
    fc = 2 * batch * 784 * 10
    return 3 * (c1 + c2 + fc)


def timeit(fn, repeats=REPEATS, warmup=3):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return np.array(out)


def peak_bytes(fn) -> int:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def train_step_pair(model, x, y, dtype):
    tmodel = mirror(model, torch.float64 if dtype == np.float64 else torch.float32)
    opt, topt = optim.SGD(model.parameters(), lr=0.01), torch.optim.SGD(tmodel.parameters(), lr=0.01)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)

    def nano_step():
        loss = softmax_cross_entropy(model(Tensor(x)), y=y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def torch_step():
        loss = torch.nn.functional.cross_entropy(tmodel(tx), ty)
        topt.zero_grad()
        loss.backward()
        topt.step()

    return nano_step, torch_step


def bench(name, nano_fn, torch_fn, flops) -> dict:
    tn, tt = timeit(nano_fn), timeit(torch_fn)
    n, t = median_iqr(tn * 1e3), median_iqr(tt * 1e3)
    row = {"name": name, "nano_ms": n, "torch_ms": t, "slowdown": float(np.median(tn) / np.median(tt)),
           "flops": flops, "nano_gflops": flops / np.median(tn) / 1e9, "torch_gflops": flops / np.median(tt) / 1e9,
           "nano_peak_mb": peak_bytes(nano_fn) / 2**20}
    print(f"  {name:30s} nano {n['value']:8.2f} ms  torch {t['value']:7.2f} ms  x{row['slowdown']:.1f}", flush=True)
    return row


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    rows = []
    for dname, dt in (("float64", np.float64), ("float32", np.float32)):
        x, y = rng.standard_normal((128, 784)).astype(dt), rng.integers(0, 10, 128)
        m = nn.mlp(784, 128, 2, 10, rng=np.random.default_rng(0)).astype(dt)
        rows.append(bench(f"MLP 784-128-128-10, B=128, {dname}", *train_step_pair(m, x, y, dt),
                          mlp_flops([784, 128, 128, 10], 128)))
        xc = rng.standard_normal((64, 1, 28, 28)).astype(dt)
        rows.append(bench(f"CNN 2xconv, B=64, {dname}", *train_step_pair(make_cnn(np.random.default_rng(0)).astype(dt),
                                                                          xc, y[:64], dt), cnn_flops(64)))
    # im2col vs the naive loop (forward + backward of one conv layer)
    x = Tensor(rng.standard_normal((8, 4, 14, 14)), requires_grad=True)
    w = Tensor(rng.standard_normal((8, 4, 3, 3)), requires_grad=True)
    b = Tensor(np.zeros(8), requires_grad=True)
    g = rng.standard_normal((8, 8, 14, 14))

    def conv_fb(fn):
        return lambda: fn(x, w, b, stride=1, padding=1).backward(g)

    fl = 3 * 2 * 8 * 14 * 14 * 8 * 4 * 9
    fast, slow = timeit(conv_fb(conv2d), repeats=REPEATS), timeit(conv_fb(conv2d_naive), repeats=max(3, REPEATS // 6))
    rows.append({"name": "conv 4->8, 3x3, 14x14, B=8: naive loop vs im2col", "nano_ms": median_iqr(slow * 1e3),
                 "torch_ms": median_iqr(fast * 1e3), "slowdown": float(np.median(slow) / np.median(fast)), "flops": fl,
                 "comparison": "naive (nano_ms) vs im2col (torch_ms column)"})
    print(f"  naive conv {np.median(slow) * 1e3:.1f} ms vs im2col {np.median(fast) * 1e3:.2f} ms", flush=True)
    evalkit.write_fragment("E5", {
        "title": "E5 cost and latency vs PyTorch CPU (single thread)",
        "config": {"repeats": REPEATS, "threads": 1, "torch_threads": torch.get_num_threads()},
        "metrics": {
            "mlp_slowdown_float64": evalkit.metric(rows[0]["slowdown"], unit="x", method="median time ratio"),
            "cnn_slowdown_float64": evalkit.metric(rows[1]["slowdown"], unit="x", method="median time ratio"),
            "mlp_step_ms_float64": evalkit.metric(rows[0]["nano_ms"]),
            "cnn_step_ms_float64": evalkit.metric(rows[1]["nano_ms"]),
            "im2col_speedup_over_naive": evalkit.metric(rows[-1]["slowdown"], unit="x"),
        },
        "tables": {"bench": rows}, "runtime_s": time.perf_counter() - t0,
        "notes": ["Peak memory is Nano's peak traced NumPy allocation during one step (tracemalloc); PyTorch's "
                  "allocator is not visible to tracemalloc, so no PyTorch peak is claimed.",
                  "FLOPs are analytic (backward counted as 2x forward)."],
    }, common.ROOT / "results")


if __name__ == "__main__":
    main()
