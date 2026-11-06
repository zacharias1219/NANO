# Decision log

Architecture decisions from the PRD (section 7) plus the open questions resolved during
the build. Each entry: what was chosen, what was rejected, why.

| # | Decision | Chosen | Rejected | Why |
|---|---|---|---|---|
| 1 | Graph construction | Dynamic tape built during the forward pass | Static graph | Simpler to read and debug, matches PyTorch, the proof needs no graph optimisation |
| 2 | Conv implementation | im2col + matmul, with a naive-loop reference | Direct loops only; FFT conv | Reuses the tested matmul backward; the naive loop is an independent oracle |
| 3 | Softmax cross entropy | Fused, max-shifted | softmax -> log -> NLL | Stable, backward is `p - onehot`; the composed version is kept for the overflow demo |
| 4 | Broadcasting gradients | One `unbroadcast(grad, shape)` applied centrally in `backward` | Per-op handling | Broadcast bugs are the most common failure; one helper, one place to test hard |
| 5 | Verification precision | float64 | float32 | float32 rounding makes long-run parity unfalsifiable; float32 is studied as a failure case |
| 6 | Extensibility | `@op`: a function returning `(out, vjp)` | Class per op | Shows the engine's idea in a few lines; the registry also lets the bug hunt swap an op |
| 7 | Reference framework | PyTorch in tests/experiments only | Reimplementing a reference | Accepted oracle; isolation keeps `nano/` pure NumPy (enforced by a test) |
| 8 | Op signature | Ops take and return ndarrays; `unbroadcast` and dtype handling live in `Op.__call__`/`backward` | Ops that manipulate Tensors | Lets the layerwise checker re-run a single op on perturbed inputs without the tape |
| 9 | Layer attribution | Each tape node records the module scope that produced it | Matching by shapes | The bug-hunt report can say "add in Sequential/7:Linear", not just "add" |
| 10 | Report toolchain (open question 4) | Python renders `report/REPORT.md` from `results.json`; Quarto renders it to HTML when installed; matplotlib for figures | Quarto + Jupyter engine | One source that is readable on GitHub without a build, and still renders with Quarto |
| 11 | CNN architecture (open question 2) | conv3x3(8) - ReLU - pool - conv3x3(16) - ReLU - pool - linear(10), batch 64 | Deeper CNN | Two conv layers exercise stacking, padding and pooling; keeps parity and timing runs short |
| 12 | E2 predictor timing (open question 3) | Step 0 only | Also later steps | Pre-registered as step 0: the useful claim is prediction *before* training |
| 13 | CI scope (open question 5) | Per commit: lint, A9, tests. Nightly: full tests + E1 + E4 + E6. Weekly/on demand: full `make repro` (A10) | Full repro per commit | The full sweep takes hours; the nightly parity suite fits a standard runner |
| 14 | Datasets | MNIST + Fashion-MNIST, pinned by SHA-256 | MNIST only | Avoids tying the CNN claim to one dataset |
