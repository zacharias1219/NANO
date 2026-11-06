# E2 pre-registration: trainability vs depth, activation and initialisation

Status: **DRAFT, committed before any E2 run.** Two values are deliberately left open
(marked `PILOT`) and will be fixed after a pilot on a seed that is not part of the study
(PRD section 15, first open question). The commit that fixes them is made before the
full sweep starts. Any later change is logged in `docs/DEVIATIONS.md`.

## Question

At what depth does each (activation, init) pair stop training, and does the step-0
gradient-norm ratio predict that failure?

## Design

| Element | Specification |
|---|---|
| Model | MLP, `depth` hidden layers of width 128, then a 128 -> 10 output layer. Every Linear layer (including the output layer) uses the cell's init; biases are zero. |
| Depths | 2, 4, 8, 16, 32, 64 hidden layers |
| Activations | sigmoid, tanh, ReLU |
| Inits | small normal (N(0, 0.01^2)), Xavier normal, He normal |
| Cells | 6 x 3 x 3 = 54 |
| Data | MNIST training subset of 5,000 images (fixed subset, `seeds.json: data_subset_seed`), standardised. 1,000 MNIST test images for a secondary accuracy readout only. |
| Seeds | 0, 1, 2, 3, 4 (5 per cell). The seed sets the init and the batch order. |
| Optimizer | SGD with momentum 0.9, batch size 128, no weight decay, no gradient clipping (clipping would hide the failure under study). |
| Learning rate | 3-value grid per cell: `PILOT` |
| Step budget | `PILOT`, identical for every cell |
| Precision | float64 |

## Outcome definitions (fixed now)

* **Final loss** `L_final`: mean cross-entropy over the full 5,000-image training subset,
  evaluated after the last step. A run whose loss becomes non-finite has `L_final = inf`.
* **Trains**: `L_final <= 0.5 * ln(10)`, i.e. the run closes at least half of the gap
  between the chance loss `ln 10` and zero.
  *Operationalisation note.* The PRD phrase "closes 50 percent of the gap between initial
  loss and chance loss" is ambiguous: with small init the initial loss is already almost
  exactly `ln 10` (the gap is ~0), and with exploding init the initial loss is far above
  chance, so collapsing to the uniform prediction would count as training. Measuring the
  gap from chance loss to zero avoids both degenerate readings. Logged as deviation D-01.
* **Per (cell, seed) outcome**: the run trains if it trains at **any** of the three
  learning rates, and its final loss is the minimum over the three. This is the control
  that keeps "does not train" from meaning "wrong learning rate". The learning rate that
  won is recorded per cell (mode over seeds).

## Predictor (fixed now)

`r = log10(||dL/dW_1||_F / ||dL/dW_last||_F)` at step 0, where `W_1` is the first Linear
layer (784 -> 128) and `W_last` the output layer (128 -> 10). It is computed on 4 fixed
batches of 128 training images and averaged (mean of the four `r` values). `r` does not
depend on the learning rate. Strongly negative `r` means gradients vanish toward the
input; strongly positive `r` means they explode toward the input. The score used for
prediction is `|r|`.

## Hypothesis H1

Runs with `|r|` above a threshold `tau` fail to train. `tau` is fitted on seeds 0-2 by
maximising balanced accuracy of the rule "predict fail iff `|r| > tau`" (candidate
thresholds are midpoints between sorted `|r|` values), and is then evaluated on
seeds 3-4 only.

## Analysis plan (fixed now)

1. **AUC** of `|r|` for predicting failure, over all 270 (cell, seed) runs, with a
   95% percentile-bootstrap CI (10,000 resamples, resampling runs).
2. **Spearman** correlation of `|r|` with `L_final`, with bootstrap CI (diverged runs
   rank as the largest loss).
3. **Held-out test of H1**: accuracy and balanced accuracy of the fitted rule on seeds
   3-4, with bootstrap CIs.
4. **Per-cell fraction of seeds that train**, with Wilson 95% intervals.
5. **Phase diagram** over depth by (activation, init), and gradient-norm heatmaps
   (per-layer `||dL/dW||` at step 0, mean over seeds).
6. **Failure inspection.** Every failing run is labelled, in this order of precedence:
   `overflow` (non-finite loss at any point) -> `dead ReLUs` (a ReLU layer with > 50% of
   units dead on 512 training images at the end of training) -> `saturation` (a
   sigmoid/tanh layer with > 50% of units saturated at step 0) -> `vanishing gradient`
   (`r < -2`) -> `exploding gradient` (`r > 2`) -> `slow / unexplained`. Only the last
   three are counted as gradient-flow failures.

## Decision rule (fixed now)

* **Confirmed**: AUC 95% CI lower bound >= 0.75 **and** held-out balanced accuracy >= 0.80.
* **Partially confirmed**: AUC 95% CI lower bound > 0.5, but the confirmed bar is not met.
* **Null**: the AUC CI includes 0.5, or every run has the same outcome (AUC undefined).

All three outcomes are accepted. A null result is reported with the same prominence as a
positive one, together with the full phase diagram.

## Cost

54 cells x 5 seeds x 3 learning rates = 810 runs. The measured wall time is recorded in
`results/E2.json` (`runtime_s`) and replaces the PRD's estimate.
