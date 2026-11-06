# E2 pre-registration: trainability vs depth, activation and initialisation

Status: **FROZEN.** The draft (commit `914ecfa`) was committed before any E2 run, with two
values left open and fixed after a pilot on seed 99, which is not a study seed (PRD
section 15, first open question). This version fixes them and is committed before the
first study-seed run. Any later change is logged in `docs/DEVIATIONS.md`.

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
| Learning rate | 3-value grid per cell: 0.003, 0.01, 0.03 (fixed after the pilot) |
| Step budget | 1,500 steps (= 38.4 epochs of the 5,000 images), identical for every cell (fixed after the pilot) |
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

## Pilot (seed 99, not a study seed) and how it fixed the open values

The pilot ran depths 2, 8, 32, 64 for all 9 (activation, init) pairs at four learning rates
(0.003, 0.01, 0.03, 0.1) for 1,500 steps (`results/E2_pilot.json`).

* **Learning-rate grid.** No pilot cell trained *only* at 0.1, while ReLU/He and tanh/He at
  depth 64 trained *only* at 0.003. The grid is therefore {0.003, 0.01, 0.03}: every value
  was exercised in the pilot, and the grid covers the best LR of every pilot cell that
  trained at any LR.
* **Step budget.** Separation between training and non-training cells was clear at 1,500
  steps, and some borderline cells (ReLU/Xavier at depth 32) were still moving between
  steps 1,000 and 1,500, so the budget is the full pilot horizon of 1,500 steps.
* **The 50% threshold** of the "trains" definition is unchanged.

**An observation the pilot made, and what was done about it.** Small-init ReLU and tanh
networks failed from depth 8 on with `r` close to 0. With small init the signal dies in
the forward pass, so the gradients at *both* ends of the network are tiny and their ratio
is unremarkable. This suggests H1 may be only partially confirmed. The pre-registered
predictor, threshold procedure and decision rule are **not** changed in response (changing
them after seeing pilot data would defeat the pre-registration). Instead, two exploratory
predictors are registered here, before the sweep, and will be reported as exploratory:

* **X1** `g1 = -log10 ||dL/dW_1||_F` at step 0 (the absolute first-layer gradient norm;
  large means the input layer receives almost no gradient).
* **X2** `q = |log10(std(h_D) / std(h_1))|` at step 0, where `h_1` and `h_D` are the first
  and last hidden activations on 512 training images (forward signal attenuation).

For each: AUC for predicting failure with a bootstrap CI over all runs. No threshold is
fitted and neither enters the H1 decision.

## Cost

54 cells x 5 seeds x 3 learning rates = 810 runs. The measured wall time is recorded in
`results/E2.json` (`runtime_s`) and replaces the PRD's estimate.
