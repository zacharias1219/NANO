# Deviations log

Every departure from the PRD or from the E2 pre-registration, with its reason. Entries are
appended, never edited away.

| ID | Date | What changed | Why | Effect on conclusions |
|---|---|---|---|---|
| D-01 | 2026-11-06 | E2 "trains" is defined as final loss <= 0.5 ln 10 (closing half the gap from chance loss to zero), not "half the gap between initial and chance loss". Written into the pre-registration draft before any run. | With small init the initial loss already equals chance (the PRD's gap is ~0); with exploding init the initial loss is far above chance, so collapsing to the uniform prediction would count as training. | None on validity; makes the outcome well defined for every cell. |
| D-02 | 2026-11-06 | E2 learning-rate grid and step budget fixed after the pilot (seed 99): LRs 0.003 / 0.01 / 0.03, 1,500 steps. | Required by the PRD (open question 1). | Planned, not a deviation in substance; recorded for completeness. |
| D-03 | 2026-11-06 | Two exploratory predictors (X1, X2) registered after the pilot and before the sweep. | The pilot showed small-init ReLU/tanh failing with r ~ 0 (both ends of the network starve). | None on H1, whose predictor, threshold procedure and decision rule are unchanged. X1/X2 are reported as exploratory only. |
| D-04 | 2026-11-06 | Handwritten derivation scans are not yet in `report/derivations/scans/`; the typeset derivations are complete. | Scans must be made by the author on paper. | Rung 1 of the ladder is documented in typeset form only until scans are added. |
| D-05 | 2026-11-06 | Published numbers were produced on a 4-core cloud container (Intel Xeon @ 2.10 GHz), not the HP Omen 16 laptop named in the PRD. | The build environment for this run. Hardware is recorded in every results fragment. | Timings (E5) are specific to this machine; `make repro` on the laptop regenerates them. All correctness results are hardware independent at the stated tolerances. |
| D-06 | 2026-11-06 | The report is rendered to Markdown by `report/build.py`; Quarto renders the same file to HTML when installed (it was not installed in the build environment). | PRD open question 4 (report toolchain). | None on content. |
| D-07 | 2026-11-06 | E1 CNN parity runs use SGD-momentum lr 0.01 instead of the 0.05 first used. | Report review found the MNIST CNN at 11% test accuracy after the run: at lr 0.05 every ReLU died within ~25 steps (failure F3 again). Parity was unaffected (both engines died identically, loss diff 7e-15), but a dead network is a poor parity witness. | A3/A4 still pass; the CNN now actually learns during the parity run. Not part of E2. |
