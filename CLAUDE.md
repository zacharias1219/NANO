# CLAUDE.md

@AGENTS.md

## Notes specific to Claude Code

- Before claiming a change works, run `make lint loc test-fast` and report the output
  honestly. For changes to ops, conv, optimizers or the tape also run `make test`.
- Long jobs (`make e2`, `make repro`) take hours. Run them in the background and keep
  working; they are resumable (cached under `results/raw/`).
- Do not regenerate `results/E2.json` with different settings than the frozen
  `experiments/e2_config.json`. If the user asks for a variant, write it as a new,
  clearly labelled experiment and log it in `docs/DEVIATIONS.md`.
- When editing `report/REPORT.md`, edit `report/build.py` instead: the Markdown is generated.
- Keep `nano/` free of convenience features; the line budget is part of the proof.
