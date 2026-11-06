# Nano - one command regenerates every number and figure: `make repro`.
# Parity and timing runs are single-threaded (PRD: determinism); the E2/E3 sweeps fan out over processes.

PY      ?= python3
JOBS    ?= $(shell $(PY) -c "import os; print(os.cpu_count())")
QUICK   ?= 0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NANO_QUICK=$(QUICK)
export NANO_JOBS=$(JOBS)

.PHONY: help install data test test-fast lint loc parity e1 e2 e2-pilot e3 e4 e5 e6 failures experiments \
        collect figures report repro repro-quick machine clean

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## install nano + dev tools (PyTorch CPU is the test oracle)
	$(PY) -m pip install -e ".[dev]"

data:  ## download MNIST + Fashion-MNIST and verify SHA-256
	$(PY) -m experiments.data

test:  ## full test suite (A1, A2, A5, A6, properties, isolation, A9)
	$(PY) -m pytest

test-fast:  ## the per-commit CI subset
	$(PY) -m pytest -m "not slow" -x

lint:  ## ruff
	ruff check .

loc:  ## A9: lines in nano/ (fails at >= 1500)
	@$(PY) scripts/loc.py

parity e1:  ## E1 + A3/A4: per-op, per-step, per-run parity vs PyTorch, float64 and float32
	$(PY) -m experiments.parity

e2-pilot:  ## E2 pilot (seed 99 only, not part of the study) used to fix the step budget
	$(PY) -m experiments.trainability --pilot

e2:  ## E2 trainability study (pre-registered: docs/PREREGISTRATION_E2.md)
	$(PY) -m experiments.trainability

e3:  ## E3 optimizer x init x schedule comparison
	$(PY) -m experiments.optim_grid

e4:  ## E4 + A7: bug-injection demo
	$(PY) -m experiments.bughunt

e5:  ## E5 cost and latency vs PyTorch CPU
	$(PY) -m experiments.speed

e6:  ## E6 finite-difference error vs step size h
	$(PY) -m experiments.fd_error

failures:  ## failure gallery
	$(PY) -m experiments.failures

experiments: parity e2 e3 e4 e5 e6 failures  ## every experiment

collect:  ## merge results/*.json into report/results.json and evaluate A1-A10
	$(PY) -m experiments.collect

report:  ## render the report from report/results.json (Quarto HTML if installed)
	$(PY) report/build.py
	@command -v quarto >/dev/null && quarto render report || echo "quarto not installed: report/REPORT.md is the rendered report"

repro: export NANO_REPRO=1
repro: export NANO_RAW_PREEXISTING=$(shell test -d results/raw/e2 && echo 1 || echo 0)
repro: data test experiments collect report  ## everything, from a clean clone
	@echo "make repro: done. See report/REPORT.md and report/results.json"

repro-quick:  ## reduced-size smoke run of the whole pipeline (CI); numbers are NOT the published ones
	$(MAKE) repro QUICK=1

machine:  ## serve the v1.1 Backprop Machine at http://localhost:8000/machine/
	$(PY) -m http.server 8000

clean:
	rm -rf results/raw report/_site .pytest_cache .ruff_cache
