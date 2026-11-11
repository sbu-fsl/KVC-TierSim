# KVC-TierSim - common tasks. Run `make` (or `make help`) for the list.
#
# Variables you can override:  make sim ARGS="--stack h200 ..."
#   PYTHON         interpreter used to create the venv          (default: python3)
#   VENV           virtualenv directory                          (default: .venv)

PYTHON       ?= python3
VENV         ?= .venv
PY           := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
STAMP        := $(VENV)/.installed
ARGS         ?=

SIM_ARGS  := $(if $(ARGS),$(ARGS),--stack h200 --total-blocks 50000 --miss-blocks 500 --request-rate 0.18 --p95-seconds 10)

.DEFAULT_GOAL := help
.PHONY: help venv install upgrade clean clean-venv sim dashboard sim-sweep sim-overhead sim-figures smoke

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^## / {printf "\n\033[1m%s\033[0m\n", substr($$0, 4)} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

## Environment
$(STAMP): requirements.txt pyproject.toml
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	$(PIP) install -q --no-deps -e .
	@touch $(STAMP)
	@echo "environment ready: source $(VENV)/bin/activate"

venv: $(STAMP) ## Create .venv and install everything (re-runs when requirements change)
install: $(STAMP) ## Alias for venv

upgrade: ## Upgrade all dependencies in the venv
	$(PIP) install -q --upgrade -r requirements.txt
	@touch $(STAMP)

## Simulator (KVC-TierSim)
sim: $(STAMP) ## Compare policies for one point (ARGS="..." to override)
	$(PY) -m simulator $(SIM_ARGS)

dashboard: $(STAMP) ## Web dashboard on http://localhost:8000
	$(VENV)/bin/uvicorn simulator.api.app:app --reload --port 8000

sim-sweep: $(STAMP) ## P95 x request-rate sweep (ARGS="..." passed through)
	$(PY) -m simulator.sweep $(ARGS)

sim-overhead: $(STAMP) ## Scheduler-overhead sweep (ARGS="..." passed through)
	$(PY) -m simulator.overhead $(ARGS)

sim-figures: $(STAMP) ## Restore-vs-recompute figures (paper)
	$(PY) -m simulator.figures

## Housekeeping
smoke: $(STAMP) ## Quick end-to-end check (simulator CLI + API imports)
	$(PY) -m simulator $(SIM_ARGS) > /dev/null && echo "simulator ok"
	$(PY) -c "import simulator.api.app, simulator.sweep, simulator.overhead; print('imports ok')"

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf *.egg-info build

clean-venv: ## Delete the virtualenv
	rm -rf $(VENV)
