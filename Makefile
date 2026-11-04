# KVC-TierSim + KVWL — common tasks. Run `make` (or `make help`) for the list.
#
# Variables you can override:  make sweep JOBS=8   make sim ARGS="--stack h200 ..."
#   PYTHON         interpreter used to create the venv          (default: python3)
#   VENV           virtualenv directory                          (default: .venv)
#   JOBS           parallel workers for sweeps                   (default: all cores)
#   SECRETS        env file with HF_TOKEN / GH_TOKEN (default: secrets.env, git-ignored)
#   DATA_DIR       data store for raw sources, streams, download cache
#                  (default: $$KVWL_DATA_DIR, else datasets/data; on ibm1.fsl: /mnt/gpfs/kvc_tiersim_kvwl)

# Per-host settings written by scripts/setup_host.sh (git-ignored), e.g. DATA_DIR=/mnt/gpfs/kvc_tiersim_kvwl
-include local.env

PYTHON       ?= python3
VENV         ?= .venv
PY           := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
STAMP        := $(VENV)/.installed
JOBS         ?=
ARGS         ?=
NAME         ?= LongBench
SECRETS      ?= secrets.env
DATA_DIR     ?= $(KVWL_DATA_DIR)
STORE        := $(if $(DATA_DIR),$(DATA_DIR),datasets/data)

# every python step sees the same data store
ifneq ($(DATA_DIR),)
export KVWL_DATA_DIR := $(DATA_DIR)
endif

# Prefix for recipes that need credentials: loads $(SECRETS) into the recipe's shell only.
WITH_SECRETS := set -a; [ ! -f "$(abspath $(SECRETS))" ] || . "$(abspath $(SECRETS))"; set +a;

JOBS_FLAG := $(if $(JOBS),--jobs $(JOBS),)
SIM_ARGS  := $(if $(ARGS),$(ARGS),--stack h200 --total-blocks 50000 --miss-blocks 500 --request-rate 0.18 --p95-seconds 10)

.DEFAULT_GOAL := help
.PHONY: help venv install upgrade clean clean-venv clean-raw data-dir \
        sim dashboard sim-sweep sim-overhead sim-figures \
        fetch fetch-hf fetch-check fetch-list \
        survey catalog catalog-check catalog-show catalog-stats \
        profile profile-hf sweep sensitivity analysis advisor atlas explorer figures pipeline \
        smoke

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^## / {printf "\n\033[1m%s\033[0m\n", substr($$0, 4)} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

## Environment
secrets.env: ## Create secrets.env from the template (then fill it in)
	@test -f $@ && echo "$@ already exists" || (cp secrets.env.example $@ && chmod 600 $@ && echo "created $@ - fill in your tokens")

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

## Data (downloaded only; big files live outside git)
data-dir: ## Show (and create) the data store: raw/, streams/, .cache/
	@mkdir -p "$(STORE)/raw" "$(STORE)/streams" "$(STORE)/.cache"
	@echo "data store: $(abspath $(STORE))"; du -sh "$(STORE)/raw" "$(STORE)/streams" "$(STORE)/.cache" 2>/dev/null || true

fetch: $(STAMP) ## Download raw sources (files, archives, GitHub dirs) into datasets/data/raw
	$(WITH_SECRETS) $(PY) -m kvwl.fetch

fetch-hf: $(STAMP) ## Download + normalize the Hugging Face datasets (HF_TOKEN for gated sets)
	$(WITH_SECRETS) $(PY) -m kvwl.fetch --hf-only

fetch-check: $(STAMP) ## Check that every source in catalog/sources.yaml is reachable
	$(WITH_SECRETS) $(PY) -m kvwl.fetch --check

fetch-list: $(STAMP) ## List the raw sources
	$(PY) -m kvwl.fetch --list


## Catalog (papers <-> workloads <-> models)
survey: $(STAMP) ## Re-rank the survey evidence into datasets/survey/out
	$(PY) -I datasets/survey/rank.py datasets/survey datasets/survey/out 50

catalog: $(STAMP) ## Regenerate papers.yaml + paper links in workloads/models.yaml
	$(PY) -m kvwl.catalog sync
	$(PY) -m kvwl.catalog check

catalog-check: $(STAMP) ## Validate catalog cross-references
	$(PY) -m kvwl.catalog check

catalog-show: $(STAMP) ## Show one workload / model / paper with its links (NAME=...)
	$(PY) -m kvwl.catalog show "$(NAME)"

catalog-stats: $(STAMP) ## Counts of papers, workloads, models
	$(PY) -m kvwl.catalog stats

## Dataset analysis pipeline (KVWL)
profile: $(STAMP) ## Build request streams + workload profiles (needs: make fetch)
	$(PY) -m kvwl.profile

profile-hf: $(STAMP) ## Rebuild `published` workloads from Hugging Face data (needs: make fetch-hf)
	$(PY) -m kvwl.profile --use-hf

sweep: $(STAMP) ## Main replay sweep: workloads x top-50 models x techniques (hours)
	$(PY) -m kvwl.sweep --grid main $(JOBS_FLAG)

sensitivity: $(STAMP) ## Sensitivity sweep (load, SLO, hardware)
	$(PY) -m kvwl.sweep --grid sensitivity $(JOBS_FLAG)

analysis: $(STAMP) ## Tables in datasets/data/analysis
	$(PY) -m kvwl.analysis

advisor: $(STAMP) ## Gain predictor + representative suites in datasets/data/advisor
	$(PY) -m kvwl.advisor

atlas: $(STAMP) ## Full-coverage figures in datasets/figures
	$(PY) -m kvwl.atlas

explorer: $(STAMP) ## Interactive explorer datasets/figures/kv_workload_atlas.html
	$(PY) -m kvwl.explorer

figures: atlas explorer ## atlas + explorer

pipeline: analysis advisor figures ## Everything downstream of the sweeps

## Housekeeping
smoke: $(STAMP) ## Quick end-to-end check (simulator, catalog, imports)
	$(PY) -m simulator $(SIM_ARGS) > /dev/null && echo "simulator ok"
	$(PY) -m kvwl.catalog check
	$(PY) -c "import simulator.api.app, kvwl.replay, kvwl.sweep, kvwl.analysis, kvwl.atlas; print('imports ok')"

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
	rm -rf *.egg-info build

clean-raw: ## Delete downloaded raw sources + download cache (refuses on a DATA_DIR store unless FORCE=1)
	@if [ -n "$(DATA_DIR)" ] && [ "$(FORCE)" != 1 ]; then echo "refusing to delete the permanent store $(DATA_DIR) (use FORCE=1)"; exit 1; fi
	rm -rf "$(STORE)/raw" "$(STORE)/.cache" datasets/.cache

clean-venv: ## Delete the virtualenv
	rm -rf $(VENV)
