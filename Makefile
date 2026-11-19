# SupplyIQ — all build steps. Every reported number comes from `make metrics`.
PY            ?= python3.11
VENV          := .venv
VPY           := $(VENV)/bin/python
VPIP          := $(VENV)/bin/pip
STREAMLIT     := $(VENV)/bin/streamlit
DEPLOY_VENV   := .deploycheck/venv

.DEFAULT_GOAL := help
.PHONY: help setup ingest run test metrics check-deploy download-data clean lint

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install dev deps, build the data sample
	$(PY) -m venv $(VENV)
	$(VPIP) install --quiet --upgrade pip
	$(VPIP) install --quiet -r requirements-dev.txt
	$(MAKE) ingest
	@echo "\nReady. Run 'make run' to start the app."

ingest: ## Build data/sample/ (real Kaggle data if data/raw/ has it, else synthetic)
	$(VPY) scripts/ingest.py

ingest-real: ## Build from raw Kaggle files only; fails if they are absent
	$(VPY) scripts/ingest.py --source kaggle

ingest-synthetic: ## Force the synthetic fallback sample
	$(VPY) scripts/ingest.py --source synthetic

download-data: ## Download the raw Kaggle datasets into data/raw/ (needs ~/.kaggle/kaggle.json)
	bash scripts/download_data.sh

run: ## Start the Streamlit app locally
	$(STREAMLIT) run streamlit_app.py

test: ## Run the test suite
	$(VPY) -m pytest -q

metrics: ## Recompute every reported number into reports/metrics.json
	$(VPY) scripts/metrics.py

check-deploy: ## Fresh venv with ONLY requirements.txt, then boot the app headless
	rm -rf .deploycheck
	mkdir -p .deploycheck
	$(PY) -m venv $(DEPLOY_VENV)
	$(DEPLOY_VENV)/bin/pip install --quiet --upgrade pip
	$(DEPLOY_VENV)/bin/pip install --quiet -r requirements.txt
	@echo "booting app headless for 20s with runtime deps only ..."
	@bash scripts/check_deploy.sh $(DEPLOY_VENV)

clean: ## Remove build scratch (keeps data/sample and reports)
	rm -rf .deploycheck .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
