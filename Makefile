# Makefile for Systematic Options Auto Hedging Engine
# This Makefile is used to build and manage the Systematic Options Auto Hedging Engine project.

# Variables
PYTHON := python3
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

# Default target
.DEFAULT_GOAL := help

## Setup & Environment
venv:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip setuptools wheel

install: venv
	$(ACTIVATE) && pip install -e .[dev]

lock:
	$(ACTIVATE) && pip freeze > requirements.lock

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info

# Quality Assurance & Testing
lint: 
	$(ACTIVATE) && ruff check src tests
	$(ACTIVATE) && black --check src tests 

format:
	$(ACTIVATE) && black src tests

typecheck:
	$(ACTIVATE) && mypy src

test: 
	$(ACTIVATE) && pytest --cov=src --cov

# Experiments & Backtests
backtest:
	$(ACTIVATE) && python -m src.cli.backtest --config configs/hedging_policy.delta_neutral.yaml

optimize:
	$(ACTIVATE) && python -m src.cli.optimize_policy --config configs/hedging_policy.gamma_scaled.yaml

# Documentation
docs:
	@echo "Docs are in ./docs. Convert with pandoc if needed."

# Utility
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?$$' Makefile | sed 's/:.*//'