.DEFAULT_GOAL := run

UV ?= uv
ARGS ?=

.PHONY: install run debug clean lint lint-flake8 lint-mypy lint-strict

install:
	$(UV) sync

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

clean:
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	find src -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache __pycache__

lint: lint-flake8 lint-mypy

lint-flake8:
	$(UV) run flake8 src

lint-mypy:
	$(UV) run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src
	$(UV) run mypy src --strict