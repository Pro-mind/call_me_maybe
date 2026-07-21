PYTHON = uv run python
SRC = src

export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export HF_HOME := $(CURDIR)/.huggingface-cache

.PHONY: install run debug lint lint-strict clean

install:
	uv sync

run:
	$(PYTHON) -m $(SRC) \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

debug:
	$(PYTHON) -m pdb -m $(SRC) \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

lint:
	uv run flake8 .
	uv run mypy $(SRC) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy $(SRC) --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
