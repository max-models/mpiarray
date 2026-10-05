# Development environment (uv), README (Quarto) and the documentation site (docs/, Astro + Starlight).
# Notebook tool: docs/tools/notebooks.py.
PYTHON ?= .venv/bin/python
NOTEBOOKS = $(PYTHON) docs/tools/notebooks.py
DOCS_ENV = DOCS_PYTHON=$(shell $(PYTHON) -c "import sys; print(sys.executable)")
FORCE ?=
# Open MPI needs --oversubscribe for more ranks than cores; with MPICH: make coverage MPIEXEC=mpiexec
MPIEXEC ?= mpiexec --oversubscribe

.PHONY: help install test coverage lint readme readme-check docs-install docs-notebooks docs-dev docs-build docs-preview docs-clean

help:  ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## uv sync with the dev extra, and the pre-commit hooks
	uv sync --extra dev
	uv run pre-commit install

test:  ## pytest with coverage
	uv run pytest --cov .

coverage:  ## coverage of the serial and the MPI runs combined, which must be 100% (needs mpiexec)
	rm -f .coverage .coverage.*
	.venv/bin/python -m coverage run -m pytest -q -p no:cacheprovider
	for n in 2 3 4 6; do \
		$(MPIEXEC) -n $$n .venv/bin/python -m coverage run -m pytest -q -p no:cacheprovider || exit 1; \
	done
	.venv/bin/python -m coverage combine -q
	.venv/bin/python -m coverage report --fail-under=100

lint:  ## ruff, pyright and ty
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright src/
	uv run ty check

readme:  ## render README.md from README.qmd with quarto
	quarto render README.qmd --to gfm

readme-check:  ## fail if README.md is out of date with README.qmd
	$(MAKE) readme
	git diff --exit-code README.md

docs-install:  ## Python docs extra and the npm packages
	$(PYTHON) -m pip install -e ".[docs]"
	cd docs && npm ci

docs-notebooks:  ## execute the changed tutorials and convert all of them (FORCE=1 reruns all)
	$(NOTEBOOKS) execute $(if $(FORCE),--force,)
	$(NOTEBOOKS) convert

docs-dev:  ## live preview at http://localhost:4321/mpiarray/
	$(NOTEBOOKS) convert
	cd docs && $(DOCS_ENV) DOCS_VALIDATE_LINKS=false npm run dev

docs-build:  ## the static site in docs/dist, as in CI
	$(NOTEBOOKS) convert
	cd docs && npm run check:mdx && $(DOCS_ENV) npm run build

docs-preview:  ## serve docs/dist
	cd docs && npm run preview

docs-clean:  ## remove generated pages, assets and build output
	$(NOTEBOOKS) clean
	rm -rf docs/dist docs/.astro
