---
title: Building the docs
description: How this documentation site is built from Markdown pages, notebooks and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has three kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **Guides**, converted from the Jupyter notebooks in `guides/` by `docs/tools/notebooks.py`.
  All guide content lives in notebooks. Each notebook is executed and its outputs (text, images, tables) are written into an MDX page.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"`.
- An MPI implementation (`mpiexec`) and the `mpi` extra, for the rank boundaries in the
  guides' plots: the notebooks run on 4 ranks (`DOCS_MPI_RANKS`). Without them they run on
  one rank and the plots show a single block.

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute guides/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/mpiarray/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove generated pages and build output
```

`make docs-notebooks` only reruns a notebook whose cells changed since the last run; pass
`FORCE=1` to rerun all of them.

Running `npm run dev`, `npm start` or `npm run build` from `docs/` also converts the
notebooks automatically, so the guide pages are available after a clean checkout or
`make docs-clean`. Conversion uses cached outputs when available; run `make docs-notebooks`
to execute the examples and populate those outputs. The npm commands use `DOCS_PYTHON`
when set, otherwise the repository's `.venv` Python or the system Python.

`npm run build` executes changed notebooks before conversion and rejects missing, stale or
failed outputs. Plotting cells therefore produce the figures embedded in the built pages.
Development preview converts immediately from cached outputs; use `make docs-notebooks`
to refresh them after editing code. GPU and PETSc examples check their optional prerequisites
and explain when a hardware-dependent section cannot run.

## Writing guides

Put a notebook in `guides/`. Its first `# Heading` becomes the page title and the first
paragraph its description. Cell tags control the output:

| Tag                | Effect                                   |
| ------------------ | ---------------------------------------- |
| `remove-cell`      | the cell is left out of the page         |
| `fold-input`      | code starts collapsed behind **Show code**, outputs stay visible |
| `hide-input`       | the code is hidden, the outputs are kept |
| `remove-output`    | the outputs are left out                 |
| `raises-exception` | an error output is shown, not a failure  |

Math in `$...$` and `$$...$$` is rendered with KaTeX.
