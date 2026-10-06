# mpiarray


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/mpiarray/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/mpiarray/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/mpiarray/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/mpiarray/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/mpiarray/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/mpiarray/)
[![codecov](https://codecov.io/gh/max-models/mpiarray/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/mpiarray)
[![PyPI](https://img.shields.io/pypi/v/mpiarray.png)](https://pypi.org/project/mpiarray/)
[![Python](https://img.shields.io/pypi/pyversions/mpiarray.png)](https://pypi.org/project/mpiarray/)

Distributed NumPy/CuPy arrays over MPI, with halo exchange, that you
create like NumPy arrays:

``` python
import numpy as np

import mpiarray as mpa

a = mpa.array([1, 2, 3, 4])  # mpiexec -n 2: rank 0 holds [1, 2], rank 1 holds [3, 4]
b = mpa.zeros((64, 48), split=(0, 1), halo=1, periodic=(True, False))
c = 2 * a + np.sin(a)  # elementwise: no communication

total = c.sum()  # collective: the same value on every rank
full = c.gather()  # collective: the whole array on every rank
print(c)  # this rank's block; printing never communicates
```

``` bash
mpiexec -n 2 python example.py   # or just: python example.py
```

- **Creation** like NumPy: `array`, `zeros`, `ones`, `full`, `empty`,
  the `*_like` versions, `arange`, `linspace` and `fromfunction`, which
  compute only the local block. Arrays are split along their first axis
  by default; `split=` chooses other axes, `split=None` gives every rank
  the whole array.
- **Halo cells** per axis: `update_halos()` before a stencil,
  `accumulate_halos()` after depositing particles near block edges.
- **Operators, ufuncs and reductions** as in NumPy (`sum`, `max`,
  `mean`, `norm`, `vdot`, …); global reductions return the same host
  scalar on every rank.
- **NumPy or CuPy:** arrays come from
  [cunumpy](https://github.com/max-models/cunumpy), so the same code
  runs on the GPU. Without a CUDA-aware MPI, device buffers are copied
  through host memory; tell cunumpy once with
  `xp.mpi.mpi_is_cuda_aware(comm)` or
  `xp.mpi.set_mpi_cuda_aware(False)`.
- **With or without MPI:** without an MPI launcher the script runs as
  one rank on cunumpy’s stand-in for `mpi4py.MPI`, without importing
  mpi4py.
- **Data in and out:** `from_local` builds an array from the pieces the
  ranks hold; `save`/`load` write and read ordinary `.npy` files in
  parallel with MPI-IO.
- **Halo boundary conditions** at walls: constant, `"edge"`,
  `"symmetric"`, `"reflect"`.
- **Debugging:** with `MPIARRAY_DEBUG=1`, a collective called on only
  some ranks raises an error instead of hanging.

Each array has a `layout` (a `mpa.Layout`) with its process grid,
neighbours and owned index ranges; most code only reads it.

Documentation: <https://max-models.github.io/mpiarray/>

## Install

``` bash
pip install "mpiarray[mpi]"   # with mpi4py, for runs under mpiexec
pip install mpiarray          # serial only, no MPI library needed
```

The `mpi` extra installs [mpi4py](https://mpi4py.readthedocs.io/), which
needs an MPI library, e.g. `brew install open-mpi` or
`sudo apt-get install libopenmpi-dev openmpi-bin`. Without it, mpiarray
runs on cunumpy’s serial stand-in for `mpi4py.MPI`, as one rank holding
the whole array. Starting such an installation with `mpiexec` gives a
warning, and every process then computes the whole problem on its own.

For development, with [uv](https://docs.astral.sh/uv/):

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
```

or with pip, in a Python 3.10+ environment:

``` bash
pip install -e ".[dev]"
```

The `test`, `docs` and `dev` extras install the test runner, the
documentation tooling and the linters; `dev` includes `mpi`.

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), type
checking [pyright](https://microsoft.github.io/pyright/) and
[ty](https://docs.astral.sh/ty/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, pyright, ty
make test     # pytest with coverage
```

The tests run serially and under MPI; some only run on 2 or 6 ranks:

``` bash
mpiexec -n 2 .venv/bin/python -m pytest
mpiexec -n 6 .venv/bin/python -m pytest
make coverage   # serial and 2, 3, 4, 6 ranks, combined; fails below 100% line coverage
```

Commit messages follow [Conventional
Commits](https://www.conventionalcommits.org/); see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
and the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-dev         # live preview at http://localhost:4321/mpiarray/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

[release-please](https://github.com/googleapis/release-please) keeps a
release PR open on `main` from the commit messages. Merging it tags the
release, updates `CHANGELOG.md` and publishes the package to PyPI with
trusted publishing (OIDC). The one-time PyPI and GitHub configuration is
described in the [publishing
guide](https://max-models.github.io/mpiarray/development/publishing/).
