---
title: Installation
description: Install mpiarray, with or without MPI.
---

In a Python 3.10+ environment:

```bash
pip install "mpiarray[mpi]"   # with mpi4py, for runs under mpiexec
pip install mpiarray          # serial only
```

The `mpi` extra installs [mpi4py](https://mpi4py.readthedocs.io/), which needs an MPI
library:

```bash
brew install open-mpi                              # macOS
sudo apt-get install libopenmpi-dev openmpi-bin    # Debian/Ubuntu
```

Without mpi4py, mpiarray uses the serial stand-in for `mpi4py.MPI` from
[maybempi](https://github.com/max-models/maybempi) (`from maybempi import MPI`): every
array lives on one rank, and no MPI library is needed. This is enough for development,
notebooks and tests of serial code. Under `mpiexec` without mpi4py, maybempi warns, and
every process then computes the whole problem as rank 0. With mpi4py installed, a plain
`python script.py` still uses the stand-in and never starts MPI; set `MAYBEMPI=1` to force
MPI or `MAYBEMPI=0` to force the stand-in.

For GPU arrays, also install the CuPy package for your CUDA version (e.g. `cupy-cuda12x`);
see [cunumpy](https://github.com/max-models/cunumpy) for selecting the backend.

## From a checkout

```bash
make install          # uv sync --extra dev, plus the pre-commit hooks
# or
pip install -e ".[dev]"
```

## Optional extras

| Extra  | Installs                                                         |
| ------ | ---------------------------------------------------------------- |
| `mpi`  | `mpi4py`, for parallel runs                                      |
| `test` | `pytest` and `coverage`                                          |
| `docs` | the notebook runner and griffe for the API reference             |
| `dev`  | formatters and linters, plus the `mpi`, `test` and `docs` extras |
