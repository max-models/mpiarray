---
title: Installation
description: Install mpiarray and an MPI library.
---

mpiarray uses [mpi4py](https://mpi4py.readthedocs.io/), which needs an MPI library:

```bash
brew install open-mpi                              # macOS
sudo apt-get install libopenmpi-dev openmpi-bin    # Debian/Ubuntu
```

Then, in a Python 3.10+ environment:

```bash
pip install mpiarray
```

For GPU arrays, also install the CuPy package for your CUDA version (e.g. `cupy-cuda12x`);
see [cunumpy](https://github.com/max-models/cunumpy) for selecting the backend.

## From a checkout

```bash
make install          # uv sync --extra dev, plus the pre-commit hooks
# or
pip install -e ".[dev]"
```

## Optional extras

| Extra  | Installs                                                  |
| ------ | --------------------------------------------------------- |
| `test` | `pytest` and `coverage`                                   |
| `docs` | the notebook runner and griffe for the API reference      |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |
