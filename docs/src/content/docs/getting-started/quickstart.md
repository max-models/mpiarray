---
title: Quickstart
description: Create a distributed array, run it on several ranks, and update its halo cells.
---

First, ensure that `mpiarray` is [installed](/mpiarray/getting-started/installation/).

## A distributed array

```python
import numpy as np

import mpiarray as mpa

a = mpa.array([1, 2, 3, 4])
print(a)  # this rank's block; printing never communicates

# a 64 x 48 grid split along x, one halo layer, periodic in x
rho = mpa.zeros((64, 48), halo=1, periodic=(True, False))
rank = rho.layout.rank
rho.local[...] = rank + 1.0  # write this rank's block
rho.update_halos()  # halo cells <- neighbours' boundary values

total = rho.sum()  # collective: the same value on every rank
full = rho.gather()  # collective: the whole array on every rank
sines = np.sin(a).gather()  # collective, so outside the `if`
if rank == 0:
    print(total, full.shape, sines)
```

Save it as `example.py` and run it on two ranks:

```bash
mpiexec -n 2 python example.py
```

Rank 0 prints its block `[1, 2]`, rank 1 prints `[3, 4]`. The same script also runs
serially (`python example.py`), as one rank holding the whole array: without an MPI
launcher mpiarray uses cunumpy's stand-in for `mpi4py.MPI` and never starts MPI.

## What to remember

- Arrays are split along their first axis by default. Pass `split=(0, 1)` to split over
  two axes, or `split=None` to give every rank the whole array.
- Elementwise arithmetic and `a.local` never communicate. Reductions, `gather()`,
  `a[i, j]` and halo updates are **collective**: every rank must call them, also when only
  rank 0 needs the result.
- `update_halos()` copies the neighbours' boundary values into the halo cells, as needed
  before a stencil; `accumulate_halos()` adds the halo cells into the neighbours, as needed
  after depositing particles.

## GPU arrays

Select CuPy with `CUNUMPY_BACKEND=cupy` (or `xp.set_backend("cupy")`), and tell cunumpy
once whether the MPI library can take device buffers:

```python
import cunumpy as xp

MPI = xp.mpi.get_mpi()
xp.mpi.mpi_is_cuda_aware(MPI.COMM_WORLD)  # collective probe, or:
xp.mpi.set_mpi_cuda_aware(False)  # copy through host memory
```

## Next steps

- [Distributed arrays](/mpiarray/guides/distributed-arrays/): creating arrays, storage,
  halo cells and gathering.
- [Layouts](/mpiarray/guides/layouts/): how arrays are split and who neighbours whom.
- [Halo exchange](/mpiarray/guides/halo-exchange/): `update_halos` for stencils,
  `accumulate_halos` for deposition.
- [Operations and reductions](/mpiarray/guides/operations/): arithmetic, indexing,
  reductions, and which calls are collective.
- [GPU arrays](/mpiarray/guides/gpu/): CuPy and CUDA-aware MPI.
- The [API reference](/mpiarray/api/mpiarray/) for every class and function.
