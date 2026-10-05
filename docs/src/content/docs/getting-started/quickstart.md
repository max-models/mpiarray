---
title: Quickstart
description: Decompose a grid over MPI ranks and work with a DistributedArray.
---

First, ensure that `mpiarray` is [installed](/mpiarray/getting-started/installation/).

## A distributed array

```python
from mpi4py import MPI

from mpiarray import DistributedArray

comm = MPI.COMM_WORLD

# a 64 x 48 grid split over the ranks, one ghost layer, periodic in x
rho = DistributedArray.zeros(
    shape=(64, 48), comm=comm, num_ghostpoints=1, periodic=(True, False)
)
print(rho.proc_sizes, rho.proc_index_bounds)  # this rank's block

rho.local[...] = comm.rank + 1.0  # write this rank's interior
rho.fill_halos()  # ghost cells <- neighbours' boundary values

total = rho.sum()  # global reductions are collective
full = rho.to_ndarray()  # gather the whole array on every rank
if comm.rank == 0:
    print(total, full.shape)
```

Save it as `example.py` and run it on four ranks:

```bash
mpiexec -n 4 python example.py
```

The same script also runs serially, as a single rank holding the whole array.

## Halo cells

- `fill_halos()` copies the neighbours' boundary values into the ghost cells (periodic
  axes wrap around), as needed before applying a stencil.
- `exchange_halos()` adds the ghost cells into the neighbours' interior and zeroes them,
  as needed after depositing particles near subdomain edges.

## The decomposition alone

`DomainDecomposition` gives the layout without any array:

```python
from mpi4py import MPI

from mpiarray import DomainDecomposition

layout = DomainDecomposition(MPI.COMM_WORLD, decompose=[True, True, False])
layout.proc_sizes  # processes along each axis
layout.neighbour_ranks  # (left, right) per axis, MPI.PROC_NULL at walls
layout.get_index_bounds((128, 128, 64))  # this rank's index ranges
```

## GPU arrays

Select CuPy with `CUNUMPY_BACKEND=cupy` (or `xp.set_backend("cupy")`), and tell cunumpy
once whether the MPI library can take device buffers:

```python
import cunumpy as xp

xp.mpi.mpi_is_cuda_aware(MPI.COMM_WORLD)  # collective probe, or:
xp.mpi.set_mpi_cuda_aware(False)  # copy through host memory
```

See the [API reference](/mpiarray/api/mpiarray/) for every class and function.
