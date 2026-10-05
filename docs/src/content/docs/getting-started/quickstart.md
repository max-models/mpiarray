---
title: Quickstart
description: Decompose a grid over MPI ranks and work with a DistributedArray.
---

First, ensure that `mpiarray` is [installed](/mpiarray/getting-started/installation/).

## A distributed array

```python
import cunumpy as xp

from mpiarray import DistributedArray

# mpi4py.MPI under mpiexec, otherwise a serial stand-in (no MPI needed)
MPI = xp.mpi.get_mpi()
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

The same script also runs serially (`python example.py`), as a single rank holding the
whole array. `xp.mpi.get_mpi()` returns `mpi4py.MPI` when the script is started by an MPI
launcher, and otherwise cunumpy's serial stand-in, so a serial run does not start MPI.

## Halo cells

- `fill_halos()` copies the neighbours' boundary values into the ghost cells (periodic
  axes wrap around), as needed before applying a stencil.
- `exchange_halos()` adds the ghost cells into the neighbours' interior and zeroes them,
  as needed after depositing particles near subdomain edges.

## The decomposition alone

`DomainDecomposition` gives the layout without any array:

```python
import cunumpy as xp

from mpiarray import DomainDecomposition

MPI = xp.mpi.get_mpi()
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

## Next steps

- [Domain decomposition](/mpiarray/guides/domain-decomposition/): how the grid is split and
  who neighbours whom.
- [Distributed arrays](/mpiarray/guides/distributed-arrays/): storage, ghost cells, filling
  and gathering.
- [Halo exchange](/mpiarray/guides/halo-exchange/): `fill_halos` for stencils,
  `exchange_halos` for deposition.
- [Operations and reductions](/mpiarray/guides/operations/): arithmetic, indexing,
  reductions, and which calls are collective.
- [GPU arrays](/mpiarray/guides/gpu/): CuPy and CUDA-aware MPI.
- The [API reference](/mpiarray/api/mpiarray/) for every class and function.
