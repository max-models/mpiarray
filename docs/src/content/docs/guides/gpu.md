---
title: GPU arrays
description: Running mpiarray on CuPy, and telling it whether the MPI library can take device buffers.
sidebar:
  order: 5
---

mpiarray creates its arrays through [cunumpy](https://github.com/max-models/cunumpy), which
is NumPy or CuPy depending on the selected backend. The same script runs on the CPU and on
the GPU; only the backend changes.

## Selecting the backend

```bash
CUNUMPY_BACKEND=cupy mpiexec -n 4 python simulation.py
```

or in the script, before any array is created:

```python
import cunumpy as xp

xp.set_backend("cupy")
```

With the CuPy backend, `a.local`, `a.local_with_halos` and `a.gather()` are CuPy arrays,
and the operators and ufuncs run on the device. `a.to_numpy()` copies the gathered array to
the host, and reductions without `axis` return host scalars, so `if a.max() > 1:` needs no
conversion.

Install the CuPy package for your CUDA version, for example `pip install cupy-cuda12x`.

## CUDA-aware MPI

Halo exchanges, gathers and `allreduce_replicated` pass array buffers to MPI. A CUDA-aware
MPI library can read device memory directly; any other needs the data copied through host
memory. mpiarray does this through `xp.mpi.mpi_buffer`, which needs to know which case
applies. Tell it once at startup:

```python
MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD

xp.mpi.mpi_is_cuda_aware(comm)  # probe the library (collective)
# or, if you know the answer:
xp.mpi.set_mpi_cuda_aware(False)  # always copy through host memory
```

Reductions without `axis` (`sum`, `max`, `norm`, …) send a single scalar, which is moved to
the host first in either case. Halo updates without CUDA-aware MPI copy each slab through
pinned host buffers, which each array keeps and reuses from one update to the next.

## One GPU per rank

Bind each rank to its own device before creating arrays. `xp.mpi.local_rank()` gives the
rank's index on its node:

```python
import cupy

cupy.cuda.Device(xp.mpi.local_rank() % cupy.cuda.runtime.getDeviceCount()).use()
```

Many job schedulers can do this for you, for example `srun --gpus-per-task=1` with Slurm.
