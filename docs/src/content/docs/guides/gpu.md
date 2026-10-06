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
conversion. `np.asarray(a)` gathers into a host `numpy.ndarray` too. Both are collective:
call them on every rank, then plot on rank 0 (`a.to_numpy(root=0)` gathers on rank 0
only).

Install the CuPy package for your CUDA version, for example `pip install cupy-cuda12x`.

Code that works on `a.local` directly should use `cunumpy` (`import cunumpy as xp`) in
place of `numpy`, so that it runs on whichever backend is selected; the
[tutorials](/mpiarray/tutorials/) do this. On the CPU, `xp` is NumPy. Arrays from the host
(for example random numbers from NumPy) go to the device with `xp.asarray`.

## Testing without a GPU

cunumpy's fake CuPy runs the CuPy code paths on the CPU and refuses implicit conversions
between host and device arrays, as CuPy does:

```bash
CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy python -m pytest
```

The GPU CI runs the test suite on NVIDIA GPUs, serially and under MPI.

## CUDA-aware MPI

Halo exchanges, gathers, redistribution, `mpa.migrate` and `allreduce_replicated` pass
array buffers to MPI. A CUDA-aware MPI library can read device memory directly; any other
needs the data copied through host memory. mpiarray does this through
`xp.mpi.mpi_buffer`. Unless the program says otherwise, mpiarray copies through host
memory, which works with every MPI library. To send device buffers directly, tell it once
at startup:

```python
from maybempi import MPI  # mpi4py.MPI under mpiexec, a serial stand-in otherwise

comm = MPI.COMM_WORLD

xp.mpi.mpi_is_cuda_aware(comm)  # probe the library (collective)
# or, if you know the answer:
xp.mpi.set_mpi_cuda_aware(True)  # pass device buffers to MPI
```

Reductions without `axis` (`sum`, `max`, `norm`, …) send a single scalar, which is moved to
the host first in either case. Halo updates without CUDA-aware MPI copy each slab through
pinned host buffers, which each array keeps and reuses from one update to the next.

## One GPU per rank

Bind each rank to its own device before creating arrays. `maybempi.local_rank()` gives the
rank's index on its node, from the launcher's environment:

```python
import cupy
import maybempi

cupy.cuda.Device(maybempi.local_rank() % cupy.cuda.runtime.getDeviceCount()).use()
```

Many job schedulers can do this for you, for example `srun --gpus-per-task=1` with Slurm.
