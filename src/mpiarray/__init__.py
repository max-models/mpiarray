"""Distributed NumPy/CuPy arrays over MPI, with halo exchange.

::

    import mpiarray as mpa

    a = mpa.array([1, 2, 3, 4])   # under mpiexec -n 2: [1, 2] on rank 0, [3, 4] on rank 1
    b = mpa.zeros((64, 48), halo=1, periodic=(True, False))
    total = (2 * a).sum()          # collective: the same value on every rank

The creation functions (`array`, `zeros`, `arange`, `fromfunction`, ...)
split the array along ``split`` (default: the first axis) over the ranks of
``comm`` (default: ``MPI.COMM_WORLD``). Without an MPI launcher, mpiarray runs
serially on cunumpy's stand-in for mpi4py, as one rank holding everything.
Arrays come from cunumpy, so the same code runs on NumPy or CuPy; on the CuPy
backend call ``xp.mpi.mpi_is_cuda_aware(comm)`` or
``xp.mpi.set_mpi_cuda_aware(...)`` once at startup.
"""

from mpiarray._mpi import default_comm
from mpiarray.creation import (
    arange,
    array,
    asarray,
    empty,
    empty_like,
    from_local,
    fromfunction,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from mpiarray.distributed_array import DistributedArray, HaloUpdate
from mpiarray.io import load, save
from mpiarray.layout import Layout, chunk_bounds, process_grid

__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "DistributedArray",
    "HaloUpdate",
    "Layout",
    "__version__",
    "arange",
    "array",
    "asarray",
    "chunk_bounds",
    "default_comm",
    "empty",
    "empty_like",
    "from_local",
    "fromfunction",
    "full",
    "full_like",
    "linspace",
    "load",
    "ones",
    "ones_like",
    "process_grid",
    "save",
    "zeros",
    "zeros_like",
]
