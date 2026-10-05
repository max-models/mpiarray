"""MPI domain decomposition and distributed NumPy/CuPy arrays.

::

    from mpiarray import DistributedArray, DomainDecomposition

`DomainDecomposition` splits a Cartesian grid over the ranks of a
communicator (process grid, neighbour ranks, owned index ranges and physical
subdomains). `DistributedArray` is a decomposed array with optional halo
cells, halo fill/accumulate exchanges, global reductions and NumPy operators.

Arrays come from cunumpy, so the same code runs on NumPy or CuPy. Device
arrays go to MPI through `cunumpy.mpi.mpi_buffer`, so on the CuPy backend
call ``xp.mpi.mpi_is_cuda_aware(comm)`` or ``xp.mpi.set_mpi_cuda_aware(...)``
once at startup.
"""

from mpiarray.darray import DistributedArray
from mpiarray.domain_decomposition import (
    DomainDecomposition,
    calculate_neighbor_ranks,
    calculate_proc_sizes,
    get_proc_bounds,
    split_array,
)

__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "DistributedArray",
    "DomainDecomposition",
    "__version__",
    "calculate_neighbor_ranks",
    "calculate_proc_sizes",
    "get_proc_bounds",
    "split_array",
]
