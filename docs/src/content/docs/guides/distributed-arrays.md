---
title: Distributed arrays
description: Creating a DistributedArray, how its local storage and ghost cells are laid out, and moving data between global and local arrays.
sidebar:
  order: 2
---

A `DistributedArray` is a global array of shape `shape` of which each rank stores only the
block it owns, plus an optional frame of ghost (halo) cells. It is a
[`DomainDecomposition`](/mpiarray/guides/domain-decomposition/), so `proc_sizes`,
`neighbour_ranks` and the other layout attributes are available on the array itself.

## Creating an array

The constructors mirror NumPy's, with the communicator and layout as extra arguments:

```python
import cunumpy as xp
import numpy as np

from mpiarray import DistributedArray

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD

a = DistributedArray.zeros((64, 48), comm)
b = DistributedArray.ones((64, 48), comm, dtype=np.float32)
c = DistributedArray.full((64, 48), 3.5, comm)
d = DistributedArray.empty((64, 48), comm)        # uninitialised
e = DistributedArray.from_array(np.random.rand(64, 48), comm)
```

`from_array` takes the *global* array, the same on every rank, and keeps the block each rank
owns. For large grids, create the array with `zeros` and write each rank's block instead
(see [Filling the array](#filling-the-array)), so no rank ever holds the whole grid.

All constructors take the same layout arguments:

| Argument          | Default        | Meaning                                                 |
| ----------------- | -------------- | ------------------------------------------------------- |
| `num_ghostpoints` | `0`            | width of the ghost frame on each side                   |
| `ghost_axes`      | all `True`     | which axes have ghost cells                             |
| `decompose`       | all `True`     | which axes may be split over ranks                      |
| `periodic`        | all `False`    | which axes wrap around in halo exchanges                |
| `dim_order`       | `None`         | see [Choosing the axes](/mpiarray/guides/domain-decomposition/#choosing-the-axes) |
| `dtype`           | `float`        | element type                                            |

Two arrays can be combined only if they have the same shape, process grid, ghost width and
ghost axes. Create them with the same arguments.

## Local storage

Each rank stores one NumPy (or CuPy) array, of shape `shape_local` plus `num_ghostpoints`
cells on both sides of every ghost axis. For one axis with two ghost points:

```text
   index in storage:  0   1 | 2   3   4   5   6   7 | 8   9
                     ghost  |       interior        |  ghost
                     (left  |   global indices      | (right
                    neighbour)  start .. end-1      neighbour)
```

The array exposes three views of it:

| Attribute           | Shape                | Contents                                     |
| ------------------- | -------------------- | -------------------------------------------- |
| `local`             | `shape_local`        | the owned interior, a writable view          |
| `local_with_halos`  | `shape_with_halos`   | the whole storage, a writable view           |
| `data`              | `shape_with_halos`   | the same as `local_with_halos`               |

and these sizes:

| Attribute            | Meaning                                         |
| -------------------- | ----------------------------------------------- |
| `shape`, `size`      | the global shape and number of elements         |
| `shape_local`        | this rank's interior shape                      |
| `local_size`         | this rank's number of interior elements         |
| `shape_with_halos`   | this rank's storage shape                       |
| `proc_index_bounds`  | this rank's `(start, end)` range along each axis |

`ghost_axes` lets some axes go without ghost cells. A vector field stored as
`(Nx, Ny, 3)` needs halos in space but not along the component axis, and should not split
the component axis either:

```python
E = DistributedArray.zeros(
    (64, 48, 3), comm,
    num_ghostpoints=2,
    ghost_axes=[True, True, False],
    decompose=[True, True, False],
)
# on 4 ranks: shape_local (32, 24, 3), shape_with_halos (36, 28, 3)
```

`get_local_slices()` returns the index that selects the interior from the storage, and
`get_local_block()` returns a contiguous copy of the interior.

## Filling the array

There are three ways to write data, which differ in what each rank must pass:

```python
# 1. Each rank writes its own block (no communication, nothing global).
(x0, x1), (y0, y1) = a.proc_index_bounds
a.local[...] = my_function(np.arange(x0, x1)[:, None], np.arange(y0, y1))

# 2. Every rank passes the same global array; each keeps its block, halos are zeroed.
a.fill(global_array)          # or: a.data = global_array

# 3. Each rank passes its whole local storage, halos included.
a.fill_local(local_storage)   # shape must be shape_with_halos
```

`fill` and `fill_local` check that the dtype matches the array's; convert with
`np.asarray(x, dtype=a.dtype)` first. Indexed assignment, `a[i, :] = value`, is described
under [Indexing](/mpiarray/guides/operations/#indexing).

After writing the interior, call [`fill_halos()`](/mpiarray/guides/halo-exchange/) before
reading the ghost cells.

## Getting the data back

```python
full = a.to_ndarray()   # the global array on every rank (NumPy or CuPy)
full = a.to_numpy()     # the global array as numpy.ndarray
full = a.to_cupy()      # the global array as cupy.ndarray
full = np.asarray(a)    # the same as to_ndarray()
```

Each of these is a collective `Allgatherv`: every rank must call it, and every rank receives
the whole array. Use them for output, plotting and tests, not inside a time loop on large
grids. The ghost cells are not included.

`copy()` returns an independent array with the same layout, and `astype(dtype)` a converted
one.
