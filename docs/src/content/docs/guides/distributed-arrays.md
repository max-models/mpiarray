---
title: Distributed arrays
description: Creating arrays with mpa.array, mpa.zeros and friends, how each rank stores its block and halo cells, and getting the data back.
sidebar:
  order: 1
---

A `DistributedArray` is a global array of which each rank stores only the block it owns,
plus an optional frame of halo cells. You create one with a function named like its NumPy
counterpart:

```python
import numpy as np

import mpiarray as mpa

a = mpa.array([1, 2, 3, 4])  # mpiexec -n 2: rank 0 holds [1, 2], rank 1 holds [3, 4]
b = mpa.zeros((64, 48))
c = mpa.ones((64, 48), dtype=np.float32)
d = mpa.full((64, 48), 3.5)
e = mpa.empty((64, 48))  # uninitialised
f = mpa.arange(100)
g = mpa.linspace(0.0, 1.0, 101)
h = mpa.fromfunction(lambda i, j: np.sin(0.1 * i) * j, (64, 48))
```

Without an MPI launcher (`python script.py`), the same code runs as one rank holding
everything, without importing mpi4py.

## Where the data comes from

- `mpa.array(data)` takes the *global* array, the same on every rank, and keeps the block
  each rank owns. Every rank briefly holds all of it, so use it for small arrays and tests.
  With `MPIARRAY_DEBUG=1` the ranks check that they really passed the same data.
- `mpa.arange`, `mpa.linspace` and `mpa.fromfunction` compute only the local block on each
  rank, so no rank ever holds the whole array. `fromfunction` gets the global indices of
  the block, as `numpy.fromfunction` does.
- `mpa.from_local(block)` builds the array from the pieces the ranks already hold, stacked
  in rank order along `split` (default: the first axis). Pieces of any length are
  redistributed to the near-even split with one `Alltoallv`; with `layout=` the pieces must
  already match it and are used as they are (`with_halos=True` for storage with halos).
- `mpa.zeros` / `mpa.empty`, then writing `a.local`, lets each rank fill its block from its
  own data.
- `mpa.load(path)` reads a `.npy` file, each rank reading only its block (see below).

`zeros_like`, `ones_like`, `full_like` and `empty_like` create an array with the layout of
another one. Given a distributed array, `mpa.array(a)` copies it with its layout, and
`mpa.array(a, halo=2)` or `mpa.array(a, split=None)` changes only the options given and
redistributes it; `mpa.asarray(a)` returns `a` itself unless an option changes.

## How the array is split

Every creation function takes the same keyword arguments:

| Argument       | Default         | Meaning                                                   |
| -------------- | --------------- | --------------------------------------------------------- |
| `split`        | `0`             | the axis or axes split over the ranks; `None`: every rank holds everything |
| `halo`         | `0`             | halo width, for every axis or one per axis                |
| `periodic`     | `False`         | whether each axis wraps around, for every axis or one per axis |
| `comm`         | `COMM_WORLD`    | the communicator                                          |
| `process_grid` | automatic       | explicit ranks per axis, instead of `split`               |
| `layout`       | —               | an existing [`Layout`](/mpiarray/guides/layouts/), instead of all of the above |

```python
p = mpa.zeros((64, 48, 3), split=(0, 1), halo=(2, 2, 0), periodic=(True, False, False))
```

splits a vector field over its two spatial axes, with two halo layers in space and none
along the component axis. Each axis is cut near-evenly: when the length does not divide,
the first ranks get one element more.

A split axis needs at least as many elements as ranks, and a halo must fit in the
smallest block along its axis; otherwise creating the array raises a `ValueError` (on
every rank), instead of leaving ranks without data. Use `split=None` for small arrays that
every rank should hold whole.

Arrays created with the same arguments have equal layouts and combine without
communication. To combine arrays, create the second one with `zeros_like(first)` or with
`layout=first.layout`.

## Local storage

Each rank stores one NumPy (or CuPy) array: its block plus `halo[axis]` cells on both sides
of every axis. For one axis with two halo cells:

```text
   index in storage:  0   1 | 2   3   4   5   6   7 | 8   9
                       halo |         block         | halo
                     (left  |    global indices     | (right
                    neighbour)    start .. end-1     neighbour)
```

| Attribute                 | Contents                                           |
| ------------------------- | -------------------------------------------------- |
| `a.local`                 | the block, a writable view                         |
| `a.local_with_halos`      | the whole storage, a writable view                 |
| `a.shape`, `a.size`       | the global shape and number of elements            |
| `a.layout.index_bounds`   | this rank's `(start, end)` range along each axis   |
| `a.layout.local_shape`    | the shape of the block                             |
| `a.layout.storage_shape`  | the shape of the storage                           |

Writing a block from local data:

```python
a = mpa.zeros((64, 48))
(x0, x1), (y0, y1) = a.layout.index_bounds
a.local[...] = my_function(np.arange(x0, x1)[:, None], np.arange(y0, y1))
```

After writing the block, call [`update_halos()`](/mpiarray/guides/halo-exchange/) before
reading the halo cells.

## Getting the data back

```python
full = a.gather()  # the global array on every rank (NumPy or CuPy)
full = a.gather(root=0)  # on rank 0 only; None on the other ranks
full = a.to_numpy()  # the same, as numpy.ndarray
value = a.get((3, 2))  # one element, the same on every rank
value = a[3, 2]  # the same as get
```

All of these are **collective**: every rank must call them, even when only rank 0 wants
the result. Use `gather` for output, plotting and tests, not inside a time loop on large
grids. Halo cells are never included.

Printing does not communicate: `repr(a)` shows the layout and this rank's block only, so
`if rank == 0: print(a)` is safe. Print `a.gather()` (on every rank) to see the whole array.

`copy()` returns an independent array with the same layout, and `astype(dtype)` a converted
one.

## Saving to files

```python
mpa.save("field.npy", a)  # an ordinary .npy file, written in parallel
b = mpa.load("field.npy", split=1, halo=2)  # any layout; the shape and dtype from the file
```

With several ranks both use MPI-IO: rank 0 writes the header, and every rank writes or
reads only its own block, so nothing global is ever built. The files are plain NumPy files:
`numpy.load` reads what `mpa.save` writes, and `mpa.load` reads what `numpy.save` writes.
The file must be on a file system that all ranks see. Both calls are collective.
