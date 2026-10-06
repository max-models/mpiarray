---
title: Layouts
description: How a Layout splits an array over the ranks, the process grid, neighbours, owned index ranges, and using a Layout without an array.
sidebar:
  order: 2
---

Every `DistributedArray` has a `layout`: an immutable description of how its global shape
is split over the ranks of a communicator. The creation functions build it from their
`split`, `halo`, `periodic`, `comm` and `process_grid` arguments, so most code only reads
it:

```python
import mpiarray as mpa

a = mpa.zeros((10, 7), split=(0, 1), halo=1, periodic=(True, False))
layout = a.layout
layout.process_grid  # ranks along each axis
layout.process_coord  # this rank's position on the process grid
layout.neighbours  # ((left, right), ...) per axis
layout.index_bounds  # this rank's (start, end) per axis
```

Layouts compare by value: the shape, process grid, halo widths, periodicity and
communicator. Two arrays with equal layouts combine without communication; otherwise
elementwise operations raise. Communicators must be the same object, or equal for MPI: an
array on `comm.Dup()` does not combine with one on `comm`, because the two communicators
keep their messages apart.

## The process grid

The split axes share the ranks as evenly as divisibility allows. `mpa.process_grid` shows
the grid for any number of ranks, without MPI:

```python
mpa.process_grid(4, 2, (0, 1))  # (2, 2)
mpa.process_grid(6, 2, (0, 1))  # (2, 3)
mpa.process_grid(12, 2, (0, 1))  # (3, 4)
mpa.process_grid(12, 3, (0, 1, 2))  # (2, 2, 3)
mpa.process_grid(15, 3, (0, 1, 2))  # (1, 3, 5)
mpa.process_grid(7, 2, (0, 1))  # (1, 7)
```

The split axes are filled in order. Each gets the largest divisor of the ranks still to
place that is not above their $d$-th root ($d$ being the number of split axes left); the
last one takes what is left. A prime number of ranks therefore ends up on the last split
axis. Choose a rank count with small factors (4, 8, 12, 16, …) for a balanced grid.

To choose the grid yourself, pass `process_grid`; its product must be the number of ranks,
and the axes with more than one rank become the split axes (if you also pass `split`, they
must be among its axes):

```python
a = mpa.zeros((120, 60), process_grid=(3, 2))  # on 6 ranks
```

Give the most ranks to the longest axis, so that the blocks stay close to square and the
halo surfaces small. Every split axis needs at least as many elements as ranks along it,
and every halo must fit in the smallest block; otherwise `Layout` raises a `ValueError` on
every rank. Leave unsplit the axes that are short, or that an algorithm needs
whole on every rank (a component axis, an axis you transform with an FFT).

## Neighbours

Ranks are placed on the process grid in row-major order: the last axis varies fastest. On
four ranks with `split=(0, 1)` and `periodic=(True, False)` the grid is $2 \times 2$:

| rank | `process_coord` | `neighbours`                   |
| ---- | --------------- | ------------------------------ |
| 0    | (0, 0)          | `((2, 2), (PROC_NULL, 1))`     |
| 1    | (0, 1)          | `((3, 3), (0, PROC_NULL))`     |
| 2    | (1, 0)          | `((0, 0), (PROC_NULL, 3))`     |
| 3    | (1, 1)          | `((1, 1), (2, PROC_NULL))`     |

Along the periodic axis 0 the neighbours wrap around (with two ranks, left and right are
the same rank). Along axis 1 a rank at the wall has `MPI.PROC_NULL` as its neighbour, and
MPI calls with `PROC_NULL` do nothing, so halo code needs no special case for walls.

`layout.coord_of(rank)` and `layout.rank_of(coord)` convert between the two.

## Owned index ranges

Each axis is cut near-evenly: when the length does not divide, the first ranks get one
element more. On the $2 \times 2$ grid above, a $10 \times 7$ array is split into rows
`0:5` and `5:10` and columns `0:4` and `4:7`.

```python
layout.index_bounds  # this rank
layout.index_bounds_of(3)  # another rank
layout.owner((6, 2))  # the rank that owns an element: 2
layout.owners(indices)  # vectorized: ranks for an (n, 2) integer array
layout.owners(indices, clip=True)  # outside the array: the nearest block
mpa.chunk_bounds(7, 2, 1)  # one axis on its own: (4, 7)
```

`owners` works on NumPy and CuPy arrays without a loop, for example to find where
particles have to be sent after a push.

## Cartesian communicators

`Layout(..., reorder=True)` builds an MPI Cartesian communicator for the process grid
(`MPI_Cart_create` with reordering), so that the MPI library can place neighbouring blocks
on nearby cores and nodes. The layout's `comm` is then that communicator, and `rank`,
`neighbours` and `owner` number the ranks in it: send your own messages (for example
particles) over `layout.comm`. Layouts made with the same arguments share the communicator,
so their arrays still combine.

```python
layout = Layout((4096, 4096), split=(0, 1), halo=2, reorder=True)
u = mpa.zeros(layout=layout)
```

## Replicated arrays

With `split=None` every rank holds the whole array: `layout.replicated` is `True` (on more
than one rank), the process grid is all ones, and along periodic axes each rank is its own
neighbour. This is the layout for particle-in-cell codes that split the particles, not the
grid: each rank deposits its own particles on a full-size array, and
[`allreduce_replicated`](/mpiarray/guides/operations/#combining-replicated-copies) sums
the copies.

## A layout without an array

A `Layout` can be built on its own, for example to decide which rank a particle belongs
to, or to create a PETSc DMDA with the same decomposition:

```python
from mpiarray import Layout

layout = Layout((128, 128), split=(0, 1), halo=1)
dest = layout.owner(cell_index)

a = mpa.zeros(layout=layout)  # an array on that layout
```

It takes the same arguments as the creation functions, the shape first.

## Without MPI

Without an MPI launcher, `comm` defaults to cunumpy's serial stand-in for
`MPI.COMM_WORLD`: one rank that owns everything. The same code therefore runs unchanged
with `python script.py` and with `mpiexec -n 8 python script.py`.
