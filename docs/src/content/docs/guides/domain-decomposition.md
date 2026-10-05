---
title: Domain decomposition
description: How DomainDecomposition splits a Cartesian grid over the MPI ranks, finds neighbours and assigns index ranges and physical subdomains.
sidebar:
  order: 1
---

`DomainDecomposition` describes how the ranks of a communicator share a Cartesian grid. It
holds no data: it answers "how many ranks along each axis", "which rank is my left
neighbour", and "which indices or which part of space do I own". `DistributedArray` is built
on top of it, so everything on this page also holds for an array.

```python
import cunumpy as xp

from mpiarray import DomainDecomposition

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD

layout = DomainDecomposition(comm, decompose=[True, True], periodic=(True, False))
```

| Argument    | Meaning                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| `comm`      | the communicator to split, or `None` for a serial layout with one rank           |
| `decompose` | one flag per axis: may this axis be split over ranks? Default: every axis        |
| `periodic`  | one flag per axis: does the axis wrap around? Default: no axis                   |
| `dim_order` | which axis gets the most ranks (see [Choosing the axes](#choosing-the-axes))     |
| `ndim`      | number of axes, needed only when `decompose` is not given                        |

## The process grid

`proc_sizes` is the number of ranks along each axis. Their product is the number of ranks.
The split is computed by `calculate_proc_sizes`, which you can also call without MPI to see
the layout for any number of ranks:

```python
from mpiarray import calculate_proc_sizes

calculate_proc_sizes(4, [True, True])  # [2, 2]
calculate_proc_sizes(6, [True, True])  # [2, 3]
calculate_proc_sizes(12, [True, True])  # [3, 4]
calculate_proc_sizes(12, [True, True, True])  # [2, 2, 3]
calculate_proc_sizes(15, [True, True, True])  # [1, 3, 5]
calculate_proc_sizes(7, [True, True])  # [1, 7]
```

The algorithm aims for the same number of ranks along every flagged axis (the $d$-th root
of the number of ranks for $d$ flagged axes), rounded down to a divisor. If that does not
use all the ranks, it drops the last flagged axis and tries again. A prime number of ranks
therefore always ends up on a single axis. Choose a rank count with small factors (4, 8,
12, 16, …) to get a balanced grid.

Axes with `decompose=False` always have one rank. Use this for axes that are short, or that
an algorithm needs whole on every rank (a component axis, or an axis you transform with an
FFT).

### Choosing the axes

By default the larger counts go to the later axes (`[2, 3]` for six ranks). `dim_order`
reorders them: entry `i` says which of the sorted counts (`0` = largest) axis `i` gets.

```python
# six ranks, decompose=[True, True]
DomainDecomposition(comm, decompose=[True, True]).proc_sizes  # [2, 3]
DomainDecomposition(comm, decompose=[True, True], dim_order=[0, 1]).proc_sizes  # [3, 2]
```

Give the most ranks to the longest axis, so that the blocks stay close to square and the
halo surfaces small.

## Rank coordinates and neighbours

Ranks are placed on the process grid in row-major order: the last axis varies fastest.

```python
layout.proc_coord  # this rank's position on the process grid
layout.get_proc_coord(r)  # the position of rank r
layout.rank_from_proc_coord((1, 0))
layout.neighbour_ranks  # [(left, right), ...], one pair per axis
```

On four ranks with `periodic=(True, False)` the grid is $2 \times 2$:

| rank | `proc_coord` | `neighbour_ranks`              |
| ---- | ------------ | ------------------------------ |
| 0    | (0, 0)       | `[(2, 2), (PROC_NULL, 1)]`     |
| 1    | (0, 1)       | `[(3, 3), (0, PROC_NULL)]`     |
| 2    | (1, 0)       | `[(0, 0), (PROC_NULL, 3)]`     |
| 3    | (1, 1)       | `[(1, 1), (2, PROC_NULL)]`     |

Along the periodic axis 0 the neighbours wrap around (with two ranks, left and right are the
same rank). Along axis 1 a rank at the wall has `MPI.PROC_NULL` as its neighbour, and MPI
calls with `PROC_NULL` do nothing, so halo code needs no special case for walls.

:::note
The module-level `calculate_neighbor_ranks(proc_sizes, rank)` ignores periodicity. Use
`DomainDecomposition.neighbour_ranks` when an axis is periodic.
:::

## Owned index ranges

`get_index_bounds(shape)` returns this rank's `(start, end)` index range along each axis of
a grid of the given shape. Each axis is split as evenly as possible; when the length does
not divide, the first ranks get one point more.

```python
layout.get_index_bounds((10, 7))  # this rank
layout.get_index_bounds((10, 7), rank=3)  # another rank
```

On the $2 \times 2$ grid above, a $10 \times 7$ grid is split into rows `0:5` and `5:10`
and columns `0:4` and `4:7`. The same split along one axis is available as
`split_array(length, num_procs)` (the sizes) and `get_proc_bounds(length, num_procs, i)`
(the range of the `i`-th chunk).

## Physical subdomains

For particle codes and other mesh-free data, the same layout splits a box in space. Each
axis of the box is cut into `proc_sizes[dim]` intervals of equal width:

```python
lower, upper = (0.0, 0.0), (1.0, 2.0)

layout.subdomain_edges(0, 0.0, 1.0)  # [0.0, 0.5, 1.0] on a 2 x 2 grid
layout.subdomain_bounds(rank, lower, upper)  # ((0.5, 1.0), (1.0, 2.0)) for rank 3
layout.owner_rank_for_position((0.6, 1.9), lower, upper, num_gridpoints=(11, 21))  # 3
```

A rank owns the half-open interval `[edges[c], edges[c + 1])`; the last rank along an axis
also owns the upper boundary. `owner_rank_for_position` is what you call to decide where to
send a particle.

:::caution
The physical split (equal widths) and the index split (`get_index_bounds`, equal numbers of
points) agree only when the number of grid points divides evenly by the number of ranks.
Otherwise a position near a subdomain edge can belong to a different rank than its nearest
grid node.
:::

## Replicated layouts

If no axis may be split (all `decompose` flags `False`) but there is more than one rank,
the layout is *replicated*: every rank holds the whole grid, `proc_sizes` is all ones and
`replicated` is `True`. Periodic axes then have the rank itself as both neighbours.

This is the layout for particle-in-cell codes that split the particles, not the grid: each
rank deposits its own particles on a full-size array and the copies are summed with
[`reduce_across_ranks`](/mpiarray/guides/operations/#combining-replicated-copies).

## Without MPI

With `comm=None`, or a serial run where `xp.mpi.get_mpi()` returns cunumpy's stand-in, the
layout has one rank that owns everything. The same code therefore runs unchanged with
`python script.py` and with `mpiexec -n 8 python script.py`.
