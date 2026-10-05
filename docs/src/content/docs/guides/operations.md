---
title: Operations and reductions
description: Arithmetic, NumPy ufuncs and broadcasting, indexing, global reductions and norms on a DistributedArray, and which of them communicate.
sidebar:
  order: 4
---

A `DistributedArray` behaves like a NumPy array for elementwise arithmetic and reductions.
Elementwise operations run on each rank's block without communication; reductions combine
the blocks with one `allreduce`.

## Arithmetic and ufuncs

The operators `+ - * / // **`, unary `- + abs`, and the comparisons `< <= == != > >=`
return a new array with the same layout. The in-place forms `+= -= *= /=` write into the
existing storage.

```python
import cunumpy as xp
import numpy as np

from mpiarray import DistributedArray

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD

a = DistributedArray.from_array(np.arange(12.0).reshape(4, 3), comm)
b = 2 * a + 1
mask = a > 5          # a boolean DistributedArray
a /= 3
```

NumPy (and CuPy) ufuncs also work and return a `DistributedArray`:

```python
np.sqrt(a)
np.maximum(a, b)
np.multiply(a, 2.0, out=a)   # in place, allocates nothing
np.add(a, b, where=mask, out=a)
```

`out=` must be a `DistributedArray` with the same layout. Ufunc methods such as
`np.add.reduce` or `np.add.at` are not supported; use the reductions below.

Elementwise operations also act on the ghost cells; see
[Halo cells and other operations](/mpiarray/guides/halo-exchange/#halo-cells-and-other-operations).

### Broadcasting

The other operand can be:

1. a `DistributedArray` with the same layout,
2. a scalar,
3. an array with the local storage shape (`shape_with_halos`), used as is on each rank,
4. any array that broadcasts against the *global* shape, as in NumPy.

For case 4 every rank must pass the same array. Each rank cuts out the part that matches
its block, so the result equals NumPy's on the gathered arrays:

```python
a * np.array([1.0, 10.0, 100.0])           # scales the last axis
a * np.arange(12.0).reshape(4, 3)          # a full global array
```

An operand that varies only along undivided axes (no ranks, no ghost cells) is applied
directly without slicing. Otherwise it is expanded to the global shape before slicing, so
pass full global arrays only when they fit in memory on every rank.

## Indexing

| Expression                    | Result                                   | Communication          |
| ----------------------------- | ---------------------------------------- | ---------------------- |
| `a[i, j]` (one int per axis)  | the value on the owning rank, `None` on the others | none          |
| `a[1:3, :]`, `a[mask]`, …     | the selection of the gathered array, on every rank | collective gather |
| `a.get_global_value((i, j), root=0)` | the value on `root`, `None` on the others | collective gather |
| `a.global_to_local((i, j))`   | the index into `a.data`, or `None` if not owned | none            |

```python
v = a[3, 2]                     # a number on one rank, None on the rest
v = a.get_global_value((3, 2))  # a number on rank 0
row = a[0, :]                   # gathers the whole array first
```

Assignment works the other way round: every rank passes the same value and writes the part
it owns.

```python
a[0, :] = -1.0                  # no communication
a[:, 1] = np.arange(4.0)        # value broadcasts to the selection
a[np.array([1, 2]), 0] = 7.0    # advanced index: gathers, assigns, refills
```

Integers, slices and `...` are written locally without communication. Index arrays and
masks gather the whole array, assign, and redistribute it (collective; the ghost cells are
zeroed).

## Reductions

`sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `all` and `any` reduce over the whole
array and return the same scalar on every rank. Only the interior cells count.

```python
a.sum()
a.mean()
a.std(ddof=1)
(a > 0).all()
```

With `axis=...` the array is gathered and reduced with NumPy, so the result is a regular
array on every rank:

```python
a.sum(axis=0)        # numpy/cupy array of shape (3,)
```

All reductions are collective: every rank must call them, even ranks that own no cells.
`out=` is not supported.

## Inner products and norms

```python
a.vdot(b)            # sum(conj(a) * b) over the whole array
a.norm()             # Euclidean norm
a.norm(1)            # sum of absolute values
a.norm(np.inf)       # largest absolute value
```

These skip the ghost cells and return the same value on every rank, which makes them
suitable for convergence checks in iterative solvers.

## Combining replicated copies

When every rank holds the whole grid (a [replicated layout](/mpiarray/guides/domain-decomposition/#replicated-layouts),
or an array created with `comm=None`) and has deposited only its own particles,
`reduce_across_ranks` sums the copies in place, halo cells included:

```python
rho = DistributedArray.zeros((128, 128), None, num_ghostpoints=2)
deposit(my_particles, rho.local_with_halos)
rho.reduce_across_ranks(comm)        # op=MPI.SUM by default
```

It raises if the array is decomposed over the same communicator, where the blocks are
different parts of the grid and summing them would be wrong.

## Which calls communicate

Calls marked collective must be made by every rank of the communicator, in the same order,
or the program hangs.

| No communication                                           | Collective                                  |
| ---------------------------------------------------------- | ------------------------------------------- |
| constructors, `copy`, `astype`                             | `fill_halos`, `exchange_halos` (and per-axis) |
| `fill`, `fill_local`, writes to `local`                    | `to_ndarray`, `to_numpy`, `to_cupy`, `np.asarray(a)`, `a.T` |
| operators and ufuncs                                       | reductions without `axis`, and with `axis` (gather) |
| `a[i, j]` with one int per axis                            | `vdot`, `norm`                              |
| `a[...] = v` with ints and slices                          | `a[...]` with slices, arrays or masks       |
| layout queries (`proc_index_bounds`, `neighbour_ranks`, …) | `a[...] = v` with arrays or masks           |
|                                                            | `get_global_value`, `reduce_across_ranks`   |
