---
title: Operations and reductions
description: Arithmetic, NumPy ufuncs and broadcasting, indexing, global reductions and norms on a distributed array, and which of them communicate.
sidebar:
  order: 4
---

A distributed array behaves like a NumPy array for elementwise arithmetic and reductions.
Elementwise operations run on each rank's block without communication; reductions combine
the blocks with one `allreduce`.

## Arithmetic and ufuncs

The operators `+ - * / // **`, unary `- + abs`, and the comparisons `< <= == != > >=`
return a new array with the same layout. The in-place forms `+= -= *= /=` write into the
existing storage.

```python
import numpy as np

import mpiarray as mpa

a = mpa.array(np.arange(12.0).reshape(4, 3))
b = 2 * a + 1
mask = a > 5  # a boolean distributed array
a /= 3
```

NumPy (and CuPy) ufuncs work too and return a distributed array:

```python
np.sqrt(a)
np.maximum(a, b)
np.multiply(a, 2.0, out=a)  # in place, allocates nothing
np.add(a, b, where=mask, out=a)
```

`out=` must be a distributed array with the same layout. Ufunc methods such as
`np.add.reduce` or `np.add.at` are not supported; use the reductions below.

Elementwise operations are computed on the blocks: a new result has zero halo cells, and
`out=` or an in-place operator leaves the halo cells of the target alone; see
[Halo cells and other operations](/mpiarray/guides/halo-exchange/#halo-cells-and-other-operations).

### Broadcasting

The other operand can be:

1. a distributed array with the same layout,
2. a scalar,
3. an array with the storage shape (`a.layout.storage_shape`), used as is on each rank,
4. any array that broadcasts against the *global* shape, as in NumPy.

For case 4 every rank must pass the same array. Each rank cuts out the part that matches
its block, so the result equals NumPy's on the gathered arrays:

```python
a * np.array([1.0, 10.0, 100.0])  # scales the last axis
a * np.arange(12.0).reshape(4, 3)  # a full global array
```

An operand that varies only along axes each rank holds whole (not split, no halo cells) is
applied directly. Otherwise it is expanded to the global shape before slicing, so pass
full global arrays only when they fit in memory on every rank.

## Indexing

Reading with global indices is collective. A single element comes back as a number on
every rank; every other selection is a new distributed array (split like `a` where it
can be, otherwise near-evenly), so slicing a large array never builds it on one rank.
Use `.gather()` or `.to_numpy()` on the result for a plain array:

| Expression                    | Result                                    |
| ----------------------------- | ----------------------------------------- |
| `a[i, j]`, `a.get((i, j))`    | the element, broadcast from its owner     |
| `a[1:3, :]`, `a[..., 0]`, …   | a new distributed array; only the selected cells are sent, to the ranks that own them in the result |
| `a[mask]`, `a[[0, 5]]`, …     | a new distributed array, made from the gathered array |
| `a.local_index((i, j))`       | the index into `a.local_with_halos`, or `None` if this rank does not own it (no communication) |

Assignment works the other way round: every rank passes the same value and writes the part
it owns.

```python
a[0, :] = -1.0  # no communication
a[:, 1] = np.arange(4.0)  # the value broadcasts to the selection
a[np.array([1, 2]), 0] = 7.0  # advanced index: gathers, assigns, redistributes
```

Integers, slices and `...` are written locally without communication. Index arrays and
masks gather the whole array, assign, and redistribute it (collective; the halo cells are
zeroed).

`len(a)`, iteration over the first axis (which gathers) and `bool(a)` for a one-element
array work as in NumPy.

## Reductions

`sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `all` and `any` reduce over the whole
array and return the same host scalar on every rank, also on the GPU backend. Only the
blocks count, not the halo cells.

```python
a.sum()
a.mean()
a.std(ddof=1)
(a > 0).all()
np.sum(a)  # NumPy's functions call the methods
```

With `axis=...` the result is again a distributed array, unless every axis is reduced
(then a scalar, as above). The array is not gathered: every rank reduces its own block,
and only these partial results travel, to the ranks that own them in the result. `var`
and `std` combine per-block means and variances (Chan et al.'s parallel formula), so they
are as accurate as NumPy's two-pass result:

```python
a.sum(axis=0)  # a distributed array of shape (3,)
a.var(axis=1, ddof=1)
a.max(axis=(0, 1), keepdims=True)  # shape (1, 1)
a.sum(axis=0).to_numpy()  # a numpy.ndarray on every rank
```

### Positions and running totals

`argmin` and `argmax` return the flat index of the first extreme element, a Python `int`
on every rank (NaN counts as the extreme, as in NumPy); with `axis=` a distributed array
of indices along that axis. `cumsum` and `cumprod` along an axis return a distributed
array with the same layout; each rank adds the totals of the blocks before it, which is
one exchange along the split axis. Without `axis` they work only on 1-D arrays, since
NumPy would flatten.

```python
i = a.argmax()  # e.g. np.unravel_index(i, a.shape)
c = a.cumsum(axis=0)
```

All reductions are collective: every rank must call them, even ranks that own no cells.
`out=` is not supported.

## NumPy functions

NumPy's functions dispatch to mpiarray (`__array_function__`), so code written for NumPy
arrays often runs unchanged:

| Kind            | Functions                                                        |
| --------------- | ---------------------------------------------------------------- |
| reductions      | `np.sum`, `prod`, `min`/`amin`, `max`/`amax`, `mean`, `var`, `std`, `all`, `any`, `argmin`, `argmax` |
| running totals  | `np.cumsum`, `np.cumprod`                                        |
| elementwise     | `np.where`, `clip`, `round`/`around`, `real`, `imag`, `isclose`  |
| comparisons     | `np.allclose`, `np.array_equal` (a `bool` on every rank)         |
| linear algebra  | `np.vdot`, `np.linalg.norm`                                      |
| creation, shape | `np.zeros_like`, `ones_like`, `empty_like`, `full_like`, `copy`, `shape`, `ndim`, `size`, `astype` |

Any other function raises `TypeError` instead of silently gathering the array; call it on
`a.gather()` when that is what you want. `np.asarray(a)` gathers explicitly (a
`numpy.ndarray`, also with CuPy storage).

## Inner products and norms

```python
a.vdot(b)  # sum(conj(a) * b) over the whole array
a.norm()  # Euclidean norm
a.norm(1)  # sum of absolute values
a.norm(np.inf)  # largest absolute value
```

These skip the halo cells and return the same value on every rank, which makes them
suitable for convergence checks in iterative solvers.

## Combining replicated copies

When every rank holds the whole grid (`split=None`) and has deposited only its own
particles, `allreduce_replicated` sums the copies in place, halo cells included:

```python
rho = mpa.zeros((128, 128), split=None, halo=2)
deposit(my_particles, rho.local_with_halos)
rho.allreduce_replicated()  # op=MPI.SUM by default
```

It raises for a split array, whose blocks are different parts of the grid.

## Which calls communicate

Collective calls must be made by every rank of the communicator, in the same order, or
the program hangs. A collective call inside `if rank == 0:` is the classic mistake.

| No communication                                           | Collective                                    |
| ---------------------------------------------------------- | --------------------------------------------- |
| creation functions (`zeros`, `array`, `arange`, …)          | `update_halos`, `accumulate_halos`            |
| `copy`, `astype`, writes to `local`                         | `gather`, `to_numpy`, `np.asarray(a)`, iteration |
| operators and ufuncs                                        | reductions, `vdot`, `norm`                    |
| `a[...] = v` with ints and slices                           | `a[...]`, `get`, `bool(a)`                    |
|                                                             | `redistribute`, `rebalance`, `mpa.migrate`    |
|                                                             | `argmin`, `argmax`, `cumsum`, `cumprod`       |
| `repr(a)`, `print(a)`, `a.layout`, `local_index`, `clear_halos` | `a[...] = v` with arrays or masks         |
| `layout.owners`                                              | `allreduce_replicated`, `from_local`          |
|                                                             | `mpa.save`, `mpa.load`, `save_hdf5`, `load_hdf5` |

## Debugging hangs

With `MPIARRAY_DEBUG=1` in the environment, every collective call first checks that all
ranks are in the same call. Ranks in different calls raise an error naming both, and a
rank left waiting (because the others never make the call) raises after
`MPIARRAY_DEBUG_TIMEOUT` seconds (default 30) instead of hanging:

```bash
MPIARRAY_DEBUG=1 MPIARRAY_DEBUG_TIMEOUT=10 mpiexec -n 4 python simulation.py
```

```text
RuntimeError: rank 0 waited 10 s in gather (collective call 7) for the other ranks. ...
```

The checks cost one barrier and one small `allgather` per collective call, so leave them
off in production runs. They also make `mpa.array` check that every rank passed the same
data.
