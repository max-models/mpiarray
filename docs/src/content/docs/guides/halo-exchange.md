---
title: Halo exchange
description: Copying neighbour values into the halo cells before a stencil with update_halos, and adding deposited values into the neighbours with accumulate_halos.
sidebar:
  order: 3
---

Halo cells hold copies of, or contributions to, cells that belong to a neighbouring rank.
mpiarray moves data through them in both directions:

|                     | `update_halos()`                       | `accumulate_halos()`                        |
| ------------------- | -------------------------------------- | ------------------------------------------- |
| direction           | neighbour's block → my halo cells      | my halo cells → neighbour's block           |
| operation           | copy (overwrite)                       | add, then zero my halo cells                |
| use it              | before reading neighbours (stencils, interpolation) | after writing into halo cells (particle deposition) |
| at a wall           | halo cells left unchanged              | halo values dropped                         |

Both are collective. They work on every axis, or on one with `axis=`, which is useful when
a stencil only reaches along some axes. On an axis with no halo cells they do nothing.
`clear_halos()` sets the halo cells to zero without communicating.

## Updating halo cells

After `update_halos()`, the halo cells of each rank equal the boundary cells of its
neighbours. For an 8-element periodic axis on four ranks, with each rank's block set to
`rank + 1` and one halo cell:

```text
rank 0: [4 | 1 1 | 2]
rank 1: [1 | 2 2 | 3]
rank 2: [2 | 3 3 | 4]
rank 3: [3 | 4 4 | 1]
```

Without periodicity the outer halo cells of rank 0 and rank 3 keep their previous value
(`0` for a new array). Zero halo cells at a wall act as a homogeneous Dirichlet boundary
for a stencil; write other boundary values into them yourself after `update_halos()`.

### Example: a Laplacian

```python
import numpy as np

import mpiarray as mpa

n = 64
h = 2 * np.pi / n
u = mpa.fromfunction(lambda i: np.sin(i * h), (n,), halo=1, periodic=True)
lap = mpa.zeros_like(u)

u.update_halos()
v = u.local_with_halos
lap.local[...] = (v[2:] - 2 * v[1:-1] + v[:-2]) / h**2
```

The stencil reads `local_with_halos` and writes `local`; with one halo cell, `v[1:-1]` is
the block. The result is the same on any number of ranks.

The axes are updated one after the other, so the corner halo cells are filled too: the
second axis exchanges rows that already contain the first axis's halo cells. A 9-point
stencil in 2D therefore works after a single `update_halos()`.

## Accumulating halo cells

When particles are deposited on a grid, a particle near the edge of a block also
contributes to cells owned by the neighbour. Deposit into `local_with_halos`, then call
`accumulate_halos()`: each rank's halo values are added to the matching cells of its
neighbour, and the halo cells are reset to zero.

```python
rho = mpa.zeros((128, 128), split=(0, 1), halo=2, periodic=True)

deposit(particles, rho.local_with_halos)  # may write into the halo cells
rho.accumulate_halos()  # halo contributions -> owners
total_charge = rho.sum()  # nothing is lost on periodic axes
```

On periodic axes the total is conserved. At a wall (`PROC_NULL` neighbour) the
contributions that fall outside the domain are dropped; apply your boundary condition to
the halo cells before accumulating if they should be reflected or kept.

## Halo cells and other operations

- Arithmetic (`a + b`, `np.sin(a)`, …) acts on the whole storage, halo cells included, so
  the halo cells of a result hold the operation applied to the operands' halo cells. They
  are meaningful only if the operands' halo cells were; call `update_halos()` on the
  result before using its halo cells.
- `mpa.array(...)` starts with zero halo cells, and assignments with array or mask indices
  zero them.
- Reductions, norms, `gather()` and `get()` never read the halo cells.
