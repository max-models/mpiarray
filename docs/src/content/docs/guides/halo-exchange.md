---
title: Halo exchange
description: Filling ghost cells before a stencil with fill_halos, and accumulating deposited values into neighbours with exchange_halos.
sidebar:
  order: 3
---

Ghost cells hold copies of, or contributions to, cells that belong to a neighbouring rank.
mpiarray has two exchanges that move data in opposite directions:

|                     | `fill_halos()`                         | `exchange_halos()`                          |
| ------------------- | -------------------------------------- | ------------------------------------------- |
| direction           | neighbour's interior → my ghost cells  | my ghost cells → neighbour's interior       |
| operation           | copy (overwrite)                       | add, then zero my ghost cells               |
| use it              | before reading neighbours (stencils, interpolation) | after writing into ghost cells (particle deposition) |
| at a wall           | ghost cells left unchanged             | ghost values discarded                      |

Both are collective over the array's communicator and work axis by axis (`fill_halo(dim)`,
`exchange_halo(dim)`). The per-axis versions are useful when a stencil only reaches along
some axes. With `num_ghostpoints=0`, or on an axis with `ghost_axes[dim] = False`, they do
nothing.

## Filling ghost cells

After `fill_halos()`, the ghost cells of each rank equal the boundary cells of its
neighbours. For an 8-point periodic axis on four ranks, with each rank's interior set to
`rank + 1` and one ghost point:

```text
rank 0: [4 | 1 1 | 2]
rank 1: [1 | 2 2 | 3]
rank 2: [2 | 3 3 | 4]
rank 3: [3 | 4 4 | 1]
```

Without periodicity the outer ghost cells of rank 0 and rank 3 keep their previous value
(`0` for a new array). Zero ghost cells at a wall act as a homogeneous Dirichlet boundary
for a stencil; write other boundary values into them yourself after `fill_halos()`.

### Example: a Laplacian

```python
import cunumpy as xp
import numpy as np

from mpiarray import DistributedArray

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD

n = 64
x = np.linspace(0, 2 * np.pi, n, endpoint=False)
h = x[1] - x[0]

u = DistributedArray.from_array(np.sin(x), comm, num_ghostpoints=1, periodic=(True,))
lap = DistributedArray.zeros(u.shape, comm, num_ghostpoints=1, periodic=(True,))

u.fill_halos()
v = u.local_with_halos
lap.local[...] = (v[2:] - 2 * v[1:-1] + v[:-2]) / h**2
```

The stencil reads `local_with_halos` and writes `local`; with one ghost point,
`v[1:-1]` is the interior. The result is the same on any number of ranks.

The exchange is done axis by axis, so the corner ghost cells are filled too: the second
axis exchanges rows that already contain the first axis's ghost cells. A 9-point stencil
in 2D therefore works after a single `fill_halos()`.

## Accumulating ghost cells

When particles are deposited on a grid, a particle near a subdomain edge also contributes to
cells owned by the neighbour. Deposit into `local_with_halos`, then call
`exchange_halos()`: each rank's ghost values are added to the matching interior cells of its
neighbour and the ghost cells are reset to zero.

```python
rho = DistributedArray.zeros((128, 128), comm, num_ghostpoints=2, periodic=(True, True))

deposit(particles, rho.local_with_halos)   # may write into the ghost cells
rho.exchange_halos()                       # ghost contributions -> owners
total_charge = rho.sum()                   # nothing is lost on periodic axes
```

On periodic axes the total is conserved. At a wall (`PROC_NULL` neighbour) the
contributions that fall outside the domain are dropped; apply your boundary condition to
the ghost cells before the exchange if they should be reflected or kept.

## Halo cells and other operations

- Arithmetic (`a + b`, `np.sin(a)`, …) acts on the whole local storage, halo cells
  included, so the ghost cells of a result hold the operation applied to the operands'
  ghost cells. They are meaningful only if the operands' ghost cells were. Call
  `fill_halos()` on the result before using its ghost cells.
- `fill(global_array)`, and assignments with array or mask indices, zero the halo cells.
- Reductions, norms and `to_ndarray()` never read the halo cells.
- `clear_halos(dim)` zeroes the ghost cells along one axis.
