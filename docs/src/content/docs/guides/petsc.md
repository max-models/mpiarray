---
title: PETSc
description: Solving on distributed arrays with PETSc - a DMDA with the layout's decomposition, vector transfer, and matrix row numbering.
sidebar:
  order: 6
---

`mpiarray.petsc` connects layouts and arrays to [petsc4py](https://petsc.org/release/petsc4py/),
so that KSP, SNES and TS can work on mpiarray data without a second, hand-made
decomposition. It needs petsc4py, which is not a dependency of mpiarray (it builds
PETSc from source where no wheel fits); install it separately:

```bash
pip install petsc4py  # or: conda install -c conda-forge petsc4py
```

petsc4py is imported on first use only; `import mpiarray` works without it.

## A DMDA with the layout's blocks

```python
import mpiarray as mpa

rho = mpa.zeros((64, 48), split=(0, 1), halo=1)
da = rho.layout.dmda()  # PETSc.DMDA; also mpa.petsc.dmda(layout)
```

The DMDA gets the layout's axes **reversed**: PETSc numbers ranks with x fastest, mpiarray
row-major with the last axis fastest, so PETSc's x is the layout's last axis. Then the rank
numbers agree, `da.getRanges()[::-1] == layout.index_bounds` on every rank (uneven and
explicit `bounds=` layouts included), and each rank's block of a PETSc global vector has the
memory order of its C-ordered mpiarray block. PETSc's natural ordering is mpiarray's global C
order.

- Periodic axes become `PERIODIC`, the others `NONE`; `boundary_type=` overrides, for every
  axis or one per axis in the layout's axis order.
- The stencil width defaults to the largest halo width, at least 1 (a DMDA has one width for
  all axes); `stencil_type` is `"star"` or `"box"`.
- On one rank, and for replicated layouts (`split=None`), the DMDA lives on `COMM_SELF` and
  holds the whole array on every rank.
- DMDAs have 1 to 3 axes.

The DMDA is cached per layout and options, so repeated calls (and `to_petsc`) return the same
object. Do not destroy or reconfigure it; take `da.clone()` for that.
`mpa.petsc.clear_dmda_cache()` (collective) destroys the cached DMDAs.

## Vectors

Each transfer is one local copy, without communication; halo cells are not part of PETSc's
global vectors:

```python
b = rho.to_petsc()  # a new global vector of rho.layout.dmda()
rho.to_petsc(b)  # or refill an existing one
x = da.createGlobalVec()
# ... solve into x ...
phi = mpa.from_petsc(x, rho.layout)  # a new array, zero halo cells
phi.copy_from_petsc(x)  # or overwrite the block; halo cells untouched
```

`mpa.copy_to_petsc(a, vec)` and `mpa.copy_from_petsc(vec, a)` are the function forms. CuPy
arrays are copied through the host.

## Assembling matrices

PETSc numbers the rows of a DMDA's vectors and matrices rank by rank, each block C-ordered.
`mpa.petsc_numbering(layout)` gives that row for every global element, computed locally from
the layout's cuts, so a matrix can be assembled from grid indices:

```python
numbering = mpa.petsc_numbering(rho.layout)
A = da.createMatrix()
(i0, i1), (j0, j1) = rho.layout.index_bounds
for i in range(i0, i1):
    for j in range(j0, j1):
        A.setValue(numbering[i, j], numbering[i, j], 4.0)
        # ... and -1.0 at numbering[i +- 1, j], numbering[i, j +- 1] inside the domain
A.assemble()
```

It is `mpa.block_numbering(layout.shape, layout.bounds)`, which needs no petsc4py and also
takes cuts with empty blocks, such as a coarse multigrid level that leaves a rank without
elements:

```python
mpa.block_numbering((4, 3), [(0, 1, 4), (0, 2, 3)])
# [[ 0  1  2]
#  [ 3  4  9]
#  [ 5  6 10]
#  [ 7  8 11]]
```
