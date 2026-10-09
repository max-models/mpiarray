# PETSc interoperability

Planned 2026-10-06, implemented 2026-10-09 in `src/mpiarray/petsc.py` (optional extra
`mpiarray[petsc]`, petsc4py imported lazily). Goal: solve on mpiarray data with PETSc (KSP,
SNES, TS) without building a second decomposition by hand.

## Done

- `mpa.petsc.dmda(layout, *, stencil_width=None, stencil_type="star", boundary_type=None)`,
  also `layout.dmda(...)`: a `PETSc.DMDA` with exactly the layout's decomposition.
  - **Axis order.** PETSc gets the axes reversed: sizes `shape[::-1]`,
    `proc_sizes = process_grid[::-1]`. PETSc's x is mpiarray's last axis, the rank numbers
    agree, and a rank's block of a global vector has the memory order of its C-ordered
    mpiarray block. PETSc's natural ordering is mpiarray's global C order.
  - **Ownership.** `ownership_ranges` from `layout.bounds` (the per-axis cuts), reversed, so
    uneven, explicit-`bounds` and weighted layouts carry over.
  - **Boundaries.** `periodic` → `PERIODIC`, else `NONE`; `boundary_type` overrides (one
    value or one per axis, in the layout's axis order).
  - **Stencil width.** Default `max(1, *layout.halo)` (DMDA has one width for all axes).
  - **Communicator.** `layout.comm` (the Cartesian one with `reorder=True`); on one rank and
    for replicated layouts `COMM_SELF`, the whole array on every rank.
  - **Dimensions.** 1–3 axes, else `ValueError`.
  - **Cache.** One DMDA per (layout, options); layouts compare by value, so equal layouts
    share it. Callers must not destroy or reconfigure it (use `da.clone()`);
    `mpa.petsc.clear_dmda_cache()` (collective) destroys them.
- Vectors, one local copy each way, no communication: `mpa.copy_to_petsc(a, vec)`,
  `mpa.copy_from_petsc(vec, a)` (halo cells untouched), `mpa.from_petsc(vec, layout)`,
  `a.to_petsc(vec=None)` (new global vector of the cached DMDA when `None`),
  `a.copy_from_petsc(vec)`. CuPy storage goes through the host.
- Numbering for matrix assembly: `mpa.petsc_numbering(layout)` (int64 array of
  `layout.shape`, the DMDA global row of every element), built on
  `mpa.block_numbering(shape, cuts)` (in `layout.py`, no petsc4py; allows empty blocks,
  e.g. a multigrid coarse level leaving a rank without nodes). `layout.bounds` is the
  public per-axis cuts.
- Tests: `src/mpiarray/tests/unit/test_petsc.py` (DMDA ranges vs `index_bounds`, AO vs
  `petsc_numbering`, round trips with/without halos on uneven layouts, a KSP Poisson solve
  against a dense NumPy solve), passing serially and on 2/3/4 ranks with petsc4py 3.24.3.
  Run them with an interpreter that has petsc4py, e.g.
  `mpiexec -n 4 python -m pytest src/mpiarray/tests/unit/test_petsc.py`.
- PICNIC (2026-10-09): `picnic/solver/petsc_dmda.py` builds its DMDA with `layout.dmda()`,
  and its Poisson matrix numbers the rows with `petsc_numbering`, its multigrid coarse levels
  with `block_numbering`, so a spatially decomposed run solves on the field arrays' blocks.
- Coverage: `petsc.py` is omitted from the coverage gate (CI has no petsc4py). The thin
  methods in `layout.py`/`distributed_array.py` and `block_numbering` are covered without
  petsc4py (ImportError path, duck-typed vector).

## Open

- **Zero-copy vectors.** Without halo cells and with C-contiguous storage a global vector
  could wrap mpiarray storage directly
  (`PETSc.Vec().createWithArray(block.reshape(-1), size=(n_local, n_global), comm=comm)`,
  keeping a reference to the array). Test that writes through one show in the other.
- **Ghosted local vectors.** When the stencil width equals the halo widths, PETSc's local
  vector has the shape of mpiarray's storage and `DMGlobalToLocal` is the same operation as
  `update_halos`: offer that as an alternative halo update, or not.
- **GPU.** CuPy storage maps to PETSc's CUDA vectors (`Vec.createCUDAWithArrays` /
  `VECCUDA`), only in a CUDA build of PETSc. Check how petsc4py exposes device pointers
  (`getCUDAHandle`), and synchronize the CuPy stream before handing memory to PETSc.
  Currently: copy through the host.
- **CI job.** petsc4py builds PETSc from source unless a wheel matches; check wheel
  availability for the CI images (Linux x86-64, Python 3.10–3.13), or install PETSc from
  conda-forge in a separate job that runs `test_petsc.py` serially and on 2/3/4 ranks and
  then drops the coverage omit.
