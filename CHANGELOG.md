# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `CopyPlan` packs and unpacks with one gather and one scatter per array (flat storage
  indices found when the plan is built; where target boxes overlap only the last copy's write
  is kept) instead of one slice per piece, and lays out its pieces with array operations: on
  the halos of 392 patches (4422 copies) a plan builds in 23 ms instead of 43 ms and executes
  in 0.7 ms instead of 10 ms.
- Faster `CopyPlan` construction: the boxes, blocks and regions of all copies are intersected
  as arrays instead of one copy and one rank at a time; 4422 copies (the halos of 392 patches)
  plan in 43 ms instead of 116 ms. The plans are the same.
- Faster halo exchanges on host storage: each array keeps the persistent MPI requests of its
  exchanges, made by the first one, so later exchanges only start and wait for them. On 4 ranks
  with 16×32×16 blocks, `update_halos(wait=False)` went from 850 to 65 µs and `update_halos()`
  from 157 to 88 µs. Starting an exchange of an array again before `wait()` on the previous one
  now raises.
- Faster elementwise operations: arithmetic and ufuncs on arrays with the same layout (and
  scalars) are computed in one pass over the whole storage, then the halo cells are zeroed, instead
  of a strided pass over the block and a copy into new storage (about 2–3× faster on large
  blocks). Floating-point warnings are not raised there, since the halo cells may hold anything
  (as on CuPy); under `np.errstate(...="raise")` the blocks alone are computed, so errors in
  them still raise.
- Faster in-place operations (`a += b`, `out=`): on C-ordered storage of at least 16384 cells
  with halo cells along more than the first axis, they go over the storage without the first
  axis's halo layers, one contiguous run, and save and restore the other halo cells, which they
  still keep (about 2× faster on 3-D blocks of 64³ cells and more).
- Faster whole-array reductions (`sum`, `min`, `max`, `dot`, `norm`, `all`, `any`, ...): the
  per-rank values are combined with a buffer `Allreduce` instead of a pickled `allreduce` (about
  10× less latency).
- The combined coverage of the serial and MPI runs must be at least 95% (was 100%).
- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (the 0.2.1 wheel reported 0.1.0).

### Fixed

- The published requirement on cunumpy is `cunumpy>=0.6.3` instead of a git URL, which PyPI rejects
  ("Can't have direct dependency"), so 0.2.0 could not be uploaded. Development environments still
  take cunumpy from its `devel` branch through `[tool.uv.sources]`.

### Added

- `mpa.compute_halos()`, a context in which elementwise operations compute the halo cells too:
  results are the operation applied to the inputs' halo cells, so arrays whose halos were up to
  date stay up to date without another `update_halos`, and in-place operations go over the whole
  storage at once. Only for halos filled from the neighbours, not for values waiting for
  `accumulate_halos`.

- A [Domain decomposition](https://max-models.github.io/mpiarray/guides/decomposition/) guide: how
  the process grid, the cut points, the rank numbering, the neighbours and the halo cells are
  decided, with figures drawn from `Layout` and an interactive explorer.
- Opt-in [scope-profiler](https://github.com/max-models/scope-profiler) regions (named
  `mpiarray.<operation>`, tag `mpiarray`) for halo exchanges, gathers, redistribution,
  reductions, copy plans, particle migration, stencils and I/O. They are off unless the
  application enables the tag, e.g. `sp.setup(enable_tags=["mpiarray"])` or
  `SCOPE_PROFILER_ENABLE_TAGS=mpiarray`. scope-profiler (>=0.7.5) is a new dependency.
- `mpa.CopyPlan` and `mpa.copy_boxes`: copy boxes between arrays that live on different
  communicators, all made of ranks of one parent communicator, e.g. the patches of an adaptive
  mesh each on its own ranks. A target box may reach into the target's halo at its walls. The
  plan is built once (one `allgather`) and executed with one `Alltoallv` over the parent.
- PETSc interoperability (`mpiarray.petsc`, needs petsc4py installed separately): `Layout.dmda()`
  builds a `PETSc.DMDA` with exactly the layout's decomposition (cached per layout);
  `DistributedArray.to_petsc()`, `copy_from_petsc()`, `mpa.copy_to_petsc`,
  `mpa.copy_from_petsc` and `mpa.from_petsc` move blocks to and from global vectors;
  `mpa.petsc_numbering` gives the DMDA row of every element for matrix assembly.
- `mpa.block_numbering(shape, cuts)`: the position of every element when blocks are stored
  one after another, allowing empty blocks.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
