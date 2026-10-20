# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `__version__` is read from the installed package metadata (`importlib.metadata`), not hardcoded,
  so it can no longer disagree with `pyproject.toml` (the 0.2.1 wheel reported 0.1.0).

### Fixed

- The published requirement on cunumpy is `cunumpy>=0.6.3` instead of a git URL, which PyPI rejects
  ("Can't have direct dependency"), so 0.2.0 could not be uploaded. Development environments still
  take cunumpy from its `devel` branch through `[tool.uv.sources]`.

### Added

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
