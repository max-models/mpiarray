# Changelog

All notable changes to this project are documented in this file and maintained manually. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- PETSc interoperability (`mpiarray.petsc`, optional extra `mpiarray[petsc]`): `Layout.dmda()`
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
