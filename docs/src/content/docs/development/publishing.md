---
title: Publishing
description: Publish the package to PyPI and the documentation to GitHub Pages.
---

## Documentation

The `Docs` workflow builds the site on every push and pull request to `main` and `devel`. It
executes the tutorials, generates the API reference and uploads the site. Pushes to `devel`
also deploy it to GitHub Pages at https://max-models.github.io/mpiarray/.

Enable GitHub Pages once in the repository settings, with "GitHub Actions" as the source.

## Releases and PyPI

The `Release` workflow uses [release-please](https://github.com/googleapis/release-please).
On every push to `main` it opens or updates a release pull request from the
[Conventional Commits](https://www.conventionalcommits.org/) since the last release, with the
next version and the changelog entry. Merging that pull request:

1. tags the release `vX.Y.Z` and creates the GitHub release,
2. updates `CHANGELOG.md`, the version in `pyproject.toml` and `__version__` in the package,
3. builds the package with `uv build` and publishes it to PyPI with trusted publishing (OIDC).

A tag pushed by hand (`git tag v1.2.3 && git push --tags`) also publishes.

### One-time configuration

1. Create an account on [PyPI](https://pypi.org/) and create or claim your project name.
2. In the PyPI project settings go to "Publishing" → "Add a new publisher" and fill in:
   - **PyPI project name**: `mpiarray` (or your project name)
   - **Owner**: your GitHub username or organization
   - **Repository name**: `mpiarray`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. Optionally, in the GitHub repository settings create an environment named `pypi` whose
   deployment branches are limited to `main`.

### Coverage

The tests upload coverage to [Codecov](https://codecov.io/). Public repositories need no
configuration; for a private one add a `CODECOV_TOKEN` repository secret.
