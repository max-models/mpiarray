# Contributing

Thanks for taking the time to contribute.

## Set up

```bash
git clone https://github.com/max-models/mpiarray.git
cd mpiarray
make install          # uv sync with the dev extra and the pre-commit hooks
```

Without uv:

```bash
python -m venv env && source env/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Develop

- Format and lint with [ruff](https://docs.astral.sh/ruff/); the pre-commit hooks run it for you.
- Type-check with `pyright src/`.
- Add tests under `src/<package>/tests/` and run `pytest .`.
- Every public module, class and function needs a Google-style docstring. They are rendered
  in the API reference and checked by ruff's `D` rules.
- Preview the documentation with `make docs-dev`; see the
  [docs guide](https://max-models.github.io/mpiarray/development/docs/).

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`. A `feat!:` or a
`BREAKING CHANGE:` footer marks a breaking change. release-please reads these to write the
changelog and choose the next version.

## Pull requests

Open the PR against `devel`. CI runs the tests on every supported Python version, ruff, pyright,
the tutorials and the documentation build. One approving review is required to merge.

## Releases

Merging to `main` lets release-please open or update a release PR. Merging that PR tags the
release, updates `CHANGELOG.md` and publishes the package to PyPI.
