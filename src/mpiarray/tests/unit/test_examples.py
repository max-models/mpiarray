"""Run the examples of the README and the docs under MPI, as users would.

A collective call that only some ranks make hangs the program; these tests
catch that (with a timeout), and check the numbers the guides quote.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mpiarray.tests.unit._mpi_jobs import SERIAL_RUN, run_job

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent / "mpi_scripts"
SOURCE = str(REPO / "src")

pytestmark = pytest.mark.skipif(not SERIAL_RUN, reason="starts its own MPI jobs")


def _first_python_block(path: Path) -> str:
    """Return the first ```python code block of a Markdown file."""
    if not path.exists():
        pytest.skip(f"{path} is not part of this installation")
    match = re.search(r"```python\n(.*?)```", path.read_text(), re.DOTALL)
    assert match is not None, f"no python example in {path}"
    return match.group(1)


@pytest.mark.parametrize(
    "document", ["README.qmd", "docs/src/content/docs/getting-started/quickstart.md"]
)
@pytest.mark.parametrize("nranks", [1, 2])
def test_example_runs_without_hanging(document: str, nranks: int, tmp_path) -> None:
    script = tmp_path / "example.py"
    script.write_text(_first_python_block(REPO / document))
    output = run_job(nranks, [str(script)], env={"PYTHONPATH": SOURCE})
    assert "DistributedArray(" in output


def test_numbers_quoted_in_the_guides() -> None:
    output = run_job(4, [str(SCRIPTS / "guide_claims.py")], env={"PYTHONPATH": SOURCE})
    assert "all guide claims hold" in output
