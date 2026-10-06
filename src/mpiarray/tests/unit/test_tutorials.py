"""Run every tutorial notebook as a script, serially and under MPI.

The docs build executes the notebooks on one rank only; here their code also
runs on several ranks, where a collective call made by only some ranks would
hang (and fail the test with a timeout).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mpiarray.tests.unit._mpi_jobs import SERIAL_RUN, run_job

REPO = Path(__file__).resolve().parents[4]
NOTEBOOKS = sorted((REPO / "tutorials").glob("*.ipynb"))

pytestmark = pytest.mark.skipif(not SERIAL_RUN, reason="starts its own MPI jobs")


@pytest.mark.parametrize("nranks", [1, 2, 4])
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_tutorial_runs(notebook: Path, nranks: int, tmp_path) -> None:
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("matplotlib")
    cells = nbformat.read(notebook, as_version=4).cells
    code = "\n\n".join(cell.source for cell in cells if cell.cell_type == "code")
    script = tmp_path / f"{notebook.stem}.py"
    script.write_text(code)
    env = {"PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg", "MPIARRAY_DEBUG": "1"}
    run_job(nranks, [str(script)], env=env)


def test_every_tutorial_is_tested() -> None:
    assert {path.stem for path in NOTEBOOKS} >= {
        "distributed_arrays",
        "heat_equation",
        "particles",
        "working_with_data",
    }
