"""mpiarray imports only its declared dependencies, and mpi4py only under MPI."""

import ast
import sys
from pathlib import Path

import maybempi
import pytest

PACKAGE = Path(__file__).resolve().parents[2]
ALLOWED = {
    "cunumpy",
    "maybempi",
    "mpi4py",
    "numpy",
    "cupy",  # lazily, in DistributedArray.to_cupy()
    "h5py",  # lazily, for save_hdf5/load_hdf5 (the hdf5 extra)
    "petsc4py",  # lazily, in mpiarray.petsc (the petsc extra)
    "typing_extensions",  # under TYPE_CHECKING only
}


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.glob("*.py")), ids=lambda path: path.name
)
def test_imports_only_declared_dependencies(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if name.startswith("mpiarray"):
                continue
            top = name.split(".")[0]
            assert top in ALLOWED or top in sys.stdlib_module_names, (
                f"{path.name} imports {name}"
            )


_SERIAL_SCRIPT = """
import sys

import cunumpy as xp
import maybempi

import mpiarray as mpa
from mpiarray import _mpi, distributed_array, layout

assert maybempi.is_serial(_mpi.MPI)
assert layout.MPI is distributed_array.MPI is _mpi.MPI
a = mpa.arange(6.0, halo=1, periodic=True)
a.update_halos()
assert a.sum() == 15.0
assert xp.to_numpy(a.local_with_halos).tolist() == [5, 0, 1, 2, 3, 4, 5, 0]
assert "mpi4py" not in sys.modules, "a serial run imported mpi4py"
assert "petsc4py" not in sys.modules, "import mpiarray imported petsc4py"
"""


@pytest.mark.skipif(maybempi.launched_under_mpi(), reason="checks a serial run")
def test_serial_run_uses_the_stand_in_and_never_imports_mpi4py() -> None:
    import os
    import subprocess

    from maybempi import OVERRIDE_VARIABLE

    env = {key: value for key, value in os.environ.items() if key != OVERRIDE_VARIABLE}
    subprocess.run([sys.executable, "-c", _SERIAL_SCRIPT], env=env, check=True)
