"""mpiarray imports only its declared dependencies (and CuPy, lazily)."""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2]
ALLOWED = {
    "array_api_compat",
    "cunumpy",
    "mpi4py",
    "numpy",
    "cupy",  # lazily, in DistributedArray.to_cupy()
    "typing_extensions",  # under TYPE_CHECKING only
}


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.glob("*.py")), ids=lambda path: path.name
)
def test_imports_only_declared_dependencies(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
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
