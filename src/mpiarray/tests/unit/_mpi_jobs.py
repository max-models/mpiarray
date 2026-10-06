"""Start separate MPI jobs from a serial test run (the examples, the GPU staging)."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys

import cunumpy as xp
import pytest

SERIAL_RUN = not xp.mpi.launched_under_mpi()


@functools.cache
def _launcher() -> list[str] | None:
    """Return the MPI launcher command, with Open MPI's oversubscribe flag."""
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        return None
    version = subprocess.run(
        [launcher, "--version"], capture_output=True, text=True, check=False
    ).stdout
    # Open MPI refuses more ranks than cores without it; MPICH has no such flag.
    return [launcher, "--oversubscribe"] if "Open MPI" in version else [launcher]


def run_job(nranks: int, args: list[str], env: dict[str, str] | None = None) -> str:
    """Run ``python args`` on ``nranks`` ranks; return its output, fail on errors or hangs.

    Skips the test without mpi4py or an MPI launcher.
    """
    pytest.importorskip("mpi4py", reason="needs mpi4py (the mpi extra)")
    launcher = _launcher()
    if launcher is None:
        pytest.skip("needs mpiexec to start an MPI job")
    result = subprocess.run(
        [*launcher, "-n", str(nranks), sys.executable, *args],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
