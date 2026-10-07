"""Start separate MPI jobs from a serial test run (the examples, the GPU staging)."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys

import maybempi
import pytest

SERIAL_RUN = not maybempi.launched_under_mpi()


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
    # Open MPI 5 calls itself "Open MPI", Open MPI 4 (Ubuntu's) "OpenRTE".
    is_open_mpi = any(
        name in version.lower() for name in ("open mpi", "openrte", "open-mpi")
    )
    return [launcher, "--oversubscribe"] if is_open_mpi else [launcher]


def run_job(
    nranks: int,
    args: list[str],
    env: dict[str, str] | None = None,
    *,
    expect_failure: bool = False,
) -> str:
    """Run ``python args`` on ``nranks`` ranks and return its output (stdout and stderr).

    Fails the test if the job fails (or, with ``expect_failure``, succeeds) or
    hangs. Skips it without mpi4py or an MPI launcher.
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
    output = result.stdout + result.stderr
    assert (result.returncode != 0) == expect_failure, output
    return output
