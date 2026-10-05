"""The MPI module of mpiarray: mpi4py under an MPI launcher, else a serial stand-in."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypeAlias

import cunumpy as xp

if TYPE_CHECKING:
    from cunumpy.mpi import SerialComm
    from mpi4py import MPI
else:
    # mpi4py.MPI under an MPI launcher, otherwise cunumpy's serial stand-in
    MPI = xp.mpi.get_mpi()

# An mpi4py communicator, or cunumpy's serial stand-in for one.
Comm: TypeAlias = "MPI.Comm | SerialComm"

#: Environment variable that turns on the (collective) consistency checks.
DEBUG_VARIABLE = "MPIARRAY_DEBUG"


def default_comm() -> Comm:
    """Return the communicator arrays use by default: ``MPI.COMM_WORLD``."""
    return MPI.COMM_WORLD


def debug_checks() -> bool:
    """Return whether ``MPIARRAY_DEBUG`` asks for the consistency checks."""
    return os.environ.get(DEBUG_VARIABLE, "").strip().lower() in {"1", "true", "yes"}
