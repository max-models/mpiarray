"""The MPI module of mpiarray: mpi4py under an MPI launcher, else a serial stand-in."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, TypeAlias, cast

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
#: Seconds a rank waits for the others in a collective call, with the checks on.
DEBUG_TIMEOUT_VARIABLE = "MPIARRAY_DEBUG_TIMEOUT"

# Collective calls made so far, per communicator, for the debug checks.
_CALLS: dict[int, int] = {}


def default_comm() -> Comm:
    """Return the communicator arrays use by default: ``MPI.COMM_WORLD``.

    Under an MPI launcher this is mpi4py's; otherwise cunumpy's serial
    stand-in, with one rank.
    """
    return MPI.COMM_WORLD


def debug_checks() -> bool:
    """Return whether ``MPIARRAY_DEBUG`` asks for the consistency checks."""
    return os.environ.get(DEBUG_VARIABLE, "").strip().lower() in {"1", "true", "yes"}


def check_collective(comm: Comm, name: str) -> None:
    """With ``MPIARRAY_DEBUG=1``, check that every rank makes the same collective call.

    Called right before mpiarray communicates. All ranks first meet in a
    non-blocking barrier: a rank that waits longer than
    ``MPIARRAY_DEBUG_TIMEOUT`` seconds (default 30) raises instead of hanging,
    which is what happens when a collective is called on only some ranks.
    Then the ranks compare the names and numbers of their calls and all raise
    if they differ. Does nothing without the variable, or on one rank.

    Args:
        comm: The communicator of the call.
        name: What is being called, for the error message.

    Raises:
        RuntimeError: If the other ranks do not arrive in time, or arrive in a
            different call.
    """
    if not debug_checks() or comm.Get_size() == 1:
        return
    count = _CALLS.get(id(comm), 0) + 1
    _CALLS[id(comm)] = count
    # several ranks only happen under a real MPI
    mpi_comm = cast("MPI.Comm", comm)
    timeout = float(os.environ.get(DEBUG_TIMEOUT_VARIABLE, "30"))
    arrived: Any = mpi_comm.Ibarrier()
    deadline = time.monotonic() + timeout
    while not arrived.Test():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"rank {comm.Get_rank()} waited {timeout:g} s in {name} (collective "
                f"call {count}) for the other ranks. A collective called on only some "
                "ranks, e.g. inside `if rank == 0:`, hangs; every rank must make it. "
                f"Set {DEBUG_TIMEOUT_VARIABLE} to wait longer.",
            )
        time.sleep(0.001)
    calls = mpi_comm.allgather((name, count))
    if len(set(calls)) > 1:
        listing = ", ".join(
            f"rank {r}: {n} (call {c})" for r, (n, c) in enumerate(calls)
        )
        raise RuntimeError(f"the ranks are in different collective calls: {listing}")
