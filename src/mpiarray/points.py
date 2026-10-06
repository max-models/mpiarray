"""Moving rows of data between ranks: particles, points, any per-item records.

`migrate` sends every row of some arrays to a destination rank, for example
each particle to the rank that owns its grid cell (see `Layout.owners`).
The arrays are sent as raw buffers (one ``Alltoallv`` each), on NumPy or
CuPy, without pickling.
"""

from __future__ import annotations

import math
from typing import Any

import cunumpy as xp
import numpy as np

from mpiarray._mpi import Comm, check_collective, default_comm


def migrate(
    destinations: Any, *arrays: Any, comm: Comm | None = None
) -> tuple[Any, ...]:
    """Send row ``i`` of every array to rank ``destinations[i]``.

    Collective. All arrays have one row per destination (their first axis);
    the other axes and the dtypes may differ between arrays but must agree
    across ranks. On return, every rank holds the rows sent to it, ordered by
    source rank and, within a source, in their original order.

    Args:
        destinations: The rank of each row, an integer array (NumPy or CuPy).
        *arrays: The arrays whose rows move together, NumPy or CuPy (or
            anything ``asarray`` takes); moved to the backend's device.
        comm: The communicator; default: ``MPI.COMM_WORLD`` (the serial
            stand-in without an MPI launcher).

    Returns:
        The received arrays, one per input array, in the same order, as the
        backend's arrays.

    Raises:
        ValueError: If a destination is not a rank, or an array does not have
            one row per destination.
    """
    comm = default_comm() if comm is None else comm
    size = comm.Get_size()
    destinations = xp.asarray(destinations)
    arrays = tuple(xp.asarray(array) for array in arrays)  # on the backend's device
    count = int(destinations.shape[0]) if destinations.ndim else 0
    if destinations.ndim != 1:
        raise ValueError("destinations must be a 1-D array of ranks")
    for array in arrays:
        if array.shape[:1] != (count,):
            raise ValueError(
                f"an array of shape {tuple(array.shape)} does not have one row per "
                f"destination ({count})",
            )
    host = xp.to_numpy(destinations).astype(np.int64)
    if bool((host < 0).any() or (host >= size).any()):
        raise ValueError(f"destinations must be ranks between 0 and {size - 1}")
    if size == 1:
        return tuple(array.copy() for array in arrays)

    order = np.argsort(host, kind="stable")
    send_rows = np.bincount(host, minlength=size).tolist()
    check_collective(comm, "migrate")
    receive_rows = comm.alltoall(send_rows)
    device_order = xp.asarray(order)
    received = []
    for array in arrays:
        row = math.prod(array.shape[1:])
        send = xp.ascontiguousarray(array[device_order]).reshape(-1)
        recv = xp.empty(sum(receive_rows) * row, dtype=array.dtype)
        send_counts = [n * row for n in send_rows]
        receive_counts = [n * row for n in receive_rows]
        send_offsets = [sum(send_counts[:r]) for r in range(size)]
        receive_offsets = [sum(receive_counts[:r]) for r in range(size)]
        receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
        with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
            comm.Alltoallv(
                [sendbuf, (send_counts, send_offsets)],
                [recvbuf, (receive_counts, receive_offsets)],
            )
        received.append(recv.reshape(sum(receive_rows), *array.shape[1:]))
    return tuple(received)
