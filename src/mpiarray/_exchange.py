"""Send boxes of a global index space to the ranks whose blocks they overlap.

The one primitive behind redistribution, distributed reductions along axes
and distributed selections: every rank holds data for one box of a target
array's global index space (or none), and receives, from every rank, the part
of that rank's box that falls into its own block of the target layout. One
``Alltoallv``; the sizes are known on every rank from the boxes, so only the
data travels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cunumpy as xp

from mpiarray._mpi import MPI, check_collective
from mpiarray.layout import Layout

#: A box of global indices: one ``(start, end)`` per axis.
Box = tuple[tuple[int, int], ...]


def overlap(a: Box, b: Box) -> Box | None:
    """Return the intersection of two boxes, or ``None`` if it is empty."""
    box = tuple(
        (max(a0, b0), min(a1, b1)) for (a0, a1), (b0, b1) in zip(a, b, strict=True)
    )
    return box if all(start < end for start, end in box) else None


def exchange_boxes(
    target: Layout,
    data: Any,
    box: Box | None,
    boxes: Sequence[Box | None],
    lead: tuple[int, ...] = (),
    name: str = "exchange",
    comm: Any = None,
) -> list[tuple[int, Box, Any]]:
    """Send the parts of every rank's box to the ranks of ``target`` that own them.

    Collective over ``target.comm`` (several ranks).

    Args:
        target: The layout whose blocks receive the data.
        data: This rank's data, of shape ``lead + extents of box``.
        box: The box ``data`` covers, in the target's global indices; ``None``
            if this rank sends nothing.
        boxes: Every rank's box (the same list on every rank).
        lead: Leading axes of ``data`` that travel along unchanged, e.g.
            several channels per element.
        name: What is being done, for the debug checks.
        comm: The communicator to send over, with ``boxes`` numbered by its
            ranks; default: ``target.comm``. It must hold the same processes
            as ``target.comm``, possibly numbered differently (e.g. a
            Cartesian communicator made with ``reorder=True``).

    Returns:
        ``(source rank, box, piece)`` for every non-empty part this rank
        receives, where ``box`` is the part in global indices and ``piece``
        its data, of shape ``lead + extents of box``.
    """
    comm = target.comm if comm is None else comm
    size = target.size
    ranks = target_ranks(comm, target.comm)
    channels = math.prod(lead)
    send_parts, send_counts = [], []
    for r in range(size):
        part = None if box is None else overlap(box, target.index_bounds_of(ranks[r]))
        if box is None or part is None:
            send_counts.append(0)
            continue
        index = (slice(None),) * len(lead) + tuple(
            slice(p0 - b0, p1 - b0) for (p0, p1), (b0, _) in zip(part, box, strict=True)
        )
        piece = xp.ascontiguousarray(data[index])
        send_parts.append(piece.reshape(-1))
        send_counts.append(int(piece.size))
    mine = target.index_bounds
    receive_parts: list[tuple[int, Box]] = []
    receive_counts = []
    for r, source in enumerate(boxes):
        part = None if source is None else overlap(source, mine)
        receive_counts.append(
            0 if part is None else channels * math.prod(e - s for s, e in part)
        )
        if part is not None:
            receive_parts.append((r, part))

    dtype = data.dtype
    send = xp.concatenate(send_parts) if send_parts else xp.empty(0, dtype=dtype)
    recv = xp.empty(sum(receive_counts), dtype=dtype)
    send_offsets = [sum(send_counts[:r]) for r in range(size)]
    receive_offsets = [sum(receive_counts[:r]) for r in range(size)]
    check_collective(comm, name)
    receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
    with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
        comm.Alltoallv(
            [sendbuf, (send_counts, send_offsets)],
            [recvbuf, (receive_counts, receive_offsets)],
        )
    pieces = []
    for r, part in receive_parts:
        count = receive_counts[r]
        shape = lead + tuple(end - start for start, end in part)
        offset = receive_offsets[r]
        pieces.append((r, part, recv[offset : offset + count].reshape(shape)))
    return pieces


def target_ranks(comm: Any, other: Any) -> list[int]:
    """Return, for each rank of ``comm``, its rank in ``other`` (the same processes).

    Raises:
        ValueError: If the communicators do not hold the same processes.
    """
    size = comm.Get_size()
    if comm is other or bool(comm == other) or size == 1:
        return list(range(size))
    if other.Get_size() == size:
        translated = MPI.Group.Translate_ranks(
            comm.Get_group(), list(range(size)), other.Get_group()
        )
        if MPI.UNDEFINED not in translated:
            return [int(r) for r in translated]
    raise ValueError("the communicators do not hold the same processes")


def local_index(
    box: Box, block: Box, halo: Sequence[int] | None = None
) -> tuple[slice, ...]:
    """Return the index of ``box`` in a rank's storage for ``block`` (with ``halo``)."""
    halo = (0,) * len(box) if halo is None else halo
    return tuple(
        slice(start - b0 + h, end - b0 + h)
        for (start, end), (b0, _), h in zip(box, block, halo, strict=True)
    )
