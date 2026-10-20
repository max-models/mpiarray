"""Copy boxes between arrays that live on different communicators.

A `CopyPlan` moves boxes of data from source arrays into boxes of target
arrays, where every array may be split over its own group of ranks: one rank,
several, or all. The groups only have to be parts of one parent communicator,
over which the plan sends. This is the transport behind exchanges between
separate arrays, such as the halos of neighbouring patches of an adaptive
mesh, each patch an array on its own ranks.

::

    plan = mpa.CopyPlan(
        [(left, ((0, 4), (60, 64)), right, ((0, 4), (-4, 0)))],  # into right's halo
        comm=MPI.COMM_WORLD,
    )
    plan.execute()  # every step; one ``Alltoallv`` over the parent communicator

The plan is built once, with one ``allgather`` of where each array's blocks
are, and then executed as often as needed. A target box may reach into the
target's halo cells at its walls (the edges of its global index range); the
halo cells between the target's own blocks are its own business
(`DistributedArray.update_halos`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cunumpy as xp
import numpy as np

from mpiarray._exchange import Box, local_index, overlap
from mpiarray._mpi import check_collective, default_comm
from mpiarray.distributed_array import DistributedArray

#: One copy: the source array (``None`` on ranks that do not hold it), the box
#: of it to copy, the target array (``None`` likewise) and the box it goes to.
Copy = tuple[
    DistributedArray | None,
    Sequence[Sequence[int]],
    DistributedArray | None,
    Sequence[Sequence[int]],
]


def _boxes(
    source_box: Sequence[Sequence[int]], target_box: Sequence[Sequence[int]]
) -> tuple[Box, Box]:
    """Normalise the two boxes of a copy and check that their extents agree."""
    source = tuple((int(start), int(end)) for start, end in source_box)
    target = tuple((int(start), int(end)) for start, end in target_box)
    if len(source) != len(target) or any(
        s1 - s0 != t1 - t0 for (s0, s1), (t0, t1) in zip(source, target, strict=False)
    ):
        raise ValueError(f"the boxes {source} and {target} differ in shape")
    if any(end <= start for start, end in source):
        raise ValueError(f"the box {source} is empty")
    return source, target


def _sending_block(array: DistributedArray | None, box: Box) -> Box | None:
    """The part of the source this rank sends from: its block, once per element.

    A replicated array (every rank holding all of it) sends from its first
    rank only.
    """
    if array is None:
        return None
    layout = array.layout
    if len(box) != len(layout.shape) or any(
        start < 0 or end > n for (start, end), n in zip(box, layout.shape, strict=True)
    ):
        raise ValueError(
            f"the source box {box} is not inside the array's shape {layout.shape}"
        )
    if not layout.distributed and layout.rank != 0:
        return None
    return layout.index_bounds


def _receiving_region(array: DistributedArray | None, box: Box) -> Box | None:
    """The part of the target's index space this rank writes.

    Its block, extended by the halo width where the block is at a wall, so that
    every element, halo cells at the walls included, has one rank that writes
    it. Every rank of a replicated array writes all of it.
    """
    if array is None:
        return None
    layout = array.layout
    halo = layout.halo
    if len(box) != len(layout.shape) or any(
        start < -h or end > n + h
        for (start, end), n, h in zip(box, layout.shape, halo, strict=True)
    ):
        raise ValueError(
            f"the target box {box} is not inside the array's shape {layout.shape} "
            f"and its halo {halo}"
        )
    region = []
    for (start, end), n, h in zip(layout.index_bounds, layout.shape, halo, strict=True):
        if not layout.distributed:
            start, end = 0, n
        region.append(
            (start - h if start == 0 else start, end + h if end == n else end)
        )
    return tuple(region)


class CopyPlan:
    """Copy boxes between arrays on parts of one communicator, again and again.

    Collective over ``comm`` (default: ``MPI.COMM_WORLD``): every rank of it
    builds the plan with the same copies in the same order. A copy is
    ``(source, source_box, target, target_box)``:

    - ``source`` and ``target`` are `DistributedArray` objects on the ranks
      that hold them and ``None`` on the others. Their communicators must be
      made of ranks of ``comm`` (e.g. by ``comm.Split``); they may differ from
      copy to copy, and a copy may go from an array to itself.
    - The boxes are ``(start, end)`` per axis in the global indices of each
      array, of equal extents. The source box lies inside the source array;
      the target box may reach into the target's halo at its walls.

    All arrays of a plan have one dtype. Where the target boxes of several
    copies overlap, the last copy wins. The sources are read before any target
    is written.

    Args:
        copies: The copies.
        comm: The parent communicator.

    Raises:
        ValueError: On every rank, if the ranks were given different numbers of
            copies or arrays of different dtypes, or if a rank found a box
            outside its array, an empty box or boxes of different shapes.
    """

    def __init__(self, copies: Sequence[Copy], comm: Any = None) -> None:
        """Find where every box goes; collective (one ``allgather``)."""
        self._comm = default_comm() if comm is None else comm
        self._size = self._comm.Get_size()
        self._copies = []
        roles = []
        dtypes = set()
        error = None
        try:
            for source, source_box, target, target_box in copies:
                source_box, target_box = _boxes(source_box, target_box)
                send = _sending_block(source, source_box)
                receive = _receiving_region(target, target_box)
                for array in (source, target):
                    if array is not None:
                        dtypes.add(np.dtype(array.dtype).str)
                self._copies.append((source, source_box, target, target_box))
                roles.append((send, receive))
        except ValueError as problem:
            error = str(problem)  # raised on every rank, after the collective
        everyone = self._comm.allgather((roles, sorted(dtypes), error))
        errors = [their_error for _, _, their_error in everyone if their_error]
        if errors:
            raise ValueError(errors[0])
        if len({len(their_roles) for their_roles, _, _ in everyone}) != 1:
            raise ValueError("the ranks were given different numbers of copies")
        all_dtypes = {
            dtype for _, their_dtypes, _ in everyone for dtype in their_dtypes
        }
        if len(all_dtypes) > 1:
            raise ValueError(
                f"the arrays of a copy plan differ in dtype: {sorted(all_dtypes)}"
            )
        self._dtype = np.dtype(all_dtypes.pop()) if all_dtypes else np.dtype(float)
        self._plan(everyone)

    def _plan(self, everyone: list) -> None:
        """What this rank sends to and receives from every rank, in order."""
        rank = self._comm.Get_rank()
        sends: list[list[tuple[DistributedArray, tuple[slice, ...]]]] = [
            [] for _ in range(self._size)
        ]
        receives: list[list[tuple[DistributedArray, tuple[slice, ...], int]]] = [
            [] for _ in range(self._size)
        ]
        for number, (source, source_box, target, target_box) in enumerate(self._copies):
            shift = [
                t0 - s0 for (s0, _), (t0, _) in zip(source_box, target_box, strict=True)
            ]

            def moved(
                block: Box | None, shift: list[int] = shift, box: Box = source_box
            ) -> Box | None:
                """The part of the source box in ``block``, in the target's indices."""
                part = None if block is None else overlap(block, box)
                if part is None:
                    return None
                return tuple(
                    (s + d, e + d) for (s, e), d in zip(part, shift, strict=True)
                )

            my_send, my_receive = everyone[rank][0][number]
            outgoing = moved(my_send)
            # a rank sends from and receives into the arrays it holds
            if outgoing is not None and source is not None:
                for other in range(self._size):
                    region = everyone[other][0][number][1]
                    part = None if region is None else overlap(outgoing, region)
                    if part is None:
                        continue
                    back = tuple(
                        (s - d, e - d) for (s, e), d in zip(part, shift, strict=True)
                    )
                    layout = source.layout
                    sends[other].append(
                        (source, local_index(back, layout.index_bounds, layout.halo))
                    )
            if my_receive is not None and target is not None:
                layout = target.layout
                for other in range(self._size):
                    incoming = moved(everyone[other][0][number][0])
                    part = None if incoming is None else overlap(incoming, my_receive)
                    if part is None:
                        continue
                    receives[other].append(
                        (
                            target,
                            local_index(part, layout.index_bounds, layout.halo),
                            number,
                        )
                    )
        self._sends = sends
        self._receives = receives

        def counts(pieces: list) -> list[int]:
            return [
                sum(math.prod(s.stop - s.start for s in piece[1]) for piece in parts)
                for parts in pieces
            ]

        self._send_counts = counts(sends)
        self._receive_counts = counts(receives)
        self._send_offsets = [sum(self._send_counts[:r]) for r in range(self._size)]
        self._receive_offsets = [
            sum(self._receive_counts[:r]) for r in range(self._size)
        ]
        # unpacked in the order of the copies, so that the last one wins
        order = []
        for other, parts in enumerate(receives):
            offset = self._receive_offsets[other]
            for target, index, number in parts:
                count = math.prod(s.stop - s.start for s in index)
                order.append((number, other, offset, target, index))
                offset += count
        order.sort(key=lambda piece: (piece[0], piece[1]))
        self._unpack = order

    @property
    def num_copies(self) -> int:
        """The number of copies in the plan."""
        return len(self._copies)

    @property
    def num_sent(self) -> int:
        """Elements this rank sends per `execute`, to itself included."""
        return sum(self._send_counts)

    def execute(self) -> None:
        """Copy the boxes. Collective over the plan's communicator."""
        parts = [
            xp.ravel(source.local_with_halos[index])
            for to_rank in self._sends
            for source, index in to_rank
        ]
        dtype = self._dtype
        send = (
            xp.concatenate(parts).astype(dtype, copy=False)
            if parts
            else xp.empty(0, dtype=dtype)
        )
        if self._size == 1:
            # everything goes to this rank, in the order it is unpacked
            receive = send
        else:
            receive = xp.empty(sum(self._receive_counts), dtype=dtype)
            check_collective(self._comm, "copy plan")
            receiving = xp.mpi.mpi_buffer(receive, send=False, recv=True)
            with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
                self._comm.Alltoallv(
                    [sendbuf, (self._send_counts, self._send_offsets)],
                    [recvbuf, (self._receive_counts, self._receive_offsets)],
                )
        for _, _, offset, target, index in self._unpack:
            shape = tuple(s.stop - s.start for s in index)
            target.local_with_halos[index] = receive[
                offset : offset + math.prod(shape)
            ].reshape(shape)


def copy_boxes(copies: Sequence[Copy], comm: Any = None) -> None:
    """Copy boxes between arrays on parts of ``comm`` once; see `CopyPlan`.

    Collective over ``comm``.
    """
    CopyPlan(copies, comm).execute()
