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
import scope_profiler as sp

from mpiarray._exchange import Box
from mpiarray._mpi import check_collective, default_comm
from mpiarray._profiling import PROFILE_TAGS
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

    def _roles(self) -> tuple[dict[int, tuple], set[str]]:
        """Where this rank sends from and writes to, for every copy, as arrays per dimension.

        The copies are grouped by the number of axes of their boxes. For each
        group: the copy numbers, the source and target boxes, the lower corners
        and halos of the arrays' blocks (for the storage index) and, per copy,
        the block this rank sends from and the region it writes (with a flag
        where it does neither). Raises the error of the first bad copy, as the
        checks of one copy at a time would.
        """
        copies = self._copies
        arrays = {
            id(array): array
            for source, _, target, _ in copies
            for array in (source, target)
            if array is not None
        }
        dtypes = {np.dtype(array.dtype).str for array in arrays.values()}
        groups: dict[int, list[int]] = {}
        for number, (_, source_box, _, target_box) in enumerate(copies):
            if len(source_box) != len(target_box):
                _boxes(source_box, target_box)  # raises: they differ in shape
            groups.setdefault(len(source_box), []).append(number)
        layouts: dict[int, tuple] = {}

        def layout(array: DistributedArray) -> tuple:
            """Shape, halo, block and the block's sending and writing parts of an array."""
            found = layouts.get(id(array))
            if found is None:
                info = array.layout
                shape = tuple(info.shape)
                halo = tuple(info.halo)
                block = tuple(info.index_bounds)
                sends = info.distributed or info.rank == 0
                region = []
                for (start, end), n, h in zip(block, shape, halo, strict=True):
                    if not info.distributed:
                        start, end = 0, n
                    region.append(
                        (
                            start - h if start == 0 else start,
                            end + h if end == n else end,
                        )
                    )
                found = (shape, halo, block, sends, tuple(region))
                layouts[id(array)] = found
            return found

        roles = {}
        self._groups = []
        for ndim, numbers in groups.items():
            k = len(numbers)
            sources = np.array([copies[n][1] for n in numbers], dtype=np.int64)
            targets = np.array([copies[n][3] for n in numbers], dtype=np.int64)
            sources = sources.reshape(k, ndim, 2)
            targets = targets.reshape(k, ndim, 2)
            # per copy: the source's shape, block, halo; the target's
            zero = ((0, 0),) * ndim
            source_info = [
                None if (source := copies[n][0]) is None else layout(source)
                for n in numbers
            ]
            target_info = [
                None if (target := copies[n][2]) is None else layout(target)
                for n in numbers
            ]
            has_source = np.array([info is not None for info in source_info])
            has_target = np.array([info is not None for info in target_info])
            source_shape = np.array(
                [(0,) * ndim if i is None else i[0] for i in source_info],
                dtype=np.int64,
            ).reshape(k, ndim)
            target_shape = np.array(
                [(0,) * ndim if i is None else i[0] for i in target_info],
                dtype=np.int64,
            ).reshape(k, ndim)
            source_halo = np.array(
                [(0,) * ndim if i is None else i[1] for i in source_info],
                dtype=np.int64,
            ).reshape(k, ndim)
            target_halo = np.array(
                [(0,) * ndim if i is None else i[1] for i in target_info],
                dtype=np.int64,
            ).reshape(k, ndim)
            source_block = np.array(
                [zero if i is None else i[2] for i in source_info], dtype=np.int64
            ).reshape(k, ndim, 2)
            target_block = np.array(
                [zero if i is None else i[2] for i in target_info], dtype=np.int64
            ).reshape(k, ndim, 2)
            writes = np.array(
                [zero if i is None else i[4] for i in target_info], dtype=np.int64
            ).reshape(k, ndim, 2)
            sending = has_source & np.array(
                [i is not None and i[3] for i in source_info], dtype=bool
            )
            extent = sources[..., 1] - sources[..., 0]
            bad = np.any(extent != targets[..., 1] - targets[..., 0], axis=1)
            bad |= np.any(extent <= 0, axis=1)
            bad |= has_source & np.any(
                (sources[..., 0] < 0) | (sources[..., 1] > source_shape), axis=1
            )
            bad |= has_target & np.any(
                (targets[..., 0] < -target_halo)
                | (targets[..., 1] > target_shape + target_halo),
                axis=1,
            )
            if bad.any():  # the message of the first bad copy, checked alone
                source, source_box, target, target_box = copies[
                    numbers[int(np.argmax(bad))]
                ]
                source_box, target_box = _boxes(source_box, target_box)
                _sending_block(source, source_box)
                _receiving_region(target, target_box)
            self._groups.append(
                (
                    np.array(numbers),
                    sources,
                    targets,
                    source_block[..., 0] - source_halo,
                    target_block[..., 0] - target_halo,
                    has_source,
                    has_target,
                )
            )
            roles[ndim] = (source_block, sending, writes, has_target)
        return roles, dtypes

    def _plan(self, everyone: list) -> None:
        """What this rank sends to and receives from every rank, in the order of the copies."""
        rank = self._comm.Get_rank()
        sends: list[list[tuple]] = [[] for _ in range(self._size)]
        receives: list[list[tuple]] = [[] for _ in range(self._size)]

        def intersect(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lower = np.maximum(a[..., 0], b[..., 0])
            upper = np.minimum(a[..., 1], b[..., 1])
            return np.stack([lower, upper], axis=-1), np.all(lower < upper, axis=-1)

        def pieces(rows: np.ndarray) -> tuple[list[tuple[slice, ...]], list[int]]:
            """The storage index of every box, and its number of elements."""
            sizes = np.prod(rows[..., 1] - rows[..., 0], axis=1).tolist()
            return [tuple(slice(a, b) for a, b in row) for row in rows.tolist()], sizes

        for (
            numbers,
            sources,
            targets,
            source_origin,
            target_origin,
            has_source,
            has_target,
        ) in self._groups:
            ndim = sources.shape[1]
            shift = (targets[..., 0] - sources[..., 0])[..., None]
            mine = everyone[rank][0][ndim]
            # what this rank sends: its block of each source, to the ranks that write it
            outgoing, sent = intersect(mine[0], sources)
            sent &= mine[1] & has_source
            outgoing = outgoing + shift
            for other in range(self._size):
                theirs = everyone[other][0][ndim]
                part, ok = intersect(outgoing, theirs[2])
                ok &= sent & theirs[3]
                where = np.flatnonzero(ok)
                if where.size:
                    local = part[where] - shift[where] - source_origin[where][..., None]
                    indices, sizes = pieces(local)
                    for n, index, size in zip(
                        numbers[where].tolist(), indices, sizes, strict=True
                    ):
                        sends[other].append((n, self._copies[n][0], index, size))
            # what this rank receives: the parts of each target it writes, from the senders
            for other in range(self._size):
                theirs = everyone[other][0][ndim]
                incoming, ok = intersect(theirs[0], sources)
                ok &= theirs[1] & mine[3] & has_target
                part, inside = intersect(incoming + shift, mine[2])
                ok &= inside
                where = np.flatnonzero(ok)
                if where.size:
                    local = part[where] - target_origin[where][..., None]
                    indices, sizes = pieces(local)
                    for n, index, size in zip(
                        numbers[where].tolist(), indices, sizes, strict=True
                    ):
                        receives[other].append((n, self._copies[n][2], index, size))
        for found in (*sends, *receives):
            found.sort(key=lambda piece: piece[0])  # the order of the copies
        self._sends = [[(array, index) for _, array, index, _ in to] for to in sends]
        self._receives = [
            [(array, index, number) for number, array, index, _ in come]
            for come in receives
        ]
        self._send_counts = [sum(piece[3] for piece in to) for to in sends]
        self._receive_counts = [sum(piece[3] for piece in come) for come in receives]
        self._send_offsets = [sum(self._send_counts[:r]) for r in range(self._size)]
        self._receive_offsets = [
            sum(self._receive_counts[:r]) for r in range(self._size)
        ]
        # unpacked in the order of the copies, so that the last one wins
        order = []
        for other, come in enumerate(receives):
            offset = self._receive_offsets[other]
            for number, target, index, size in come:
                order.append((number, other, offset, target, index))
                offset += size
        order.sort(key=lambda piece: (piece[0], piece[1]))
        self._unpack = order

    def __init__(self, copies: Sequence[Copy], comm: Any = None) -> None:
        """Find where every box goes; collective (one ``allgather``)."""
        self._comm = default_comm() if comm is None else comm
        self._size = self._comm.Get_size()
        self._copies = list(copies)
        roles: dict[int, tuple] = {}
        dtypes: set[str] = set()
        error = None
        try:
            roles, dtypes = self._roles()
        except ValueError as problem:
            error = str(problem)  # raised on every rank, after the collective
        with sp.region("mpiarray.CopyPlan.plan", tags=PROFILE_TAGS):
            everyone = self._comm.allgather(
                (roles, len(self._copies), sorted(dtypes), error)
            )
            errors = [their_error for *_, their_error in everyone if their_error]
            if errors:
                raise ValueError(errors[0])
            if len({count for _, count, _, _ in everyone}) != 1:
                raise ValueError("the ranks were given different numbers of copies")
            all_dtypes = {
                dtype for _, _, their_dtypes, _ in everyone for dtype in their_dtypes
            }
            if len(all_dtypes) > 1:
                raise ValueError(
                    f"the arrays of a copy plan differ in dtype: {sorted(all_dtypes)}"
                )
            self._dtype = np.dtype(all_dtypes.pop()) if all_dtypes else np.dtype(float)
            self._plan(everyone)

    @property
    def num_copies(self) -> int:
        """The number of copies in the plan."""
        return len(self._copies)

    @property
    def num_sent(self) -> int:
        """Elements this rank sends per `execute`, to itself included."""
        return sum(self._send_counts)

    @sp.profile("mpiarray.CopyPlan.execute", tags=PROFILE_TAGS)
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
