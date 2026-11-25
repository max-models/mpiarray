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


def _elements(array: DistributedArray) -> Any:
    """The storage of an array, halos included, as one flat view."""
    storage = array.local_with_halos
    if not storage.flags.c_contiguous:
        raise ValueError("a copy plan needs arrays with C-contiguous storage")
    return storage.reshape(-1)


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

        # every copy's source and target as an index into the arrays (-1: none here)
        order: dict[int, int] = {}
        known: list[DistributedArray] = []

        def number_of(array: DistributedArray | None) -> int:
            if array is None:
                return -1
            found = order.get(id(array))
            if found is None:
                found = order[id(array)] = len(known)
                known.append(array)
            return found

        source_of = np.array([number_of(c[0]) for c in copies], dtype=np.int64)
        target_of = np.array([number_of(c[2]) for c in copies], dtype=np.int64)
        infos = [layout(array) for array in known]

        def table(ndim: int, item: int, empty: Any) -> np.ndarray:
            """One item of every array's layout, a last row for none (index -1)."""
            rows = [info[item] if len(info[0]) == ndim else empty for info in infos] + [
                empty
            ]
            return np.array(rows, dtype=np.int64)

        roles = {}
        self._groups = []
        for ndim, numbers in groups.items():
            k = len(numbers)
            sources = np.array([copies[n][1] for n in numbers], dtype=np.int64)
            targets = np.array([copies[n][3] for n in numbers], dtype=np.int64)
            sources = sources.reshape(k, ndim, 2)
            targets = targets.reshape(k, ndim, 2)
            # per copy: the source's shape, block, halo; the target's
            zero, none = ((0, 0),) * ndim, (0,) * ndim
            shapes, halos = table(ndim, 0, none), table(ndim, 1, none)
            blocks, regions = table(ndim, 2, zero), table(ndim, 4, zero)
            sends = np.array([bool(info[3]) for info in infos] + [False])
            fits = np.array([len(info[0]) == ndim for info in infos] + [True])
            source = source_of[np.asarray(numbers)]
            target = target_of[np.asarray(numbers)]
            has_source, has_target = source >= 0, target >= 0
            source_shape, target_shape = shapes[source], shapes[target]
            source_halo, target_halo = halos[source], halos[target]
            source_block, target_block = blocks[source], blocks[target]
            writes = regions[target]
            sending = has_source & sends[source]
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
            bad |= ~fits[source] | ~fits[target]  # an array of other dimensions
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

    def _layout(
        self, parts: list[tuple], role: int, width: int, last: bool
    ) -> tuple[list[tuple], list[int]]:
        """The flat indices and buffer places per array of pieces, and the counts per rank.

        A buffer holds the pieces for rank 0, then rank 1, ...; for each rank in
        the order of the copies, the elements of a box in C order. ``role`` picks
        the source (0) or the target (2) of a copy. With ``last``, an element that
        several pieces write keeps only the last piece in the order of the
        copies, so that the writes are one assignment and the last copy wins.
        """
        counts = [0] * self._size
        if not parts:
            return [], counts
        numbers = np.concatenate([n for n, _, _ in parts])
        others = np.concatenate([np.full(n.size, o) for n, o, _ in parts])
        boxes = np.concatenate([b for _, _, b in parts])
        # the arrays, and the strides of their storage (padded like the boxes)
        index: dict[int, int] = {}
        arrays: list[DistributedArray] = []
        owner = np.empty(numbers.size, dtype=np.int64)
        for k, n in enumerate(numbers.tolist()):
            array = self._copies[n][role]
            found = index.get(id(array))
            if found is None:
                found = index[id(array)] = len(arrays)
                arrays.append(array)
            owner[k] = found
        strides = np.zeros((len(arrays), width), dtype=np.int64)
        for a, array in enumerate(arrays):
            shape = array.local_with_halos.shape
            strides[a, width - len(shape) :] = np.cumprod(
                (tuple(shape[1:]) + (1,))[::-1]
            )[::-1]
        extent = boxes[..., 1] - boxes[..., 0]
        sizes = np.prod(extent, axis=1)
        # buffer order: by rank, then by copy
        order = np.lexsort((numbers, others))
        offsets = np.empty(numbers.size, dtype=np.int64)
        offsets[order] = np.cumsum(sizes[order]) - sizes[order]
        for other, size in zip(others.tolist(), sizes.tolist(), strict=True):
            counts[other] += size
        # every element: its piece, its place in the buffer, its flat index
        if last:  # written in the order of the copies (then of the ranks)
            pieces = np.lexsort((others, numbers))
        else:
            pieces = order
        piece = np.repeat(pieces, sizes[pieces])
        start = np.repeat(np.cumsum(sizes[pieces]) - sizes[pieces], sizes[pieces])
        within = np.arange(piece.size) - start
        at = offsets[piece] + within
        flat = np.zeros(piece.size, dtype=np.int64)
        rest = within
        for axis in range(width - 1, -1, -1):
            span = extent[piece, axis]
            flat += (boxes[piece, axis, 0] + rest % span) * strides[owner[piece], axis]
            rest = rest // span
        target = owner[piece]
        if last:  # the last write of every element of every array
            key = target * (int(flat.max()) + 1) + flat
            _, keep = np.unique(key[::-1], return_index=True)
            keep = np.sort(key.size - 1 - keep)
            flat, at, target = flat[keep], at[keep], target[keep]
        grouped = np.argsort(target, kind="stable")
        bounds = np.searchsorted(target[grouped], np.arange(len(arrays) + 1))
        result = []
        for a, array in enumerate(arrays):
            chosen = grouped[bounds[a] : bounds[a + 1]]
            if chosen.size:
                result.append((array, xp.asarray(flat[chosen]), xp.asarray(at[chosen])))
        return result, counts

    def _plan(self, everyone: list) -> None:
        """What this rank sends to and receives from every rank, in the order of the copies.

        The pieces (a copy's part between two ranks) are rows of arrays: copy
        number, the other rank, the box in the storage of the array; their
        elements become flat storage indices and places in the buffers, per array.
        """
        rank = self._comm.Get_rank()
        width = max((group[1].shape[1] for group in self._groups), default=1)
        sends: list[tuple] = []  # (copy numbers, other rank, storage boxes)
        receives: list[tuple] = []

        def intersect(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            lower = np.maximum(a[..., 0], b[..., 0])
            upper = np.minimum(a[..., 1], b[..., 1])
            return np.stack([lower, upper], axis=-1), np.all(lower < upper, axis=-1)

        def padded(boxes: np.ndarray) -> np.ndarray:
            """Boxes with ``(0, 1)`` on the axes their arrays do not have."""
            extra = width - boxes.shape[1]
            if not extra:
                return boxes
            pad = np.zeros((boxes.shape[0], extra, 2), dtype=np.int64)
            pad[..., 1] = 1
            return np.concatenate([pad, boxes], axis=1)

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
                    sends.append((numbers[where], other, padded(local)))
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
                    receives.append((numbers[where], other, padded(local)))
        self._pack, self._send_counts = self._layout(sends, 0, width, last=False)
        self._place, self._receive_counts = self._layout(receives, 2, width, last=True)
        self._send_offsets = [sum(self._send_counts[:r]) for r in range(self._size)]
        self._receive_offsets = [
            sum(self._receive_counts[:r]) for r in range(self._size)
        ]

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
        dtype = self._dtype
        send = xp.empty(sum(self._send_counts), dtype=dtype)
        for source, flat, at in self._pack:
            send[at] = _elements(source)[flat]
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
        for target, flat, at in self._place:
            _elements(target)[flat] = receive[at]


def copy_boxes(copies: Sequence[Copy], comm: Any = None) -> None:
    """Copy boxes between arrays on parts of ``comm`` once; see `CopyPlan`.

    Collective over ``comm``.
    """
    CopyPlan(copies, comm).execute()
