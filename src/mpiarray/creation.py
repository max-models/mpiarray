"""Functions that create distributed arrays, like their NumPy namesakes.

Every function takes the same keyword arguments for the layout:

- ``split``: the axis or axes split over the ranks (default: axis ``0``);
  ``None`` makes every rank hold the whole array. A split axis needs at least
  as many elements as ranks, and a halo must fit in the smallest block, or
  `Layout` raises.
- ``halo``: the halo width, for every axis or one per axis (default ``0``).
- ``periodic``: whether each axis wraps around in halo updates.
- ``comm``: the communicator (default: ``MPI.COMM_WORLD``, or maybempi's serial
  stand-in when not started by an MPI launcher).
- ``process_grid``: explicit process counts per axis, instead of ``split``.
- ``layout``: an existing `Layout`, instead of all of the above.

All of them are collective in the sense that every rank must call them with
the same arguments, but only `array` with ``MPIARRAY_DEBUG=1`` communicates.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import cunumpy as xp
import numpy as np
from numpy.typing import DTypeLike

from mpiarray._mpi import MPI, Comm, check_collective, debug_checks, default_comm
from mpiarray.distributed_array import Array, DistributedArray
from mpiarray.layout import (
    DEFAULT,
    HaloLike,
    Layout,
    PeriodicLike,
    ShapeLike,
    SplitArg,
    _Default,
)

# `array` and `asarray` tell "not given" from a value, to keep a distributed
# input's layout unless an option is given.
_HaloArg = HaloLike | Literal[_Default.DEFAULT]
_PeriodicArg = PeriodicLike | Literal[_Default.DEFAULT]


def _make_layout(
    shape: ShapeLike | None,
    layout: Layout | None,
    split: SplitArg,
    halo: HaloLike,
    periodic: PeriodicLike,
    comm: Comm | None,
    process_grid: Sequence[int] | None,
) -> Layout:
    """Return ``layout``, checked against ``shape``, or a new one."""
    if layout is not None:
        if shape is not None and Layout(shape, comm=layout.comm).shape != layout.shape:
            raise ValueError(
                f"shape {shape} does not match the layout's {layout.shape}"
            )
        return layout
    if shape is None:
        raise TypeError("a shape or a layout is required")
    return Layout(
        shape,
        comm=comm,
        split=split,
        halo=halo,
        periodic=periodic,
        process_grid=process_grid,
    )


def empty(
    shape: ShapeLike | None = None,
    dtype: DTypeLike = float,
    *,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return a new array without initializing its values.

    Args:
        shape: The global shape; may be left out when ``layout`` is given.
        dtype: The element type.
        split: The split axis or axes; see the module docstring.
        halo: The halo width.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.

    Returns:
        The array; its values, halo cells included, are undefined.
    """
    layout = _make_layout(shape, layout, split, halo, periodic, comm, process_grid)
    return DistributedArray(layout, xp.empty(layout.storage_shape, dtype=dtype))


def zeros(
    shape: ShapeLike | None = None,
    dtype: DTypeLike = float,
    *,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return a new array of zeros.

    Args:
        shape: The global shape; may be left out when ``layout`` is given.
        dtype: The element type.
        split: The split axis or axes; see the module docstring.
        halo: The halo width.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.

    Returns:
        The array, halo cells included.
    """
    layout = _make_layout(shape, layout, split, halo, periodic, comm, process_grid)
    return DistributedArray(layout, xp.zeros(layout.storage_shape, dtype=dtype))


def ones(
    shape: ShapeLike | None = None,
    dtype: DTypeLike = float,
    *,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return a new array of ones.

    Args:
        shape: The global shape; may be left out when ``layout`` is given.
        dtype: The element type.
        split: The split axis or axes; see the module docstring.
        halo: The halo width.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.

    Returns:
        The array, halo cells included.
    """
    layout = _make_layout(shape, layout, split, halo, periodic, comm, process_grid)
    return DistributedArray(layout, xp.ones(layout.storage_shape, dtype=dtype))


def full(
    shape: ShapeLike | None,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    *,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return a new array filled with ``fill_value``.

    Args:
        shape: The global shape; may be ``None`` when ``layout`` is given.
        fill_value: The value of every element, halo cells included.
        dtype: The element type; default: that of ``fill_value``.
        split: The split axis or axes; see the module docstring.
        halo: The halo width.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.

    Returns:
        The array.
    """
    layout = _make_layout(shape, layout, split, halo, periodic, comm, process_grid)
    storage = xp.full(layout.storage_shape, fill_value, dtype=dtype)
    return DistributedArray(layout, storage)


def empty_like(a: DistributedArray, dtype: DTypeLike | None = None) -> DistributedArray:
    """Return a new array with the layout of ``a``, without initializing it.

    Args:
        a: The array whose layout to use.
        dtype: The element type; default: that of ``a``.
    """
    return empty(dtype=a.dtype if dtype is None else dtype, layout=a.layout)


def zeros_like(a: DistributedArray, dtype: DTypeLike | None = None) -> DistributedArray:
    """Return a new array of zeros with the layout of ``a``.

    Args:
        a: The array whose layout to use.
        dtype: The element type; default: that of ``a``.
    """
    return zeros(dtype=a.dtype if dtype is None else dtype, layout=a.layout)


def ones_like(a: DistributedArray, dtype: DTypeLike | None = None) -> DistributedArray:
    """Return a new array of ones with the layout of ``a``.

    Args:
        a: The array whose layout to use.
        dtype: The element type; default: that of ``a``.
    """
    return ones(dtype=a.dtype if dtype is None else dtype, layout=a.layout)


def full_like(
    a: DistributedArray, fill_value: Any, dtype: DTypeLike | None = None
) -> DistributedArray:
    """Return a new array filled with ``fill_value``, with the layout of ``a``.

    Args:
        a: The array whose layout to use.
        fill_value: The value of every element, halo cells included.
        dtype: The element type; default: that of ``a``.
    """
    return full(
        None, fill_value, dtype=a.dtype if dtype is None else dtype, layout=a.layout
    )


def _check_identical(data: Array, comm: Comm) -> None:
    """Raise on every rank unless every rank passed the same global data."""
    digest = hashlib.sha256(f"{data.shape} {data.dtype.str}".encode())
    digest.update(np.ascontiguousarray(xp.to_numpy(data)).tobytes())
    digests = comm.allgather(digest.hexdigest())
    different = [rank for rank, d in enumerate(digests) if d != digests[0]]
    if different:
        raise ValueError(
            f"mpiarray.array got different data on ranks {different} than on rank 0; "
            "every rank must pass the same global array",
        )


def _layout_for(
    source: DistributedArray,
    split: SplitArg,
    halo: _HaloArg,
    periodic: _PeriodicArg,
    comm: Comm | None,
    process_grid: Sequence[int] | None,
) -> Layout | None:
    """Return ``source``'s layout with the given options changed, or None if none is."""
    if (
        split is DEFAULT
        and halo is DEFAULT
        and periodic is DEFAULT
        and comm is None
        and process_grid is None
    ):
        return None
    old = source.layout
    if split is DEFAULT and process_grid is None:
        split = old.split if old.split else None
    return Layout(
        old.shape,
        comm=old.comm if comm is None else comm,
        split=split,
        halo=old.halo if halo is DEFAULT else halo,
        periodic=old.periodic if periodic is DEFAULT else periodic,
        process_grid=process_grid,
    )


def array(
    obj: Any,
    dtype: DTypeLike | None = None,
    *,
    split: SplitArg = DEFAULT,
    halo: _HaloArg = DEFAULT,
    periodic: _PeriodicArg = DEFAULT,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return a distributed array with the values of a global array.

    Every rank passes the same global array (a list, a NumPy or CuPy array, a
    scalar) and keeps the block it owns, so each rank briefly holds all of
    it; for large arrays, compute the blocks locally with `fromfunction`,
    `arange` or `linspace`, or write ``a.local`` of an `empty` array. With
    ``MPIARRAY_DEBUG=1`` the ranks check, collectively, that their data is the
    same.

    A `DistributedArray` is copied, keeping its layout except for the options
    given; if they change the layout, the array is redistributed, which
    gathers it (collective).

    Args:
        obj: The global array, or a distributed array.
        dtype: The element type; default: that of ``obj``.
        split: The split axis or axes; see the module docstring.
        halo: The halo width (default 0); the halo cells start at zero.
        periodic: Whether each axis wraps around (default ``False``).
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.

    Returns:
        The array.

    Raises:
        ValueError: With ``MPIARRAY_DEBUG=1``, on every rank, if the ranks
            passed different data.
    """
    if isinstance(obj, DistributedArray):
        if layout is None:
            layout = _layout_for(obj, split, halo, periodic, comm, process_grid)
        if layout is None or layout == obj.layout:
            return obj.astype(obj.dtype if dtype is None else dtype)
        obj = obj.gather()
    data = xp.asarray(obj, dtype=dtype)
    layout = _make_layout(
        data.shape,
        layout,
        split,
        0 if halo is DEFAULT else halo,
        False if periodic is DEFAULT else periodic,
        comm,
        process_grid,
    )
    if debug_checks():
        _check_identical(data, layout.comm)
    storage = xp.zeros(layout.storage_shape, dtype=data.dtype)
    storage[layout.interior] = data[layout.global_slices()]
    return DistributedArray(layout, storage)


def asarray(
    obj: Any,
    dtype: DTypeLike | None = None,
    *,
    split: SplitArg = DEFAULT,
    halo: _HaloArg = DEFAULT,
    periodic: _PeriodicArg = DEFAULT,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return ``obj`` as a distributed array, without a copy if it already is one.

    A `DistributedArray` is returned as it is if it already has the requested
    ``dtype`` and layout options; anything else goes through `array`.

    Args:
        obj: A distributed array, or a global array.
        dtype: The element type; default: that of ``obj``.
        split: The split axis or axes; see the module docstring.
        halo: The halo width.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, instead of the arguments above.
    """
    if isinstance(obj, DistributedArray) and (
        dtype is None or xp.dtype(dtype) == obj.dtype
    ):
        wanted = layout or _layout_for(obj, split, halo, periodic, comm, process_grid)
        if wanted is None or wanted == obj.layout:
            return obj
    return array(
        obj,
        dtype,
        split=split,
        halo=halo,
        periodic=periodic,
        comm=comm,
        process_grid=process_grid,
        layout=layout,
    )


def from_local(
    block: Any,
    *,
    split: int | None = 0,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    layout: Layout | None = None,
    with_halos: bool = False,
) -> DistributedArray:
    """Return a distributed array made of the blocks the ranks already hold.

    The inverse of ``a.local``: each rank passes its own piece, and nothing
    global is ever built. Without ``layout``, the pieces are stacked in rank
    order along the ``split`` axis (their other extents must agree), the
    global shape is the result, and pieces whose lengths differ from the
    near-even split are redistributed (one ``Alltoallv``). With ``split=None``
    every rank passes the whole array.

    Collective.

    Args:
        block: This rank's piece, a NumPy or CuPy array (or anything
            ``asarray`` takes).
        split: The axis along which the pieces are stacked, or ``None``.
        halo: The halo width of the result; the halo cells start at zero.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        layout: An existing layout; then ``block`` must have its local shape
            (its storage shape with ``with_halos``) and is used as it is.
        with_halos: Whether ``block`` includes the halo cells (only with
            ``layout``).

    Returns:
        The array; the pieces are copied, except with ``layout``.

    Raises:
        ValueError: On every rank, if the pieces do not fit together or do not
            match ``layout``.
    """
    block = xp.asarray(block)
    if layout is not None:
        expected = layout.storage_shape if with_halos else layout.local_shape
        fits = tuple(block.shape) == expected
        if layout.distributed:
            check_collective(layout.comm, "from_local")
            fits = bool(layout.comm.allreduce(fits, op=MPI.LAND))
        if not fits:
            raise ValueError(
                f"the blocks do not match the layout's {'storage' if with_halos else 'local'} "
                f"shapes (rank {layout.rank}: {tuple(block.shape)}, expected {expected})",
            )
        if with_halos:
            return DistributedArray(layout, block)
        storage = xp.zeros(layout.storage_shape, dtype=block.dtype)
        storage[layout.interior] = block
        return DistributedArray(layout, storage)
    if with_halos:
        raise ValueError("with_halos=True needs a layout")

    comm = default_comm() if comm is None else comm
    size = comm.Get_size()
    if split is not None and not -block.ndim <= split < block.ndim:
        raise ValueError(
            f"the blocks cannot be stacked along axis {split}: they have "
            f"{block.ndim} dimensions",
        )
    if split is None or size == 1:
        layout = Layout(
            block.shape,
            comm=comm,
            split=None if split is None else split % block.ndim,
            halo=halo,
            periodic=periodic,
        )
        return from_local(block, layout=layout)

    axis = split % block.ndim
    check_collective(comm, "from_local")
    pieces = comm.allgather((tuple(block.shape), block.dtype.str))
    shapes = [shape for shape, _ in pieces]
    others = {shape[:axis] + shape[axis + 1 :] for shape in shapes if len(shape) > axis}
    if (
        len({len(shape) for shape in shapes}) != 1
        or len(others) != 1
        or len({dtype for _, dtype in pieces}) != 1
    ):
        raise ValueError(
            f"the blocks cannot be stacked along axis {split}: shapes "
            f"{shapes}, dtypes {[dtype for _, dtype in pieces]}",
        )
    lengths = [shape[axis] for shape in shapes]
    shape = list(shapes[0])
    shape[axis] = sum(lengths)
    layout = Layout(tuple(shape), comm=comm, split=axis, halo=halo, periodic=periodic)
    if lengths == [layout.local_shape_of(r)[axis] for r in range(size)]:
        return from_local(block, layout=layout)

    # Redistribute along the axis: send each rank the rows it owns.
    rank = comm.Get_rank()
    row = math.prod(shape) // shape[axis] if shape[axis] else 0
    starts = [sum(lengths[:r]) for r in range(size)]
    mine = (starts[rank], starts[rank] + lengths[rank])
    targets = [layout.index_bounds_of(r)[axis] for r in range(size)]

    def overlaps(
        have: tuple[int, int], owners: list[tuple[int, int]]
    ) -> tuple[list, list]:
        """Return element counts and offsets in ``have`` for each range of ``owners``."""
        counts, displacements = [], []
        for start, end in owners:
            low, high = max(have[0], start), min(have[1], end)
            counts.append(max(0, high - low) * row)
            displacements.append(max(0, low - have[0]) * row)
        return counts, displacements

    send_counts, send_displacements = overlaps(mine, targets)
    target = targets[rank]
    sources = [(starts[r], starts[r] + lengths[r]) for r in range(size)]
    recv_counts, recv_displacements = overlaps(target, sources)
    send = xp.ascontiguousarray(xp.moveaxis(block, axis, 0)).reshape(-1)
    received = xp.empty((target[1] - target[0]) * row, dtype=block.dtype)
    receiving = xp.mpi.mpi_buffer(received, send=False, recv=True)
    with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
        comm.Alltoallv(
            [sendbuf, (send_counts, send_displacements)],
            [recvbuf, (recv_counts, recv_displacements)],
        )
    moved_shape = (target[1] - target[0], *shape[:axis], *shape[axis + 1 :])
    local = xp.moveaxis(received.reshape(moved_shape), 0, axis)
    return from_local(local, layout=layout)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike | None = None,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
) -> DistributedArray:
    """Return evenly spaced values in ``[start, stop)``, like ``numpy.arange``.

    Each rank computes only its own block.

    Args:
        start: The first value; the stop value if ``stop`` is not given.
        stop: The end of the interval, not included.
        step: The spacing.
        dtype: The element type; default: inferred from the arguments.
        split: The split axis; see the module docstring.
        halo: The halo width.
        periodic: Whether the axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts.

    Returns:
        A 1-D array.

    Raises:
        ValueError: If ``step`` is zero.
    """
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("step must not be zero")
    if dtype is None:
        dtype = np.result_type(start, stop, step)
    length = max(0, math.ceil((stop - start) / step))
    layout = Layout(
        length,
        comm=comm,
        split=split,
        halo=halo,
        periodic=periodic,
        process_grid=process_grid,
    )
    first, last = layout.index_bounds[0]
    values = start + xp.arange(first, last) * step
    storage = xp.zeros(layout.storage_shape, dtype=dtype)
    storage[layout.interior] = values
    return DistributedArray(layout, storage)


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = True,
    *,
    dtype: DTypeLike | None = None,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
) -> DistributedArray:
    """Return ``num`` evenly spaced values from ``start`` to ``stop``, like ``numpy.linspace``.

    Each rank computes only its own block, with NumPy's formula.

    Args:
        start: The first value.
        stop: The last value, or the end of the interval if not ``endpoint``.
        num: The number of values.
        endpoint: Whether ``stop`` is the last value.
        dtype: The element type; default: floating point.
        split: The split axis; see the module docstring.
        halo: The halo width.
        periodic: Whether the axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts.

    Returns:
        A 1-D array.

    Raises:
        ValueError: If ``num`` is negative.
    """
    if num < 0:
        raise ValueError(f"number of samples, {num}, must be non-negative")
    if dtype is None:
        dtype = np.result_type(start, stop, float)
    layout = Layout(
        num,
        comm=comm,
        split=split,
        halo=halo,
        periodic=periodic,
        process_grid=process_grid,
    )
    first, last = layout.index_bounds[0]
    div = num - 1 if endpoint else num
    delta = stop - start
    index = xp.arange(first, last, dtype=float)
    values = start + (index * (delta / div) if div > 0 else index * delta)
    if endpoint and num > 1 and last == num and last > first:
        values[-1] = stop
    storage = xp.zeros(layout.storage_shape, dtype=dtype)
    storage[layout.interior] = values
    return DistributedArray(layout, storage)


def fromfunction(
    function: Callable[..., Any],
    shape: ShapeLike,
    *,
    dtype: DTypeLike = float,
    split: SplitArg = DEFAULT,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
) -> DistributedArray:
    """Return an array whose values are ``function`` of the global indices.

    Like ``numpy.fromfunction``: ``function`` gets one array of global
    indices per axis (of type ``dtype``). Each rank calls it for its own block
    only, so nothing global is ever built.

    Args:
        function: Maps the index arrays to the values; elementwise.
        shape: The global shape.
        dtype: The type of the index arrays.
        split: The split axis or axes; see the module docstring.
        halo: The halo width; the halo cells start at zero.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.

    Returns:
        The array, with the dtype of the values ``function`` returns.
    """
    layout = Layout(
        shape,
        comm=comm,
        split=split,
        halo=halo,
        periodic=periodic,
        process_grid=process_grid,
    )
    axes = [xp.arange(start, end, dtype=dtype) for start, end in layout.index_bounds]
    indices = xp.meshgrid(*axes, indexing="ij") if axes else []
    values = xp.asarray(function(*indices))
    storage = xp.zeros(layout.storage_shape, dtype=values.dtype)
    storage[layout.interior] = xp.broadcast_to(values, layout.local_shape)
    return DistributedArray(layout, storage)
