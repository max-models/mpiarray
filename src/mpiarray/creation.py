"""Functions that create distributed arrays, like their NumPy namesakes.

Every function takes the same keyword arguments for the layout:

- ``split``: the axis or axes split over the ranks (default ``0``); ``None``
  makes every rank hold the whole array.
- ``halo``: the halo width, for every axis or one per axis (default ``0``).
- ``periodic``: whether each axis wraps around in halo updates.
- ``comm``: the communicator (default: ``MPI.COMM_WORLD``, or cunumpy's serial
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
from typing import Any

import cunumpy as xp
import numpy as np
from numpy.typing import DTypeLike

from mpiarray._mpi import Comm, debug_checks
from mpiarray.distributed_array import Array, DistributedArray
from mpiarray.layout import HaloLike, Layout, PeriodicLike, ShapeLike, SplitLike


def _make_layout(
    shape: ShapeLike | None,
    layout: Layout | None,
    split: SplitLike,
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
    split: SplitLike = 0,
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
    split: SplitLike = 0,
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
    split: SplitLike = 0,
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
    split: SplitLike = 0,
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


def array(
    obj: Any,
    dtype: DTypeLike | None = None,
    *,
    split: SplitLike = 0,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
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

    A `DistributedArray` is copied; with a different ``layout`` it is
    redistributed, which gathers it (collective).

    Args:
        obj: The global array.
        dtype: The element type; default: that of ``obj``.
        split: The split axis or axes; see the module docstring.
        halo: The halo width; the halo cells start at zero.
        periodic: Whether each axis wraps around.
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
        if layout is None or layout == obj.layout:
            return obj.astype(obj.dtype if dtype is None else dtype)
        obj = obj.gather()
    data = xp.asarray(obj, dtype=dtype)
    layout = _make_layout(data.shape, layout, split, halo, periodic, comm, process_grid)
    if debug_checks():
        _check_identical(data, layout.comm)
    storage = xp.zeros(layout.storage_shape, dtype=data.dtype)
    storage[layout.interior] = data[layout.global_slices()]
    return DistributedArray(layout, storage)


def asarray(
    obj: Any,
    dtype: DTypeLike | None = None,
    *,
    split: SplitLike = 0,
    halo: HaloLike = 0,
    periodic: PeriodicLike = False,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Return ``obj`` as a distributed array, without a copy if it already is one.

    A `DistributedArray` with the requested ``dtype`` (and ``layout``, if
    given) is returned as it is; anything else goes through `array`.

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
    if (
        isinstance(obj, DistributedArray)
        and (dtype is None or xp.dtype(dtype) == obj.dtype)
        and (layout is None or layout == obj.layout)
    ):
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


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike | None = None,
    split: SplitLike = 0,
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
    split: SplitLike = 0,
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
    split: SplitLike = 0,
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
