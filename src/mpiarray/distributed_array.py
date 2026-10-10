"""The distributed array: each rank's block of a global array, with halo cells.

Create arrays with the functions of the package (`mpiarray.zeros`,
`mpiarray.array`, `mpiarray.arange`, ...), not with the constructor.

Calls that communicate are collective: every rank of the layout's
communicator must make them, in the same order, or the program hangs. They
are marked "Collective." in their docstrings: gathers, reading with global
indices, reductions and halo updates. Elementwise arithmetic, ``local`` and
``repr`` never communicate, so ``if rank == 0: print(a)`` is safe.
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, cast, overload

import cunumpy as xp
import numpy as np
from numpy.typing import DTypeLike

from mpiarray._exchange import Box, exchange_boxes, local_index, target_ranks
from mpiarray._mpi import MPI, check_collective
from mpiarray.layout import Layout

if TYPE_CHECKING:
    from typing_extensions import Self  # typing.Self needs Python 3.11

# A NumPy or CuPy array.
Array: TypeAlias = Any
Number: TypeAlias = int | float
# One entry of an index: basic (int, slice, Ellipsis, None) or advanced (an
# integer or boolean array, NumPy or CuPy).
IndexItem: TypeAlias = int | np.integer | slice | EllipsisType | None | Array
IndexLike: TypeAlias = IndexItem | tuple[IndexItem, ...]
# A host scalar: a NumPy scalar, Python number or bool.
Scalar: TypeAlias = Any

#: A condition for the halo cells at walls; see `DistributedArray.update_halos`.
Boundary: TypeAlias = "str | float | complex | None"
_BOUNDARIES = ("zero", "edge", "symmetric", "reflect")

# Tag offsets so halo updates and halo accumulations never share a message tag.
_UPDATE_TAG = 0
_ACCUMULATE_TAG = 1000
_NEIGHBOUR_TAG = 2000


@functools.lru_cache(maxsize=1024)
def _subarray(
    shape: tuple[int, ...], starts: tuple[int, ...], sizes: tuple[int, ...], dtype: str
) -> Any:
    """Return a committed MPI datatype for a box of a C-ordered array of ``shape``.

    Cached and never freed: there is one per halo region, layout and dtype.
    """
    from mpi4py.util import dtlib

    base = dtlib.from_numpy_dtype(np.dtype(dtype))
    return base.Create_subarray(list(shape), list(sizes), list(starts)).Commit()


class HaloUpdate:
    """A halo update in flight, from ``update_halos(wait=False)``.

    Args:
        finish: Completes the exchange; called once, by `wait`.
    """

    def __init__(self, finish: Callable[[], None]) -> None:
        """Keep ``finish`` for `wait`."""
        self._finish: Callable[[], None] | None = finish

    @property
    def done(self) -> bool:
        """Whether `wait` has completed the update."""
        return self._finish is None

    def wait(self) -> None:
        """Wait for the messages and write the halo cells; later calls do nothing."""
        if self._finish is not None:
            finish, self._finish = self._finish, None
            finish()


def _positions_within(selected: range, start: int, end: int) -> tuple[int, int]:
    """Return the positions ``[lo, hi)`` of ``selected`` whose values lie in ``[start, end)``.

    O(log n): a range is a sorted sequence, so bisection needs no array.
    """
    if selected.step > 0:
        return bisect.bisect_left(selected, start), bisect.bisect_left(selected, end)
    ascending = selected[::-1]
    lo, hi = bisect.bisect_left(ascending, start), bisect.bisect_left(ascending, end)
    return len(selected) - hi, len(selected) - lo


def _normalize_axes(axis: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Return ``axis`` as non-negative axes; NumPy's ``AxisError`` if out of range."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = []
    for a in axes:
        if not -ndim <= a < ndim:
            raise np.exceptions.AxisError(a, ndim)
        normalized.append(a % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate value in 'axis'")
    return tuple(normalized)


# The local reduction and how two partial results combine, per reduction.
_AXIS_REDUCTIONS: dict[str, tuple[Callable, Callable]] = {
    "sum": (lambda a, **kw: xp.sum(a, **kw), lambda x, y: x + y),
    "prod": (lambda a, **kw: xp.prod(a, **kw), lambda x, y: x * y),
    "min": (lambda a, **kw: xp.min(a, **kw), lambda x, y: xp.minimum(x, y)),
    "max": (lambda a, **kw: xp.max(a, **kw), lambda x, y: xp.maximum(x, y)),
    "all": (lambda a, **kw: xp.all(a, **kw), lambda x, y: xp.logical_and(x, y)),
    "any": (lambda a, **kw: xp.any(a, **kw), lambda x, y: xp.logical_or(x, y)),
}


def _identity(name: str, dtype: np.dtype) -> Any:
    """Return the value that leaves a partial result unchanged when combined."""
    kind = np.dtype(dtype).kind
    if name in ("sum", "any"):
        return 0
    if name in ("prod", "all"):
        return 1
    largest = name == "min"
    if kind == "b":
        return largest
    if kind in "iu":
        info = np.iinfo(dtype)
        return info.max if largest else info.min
    infinity = np.inf if largest else -np.inf
    return complex(infinity, infinity) if kind == "c" else infinity


def _arg_key(candidate: tuple[Any, int], is_min: bool) -> tuple:
    """Order (value, index) candidates as argmin/argmax do: NaN first, then the first index."""
    value, index = candidate
    value = value.item() if isinstance(value, np.generic) else value
    if value != value:  # NaN
        return (0, 0, index)
    return (1, value if is_min else -value, index)


def _better(piece: Any, current: Any, is_min: bool) -> Any:
    """Return where ``piece`` (values, indices) beats ``current`` for argmin/argmax."""
    value, index = piece[0], piece[1]
    best, best_index = current[0], current[1]
    strictly = value < best if is_min else value > best
    tie = (value == best) & (index < best_index)
    if np.dtype(piece.dtype).kind != "f":
        return strictly | tie
    nan, best_nan = xp.isnan(value), xp.isnan(best)
    return xp.where(best_nan, nan & (index < best_index), nan | strictly | tie)


def _host(value: Any) -> Any:
    """Return a device scalar or 0-d array as a host scalar; other values unchanged."""
    if xp.is_gpu(value):
        return xp.to_numpy(value)[()]
    return value


def _is_int(item: object) -> TypeGuard[int | np.integer]:
    """Return whether ``item`` is an integer index (not a bool)."""
    return isinstance(item, int | np.integer) and not isinstance(item, bool | np.bool_)


class DistributedArray:
    """A global array split over the ranks of a communicator.

    Each rank stores the block it owns (see `Layout`), padded with halo
    cells. Elementwise operations work on the local storage without
    communication; gathers, global indexing, reductions and halo updates are
    collective.

    Args:
        layout: How the array is split.
        storage: This rank's storage, of shape ``layout.storage_shape``
            (the block plus its halo cells); used without a copy.

    Raises:
        ValueError: If ``storage`` does not have the layout's storage shape.
    """

    __array_priority__ = 1000

    def __init__(self, layout: Layout, storage: Array) -> None:
        """Wrap ``storage``; see the class docstring."""
        if tuple(storage.shape) != layout.storage_shape:
            raise ValueError(
                f"storage shape {tuple(storage.shape)} does not match the layout's "
                f"storage shape {layout.storage_shape}",
            )
        self._layout = layout
        self._data = storage
        # receive buffers and host staging, reused by every halo exchange
        self._halo_buffers: dict[tuple, Any] = {}

    def _with_storage(self, storage: Array) -> Self:
        """Return an array with this layout and type that owns ``storage``."""
        return type(self)(self._layout, storage)

    # ------------------------------------------------------------------ #
    # Properties

    @property
    def layout(self) -> Layout:
        """How the array is split over the ranks."""
        return self._layout

    @property
    def shape(self) -> tuple[int, ...]:
        """The global shape."""
        return self._layout.shape

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return self._layout.ndim

    @property
    def dtype(self) -> np.dtype:
        """The element type."""
        return self._data.dtype

    @property
    def size(self) -> int:
        """The number of elements of the global array."""
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        """The size of one element in bytes."""
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """The size of the global array in bytes, halo cells not included."""
        return self.size * self.itemsize

    @property
    def local(self) -> Array:
        """A writable view of this rank's block, without halo cells."""
        return self._data[self._layout.interior]

    @property
    def local_with_halos(self) -> Array:
        """This rank's storage: the block and its halo cells (writable)."""
        return self._data

    def copy(self) -> Self:
        """Return an independent copy with the same layout, halo cells included."""
        return self._with_storage(self._data.copy())

    def astype(self, dtype: DTypeLike, copy: bool = True) -> Self:
        """Return the array converted to ``dtype``.

        Args:
            dtype: The new element type.
            copy: If ``False`` and the array already has ``dtype``, return the
                array itself.

        Returns:
            An array with the same layout, halo cells included.
        """
        if not copy and xp.dtype(dtype) == self.dtype:
            return self
        return self._with_storage(self._data.astype(dtype))

    # ------------------------------------------------------------------ #
    # Gathering and global indexing

    @overload
    def gather(self, root: None = None) -> Array: ...
    @overload
    def gather(self, root: int) -> Array | None: ...
    def gather(self, root: int | None = None) -> Array | None:
        """Return the whole global array, halo cells not included.

        Collective.

        Args:
            root: The rank that receives the array; ``None``: every rank.

        Returns:
            A new NumPy or CuPy array (as the backend selects), on ``root`` or
            on every rank; ``None`` on the other ranks.
        """
        layout = self._layout
        if not layout.distributed:
            if root is not None and layout.rank != root:
                return None
            return self.local.copy()

        counts = [math.prod(layout.local_shape_of(r)) for r in range(layout.size)]
        send = xp.ascontiguousarray(self.local)
        receives = root is None or layout.rank == root
        recv = xp.empty(sum(counts) if receives else 0, dtype=self.dtype)
        check_collective(
            layout.comm, "gather" if root is None else f"gather(root={root})"
        )
        receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
        with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
            if root is None:
                layout.comm.Allgatherv(sendbuf, [recvbuf, counts])
            else:
                layout.comm.Gatherv(sendbuf, [recvbuf, counts], root=root)
        if not receives:
            return None

        full = xp.empty(self.shape, dtype=self.dtype)
        offset = 0
        for rank, count in enumerate(counts):
            block = recv[offset : offset + count].reshape(layout.local_shape_of(rank))
            full[layout.global_slices(rank)] = block
            offset += count
        return full

    def _gather_all(self) -> Array:
        """Return the global array on every rank (collective); see `gather`."""
        return self.gather()

    @overload
    def to_numpy(self, root: None = None) -> np.ndarray: ...
    @overload
    def to_numpy(self, root: int) -> np.ndarray | None: ...
    def to_numpy(self, root: int | None = None) -> np.ndarray | None:
        """Return the whole global array as a ``numpy.ndarray``.

        Collective; see `gather`.

        Args:
            root: The rank that receives the array; ``None``: every rank.
        """
        gathered = self.gather(root)
        return None if gathered is None else xp.to_numpy(gathered)

    def to_petsc(self, vec: Any = None) -> Any:
        """Copy this rank's block into a global PETSc vector (needs petsc4py).

        Local, without communication; halo cells are left out. See
        `mpiarray.petsc.copy_to_petsc`.

        Args:
            vec: The ``PETSc.Vec`` to fill; default: a new global vector of
                ``self.layout.dmda()`` (the cached DMDA; collective the first
                time it is built for this layout).

        Returns:
            The filled vector.
        """
        from mpiarray.petsc import copy_to_petsc

        if vec is None:
            vec = self._layout.dmda().createGlobalVec()
        copy_to_petsc(self, vec)
        return vec

    def copy_from_petsc(self, vec: Any) -> None:
        """Overwrite this rank's block from a global PETSc vector (needs petsc4py).

        Local; halo cells are left untouched. See
        `mpiarray.petsc.copy_from_petsc`.

        Args:
            vec: A global vector of ``self.layout.dmda()``.
        """
        from mpiarray.petsc import copy_from_petsc

        copy_from_petsc(vec, self)

    def local_index(self, index: int | tuple[int, ...]) -> tuple[int, ...] | None:
        """Return where a global index lies in this rank's storage.

        No communication.

        Args:
            index: One integer per axis (an int for 1-D); negative values count
                from the end.

        Returns:
            The index into `local_with_halos`, or ``None`` if this rank does not
            own the element.

        Raises:
            IndexError: If ``index`` has the wrong length or is out of bounds.
        """
        self._layout.owner(index)  # validates the index on every rank alike
        index = (index,) if isinstance(index, int) else index
        local = []
        for i, length, (start, end), halo in zip(
            index,
            self.shape,
            self._layout.index_bounds,
            self._layout.halo,
            strict=True,
        ):
            position = i + length if i < 0 else i
            if not start <= position < end:
                return None
            local.append(position - start + halo)
        return tuple(local)

    def get(self, index: int | tuple[int, ...]) -> Scalar:
        """Return one element of the global array, on every rank.

        Collective: the owning rank broadcasts the value.

        Args:
            index: One integer per axis (an int for 1-D); negative values count
                from the end.

        Returns:
            The element, as a host scalar.

        Raises:
            IndexError: On every rank, if ``index`` is out of bounds.
        """
        owner = self._layout.owner(index)
        local = self.local_index(index)
        value = None if local is None else _host(self._data[local])
        if not self._layout.distributed:
            return value
        check_collective(self._layout.comm, f"get({index!r})")
        return self._layout.comm.bcast(value, root=owner)

    def __getitem__(self, index: IndexLike) -> Scalar | Array:
        """Return an element or a selection of the global array, on every rank.

        Collective. An integer per axis is `get`: a host scalar, the same on
        every rank. Any other index returns a new `DistributedArray`: basic
        indices (integers, slices, Ellipsis) move only the selected cells to the
        ranks that own them in the result; index arrays and masks gather the
        array first. Call ``gather()`` on the result for a NumPy array.

        Args:
            index: A global index, as for a NumPy array of shape `shape`.

        Returns:
            A host scalar, or a `DistributedArray`.
        """
        basic = self._normalize_basic_index(index)
        if basic is not None and all(_is_int(item) for item in basic):
            return self.get(tuple(int(item) for item in basic if _is_int(item)))
        if basic is not None:
            return self._select(basic)
        return self._spread(self._gather_all()[index])

    def __setitem__(self, index: IndexLike, value: Any) -> None:
        """Set an element or a selection of the global array.

        Integers, slices and Ellipsis are written by each rank into the cells
        it owns, without communication; ``value`` must then be the same on
        every rank. Advanced indices (arrays, masks) gather the whole array,
        are collective, and zero the halo cells.

        Args:
            index: A global index, as for a NumPy array of shape `shape`.
            value: The value to write, broadcast to the selection.
        """
        basic = self._normalize_basic_index(index)
        if basic is not None:
            self._assign_basic(basic, value)
            return
        data = self._gather_all()
        data[index] = value
        self._data[...] = 0
        self._data[self._layout.interior] = data[self._layout.global_slices()]

    def _normalize_basic_index(self, index: Any) -> tuple[int | slice, ...] | None:
        """Return ``index`` as one int or slice per axis, or None if it is not basic.

        Basic means integers, slices and at most one Ellipsis; anything else
        (arrays, masks, ``None``) returns None.
        """
        items = index if isinstance(index, tuple) else (index,)
        if not all(
            item is Ellipsis or isinstance(item, slice) or _is_int(item)
            for item in items
        ):
            return None
        num_ellipsis = sum(item is Ellipsis for item in items)
        if num_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        explicit = len(items) - num_ellipsis
        if explicit > self.ndim:
            raise IndexError(
                f"too many indices: array is {self.ndim}-dimensional, "
                f"but {explicit} were indexed",
            )
        fill = (slice(None),) * (self.ndim - explicit)
        if num_ellipsis:
            position = next(i for i, item in enumerate(items) if item is Ellipsis)
            items = items[:position] + fill + items[position + 1 :]
        else:
            items = items + fill
        return tuple(int(item) if _is_int(item) else item for item in items)

    def _selection_plan(
        self, index: tuple[int | slice, ...], rank: int
    ) -> tuple[tuple[int, ...], tuple[tuple, tuple, tuple[int, ...]] | None]:
        """Return how the global selection ``index`` meets the block of ``rank``.

        Returns:
            The shape of the whole selection, and ``None`` if ``rank`` owns none
            of it, or else ``(storage_index, result_index, piece_shape)``: where
            the owned part lies in that rank's storage and in the selection.

        Raises:
            IndexError: If an integer is out of bounds (on every rank alike).
        """
        bounds = self._layout.index_bounds_of(rank)
        selection_shape: list[int] = []
        storage_index: list[int | slice] = []
        result_index: list[slice] = []
        piece_shape: list[int] = []
        owned = True
        for axis, (item, (start, end), halo) in enumerate(
            zip(index, bounds, self._layout.halo, strict=True),
        ):
            extent = self.shape[axis]
            if isinstance(item, int):
                position = item + extent if item < 0 else item
                if not 0 <= position < extent:
                    raise IndexError(
                        f"index {item} is out of bounds for axis {axis} "
                        f"with size {extent}",
                    )
                owned = owned and start <= position < end
                storage_index.append(position - start + halo)
                continue

            selected = range(*item.indices(extent))
            selection_shape.append(len(selected))
            # The selection is monotonic, so the owned entries are contiguous.
            lo, hi = _positions_within(selected, start, end)
            if lo == hi:
                owned = False
                continue
            first = selected[lo] - start + halo
            last = selected[hi - 1] - start + halo
            stop = last + selected.step
            storage_index.append(
                slice(first, None if stop < 0 else stop, selected.step)
            )
            result_index.append(slice(lo, hi))
            piece_shape.append(hi - lo)
        if not owned:
            return tuple(selection_shape), None
        return tuple(selection_shape), (
            tuple(storage_index),
            tuple(result_index),
            tuple(piece_shape),
        )

    def _layout_like(self, shape: tuple[int, ...], split: Sequence[int]) -> Layout:
        """Return a layout for a result: split along ``split`` if it fits, else replicated."""
        layout = self._layout
        if layout.distributed and split:
            try:
                return Layout(shape, comm=layout.comm, split=tuple(split))
            except ValueError:
                pass
        return Layout(shape, comm=layout.comm, split=None)

    def _spread(self, data: Array) -> Self:
        """Return a global array that every rank holds as a distributed array."""
        target = self._layout_like(tuple(data.shape), (0,) if data.ndim else ())
        return self._with_layout(target, data[target.global_slices()])

    def _with_layout(self, layout: Layout, block: Array) -> Self:
        """Return an array of this type with ``layout`` around a block (halos zero)."""
        storage = xp.zeros(layout.storage_shape, dtype=block.dtype)
        storage[layout.interior] = block
        return type(self)(layout, storage)

    def _select(self, index: tuple[int | slice, ...]) -> Self:
        """Return a basic selection as a distributed array, moving only its cells."""
        layout = self._layout
        slice_axes = [
            axis for axis, item in enumerate(index) if isinstance(item, slice)
        ]
        selection_shape, own = self._selection_plan(index, layout.rank)
        split = [k for k, axis in enumerate(slice_axes) if axis in layout.split]
        target = self._layout_like(selection_shape, split)
        if not layout.distributed:
            if own is None:
                block = xp.empty(selection_shape, dtype=self.dtype)
            else:
                block = self._data[own[0]]
            return self._with_layout(target, block[target.global_slices()])

        def box_of(plan: Any) -> Box | None:
            return None if plan is None else tuple((i.start, i.stop) for i in plan[1])

        boxes = [box_of(self._selection_plan(index, r)[1]) for r in range(layout.size)]
        data = self._data[own[0]] if own is not None else self._data
        pieces = exchange_boxes(
            target, data, boxes[layout.rank], boxes, name="a[...] (selection)"
        )
        storage = xp.zeros(target.storage_shape, dtype=self.dtype)
        for _, box, piece in pieces:
            storage[local_index(box, target.index_bounds, target.halo)] = piece
        return type(self)(target, storage)

    def redistribute(self, layout: Layout) -> Self:
        """Return the array moved to another layout of the same shape and communicator.

        Every rank sends each other rank only the part of its block that the
        other owns in ``layout`` (one ``Alltoallv``); nothing global is built.
        The halo cells of the result start at zero. Collective.

        Args:
            layout: The new layout.

        Returns:
            A new array; a copy if ``layout`` is already the array's.

        Raises:
            ValueError: If ``layout`` has another shape, or a communicator with
                other processes (a renumbered one, e.g. from ``reorder=True``,
                is fine).
        """
        source = self._layout
        if layout.shape != self.shape:
            raise ValueError(
                f"cannot redistribute shape {self.shape} to {layout.shape}"
            )
        try:
            target_ranks(source.comm, layout.comm)
        except ValueError:
            raise ValueError("cannot redistribute to another communicator") from None
        if layout == source:
            return self.copy()
        storage = xp.zeros(layout.storage_shape, dtype=self.dtype)
        if not source.distributed:  # every rank holds the whole array
            storage[layout.interior] = self.local[layout.global_slices()]
            return type(self)(layout, storage)
        boxes: list[Box | None] = [
            source.index_bounds_of(r) for r in range(source.size)
        ]
        pieces = exchange_boxes(
            layout,
            self.local,
            source.index_bounds,
            boxes,
            name="redistribute",
            comm=source.comm,
        )
        for _, box, piece in pieces:
            storage[local_index(box, layout.index_bounds, layout.halo)] = piece
        return type(self)(layout, storage)

    def rebalance(self, weights: Any) -> Self:
        """Return the array moved to blocks of about equal weight; see `Layout.weighted`.

        Collective.

        Args:
            weights: The cost of each element: an array of the same shape (a
                `DistributedArray` too), or one 1-D profile per axis.

        Returns:
            The redistributed array, with explicit `Layout.bounds`.
        """
        layout = self._layout
        balanced = Layout.weighted(
            self.shape,
            weights,
            comm=layout.comm,
            split=layout.split if layout.split else None,
            halo=layout.halo,
            periodic=layout.periodic,
            process_grid=layout.process_grid if layout.distributed else None,
        )
        return self.redistribute(balanced)

    def _assign_basic(self, index: tuple[int | slice, ...], value: Any) -> None:
        """Write ``value`` into the owned part of the global selection ``index``.

        ``value`` must be the same on every rank and broadcast to the shape of
        the selection. No communication happens, and halo cells are untouched.
        """
        if isinstance(value, DistributedArray):
            value = value._gather_all()
        selection_shape, own = self._selection_plan(index, self._layout.rank)
        # Broadcast on every rank, so a bad value raises everywhere alike.
        value = xp.broadcast_to(xp.asarray(value), selection_shape)
        if own is not None:
            self._data[own[0]] = value[own[1]]

    # ------------------------------------------------------------------ #
    # Halo cells

    def _axes(self, axis: int | None) -> range | tuple[int]:
        """Return the axes a halo method works on."""
        return range(self.ndim) if axis is None else (axis,)

    def _region_bounds(
        self, axis: int, name: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the ``(starts, sizes)`` in the storage of a halo region along ``axis``.

        ``name`` is one of ``lower_halo``, ``lower_interior``, ``upper_interior``
        and ``upper_halo``: the halo cells along ``axis`` or the layers of the
        block next to them, across the whole storage along the other axes
        (their halos included, so that corners travel with the faces).
        """
        layout = self._layout
        h = layout.halo[axis]
        extent = layout.storage_shape[axis]
        offset = {
            "lower_halo": 0,
            "lower_interior": h,
            "upper_interior": extent - 2 * h,
            "upper_halo": extent - h,
        }[name]
        starts = [offset if ax == axis else 0 for ax in range(self.ndim)]
        sizes = [h if ax == axis else n for ax, n in enumerate(layout.storage_shape)]
        return tuple(starts), tuple(sizes)

    def _region(self, axis: int, name: str) -> tuple[slice, ...]:
        """Return the index of a halo region in the storage; see `_region_bounds`."""
        starts, sizes = self._region_bounds(axis, name)
        return tuple(slice(a, a + n) for a, n in zip(starts, sizes, strict=True))

    def _is_self_neighbour(self, axis: int) -> bool:
        """Return whether both neighbours along ``axis`` are this rank."""
        left, right = self._layout.neighbours[axis]
        return left == right == self._layout.rank

    def _sends_in_place(self) -> bool:
        """Return whether halo slabs can go to MPI straight from the storage.

        True for C-ordered host storage, described by MPI subarray datatypes;
        device or other storage goes through contiguous buffers instead.
        """
        return not xp.is_gpu(self._data) and bool(self._data.flags.c_contiguous)

    def _buffer(self, key: tuple, shape: tuple[int, ...]) -> Array:
        """Return a contiguous buffer kept for repeated halo exchanges."""
        buffer = self._halo_buffers.get(key)
        if buffer is None:
            buffer = xp.empty(shape, dtype=self.dtype)
            self._halo_buffers[key] = buffer
        return buffer

    def _staging(self, key: tuple, shape: tuple[int, ...]) -> Any:
        """Return the host staging kept for a device buffer, or None on the host."""
        if not xp.is_gpu(self._data):
            return None
        key = ("staging", *key)
        staging = self._halo_buffers.get(key)
        if staging is None:
            staging = xp.mpi.MPIStaging(shape, self.dtype)
            self._halo_buffers[key] = staging
        return staging

    def _post_axis(
        self,
        axis: int,
        accumulate: bool,
        requests: list,
        arrivals: list[tuple[tuple[slice, ...], Array]],
        stack: contextlib.ExitStack,
    ) -> None:
        """Post the two receives and two sends of a halo exchange along ``axis``.

        Fills ``requests`` with the MPI requests and ``arrivals`` with the
        ``(storage region, buffer)`` pairs whose received values are written
        (update) or added (accumulate) once the requests are done.
        """
        layout = self._layout
        # Exchanges only run between different ranks, so under a real MPI
        # (maybempi's serial stand-in has no point-to-point calls).
        comm = cast("MPI.Comm", layout.comm)
        left, right = layout.neighbours[axis]
        tag = (_ACCUMULATE_TAG if accumulate else _UPDATE_TAG) + 2 * axis
        if accumulate:
            # halo cells go to the neighbour whose block they belong to
            transfers = [
                ("lower_halo", left, "upper_interior", right, tag),
                ("upper_halo", right, "lower_interior", left, tag + 1),
            ]
        else:
            # boundary layers go to the neighbour whose halo they fill
            transfers = [
                ("upper_interior", right, "lower_halo", left, tag),
                ("lower_interior", left, "upper_halo", right, tag + 1),
            ]
        in_place = self._sends_in_place()
        for send_name, dest, recv_name, source, message_tag in transfers:
            starts, sizes = self._region_bounds(axis, recv_name)
            if in_place and not accumulate:
                datatype = _subarray(self._data.shape, starts, sizes, self.dtype.str)
                requests.append(
                    comm.Irecv(
                        [self._data, 1, datatype], source=source, tag=message_tag
                    )
                )
            else:
                key = (axis, recv_name)
                buffer = self._buffer(key, sizes)
                receiving = xp.mpi.mpi_buffer(
                    buffer, send=False, recv=True, staging=self._staging(key, sizes)
                )
                requests.append(
                    comm.Irecv(
                        stack.enter_context(receiving), source=source, tag=message_tag
                    )
                )
                if source != MPI.PROC_NULL:
                    arrivals.append((self._region(axis, recv_name), buffer))

            starts, sizes = self._region_bounds(axis, send_name)
            if in_place:
                datatype = _subarray(self._data.shape, starts, sizes, self.dtype.str)
                requests.append(
                    comm.Isend([self._data, 1, datatype], dest=dest, tag=message_tag)
                )
            else:
                key = (axis, send_name, "send")
                buffer = self._buffer(key, sizes)
                buffer[...] = self._data[self._region(axis, send_name)]
                sending = xp.mpi.mpi_buffer(buffer, staging=self._staging(key, sizes))
                requests.append(
                    comm.Isend(stack.enter_context(sending), dest=dest, tag=message_tag)
                )

    def _exchange(self, axes: Sequence[int], accumulate: bool) -> HaloUpdate:
        """Start the halo exchange along ``axes``; return the handle that finishes it."""
        requests: list = []
        arrivals: list[tuple[tuple[slice, ...], Array]] = []
        stack = contextlib.ExitStack()
        if axes:
            name = "accumulate_halos" if accumulate else "update_halos"
            check_collective(self._layout.comm, f"{name} (axes {tuple(axes)})")
        for axis in axes:
            self._post_axis(axis, accumulate, requests, arrivals, stack)

        def finish() -> None:
            MPI.Request.Waitall(requests)
            stack.close()  # copies staged receives back to the device
            for region, buffer in arrivals:
                if accumulate:
                    self._data[region] += buffer
                else:
                    self._data[region] = buffer
            if accumulate:
                for axis in axes:
                    self.clear_halos(axis)

        return HaloUpdate(finish)

    def _exchanged_axes(self, axis: int | None) -> list[int]:
        """Return the axes with halo cells that exchange with other ranks.

        Axes held whole by this rank are handled locally right away: their
        periodic halos are copies of the block's own boundary layers.
        """
        axes = []
        for ax in self._axes(axis):
            if self._layout.halo[ax] == 0:
                continue
            if self._layout.distributed and not self._is_self_neighbour(ax):
                axes.append(ax)
            elif self._layout.periodic[ax]:
                self._data[self._region(ax, "lower_halo")] = self._data[
                    self._region(ax, "upper_interior")
                ]
                self._data[self._region(ax, "upper_halo")] = self._data[
                    self._region(ax, "lower_interior")
                ]
        return axes

    def _check_boundary(self, boundary: Boundary) -> None:
        """Raise on every rank alike unless ``boundary`` is a supported condition."""
        if boundary is None or (
            isinstance(boundary, int | float | complex)
            and not isinstance(boundary, bool)
        ):
            return
        if boundary not in _BOUNDARIES:
            raise ValueError(
                f"unknown boundary {boundary!r}; use None, a number, "
                + ", ".join(repr(b) for b in _BOUNDARIES),
            )
        if boundary == "reflect":
            layout = self._layout
            for axis, (length, n, h) in enumerate(
                zip(layout.shape, layout.process_grid, layout.halo, strict=True)
            ):
                if h and length // n < h + 1:
                    raise ValueError(
                        f"boundary='reflect' needs blocks of at least {h + 1} elements "
                        f"along axis {axis}, the smallest has {length // n}",
                    )

    def _fill_walls(self, axis: int, boundary: Boundary) -> None:
        """Write the boundary condition into the halo cells at walls along ``axis``."""
        layout = self._layout
        h = layout.halo[axis]
        if boundary is None or h == 0 or layout.periodic[axis]:
            return
        extent = layout.storage_shape[axis]
        left, right = layout.neighbours[axis]
        for side, neighbour in (("lower", left), ("upper", right)):
            if neighbour != MPI.PROC_NULL:
                continue
            region = list(self._region(axis, f"{side}_halo"))
            if not isinstance(boundary, str) or boundary == "zero":
                self._data[tuple(region)] = 0 if boundary == "zero" else boundary
                continue
            # where the halo values come from, as a slice along the axis
            source = {
                ("lower", "edge"): slice(h, h + 1),
                ("lower", "symmetric"): slice(2 * h - 1, h - 1, -1),
                ("lower", "reflect"): slice(2 * h, h, -1),
                ("upper", "edge"): slice(extent - h - 1, extent - h),
                ("upper", "symmetric"): slice(extent - h - 1, extent - 2 * h - 1, -1),
                ("upper", "reflect"): slice(extent - h - 2, extent - 2 * h - 2, -1),
            }[side, boundary]
            values = list(region)
            values[axis] = source
            self._data[tuple(region)] = self._data[tuple(values)]

    def _box(self, offset: Sequence[int], halo_side: bool) -> tuple[slice, ...]:
        """Return the storage box on side ``offset`` of the block.

        Its halo cells, or (``halo_side=False``) the block cells next to them;
        along axes where the offset is 0, the whole block.
        """
        layout = self._layout
        index = []
        for o, h, n in zip(offset, layout.halo, layout.storage_shape, strict=True):
            if o == 0:
                index.append(slice(h, n - h))
            elif halo_side:
                index.append(slice(0, h) if o < 0 else slice(n - h, n))
            else:
                index.append(slice(h, 2 * h) if o < 0 else slice(n - 2 * h, n - h))
        return tuple(index)

    def _neighbour(self, offset: Sequence[int]) -> int:
        """Return the rank at ``offset`` on the process grid, or PROC_NULL past a wall."""
        layout = self._layout
        coord = []
        for c, o, n, wraps in zip(
            layout.process_coord,
            offset,
            layout.process_grid,
            layout.periodic,
            strict=True,
        ):
            c += o
            if not 0 <= c < n:
                if not wraps:
                    return MPI.PROC_NULL
                c %= n
            coord.append(c)
        return layout.rank_of(coord)

    def _neighbourhood_exchange(
        self, accumulate: bool, boundary: Boundary
    ) -> HaloUpdate:
        """Exchange every halo region directly with its neighbour, diagonals included.

        One round of messages for all axes (up to 26 neighbours in 3-D), so
        corners need no second step; see `update_halos` and `accumulate_halos`
        with ``wait=False``.
        """
        layout = self._layout
        comm = cast("MPI.Comm", layout.comm)
        offsets = [
            offset
            for offset in itertools.product((-1, 0, 1), repeat=self.ndim)
            if any(offset)
            and all(h > 0 or o == 0 for o, h in zip(offset, layout.halo, strict=True))
        ]
        tags: dict[tuple[int, ...], int] = {
            offset: _NEIGHBOUR_TAG + k for k, offset in enumerate(offsets)
        }
        name = "accumulate_halos" if accumulate else "update_halos"
        check_collective(comm, f"{name} (neighbourhood)")
        in_place = self._sends_in_place() and not accumulate
        requests: list = []
        arrivals: list[tuple[tuple[slice, ...], Array]] = []
        stack = contextlib.ExitStack()
        for offset in offsets:
            opposite = tuple(-o for o in offset)
            source, dest = self._neighbour(offset), self._neighbour(offset)
            # receive: what the neighbour at ``offset`` sent towards us
            target = self._box(offset, halo_side=not accumulate)
            sizes = tuple(sl.stop - sl.start for sl in target)
            if in_place:
                datatype = _subarray(
                    self._data.shape,
                    tuple(sl.start for sl in target),
                    sizes,
                    self.dtype.str,
                )
                requests.append(
                    comm.Irecv(
                        [self._data, 1, datatype], source=source, tag=tags[opposite]
                    )
                )
            else:
                key = ("neighbourhood", offset, accumulate)
                buffer = self._buffer(key, sizes)
                receiving = xp.mpi.mpi_buffer(
                    buffer, send=False, recv=True, staging=self._staging(key, sizes)
                )
                requests.append(
                    comm.Irecv(
                        stack.enter_context(receiving),
                        source=source,
                        tag=tags[opposite],
                    )
                )
                if source != MPI.PROC_NULL:
                    arrivals.append((target, buffer))
            # send: our cells for the neighbour at ``offset``
            box = self._box(offset, halo_side=accumulate)
            sizes = tuple(sl.stop - sl.start for sl in box)
            if self._sends_in_place():
                datatype = _subarray(
                    self._data.shape,
                    tuple(sl.start for sl in box),
                    sizes,
                    self.dtype.str,
                )
                requests.append(
                    comm.Isend([self._data, 1, datatype], dest=dest, tag=tags[offset])
                )
            else:
                key = ("neighbourhood send", offset, accumulate)
                buffer = self._buffer(key, sizes)
                buffer[...] = self._data[box]
                sending = xp.mpi.mpi_buffer(buffer, staging=self._staging(key, sizes))
                requests.append(
                    comm.Isend(
                        stack.enter_context(sending), dest=dest, tag=tags[offset]
                    )
                )

        def finish() -> None:
            MPI.Request.Waitall(requests)
            stack.close()
            for region, buffer in arrivals:
                if accumulate:
                    self._data[region] += buffer
                else:
                    self._data[region] = buffer
            if accumulate:
                self.clear_halos()
            else:
                for ax in range(self.ndim):
                    self._fill_walls(ax, boundary)

        return HaloUpdate(finish)

    def update_halos(
        self,
        axis: int | None = None,
        *,
        wait: bool = True,
        boundary: Boundary = None,
    ) -> HaloUpdate | None:
        """Copy the neighbours' boundary values into the halo cells.

        This is what a finite-difference stencil needs before it is applied.
        At walls (non-periodic boundaries) the halo cells are left unchanged.
        Both directions of an axis are exchanged at the same time, and host
        storage is sent and received in place, without copies.

        With ``wait=True`` the axes are updated one after the other, so the
        corner halo cells are filled too. With ``wait=False`` the messages to
        every neighbour, diagonal ones included, are started at once and a
        handle is returned: compute on the block, then call its ``wait()``
        before reading the halo cells. The result is the same as with
        ``wait=True``, corners included; the array must not be written until
        ``wait()`` returns.

        ``boundary`` fills the halo cells at walls (non-periodic boundaries),
        where there is no neighbour to copy from:

        - ``None``: leave them as they are (default);
        - a number, or ``"zero"``: that constant (a Dirichlet condition);
        - ``"edge"``: the nearest value of the block (zero gradient, Neumann);
        - ``"symmetric"``: the block mirrored, edge value included;
        - ``"reflect"``: the block mirrored about the edge value, as
          ``numpy.pad`` does (needs blocks one cell wider than the halo).

        Collective.

        Args:
            axis: The axis to update; default: every axis.
            wait: Whether to finish before returning.
            boundary: What to write into the halo cells at walls.

        Returns:
            ``None``, or with ``wait=False`` the `HaloUpdate` to wait for.

        Raises:
            ValueError: For an unknown ``boundary``, or ``"reflect"`` with too
                narrow blocks (on every rank).
        """
        self._check_boundary(boundary)
        if wait:
            for ax in self._axes(axis):
                self._exchange(self._exchanged_axes(ax), False).wait()
                self._fill_walls(ax, boundary)
            return None
        if not self._layout.distributed:  # nothing to send: done at once
            self.update_halos(axis, boundary=boundary)
            return HaloUpdate(lambda: None)
        if axis is not None:
            pending = self._exchange(self._exchanged_axes(axis), False)

            def finish() -> None:
                pending.wait()
                self._fill_walls(axis, boundary)

            return HaloUpdate(finish)
        return self._neighbourhood_exchange(accumulate=False, boundary=boundary)

    def accumulate_halos(
        self, axis: int | None = None, *, wait: bool = True
    ) -> HaloUpdate | None:
        """Add the halo cells into the neighbours' boundary cells, then zero them.

        This is the scatter-add after depositing particles near the edges of a
        block. At walls (non-periodic boundaries) the halo values are dropped.
        With ``wait=True`` the axes are handled one after the other, so values
        deposited in corner halo cells reach the diagonal neighbour in two
        steps. With ``wait=False`` (all axes) every halo region goes straight
        to the neighbour it belongs to, diagonal ones included, and a handle is
        returned: call its ``wait()`` before using the array, and do not write
        the array until then.

        Collective.

        Args:
            axis: The axis to accumulate along; default: every axis.
            wait: Whether to finish before returning.

        Returns:
            ``None``, or with ``wait=False`` the `HaloUpdate` to wait for.
        """
        if not wait and axis is None and self._layout.distributed:
            return self._neighbourhood_exchange(accumulate=True, boundary=None)
        for ax in self._axes(axis):
            if self._layout.halo[ax] == 0:
                continue
            if self._layout.distributed and not self._is_self_neighbour(ax):
                self._exchange([ax], True).wait()
                continue
            if self._layout.periodic[ax]:
                lower_halo = self._data[self._region(ax, "lower_halo")].copy()
                self._data[self._region(ax, "lower_interior")] += self._data[
                    self._region(ax, "upper_halo")
                ]
                self._data[self._region(ax, "upper_interior")] += lower_halo
            self.clear_halos(ax)
        return None if wait else HaloUpdate(lambda: None)

    def clear_halos(self, axis: int | None = None) -> None:
        """Set the halo cells to zero; no communication.

        Args:
            axis: The axis whose halo cells are cleared; default: every axis.
        """
        for ax in self._axes(axis):
            if self._layout.halo[ax]:
                self._data[self._region(ax, "lower_halo")] = 0
                self._data[self._region(ax, "upper_halo")] = 0

    # ------------------------------------------------------------------ #
    # Elementwise operations

    def _check_compatibility(self, other: DistributedArray) -> None:
        """Raise unless ``other`` has the same layout."""
        if self._layout == other._layout:
            return
        if self.shape != other.shape:
            raise ValueError(f"Shapes must match: {self.shape} and {other.shape}")
        raise ValueError(
            f"Layouts must match: {self._layout!r} and {other._layout!r}",
        )

    def _axis_is_whole(self, axis: int) -> bool:
        """Return whether the storage spans the whole global extent of ``axis``."""
        return self._layout.process_grid[axis] == 1 and self._layout.halo[axis] == 0

    def _coerce_other_data(self, other: Any) -> Array | Number:
        """Return an operand as local data matching this rank's storage.

        Accepted operands, in order of precedence: distributed arrays with the
        same layout, scalars, arrays with the storage shape (halos included),
        arrays that broadcast only along axes the storage spans whole (applied
        to the storage directly), and arrays that broadcast to the global shape
        (sliced to the owned block, with zero halos).
        """
        if isinstance(other, DistributedArray):
            self._check_compatibility(other)
            return other._data
        if xp.isscalar(other):
            return other

        other = xp.asarray(other)
        if other.ndim == 0 or other.shape == self._data.shape:
            return other
        if other.ndim > self.ndim:
            raise ValueError(
                f"operand with shape {other.shape} has more dimensions than the "
                f"distributed array with shape {self.shape}",
            )
        try:
            global_view = xp.broadcast_to(other, self.shape)
        except ValueError as exc:
            raise ValueError(
                f"operand with shape {other.shape} cannot be broadcast to the "
                f"global shape {self.shape}",
            ) from exc
        axes = range(self.ndim - other.ndim, self.ndim)
        if all(
            extent == 1 or self._axis_is_whole(axis)
            for axis, extent in zip(axes, other.shape, strict=True)
        ):
            return other  # the same result on every rank as on the global array
        data = xp.zeros(self._data.shape, dtype=other.dtype)
        data[self._layout.interior] = global_view[self._layout.global_slices()]
        return data

    def _interior_of(self, operand: Any) -> Any:
        """Return the part of a coerced operand that lines up with the block."""
        if xp.isscalar(operand) or operand.ndim == 0:
            return operand
        storage_shape = self._layout.storage_shape
        offset = self.ndim - operand.ndim
        return operand[
            tuple(
                self._layout.interior[offset + k]
                if extent == storage_shape[offset + k]
                else slice(None)
                for k, extent in enumerate(operand.shape)
            )
        ]

    def _elementwise(
        self,
        op: Callable,
        inputs: tuple[Any, ...],
        out: tuple[DistributedArray, ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply ``op`` to the blocks of ``inputs``; halo cells are not computed.

        Without ``out`` the results get zero halo cells; with ``out`` the
        results are written into the blocks of ``out``, whose halo cells are
        left as they are.
        """
        local_inputs = [self._interior_of(self._coerce_other_data(x)) for x in inputs]
        where = kwargs.get("where", True)
        if where is not True:
            kwargs["where"] = self._interior_of(self._coerce_other_data(where))
        interior = self._layout.interior
        if out is not None:
            for item in out:
                self._check_compatibility(item)
            views = tuple(item._data[interior] for item in out)
            op(*local_inputs, out=views[0] if len(views) == 1 else views, **kwargs)
            return out[0] if len(out) == 1 else out
        result = op(*local_inputs, **kwargs)
        wrapped = []
        for block in result if isinstance(result, tuple) else (result,):
            block = xp.asarray(block)
            storage = xp.zeros(self._layout.storage_shape, dtype=block.dtype)
            storage[interior] = block
            wrapped.append(self._with_storage(storage))
        return tuple(wrapped) if isinstance(result, tuple) else wrapped[0]

    def _binary_op(self, other: Any, op: Callable) -> Self:
        """Apply an elementwise binary operation."""
        return self._elementwise(op, (self, other))

    def _binary_rop(self, other: Any, op: Callable) -> Self:
        """Apply a reflected elementwise binary operation."""
        return self._elementwise(op, (other, self))

    def _binary_iop(self, other: Any, op: Callable) -> Self:
        """Apply an in-place elementwise binary operation; halo cells are kept."""
        return self._elementwise(op, (self, other), out=(self,))

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> DistributedArray | tuple[DistributedArray, ...] | NotImplementedType:
        """Apply a NumPy ufunc elementwise to the blocks.

        The result gets zero halo cells. ``out=`` accepts distributed arrays
        with the same layout and writes into their blocks in place (their halo
        cells are kept), so ``np.multiply(a, 2.0, out=a)`` allocates nothing.

        Args:
            ufunc: The ufunc being called.
            method: How it is called; only ``"__call__"`` is supported.
            *inputs: The operands; see `_coerce_other_data` for the accepted ones.
            **kwargs: Keyword arguments of the ufunc, ``out`` and ``where`` included.

        Returns:
            A `DistributedArray` (a tuple for ufuncs with several outputs), or
            ``NotImplemented`` for other methods or a non-distributed ``out``.
        """
        if method != "__call__":
            return NotImplemented
        if ufunc is np.matmul:
            # a generalized ufunc, not a function, so __array_function__ never
            # sees it; dispatched here instead, to the same `dot` as np.dot.
            if kwargs.pop("out", None) is not None:
                raise TypeError("np.matmul on distributed arrays takes no out")
            if kwargs:
                return NotImplemented
            return _dot(*inputs)
        template = next(
            (item for item in inputs if isinstance(item, DistributedArray)),
            self,
        )
        out = kwargs.pop("out", None)
        if out is not None and not all(
            isinstance(item, DistributedArray) for item in out
        ):
            return NotImplemented
        return template._elementwise(ufunc, inputs, out=out, **kwargs)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> Array:
        """Return the gathered global array as a NumPy array, for ``np.asarray(a)``.

        Collective. On the CuPy backend the array is copied to the host, as
        NumPy's protocol requires; use `gather` to keep it on the device.

        Args:
            dtype: Convert to this element type.
            copy: Whether to copy; the gathered array is always new, so only
                ``True`` with a ``dtype`` makes a difference.

        Returns:
            The global array, a ``numpy.ndarray``.
        """
        array = xp.to_numpy(self._gather_all())
        if dtype is not None:
            return array.astype(dtype, copy=False if copy is None else copy)
        if copy:
            return array.copy()
        return array

    # ------------------------------------------------------------------ #
    # Reductions

    def _allreduce(self, local_value: Scalar, mpi_op: MPI.Op, name: str) -> Scalar:
        """Combine a per-rank value over all ranks; a host scalar on every rank."""
        local_value = _host(local_value)
        if not self._layout.distributed:
            return local_value
        check_collective(self._layout.comm, name)
        return self._layout.comm.allreduce(local_value, op=mpi_op)

    def _global_reduction(
        self, name: str, op: Callable, mpi_op: MPI.Op, **op_kwargs: Any
    ) -> Scalar:
        """Reduce all blocks to one host scalar on every rank.

        Every rank owns at least one cell of a non-empty array (see `Layout`),
        so each one reduces its own block and the results are combined.
        """
        if mpi_op in (MPI.MIN, MPI.MAX):
            # Raised on every rank alike, so that no rank waits in allreduce.
            if self.size == 0:
                raise ValueError("zero-size array has no minimum or maximum")
            if self.dtype.kind == "c" and self._layout.distributed:
                # MPI has no complex MIN/MAX: compare the candidates, ordered
                # like NumPy orders complex numbers (real part, then imaginary)
                check_collective(self._layout.comm, name)
                candidates = self._layout.comm.allgather(_host(op(self.local)))
                key = lambda z: (z.real, z.imag)  # noqa: E731
                return (
                    min(candidates, key=key)
                    if name == "min"
                    else max(candidates, key=key)
                )
        return self._allreduce(op(self.local, **op_kwargs), mpi_op, name)

    def _axis_result(
        self, axes: tuple[int, ...], keepdims: bool
    ) -> tuple[Layout, list[Box | None], bool]:
        """Return a reduction's result layout, every rank's box in it, and whether it is local.

        "Local" means each rank's partial result already is its block of the
        result. Reducing along axes that are not split leaves every block where it is;
        otherwise the result is split along the remaining split axes (or held
        whole, if they cannot be split).
        """
        layout = self._layout
        kept = [a for a in range(self.ndim) if a not in axes]
        result_axes = list(range(self.ndim)) if keepdims else kept

        def project(bounds: Sequence[tuple[int, int]]) -> Box:
            return tuple(
                (0, 1) if a in axes else (bounds[a][0], bounds[a][1])
                for a in result_axes
            )

        shape = tuple(end for _, end in project([(0, n) for n in self.shape]))
        boxes: list[Box | None] = [
            project(layout.index_bounds_of(r)) for r in range(layout.size)
        ]
        if not layout.distributed:
            return Layout(shape, comm=layout.comm, split=None), boxes, True
        if all(layout.process_grid[a] == 1 for a in axes):
            cuts = [(0, 1) if a in axes else layout.bounds[a] for a in result_axes]
            return Layout(shape, comm=layout.comm, bounds=cuts), boxes, True
        split = [
            k for k, a in enumerate(result_axes) if a in layout.split and a not in axes
        ]
        return self._layout_like(shape, split), boxes, False

    def _reduce_along(
        self,
        name: str,
        axis: int | tuple[int, ...],
        keepdims: bool,
        **kwargs: Any,
    ) -> Any:
        """Reduce along ``axis`` without gathering: a `DistributedArray` (or a scalar).

        Each rank reduces its own block; the partial results go to the ranks
        that own them in the result and are combined there.
        """
        axes = _normalize_axes(axis, self.ndim)
        if name in ("min", "max") and any(self.shape[a] == 0 for a in axes):
            raise ValueError(f"zero-size array to reduction operation {name}")
        if len(axes) == self.ndim and not keepdims:
            return getattr(self, name)(**kwargs)
        local_op, combine = _AXIS_REDUCTIONS[name]
        partial = local_op(self.local, axis=axes, keepdims=keepdims, **kwargs)
        target, boxes, local_only = self._axis_result(axes, keepdims)
        if local_only:
            return self._with_layout(target, partial)
        result = xp.full(
            target.local_shape, _identity(name, partial.dtype), dtype=partial.dtype
        )
        pieces = exchange_boxes(
            target,
            partial,
            boxes[self._layout.rank],
            boxes,
            name=f"{name}(axis={axis!r})",
        )
        for _, box, piece in pieces:
            index = local_index(box, target.index_bounds)
            result[index] = combine(result[index], piece)
        return self._with_layout(target, result)

    def _var_along(
        self,
        axis: int | tuple[int, ...],
        dtype: DTypeLike | None,
        ddof: int,
        keepdims: bool,
    ) -> Any:
        """Return the variance along ``axis``: blocks combined with Chan et al.'s update."""
        axes = _normalize_axes(axis, self.ndim)
        if len(axes) == self.ndim and not keepdims:
            return self.var(dtype=dtype, ddof=ddof)
        local = self.local
        total = math.prod(self.shape[a] for a in axes)
        count = math.prod(local.shape[a] for a in axes)
        mean = xp.sum(local, axis=axes, keepdims=True, dtype=dtype) / max(count, 1)
        deviation = local - mean
        if deviation.dtype.kind == "c":
            squares = deviation.real**2 + deviation.imag**2
        else:
            squares = deviation * deviation
        m2 = xp.sum(squares, axis=axes, keepdims=keepdims, dtype=dtype)
        if not keepdims:
            mean = mean.reshape(m2.shape)
        target, boxes, local_only = self._axis_result(axes, keepdims)
        if local_only:
            return self._with_layout(target, m2 / (total - ddof))
        layout = self._layout
        counts = [
            math.prod(
                end - start
                for a, (start, end) in enumerate(layout.index_bounds_of(r))
                if a in axes
            )
            for r in range(layout.size)
        ]
        dtype_ = np.result_type(mean.dtype, m2.dtype)
        stacked = xp.stack([mean.astype(dtype_), m2.astype(dtype_)])
        seen = xp.zeros(target.local_shape)
        state_mean = xp.zeros(target.local_shape, dtype=dtype_)
        state_m2 = xp.zeros(target.local_shape, dtype=dtype_)
        pieces = exchange_boxes(
            target,
            stacked,
            boxes[layout.rank],
            boxes,
            lead=(2,),
            name=f"var(axis={axis!r})",
        )
        for source, box, piece in pieces:
            index = local_index(box, target.index_bounds)
            n = counts[source]
            delta = piece[0] - state_mean[index]
            together = seen[index] + n
            state_m2[index] += (
                piece[1] + xp.abs(delta) ** 2 * seen[index] * n / together
            )
            state_mean[index] += delta * n / together
            seen[index] = together
        return self._with_layout(target, xp.real(state_m2) / (total - ddof))

    def _arg_extremum(self, name: str, axis: int | None, keepdims: bool) -> Any:
        """Return ``argmin``/``argmax``: a global flat index, or indices along ``axis``."""
        if self.dtype.kind == "c":
            raise TypeError(f"{name} is not supported for dtype {self.dtype}")
        if self.size == 0:
            raise ValueError(f"attempt to get {name} of an empty sequence")
        local = self.local
        is_min = name == "argmin"
        layout = self._layout
        if axis is None:
            position = np.unravel_index(int(getattr(xp, name)(local)), local.shape)
            value = _host(local[position])
            index = int(
                np.ravel_multi_index(
                    tuple(
                        int(p) + start
                        for p, (start, _) in zip(
                            position, layout.index_bounds, strict=True
                        )
                    ),
                    self.shape,
                )
            )
            candidates = [(value, index)]
            if layout.distributed:
                check_collective(layout.comm, name)
                candidates = layout.comm.allgather((value, index))
            return min(candidates, key=lambda c: _arg_key(c, is_min))[1]
        (axis,) = _normalize_axes(axis, self.ndim)
        arg = getattr(xp, name)(local, axis=axis, keepdims=True)
        values = xp.take_along_axis(local, arg, axis=axis)
        indices = arg + layout.index_bounds[axis][0]
        if not keepdims:
            values, indices = values.squeeze(axis), indices.squeeze(axis)
        target, boxes, local_only = self._axis_result((axis,), keepdims)
        if local_only:
            return self._with_layout(target, indices.astype(np.int64))
        dtype_ = np.float64 if self.dtype.kind == "f" else np.int64
        stacked = xp.stack([values.astype(dtype_), indices.astype(dtype_)])
        best = xp.zeros((2, *target.local_shape), dtype=dtype_)
        empty = xp.ones(target.local_shape, dtype=bool)
        pieces = exchange_boxes(
            target,
            stacked,
            boxes[layout.rank],
            boxes,
            lead=(2,),
            name=f"{name}(axis={axis})",
        )
        for _, box, piece in pieces:
            index = local_index(box, target.index_bounds)
            current = best[(slice(None), *index)]
            better = empty[index] | _better(piece, current, is_min)
            best[(slice(None), *index)] = xp.where(better, piece, current)
            empty[index] = False
        return self._with_layout(target, best[1].astype(np.int64))

    def _finish(self, value: Scalar, keepdims: bool) -> Scalar | Array:
        """Apply ``keepdims`` to the result of a whole-array reduction."""
        if keepdims:
            return xp.full((1,) * self.ndim, value)
        return value

    @staticmethod
    def _reject_out(out: object) -> None:
        """Raise for NumPy's ``out=`` argument, which reductions do not support."""
        if out is not None:
            raise TypeError("DistributedArray reductions do not support out=")

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the sum of the elements.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to sum along, on the gathered array; ``None``
                sums the whole array without gathering.
            dtype: Type of the accumulator and the result.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is None:
            return self._finish(
                self._global_reduction("sum", xp.sum, MPI.SUM, dtype=dtype), keepdims
            )
        return self._reduce_along("sum", axis, keepdims, dtype=dtype)

    def prod(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the product of the elements.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to multiply along, on the gathered array;
                ``None`` reduces the whole array without gathering.
            dtype: Type of the accumulator and the result.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is None:
            return self._finish(
                self._global_reduction("prod", xp.prod, MPI.PROD, dtype=dtype), keepdims
            )
        return self._reduce_along("prod", axis, keepdims, dtype=dtype)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the smallest element.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given, or for complex values.
            ValueError: If the array has no elements.
        """
        self._reject_out(out)
        if axis is None:
            return self._finish(
                self._global_reduction("min", xp.min, MPI.MIN), keepdims
            )
        return self._reduce_along("min", axis, keepdims)

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the largest element.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given, or for complex values.
            ValueError: If the array has no elements.
        """
        self._reject_out(out)
        if axis is None:
            return self._finish(
                self._global_reduction("max", xp.max, MPI.MAX), keepdims
            )
        return self._reduce_along("max", axis, keepdims)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the mean of the elements.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to average along, on the gathered array;
                ``None`` reduces the whole array without gathering.
            dtype: Type of the accumulator and the result.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is None:
            return self._finish(self.sum(dtype=dtype) / self.size, keepdims)
        total = self._reduce_along("sum", axis, keepdims, dtype=dtype)
        return total / math.prod(
            self.shape[a] for a in _normalize_axes(axis, self.ndim)
        )

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the variance of the elements.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            dtype: Type of the accumulator and the result.
            out: Not supported; must be ``None``.
            ddof: Delta degrees of freedom; the divisor is ``size - ddof``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is not None:
            return self._var_along(axis, dtype, ddof, keepdims)
        # Each rank's count, mean and sum of squared deviations, combined in
        # one collective with Chan et al.'s parallel update (numerically like
        # two passes, but one round of communication).
        local = self.local
        count = int(local.size)
        local_mean = _host(xp.sum(local, dtype=dtype)) / count if count else 0.0
        deviation = local - local_mean
        if deviation.dtype.kind == "c":
            squares = xp.sum(deviation.real**2 + deviation.imag**2, dtype=dtype)
        else:
            squares = xp.sum(deviation * deviation, dtype=dtype)
        squares = _host(squares)
        stats = [(count, local_mean, squares)]
        if self._layout.distributed:
            check_collective(self._layout.comm, "var")
            stats = self._layout.comm.allgather(stats[0])
        total, mean, m2 = 0, 0.0, 0.0
        for n, block_mean, block_m2 in stats:
            delta = block_mean - mean
            mean = mean + delta * n / (total + n)
            m2 = m2 + block_m2 + abs(delta) ** 2 * total * n / (total + n)
            total += n
        return self._finish(m2 / (total - ddof), keepdims)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the standard deviation of the elements.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            dtype: Type of the accumulator and the result.
            out: Not supported; must be ``None``.
            ddof: Delta degrees of freedom; the divisor is ``size - ddof``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a host scalar, the same on every rank;
            otherwise a NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        variance = self.var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )
        if isinstance(variance, DistributedArray) or (axis is None and not keepdims):
            return np.sqrt(variance)
        return xp.sqrt(variance)

    def all(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> bool | Array:
        """Return whether every element is true.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a ``bool``, the same on every rank; otherwise a
            NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is None:
            value = self._allreduce(bool(xp.all(self.local)), MPI.LAND, "all")
            return self._finish(value, keepdims)
        return self._reduce_along("all", axis, keepdims)

    def any(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> bool | Array:
        """Return whether any element is true.

        Collective. Halo cells are not included.

        Args:
            axis: Axis or axes to reduce along, on the gathered array; ``None``
                reduces the whole array without gathering.
            out: Not supported; must be ``None``.
            keepdims: Keep the reduced axes with length one.

        Returns:
            With ``axis=None``, a ``bool``, the same on every rank; otherwise a
            NumPy or CuPy array on every rank.

        Raises:
            TypeError: If ``out`` is given.
        """
        self._reject_out(out)
        if axis is None:
            value = self._allreduce(bool(xp.any(self.local)), MPI.LOR, "any")
            return self._finish(value, keepdims)
        return self._reduce_along("any", axis, keepdims)

    def argmin(
        self, axis: int | None = None, out: None = None, *, keepdims: bool = False
    ) -> Any:
        """Return the index of the smallest element, like ``numpy.argmin``.

        Collective. Without ``axis``, the index into the flattened global array
        (a host int, the same on every rank); with ``axis``, a `DistributedArray`
        of indices along it. The first occurrence wins, NaN first, as in NumPy.

        Args:
            axis: The axis to search along; ``None``: the whole array.
            out: Not supported; must be ``None``.
            keepdims: Keep the searched axis with length one.

        Raises:
            TypeError: If ``out`` is given, or for complex values.
            ValueError: If the array is empty.
        """
        self._reject_out(out)
        return self._arg_extremum("argmin", axis, keepdims)

    def argmax(
        self, axis: int | None = None, out: None = None, *, keepdims: bool = False
    ) -> Any:
        """Return the index of the largest element, like ``numpy.argmax``.

        Collective; see `argmin`.

        Args:
            axis: The axis to search along; ``None``: the whole array.
            out: Not supported; must be ``None``.
            keepdims: Keep the searched axis with length one.

        Raises:
            TypeError: If ``out`` is given, or for complex values.
            ValueError: If the array is empty.
        """
        self._reject_out(out)
        return self._arg_extremum("argmax", axis, keepdims)

    def _accumulate(self, name: str, axis: int | None, dtype: DTypeLike | None) -> Self:
        """Return the running sum or product along ``axis``; see `cumsum`."""
        if axis is None:
            if self.ndim != 1:
                raise TypeError(
                    f"{name} without an axis flattens the array, which a distributed "
                    "array does not do; give an axis, or use gather()",
                )
            axis = 0
        (axis,) = _normalize_axes(axis, self.ndim)
        layout = self._layout
        running = getattr(xp, name)(self.local, axis=axis, dtype=dtype)
        if layout.distributed and layout.process_grid[axis] > 1 and running.size:
            # add the totals of the blocks before this one along the axis
            last = xp.to_numpy(xp.take(running, [-1], axis=axis))
            check_collective(layout.comm, f"{name}(axis={axis})")
            totals = layout.comm.allgather((layout.process_coord, last))
            mine = layout.process_coord
            before = [
                total
                for coord, total in totals
                if coord[axis] < mine[axis]
                and all(
                    c == m
                    for k, (c, m) in enumerate(zip(coord, mine, strict=True))
                    if k != axis
                )
            ]
            if before:
                offset = before[0]
                for total in before[1:]:
                    offset = offset + total if name == "cumsum" else offset * total
                offset = xp.asarray(offset)
                running = running + offset if name == "cumsum" else running * offset
        return self._with_layout(layout, running)

    def cumsum(self, axis: int | None = None, dtype: DTypeLike | None = None) -> Self:
        """Return the cumulative sum along ``axis``, a new array with this layout.

        Each rank sums its block, then adds the totals of the blocks before it
        along the axis (one small ``allgather``); nothing is gathered.
        Collective.

        Args:
            axis: The axis; ``None`` only for 1-D arrays (NumPy would flatten).
            dtype: The type of the accumulator and the result.

        Raises:
            TypeError: Without ``axis`` for more than one dimension.
        """
        return self._accumulate("cumsum", axis, dtype)

    def cumprod(self, axis: int | None = None, dtype: DTypeLike | None = None) -> Self:
        """Return the cumulative product along ``axis``; see `cumsum`.

        Args:
            axis: The axis; ``None`` only for 1-D arrays (NumPy would flatten).
            dtype: The type of the accumulator and the result.

        Raises:
            TypeError: Without ``axis`` for more than one dimension.
        """
        return self._accumulate("cumprod", axis, dtype)

    def __array_function__(
        self, func: Callable, types: tuple, args: tuple, kwargs: dict[str, Any]
    ) -> Any:
        """Run the distributed version of a NumPy function, if there is one.

        Reductions, ``argmin``/``argmax``, ``cumsum``/``cumprod``, ``where``,
        ``clip``, ``round``, ``real``/``imag``/``angle``, ``isclose``,
        ``allclose``, ``array_equal``, ``linalg.norm``, ``vdot``,
        ``dot``/``matmul`` (two 1-D arrays), ``concatenate``/``stack`` (along
        an axis none of the arrays split), ``copy``, the ``*_like`` functions,
        ``shape``/``ndim``/``size`` and ``astype`` work like their NumPy
        versions. NumPy's ufuncs (``sin``, ``maximum``, ``isnan``, ...) work
        through `__array_ufunc__` instead, and are aliased at module level in
        `mpiarray.ufuncs`. Any other NumPy function raises ``TypeError``
        instead of quietly gathering the array; call ``gather()`` first for
        those.

        Args:
            func: The NumPy function.
            types: The argument types that implement this protocol.
            args: The positional arguments.
            kwargs: The keyword arguments.

        Returns:
            The result, or ``NotImplemented`` for unsupported functions.
        """
        handler = _ARRAY_FUNCTIONS.get(func)
        if handler is None:
            return NotImplemented
        return handler(*args, **kwargs)

    def vdot(self, other: DistributedArray) -> Scalar:
        """Return ``sum(conj(self) * other)`` over the whole array, on every rank.

        Like ``numpy.vdot`` on the gathered arrays; halo cells are excluded.
        Collective.

        Args:
            other: An array with the same layout.

        Returns:
            The inner product, a host scalar.

        Raises:
            TypeError: If ``other`` is not a `DistributedArray`.
            ValueError: If the layouts differ.
        """
        if not isinstance(other, DistributedArray):
            raise TypeError("vdot needs another DistributedArray")
        self._check_compatibility(other)
        return self._allreduce(xp.vdot(self.local, other.local), MPI.SUM, "vdot")

    def dot(self, other: DistributedArray) -> Scalar:
        """Return ``sum(self * other)`` over the whole array, on every rank.

        Like ``numpy.dot`` on two 1-D arrays: unlike `vdot`, complex values
        are not conjugated. Halo cells are excluded. Collective.

        Args:
            other: A 1-D array with the same layout.

        Returns:
            The dot product, a host scalar.

        Raises:
            TypeError: If ``other`` is not a `DistributedArray`, or either
                array is not 1-D.
            ValueError: If the layouts differ.
        """
        if not isinstance(other, DistributedArray):
            raise TypeError("dot needs another DistributedArray")
        if self.ndim != 1 or other.ndim != 1:
            raise TypeError("dot on distributed arrays needs two 1-D arrays (vectors)")
        self._check_compatibility(other)
        return self._allreduce(xp.dot(self.local, other.local), MPI.SUM, "dot")

    def norm(self, ord: float = 2) -> float:
        """Return the vector norm of the whole array, on every rank.

        Halo cells are excluded. Collective.

        Args:
            ord: 2 (Euclidean), 1 (sum of absolute values) or ``numpy.inf``
                (largest absolute value).

        Returns:
            The norm.

        Raises:
            ValueError: For any other ``ord``.
        """
        if ord == 2:
            return float(np.sqrt(np.real(self.vdot(self))))
        local = xp.abs(self.local)
        if ord == 1:
            return float(self._allreduce(xp.sum(local), MPI.SUM, "norm(1)"))
        if ord == np.inf:
            local_max = xp.max(local) if local.size else 0.0
            return float(self._allreduce(local_max, MPI.MAX, "norm(inf)"))
        raise ValueError(f"unsupported norm order {ord!r}; use 1, 2 or numpy.inf")

    def allreduce_replicated(self, op: MPI.Op | None = None) -> None:
        """Combine every rank's copy of a replicated array in place, halos included.

        For an array that every rank holds whole (``split=None``), into which
        each rank has deposited only its own particles: afterwards every rank
        holds the total. A no-op on a single rank.

        Collective.

        Args:
            op: The MPI reduction; default: ``MPI.SUM``.

        Raises:
            ValueError: If the array is split, so that the ranks hold different
                blocks rather than copies.
        """
        layout = self._layout
        if layout.size == 1:
            return
        if layout.distributed:
            raise ValueError(
                "allreduce_replicated needs every rank to hold the whole array "
                "(split=None), but this array is split over the ranks",
            )
        data = xp.ascontiguousarray(self._data)
        check_collective(layout.comm, "allreduce_replicated")
        with xp.mpi.mpi_buffer(data, recv=True) as buf:
            layout.comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM if op is None else op)
        if data is not self._data:
            self._data[...] = data

    # ------------------------------------------------------------------ #
    # Python protocols

    def __len__(self) -> int:
        """Return the length of the first axis."""
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the first axis of the gathered array. Collective."""
        return iter(self._gather_all())

    def __bool__(self) -> bool:
        """Return the truth value of a one-element array. Collective.

        Raises:
            ValueError: For more or fewer than one element, as NumPy does.
        """
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is "
                "ambiguous. Use a.any() or a.all()",
            )
        return bool(self.get((0,) * self.ndim))

    def _summary(self, edge: int = 3) -> str:
        """Return this rank's block as text, copying at most ``2 * edge`` values."""
        block = self.local
        if block.size <= 2 * edge:
            return np.array2string(xp.to_numpy(block), separator=", ")
        flat = block.reshape(-1)
        head = np.array2string(xp.to_numpy(flat[:edge]), separator=", ")[1:-1]
        tail = np.array2string(xp.to_numpy(flat[-edge:]), separator=", ")[1:-1]
        return f"[{head}, ..., {tail}] ({block.size} values)"

    def __repr__(self) -> str:
        """Return the layout and this rank's block; no communication."""
        layout = self._layout
        bounds = ", ".join(f"{start}:{end}" for start, end in layout.index_bounds)
        local = self._summary()
        return (
            f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"split={layout.split}, rank {layout.rank} of {layout.size} holds "
            f"[{bounds}]: {local})"
        )

    def __add__(self, other: Any) -> Self:
        """Add elementwise."""
        return self._binary_op(other, xp.add)

    def __radd__(self, other: Any) -> Self:
        """Add elementwise, with reflected operands."""
        return self._binary_rop(other, xp.add)

    def __iadd__(self, other: Any) -> Self:
        """Add elementwise, in place."""
        return self._binary_iop(other, xp.add)

    def __sub__(self, other: Any) -> Self:
        """Subtract elementwise."""
        return self._binary_op(other, xp.subtract)

    def __rsub__(self, other: Any) -> Self:
        """Subtract elementwise, with reflected operands."""
        return self._binary_rop(other, xp.subtract)

    def __isub__(self, other: Any) -> Self:
        """Subtract elementwise, in place."""
        return self._binary_iop(other, xp.subtract)

    def __mul__(self, other: Any) -> Self:
        """Multiply elementwise."""
        return self._binary_op(other, xp.multiply)

    def __rmul__(self, other: Any) -> Self:
        """Multiply elementwise, with reflected operands."""
        return self._binary_rop(other, xp.multiply)

    def __imul__(self, other: Any) -> Self:
        """Multiply elementwise, in place."""
        return self._binary_iop(other, xp.multiply)

    def __truediv__(self, other: Any) -> Self:
        """Divide elementwise."""
        return self._binary_op(other, xp.divide)

    def __rtruediv__(self, other: Any) -> Self:
        """Divide elementwise, with reflected operands."""
        return self._binary_rop(other, xp.divide)

    def __itruediv__(self, other: Any) -> Self:
        """Divide elementwise, in place."""
        return self._binary_iop(other, xp.divide)

    def __floordiv__(self, other: Any) -> Self:
        """Floor-divide elementwise."""
        return self._binary_op(other, xp.floor_divide)

    def __rfloordiv__(self, other: Any) -> Self:
        """Floor-divide elementwise, with reflected operands."""
        return self._binary_rop(other, xp.floor_divide)

    def __pow__(self, other: Any) -> Self:
        """Raise elementwise to a power."""
        return self._binary_op(other, xp.power)

    def __rpow__(self, other: Any) -> Self:
        """Raise elementwise to a power, with reflected operands."""
        return self._binary_rop(other, xp.power)

    def __neg__(self) -> Self:
        """Negate elementwise."""
        return self._elementwise(xp.negative, (self,))

    def __pos__(self) -> Self:
        """Return the values elementwise (a new array)."""
        return self._elementwise(xp.positive, (self,))

    def __abs__(self) -> Self:
        """Return the absolute values."""
        return self._elementwise(xp.abs, (self,))

    def __lt__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.less)

    def __le__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.less_equal)

    def __eq__(self, other: object) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        """Compare elementwise for equality."""
        return self._binary_op(other, xp.equal)

    def __ne__(self, other: object) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        """Compare elementwise for inequality."""
        return self._binary_op(other, xp.not_equal)

    def __gt__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.greater)

    def __ge__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.greater_equal)

    __hash__ = None  # type: ignore[assignment]  # mutable, and == is elementwise


# --------------------------------------------------------------------------- #
# NumPy functions with a distributed version (see DistributedArray.__array_function__)


def _template(*values: Any) -> DistributedArray:
    """Return the first distributed array among ``values``."""
    return next(v for v in values if isinstance(v, DistributedArray))


def _where(condition: Any, x: Any = None, y: Any = None) -> Any:
    if x is None or y is None:
        raise TypeError(
            "np.where(condition) (nonzero) is not supported on distributed arrays"
        )
    return _template(condition, x, y)._elementwise(xp.where, (condition, x, y))


def _clip(a: Any, a_min: Any = None, a_max: Any = None, **kwargs: Any) -> Any:
    low, high = kwargs.pop("min", a_min), kwargs.pop("max", a_max)
    if kwargs:
        raise TypeError(f"np.clip on distributed arrays does not take {sorted(kwargs)}")
    operands = [a] + [v for v in (low, high) if v is not None]

    def clip(x: Any, *bounds: Any) -> Any:
        values = iter(bounds)
        return xp.clip(
            x,
            next(values) if low is not None else None,
            next(values) if high is not None else None,
        )

    return _template(*operands)._elementwise(clip, tuple(operands))


def _isclose(
    a: Any, b: Any, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> Any:
    def isclose(x: Any, y: Any) -> Any:
        return xp.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    return _template(a, b)._elementwise(isclose, (a, b))


def _allclose(
    a: Any, b: Any, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    return bool(_isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan).all())


def _array_equal(a: Any, b: Any, equal_nan: bool = False) -> bool:
    template = _template(a, b)
    other = b if template is a else a
    if np.shape(other) != template.shape:
        return False
    if isinstance(other, DistributedArray) and other.layout != template.layout:
        other = other.redistribute(template.layout)
    if equal_nan:
        return _allclose(template, other, rtol=0, atol=0, equal_nan=True)
    return bool((template == other).all())


def _round(a: DistributedArray, decimals: int = 0) -> Any:
    return a._elementwise(lambda x: x.round(decimals), (a,))


def _norm(
    x: DistributedArray, ord: Any = None, axis: Any = None, keepdims: bool = False
) -> Any:
    if axis is not None or keepdims:
        raise TypeError(
            "np.linalg.norm on distributed arrays takes no axis or keepdims"
        )
    if ord in (None, "fro") or (x.ndim == 1 and ord == 2):
        return x.norm(2)
    if x.ndim == 1 and ord in (1, np.inf):
        return x.norm(ord)
    raise TypeError(
        f"np.linalg.norm(ord={ord!r}) is not supported for {x.ndim}-D distributed arrays"
    )


def _dot(a: Any, b: Any, out: Any = None) -> Scalar:
    if out is not None:
        raise TypeError("np.dot/np.matmul on distributed arrays take no out")
    template = _template(a, b)
    other = b if template is a else a
    if not isinstance(other, DistributedArray):
        raise TypeError(
            "np.dot/np.matmul on distributed arrays need two DistributedArrays"
        )
    return template.dot(other)


def _axis_matches(a: Layout, b: Layout, skip: int) -> bool:
    """Return whether layouts ``a`` and ``b`` agree everywhere but axis ``skip``."""
    same_comm = a.comm is b.comm or bool(a.comm == b.comm)
    return (
        same_comm
        and a.halo == b.halo
        and a.periodic == b.periodic
        and a.process_grid == b.process_grid
        and all(
            a.bounds[axis] == b.bounds[axis] for axis in range(a.ndim) if axis != skip
        )
    )


def _bounds_layout(
    layout: Layout,
    shape: tuple[int, ...],
    bounds: Sequence[Sequence[int] | None],
    halo: tuple[int, ...],
    periodic: tuple[bool, ...],
) -> Layout:
    """Return a layout of ``shape``, replicated like ``layout`` or split at ``bounds``.

    A non-distributed ``layout`` (one rank, or every rank holding everything)
    has no process grid for explicit ``bounds`` to imply, so it is rebuilt
    with ``split=None`` instead.
    """
    if not layout.distributed:
        return Layout(shape, comm=layout.comm, split=None, halo=halo, periodic=periodic)
    return Layout(shape, comm=layout.comm, bounds=bounds, halo=halo, periodic=periodic)


def _concatenate(arrays: Sequence[Any], axis: int = 0, **kwargs: Any) -> Any:
    if kwargs:
        raise TypeError(
            "np.concatenate on distributed arrays takes no further keyword arguments"
        )
    # NumPy's dispatch needs a relevant argument to route here at all, so
    # ``arrays`` is never empty (``np.concatenate([])`` never reaches this
    # function; it is NumPy's own ValueError, before any dispatch).
    arrays = list(arrays)
    if not all(isinstance(a, DistributedArray) for a in arrays):
        raise TypeError(
            "np.concatenate on distributed arrays needs every array to be one"
        )
    template = arrays[0]
    ndim = template.ndim
    axis = _normalize_axes(axis, ndim)[0]
    layout = template._layout
    if layout.distributed and axis in layout.split:
        raise TypeError(
            "np.concatenate on distributed arrays needs an axis outside the split axes"
        )
    for other in arrays[1:]:
        if other.ndim != ndim or not _axis_matches(other._layout, layout, axis):
            raise ValueError(
                "np.concatenate needs arrays with the same layout except along axis"
            )
    local = xp.concatenate([a.local for a in arrays], axis=axis)
    new_length = local.shape[axis]
    shape = tuple(
        new_length if ax == axis else template.shape[ax] for ax in range(ndim)
    )
    bounds = [
        (0, new_length) if ax == axis else layout.bounds[ax] for ax in range(ndim)
    ]
    target = _bounds_layout(layout, shape, bounds, layout.halo, layout.periodic)
    return template._with_layout(target, local)


def _stack(arrays: Sequence[Any], axis: int = 0, **kwargs: Any) -> Any:
    if kwargs:
        raise TypeError(
            "np.stack on distributed arrays takes no further keyword arguments"
        )
    # see _concatenate: NumPy's dispatch guarantees ``arrays`` is non-empty.
    arrays = list(arrays)
    if not all(isinstance(a, DistributedArray) for a in arrays):
        raise TypeError("np.stack on distributed arrays needs every array to be one")
    template = arrays[0]
    ndim = template.ndim + 1
    axis = _normalize_axes(axis, ndim)[0]
    layout = template._layout
    for other in arrays[1:]:
        if other.shape != template.shape:
            raise ValueError("all input arrays must have the same shape")
        template._check_compatibility(other)
    local = xp.concatenate([xp.expand_dims(a.local, axis) for a in arrays], axis=axis)
    shape: list[int] = []
    bounds: list[tuple[int, ...]] = []
    halo: list[int] = []
    periodic: list[bool] = []
    source = 0
    for ax in range(ndim):
        if ax == axis:
            shape.append(len(arrays))
            bounds.append((0, len(arrays)))
            halo.append(0)
            periodic.append(False)
        else:
            shape.append(template.shape[source])
            bounds.append(layout.bounds[source])
            halo.append(layout.halo[source])
            periodic.append(layout.periodic[source])
            source += 1
    target = _bounds_layout(layout, tuple(shape), bounds, tuple(halo), tuple(periodic))
    return template._with_layout(target, local)


def _like(name: str) -> Callable:
    def create(a: DistributedArray, *args: Any, **kwargs: Any) -> Any:
        from mpiarray import creation

        if (
            kwargs.pop("shape", None) is not None
            or kwargs.pop("subok", True) is not True
        ):
            raise TypeError(f"np.{name} on distributed arrays takes no shape or subok")
        return getattr(creation, name)(a, *args, **kwargs)

    return create


def _method(name: str) -> Callable:
    def call(a: DistributedArray, *args: Any, **kwargs: Any) -> Any:
        return getattr(a, name)(*args, **kwargs)

    return call


_ARRAY_FUNCTIONS: dict[Callable, Callable] = {
    **{
        getattr(np, name): _method(name)
        for name in (
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "var",
            "std",
            "all",
            "any",
            "argmin",
            "argmax",
            "cumsum",
            "cumprod",
            "copy",
        )
    },
    np.amin: _method("min"),
    np.amax: _method("max"),
    np.where: _where,
    np.clip: _clip,
    np.round: _round,
    np.around: _round,
    np.real: lambda a: a._elementwise(lambda x: x.real, (a,)),
    np.imag: lambda a: a._elementwise(lambda x: x.imag, (a,)),
    np.angle: lambda a, deg=False: a._elementwise(lambda x: xp.angle(x, deg=deg), (a,)),
    np.isclose: _isclose,
    np.allclose: _allclose,
    np.array_equal: _array_equal,
    np.linalg.norm: _norm,
    np.vdot: lambda a, b: _template(a, b).vdot(b if _template(a, b) is a else a),
    np.dot: _dot,
    # np.matmul is a generalized ufunc, not a function: see __array_ufunc__.
    np.concatenate: _concatenate,
    np.stack: _stack,
    np.shape: lambda a: a.shape,
    np.ndim: lambda a: a.ndim,
    np.size: lambda a, axis=None: a.size if axis is None else a.shape[axis],
    np.zeros_like: _like("zeros_like"),
    np.ones_like: _like("ones_like"),
    np.empty_like: _like("empty_like"),
    np.full_like: _like("full_like"),
}
if hasattr(np, "astype"):  # pragma: no branch - NumPy 2 (CI has no NumPy 1)
    _ARRAY_FUNCTIONS[np.astype] = lambda a, dtype, copy=True, **_: a.astype(
        dtype, copy=copy
    )
