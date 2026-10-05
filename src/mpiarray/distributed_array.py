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

import math
from collections.abc import Callable, Iterator
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, overload

import cunumpy as xp
import numpy as np
from numpy.typing import DTypeLike

from mpiarray._mpi import MPI
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

# Tag offsets so halo updates and halo accumulations never share a message tag.
_UPDATE_TAG = 0
_ACCUMULATE_TAG = 1000


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
        return self._layout.comm.bcast(value, root=owner)

    def __getitem__(self, index: IndexLike) -> Scalar | Array:
        """Return an element or a selection of the global array, on every rank.

        Collective. An integer per axis is `get`; any other index selects from
        the gathered array.

        Args:
            index: A global index, as for a NumPy array of shape `shape`.

        Returns:
            A host scalar, or a NumPy or CuPy array.
        """
        basic = self._normalize_basic_index(index)
        if basic is not None and all(_is_int(item) for item in basic):
            return self.get(tuple(int(item) for item in basic if _is_int(item)))
        return self._gather_all()[index]

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

    def _assign_basic(self, index: tuple[int | slice, ...], value: Any) -> None:
        """Write ``value`` into the owned part of the global selection ``index``.

        ``value`` must be the same on every rank and broadcast to the shape of
        the selection. No communication happens, and halo cells are untouched.
        """
        if isinstance(value, DistributedArray):
            value = value._gather_all()
        selection_shape = []
        data_index: list[int | slice] = []
        value_index: list[slice] = []
        owned = True
        for axis, (item, (start, end), halo) in enumerate(
            zip(index, self._layout.index_bounds, self._layout.halo, strict=True),
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
                data_index.append(position - start + halo)
                continue

            selected = np.arange(extent)[item]
            selection_shape.append(selected.size)
            # The selection is monotonic, so the owned entries are contiguous.
            positions = np.flatnonzero((selected >= start) & (selected < end))
            if positions.size == 0:
                owned = False
                continue
            first = int(selected[positions[0]]) - start + halo
            last = int(selected[positions[-1]]) - start + halo
            step = item.indices(extent)[2]
            stop = last + step
            data_index.append(slice(first, None if stop < 0 else stop, step))
            value_index.append(slice(int(positions[0]), int(positions[-1]) + 1))

        # Broadcast on every rank, so a bad value raises everywhere alike.
        value = xp.broadcast_to(xp.asarray(value), tuple(selection_shape))
        if owned:
            self._data[tuple(data_index)] = value[tuple(value_index)]

    # ------------------------------------------------------------------ #
    # Halo cells

    def _axes(self, axis: int | None) -> range | tuple[int]:
        """Return the axes a halo method works on."""
        return range(self.ndim) if axis is None else (axis,)

    def _axis_slice(self, axis: int, start: int, stop: int | None) -> tuple[slice, ...]:
        """Return an index selecting ``start:stop`` along ``axis`` and everything else."""
        index = [slice(None)] * self.ndim
        index[axis] = slice(start, stop)
        return tuple(index)

    def _halo_regions(self, axis: int) -> dict[str, tuple[slice, ...]]:
        """Return the halos along ``axis`` and the interior layers next to them."""
        h = self._layout.halo[axis]
        return {
            "lower_halo": self._axis_slice(axis, 0, h),
            "lower_interior": self._axis_slice(axis, h, 2 * h),
            "upper_interior": self._axis_slice(axis, -2 * h, -h),
            "upper_halo": self._axis_slice(axis, -h, None),
        }

    def _is_self_neighbour(self, axis: int) -> bool:
        """Return whether both neighbours along ``axis`` are this rank."""
        left, right = self._layout.neighbours[axis]
        return left == right == self._layout.rank

    def _sendrecv(self, send: Array, dest: int, source: int, tag: int) -> Array:
        """Send ``send`` to ``dest`` and return the matching buffer from ``source``."""
        send = xp.ascontiguousarray(send)
        recv = xp.empty(send.shape, dtype=send.dtype)
        receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
        with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
            self._layout.comm.Sendrecv(
                sendbuf,
                dest=dest,
                sendtag=tag,
                recvbuf=recvbuf,
                source=source,
                recvtag=tag,
            )
        return recv

    def update_halos(self, axis: int | None = None) -> None:
        """Copy the neighbours' boundary values into the halo cells.

        This is what a finite-difference stencil needs before it is applied.
        Axes are updated one after the other, so corner halos are filled too.
        At walls (non-periodic boundaries) the halo cells are left unchanged.

        Collective.

        Args:
            axis: The axis to update; default: every axis.
        """
        for ax in self._axes(axis):
            self._update_halo(ax)

    def _update_halo(self, axis: int) -> None:
        """Update the halo cells along one axis; see `update_halos`."""
        if self._layout.halo[axis] == 0:
            return
        regions = self._halo_regions(axis)
        left, right = self._layout.neighbours[axis]

        if not self._layout.distributed or self._is_self_neighbour(axis):
            if self._layout.periodic[axis]:
                self._data[regions["lower_halo"]] = self._data[
                    regions["upper_interior"]
                ]
                self._data[regions["upper_halo"]] = self._data[
                    regions["lower_interior"]
                ]
            return

        # Upper interior goes right; the left neighbour's upper interior fills our lower halo.
        recv = self._sendrecv(
            self._data[regions["upper_interior"]],
            dest=right,
            source=left,
            tag=_UPDATE_TAG + 2 * axis,
        )
        if left != MPI.PROC_NULL:
            self._data[regions["lower_halo"]] = recv
        # Lower interior goes left; the right neighbour's lower interior fills our upper halo.
        recv = self._sendrecv(
            self._data[regions["lower_interior"]],
            dest=left,
            source=right,
            tag=_UPDATE_TAG + 2 * axis + 1,
        )
        if right != MPI.PROC_NULL:
            self._data[regions["upper_halo"]] = recv

    def accumulate_halos(self, axis: int | None = None) -> None:
        """Add the halo cells into the neighbours' boundary cells, then zero them.

        This is the scatter-add after depositing particles near the edges of a
        block. At walls (non-periodic boundaries) the halo values are dropped.

        Collective.

        Args:
            axis: The axis to accumulate along; default: every axis.
        """
        for ax in self._axes(axis):
            self._accumulate_halo(ax)

    def _accumulate_halo(self, axis: int) -> None:
        """Accumulate the halo cells along one axis; see `accumulate_halos`."""
        if self._layout.halo[axis] == 0:
            return
        regions = self._halo_regions(axis)
        left, right = self._layout.neighbours[axis]

        if not self._layout.distributed or self._is_self_neighbour(axis):
            if self._layout.periodic[axis]:
                lower_halo = self._data[regions["lower_halo"]].copy()
                self._data[regions["lower_interior"]] += self._data[
                    regions["upper_halo"]
                ]
                self._data[regions["upper_interior"]] += lower_halo
            self.clear_halos(axis)
            return

        # Lower halo goes left; the right neighbour's lower halo lands in our upper interior.
        recv = self._sendrecv(
            self._data[regions["lower_halo"]],
            dest=left,
            source=right,
            tag=_ACCUMULATE_TAG + 2 * axis,
        )
        if right != MPI.PROC_NULL:
            self._data[regions["upper_interior"]] += recv
        # Upper halo goes right; the left neighbour's upper halo lands in our lower interior.
        recv = self._sendrecv(
            self._data[regions["upper_halo"]],
            dest=right,
            source=left,
            tag=_ACCUMULATE_TAG + 2 * axis + 1,
        )
        if left != MPI.PROC_NULL:
            self._data[regions["lower_interior"]] += recv
        self.clear_halos(axis)

    def clear_halos(self, axis: int | None = None) -> None:
        """Set the halo cells to zero; no communication.

        Args:
            axis: The axis whose halo cells are cleared; default: every axis.
        """
        for ax in self._axes(axis):
            if self._layout.halo[ax]:
                regions = self._halo_regions(ax)
                self._data[regions["lower_halo"]] = 0
                self._data[regions["upper_halo"]] = 0

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

    def _new_like(self, storage: Array) -> Self:
        """Return an array with this layout around a ufunc result (always new memory)."""
        return self._with_storage(xp.asarray(storage))

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

    def _binary_op(self, other: Any, op: Callable) -> Self:
        """Apply an elementwise binary operation."""
        return self._new_like(op(self._data, self._coerce_other_data(other)))

    def _binary_rop(self, other: Any, op: Callable) -> Self:
        """Apply a reflected elementwise binary operation."""
        return self._new_like(op(self._coerce_other_data(other), self._data))

    def _binary_iop(self, other: Any, op: Callable) -> Self:
        """Apply an in-place elementwise binary operation."""
        op(self._data, self._coerce_other_data(other), out=self._data)
        return self

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> DistributedArray | tuple[DistributedArray, ...] | NotImplementedType:
        """Apply a NumPy ufunc elementwise to the local storage.

        ``out=`` accepts distributed arrays with the same layout and writes in
        place, so ``np.multiply(a, 2.0, out=a)`` allocates nothing.

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
        template = next(
            (item for item in inputs if isinstance(item, DistributedArray)),
            self,
        )
        out = kwargs.pop("out", None)
        if out is not None:
            if not all(isinstance(item, DistributedArray) for item in out):
                return NotImplemented
            for item in out:
                template._check_compatibility(item)
            kwargs["out"] = tuple(item._data for item in out)
        where = kwargs.get("where", True)
        if where is not True:
            kwargs["where"] = template._coerce_other_data(where)
        local_inputs = [template._coerce_other_data(item) for item in inputs]

        result = getattr(ufunc, method)(*local_inputs, **kwargs)
        if out is not None:
            return out[0] if len(out) == 1 else out
        if isinstance(result, tuple):
            return tuple(template._new_like(item) for item in result)
        return template._new_like(result)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> Array:
        """Return the gathered global array, for NumPy functions without a method here.

        Collective.

        Args:
            dtype: Convert to this element type.
            copy: Whether to copy; the gathered array is always new, so only
                ``True`` with a ``dtype`` makes a difference.

        Returns:
            The global array.
        """
        array = self._gather_all()
        if dtype is not None:
            return array.astype(dtype, copy=False if copy is None else copy)
        if copy:
            return array.copy()
        return array

    # ------------------------------------------------------------------ #
    # Reductions

    def _allreduce(self, local_value: Scalar, mpi_op: MPI.Op) -> Scalar:
        """Combine a per-rank value over all ranks; a host scalar on every rank."""
        local_value = _host(local_value)
        if not self._layout.distributed:
            return local_value
        return self._layout.comm.allreduce(local_value, op=mpi_op)

    def _extremum_identity(self, mpi_op: MPI.Op) -> Scalar:
        """Return the neutral element of ``MPI.MIN``/``MPI.MAX`` for this dtype."""
        dtype: np.dtype[Any] = np.dtype(self.dtype)
        is_min = mpi_op == MPI.MIN
        if dtype.kind == "b":
            return np.True_ if is_min else np.False_
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            return dtype.type(info.max if is_min else info.min)
        if dtype.kind == "f":
            return dtype.type(np.inf if is_min else -np.inf)
        raise TypeError(f"min/max are not supported for dtype {dtype}")

    def _global_reduction(
        self, op: Callable, mpi_op: MPI.Op, **op_kwargs: Any
    ) -> Scalar:
        """Reduce all owned cells to one host scalar on every rank.

        A rank that owns no cells contributes the neutral element, so layouts
        that leave some ranks empty still reduce correctly.
        """
        identity = None
        if mpi_op in (MPI.MIN, MPI.MAX):
            # Raised on every rank alike, so that no rank waits in allreduce.
            if self.size == 0:
                raise ValueError("zero-size array has no minimum or maximum")
            identity = self._extremum_identity(mpi_op)
        local = self.local
        if identity is not None and local.size == 0:
            value = identity
        else:
            # sum/prod of an empty block already return their neutral element.
            value = op(local, **op_kwargs)
        return self._allreduce(value, mpi_op)

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
                self._global_reduction(xp.sum, MPI.SUM, dtype=dtype), keepdims
            )
        return xp.sum(self._gather_all(), axis=axis, dtype=dtype, keepdims=keepdims)

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
                self._global_reduction(xp.prod, MPI.PROD, dtype=dtype), keepdims
            )
        return xp.prod(self._gather_all(), axis=axis, dtype=dtype, keepdims=keepdims)

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
            return self._finish(self._global_reduction(xp.min, MPI.MIN), keepdims)
        return xp.min(self._gather_all(), axis=axis, keepdims=keepdims)

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
            return self._finish(self._global_reduction(xp.max, MPI.MAX), keepdims)
        return xp.max(self._gather_all(), axis=axis, keepdims=keepdims)

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
        return xp.mean(self._gather_all(), axis=axis, dtype=dtype, keepdims=keepdims)

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
            return xp.var(
                self._gather_all(),
                axis=axis,
                dtype=dtype,
                correction=ddof,  # the array API name of numpy's ddof
                keepdims=keepdims,
            )
        mean = self.mean(dtype=dtype)
        local = xp.sum(xp.abs(self.local - mean) ** 2, dtype=dtype)
        return self._finish(
            self._allreduce(local, MPI.SUM) / (self.size - ddof), keepdims
        )

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
        if axis is None and not keepdims:
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
            value = self._allreduce(bool(xp.all(self.local)), MPI.LAND)
            return self._finish(value, keepdims)
        return xp.all(self._gather_all(), axis=axis, keepdims=keepdims)

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
            value = self._allreduce(bool(xp.any(self.local)), MPI.LOR)
            return self._finish(value, keepdims)
        return xp.any(self._gather_all(), axis=axis, keepdims=keepdims)

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
        return self._allreduce(xp.vdot(self.local, other.local), MPI.SUM)

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
            return float(self._allreduce(xp.sum(local), MPI.SUM))
        if ord == np.inf:
            local_max = xp.max(local) if local.size else 0.0
            return float(self._allreduce(local_max, MPI.MAX))
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

    def __repr__(self) -> str:
        """Return the layout and this rank's block; no communication."""
        layout = self._layout
        bounds = ", ".join(f"{start}:{end}" for start, end in layout.index_bounds)
        local = np.array2string(xp.to_numpy(self.local), separator=", ", threshold=20)
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
        return self._with_storage(-self._data)

    def __pos__(self) -> Self:
        """Return a copy."""
        return self.copy()

    def __abs__(self) -> Self:
        """Return the absolute values."""
        return self._with_storage(xp.abs(self._data))

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
