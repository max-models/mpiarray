"""MPI-distributed arrays with optional halo cells."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import cunumpy as xp
import numpy as np
from mpi4py import MPI
from numpy.typing import DTypeLike

from mpiarray.domain_decomposition import DomainDecomposition

if TYPE_CHECKING:
    from typing_extensions import Self  # typing.Self needs Python 3.11

# A NumPy or CuPy array.
Array: TypeAlias = Any
Number: TypeAlias = int | float
IndexLike: TypeAlias = int | slice | tuple[int | slice, ...]
# Scalar returned by a global reduction (numpy/cupy scalar, Python number or bool).
Scalar = Any

# Tag offsets so ghost fills and halo accumulations never share a message tag.
_FILL_TAG = 0
_EXCHANGE_TAG = 1000


class DistributedArray(DomainDecomposition):
    """Represent an MPI-distributed array with optional halo cells."""

    __array_priority__ = 1000

    def __init__(
        self,
        shape: tuple[int, ...],
        comm: MPI.Comm | None,
        data: Array | None = None,
        data_local: Array | None = None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike = float,
    ) -> None:
        """Initialize the distributed array instance."""
        if ndim is None and decompose is None:
            ndim = len(shape)
        super().__init__(
            comm=comm,
            decompose=decompose,
            dim_order=dim_order,
            periodic=periodic,
            ndim=ndim,
        )

        self._num_ghostpoints = num_ghostpoints
        self._dtype = xp.dtype(dtype)
        if ghost_axes is None:
            ghost_axes = [True] * self.ndim
        self._ghost_axes = list(ghost_axes)
        if len(self.ghost_axes) != self.ndim:
            raise ValueError(
                f"ghost_axes has {len(self.ghost_axes)} entries, expected {self.ndim}",
            )
        # shape is e.g. (Nx, Ny, Nz, Ncomponents)
        self._shape = tuple(shape)
        if len(self.shape) != self.ndim:
            raise ValueError(f"shape {self.shape} does not have {self.ndim} dimensions")

        self._shape_local = self.get_shape_local(self.mpi_rank)
        self._proc_index_bounds = self.get_index_bounds(self.mpi_rank)

        shape_local_with_ghostpoints = tuple(
            sh + 2 * num_ghostpoints if ghost_axis else sh
            for sh, ghost_axis in zip(self.shape_local, self.ghost_axes, strict=True)
        )
        self._data = xp.zeros(shape_local_with_ghostpoints, dtype=self.dtype)

        if data is not None:
            self.fill(data=data)
        if data_local is not None:
            self.fill_local(data=data_local)

    @property
    def _distributed_comm(self) -> MPI.Comm:
        """Return the communicator of a distributed array (never ``None`` there)."""
        assert self.comm is not None
        return self.comm

    @property
    def is_distributed(self) -> bool:
        """Return whether ranks hold different parts of the array.

        ``False`` for a single rank, no communicator, or a replicated layout
        (no decomposed axis), where every rank holds the whole array.
        """
        return self.comm is not None and self.mpi_size > 1 and not self.replicated

    # Narrower than DomainDecomposition.get_index_bounds: the shape is this array's.
    def get_index_bounds(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, rank: int | None = None
    ) -> list[tuple[int, int]]:
        """Return the global ``(start, end)`` index bounds owned by ``rank``."""
        return super().get_index_bounds(self.shape, rank)

    def get_shape_local(self, rank: int | None = None) -> tuple[int, ...]:
        """Return the interior array shape owned by ``rank``."""
        return tuple(end - start for start, end in self.get_index_bounds(rank))

    def _wrap_local(self, data: Array) -> Self:
        """Return an array with this layout that takes ownership of ``data``.

        No copy is made and the decomposition is not recomputed, so this is the
        cheap path for creating results of elementwise operations.
        """
        if data.shape != self._data.shape:
            raise ValueError(
                f"local data shape {data.shape} does not match the local storage "
                f"shape {self._data.shape}",
            )
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result._data = data
        result._dtype = data.dtype
        return result

    def copy(self) -> Self:
        """Return a copy of this object."""
        return self._wrap_local(self._data.copy())

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        comm: MPI.Comm | None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike = float,
    ) -> Self:
        """Create an uninitialized distributed array."""
        result = cls(
            shape=shape,
            comm=comm,
            num_ghostpoints=num_ghostpoints,
            ghost_axes=ghost_axes,
            decompose=decompose,
            periodic=periodic,
            dim_order=dim_order,
            ndim=ndim,
            dtype=dtype,
        )
        result._data = xp.empty(result._data.shape, dtype=result.dtype)
        return result

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        comm: MPI.Comm | None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike = float,
    ) -> Self:
        """Create a distributed array filled with zeros."""
        return cls(
            shape=shape,
            comm=comm,
            num_ghostpoints=num_ghostpoints,
            ghost_axes=ghost_axes,
            decompose=decompose,
            periodic=periodic,
            dim_order=dim_order,
            ndim=ndim,
            dtype=dtype,
        )

    @classmethod
    def full(
        cls,
        shape: tuple[int, ...],
        fill_value: Number,
        comm: MPI.Comm | None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        """Create a distributed array filled with a scalar value."""
        if dtype is None:
            dtype = xp.asarray(fill_value).dtype
        result = cls.zeros(
            shape=shape,
            comm=comm,
            num_ghostpoints=num_ghostpoints,
            ghost_axes=ghost_axes,
            decompose=decompose,
            periodic=periodic,
            dim_order=dim_order,
            ndim=ndim,
            dtype=dtype,
        )
        result.data[...] = fill_value
        return result

    @classmethod
    def ones(
        cls,
        shape: tuple[int, ...],
        comm: MPI.Comm | None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike = float,
    ) -> Self:
        """Create a distributed array filled with ones."""
        return cls.full(
            shape=shape,
            fill_value=1,
            comm=comm,
            num_ghostpoints=num_ghostpoints,
            ghost_axes=ghost_axes,
            decompose=decompose,
            periodic=periodic,
            dim_order=dim_order,
            ndim=ndim,
            dtype=dtype,
        )

    @classmethod
    def from_array(
        cls,
        data: Array,
        comm: MPI.Comm | None,
        num_ghostpoints: int = 0,
        ghost_axes: list[bool] | None = None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        """Create a distributed array from a global ndarray."""
        data = xp.asarray(data, dtype=dtype)
        return cls(
            shape=data.shape,
            comm=comm,
            data=data,
            num_ghostpoints=num_ghostpoints,
            ghost_axes=ghost_axes,
            decompose=decompose,
            periodic=periodic,
            dim_order=dim_order,
            ndim=ndim,
            dtype=data.dtype,
        )

    def fill_local(self, data: Array) -> None:
        """Fill the local storage (including halo cells) from local data."""
        if data.dtype != self.dtype:
            raise ValueError(f"dtype {data.dtype} does not match {self.dtype}")
        if data.shape != self.data.shape:
            raise ValueError(
                f"local data shape {data.shape} does not match {self.data.shape}",
            )
        self.data[:] = data

    def _global_slices(self, rank: int | None = None) -> tuple[slice, ...]:
        """Return slices selecting the block owned by ``rank`` in a global array."""
        return tuple(slice(start, end) for start, end in self.get_index_bounds(rank))

    def fill(self, data: Array) -> None:
        """Fill the distributed array from global data; halo cells are zeroed."""
        if data.dtype != self.dtype:
            raise ValueError(f"dtype {data.dtype} does not match {self.dtype}")
        if data.shape != self.shape:
            raise ValueError(
                f"global data shape {data.shape} does not match {self.shape}"
            )
        self._data.fill(0)
        self._data[self.get_local_slices()] = data[self._global_slices()]

    def get_local_slices(self) -> tuple[slice, ...]:
        """Return slices selecting the local interior region."""
        g = self.num_ghostpoints
        if g == 0:
            return (slice(None),) * self.ndim
        return tuple(
            slice(g, -g) if ghost_axis else slice(None)
            for ghost_axis in self.ghost_axes
        )

    def get_local_block(self) -> Array:
        """Return the local interior block without halo cells, as a contiguous array."""
        return xp.ascontiguousarray(self._data[self.get_local_slices()])

    @property
    def local(self) -> Array:
        """Return a writable view of the local interior without halo cells."""
        return self._data[self.get_local_slices()]

    @property
    def local_with_halos(self) -> Array:
        """Return the writable local array including halo cells."""
        return self._data

    def to_ndarray(self) -> Array:
        """Gather the distributed array into a global ndarray on every rank."""
        if not self.is_distributed:
            return self._data[self.get_local_slices()].copy()

        counts = [
            math.prod(self.get_shape_local(rank)) for rank in range(self.mpi_size)
        ]
        send = self.get_local_block()
        recv = xp.empty(sum(counts), dtype=self.dtype)
        receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
        with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
            self._distributed_comm.Allgatherv(sendbuf, [recvbuf, counts])

        full = xp.empty(self.shape, dtype=self.dtype)
        offset = 0
        for rank, count in enumerate(counts):
            block = recv[offset : offset + count]
            full[self._global_slices(rank)] = block.reshape(self.get_shape_local(rank))
            offset += count
        return full

    def _axis_slice(self, dim: int, start: int, stop: int | None) -> tuple[slice, ...]:
        """Return an index selecting ``start:stop`` along ``dim`` and everything else."""
        index = [slice(None)] * self._data.ndim
        index[dim] = slice(start, stop)
        return tuple(index)

    def _halo_regions(self, dim: int) -> dict[str, tuple[slice, ...]]:
        """Return the four halo-related regions along ``dim``."""
        g = self.num_ghostpoints
        return {
            "lower_halo": self._axis_slice(dim, 0, g),
            "lower_interior": self._axis_slice(dim, g, 2 * g),
            "upper_interior": self._axis_slice(dim, -2 * g, -g),
            "upper_halo": self._axis_slice(dim, -g, None),
        }

    def _is_self_neighbour(self, dim: int) -> bool:
        """Return whether both neighbours along ``dim`` are this rank (periodic, undivided)."""
        left, right = self.neighbour_ranks[dim]
        return left == right == self.mpi_rank

    def _sendrecv(
        self,
        send: Array,
        dest: int,
        source: int,
        tag: int,
    ) -> Array:
        """Send ``send`` to ``dest`` and return the matching buffer from ``source``."""
        send = xp.ascontiguousarray(send)
        recv = xp.empty(send.shape, dtype=send.dtype)
        receiving = xp.mpi.mpi_buffer(recv, send=False, recv=True)
        with xp.mpi.mpi_buffer(send) as sendbuf, receiving as recvbuf:
            self._distributed_comm.Sendrecv(
                sendbuf,
                dest=dest,
                sendtag=tag,
                recvbuf=recvbuf,
                source=source,
                recvtag=tag,
            )
        return recv

    def exchange_halo(self, dim: int) -> None:
        """Accumulate halo cells along ``dim`` into the neighbours' interior cells.

        This is the scatter-add used after particle deposition: the lower halo
        is added to the left neighbour's upper interior, the upper halo to the
        right neighbour's lower interior. Halos along ``dim`` are zeroed
        afterwards. At physical (non-periodic) boundaries the halo values are
        discarded.
        """
        if self.num_ghostpoints == 0 or not self.ghost_axes[dim]:
            return

        regions = self._halo_regions(dim)
        left, right = self.neighbour_ranks[dim]

        if not self.is_distributed or self._is_self_neighbour(dim):
            if self.periodic[dim]:
                lower_halo = self._data[regions["lower_halo"]].copy()
                self._data[regions["lower_interior"]] += self._data[
                    regions["upper_halo"]
                ]
                self._data[regions["upper_interior"]] += lower_halo
            self.clear_halos(dim)
            return

        # Lower halo goes left; the right neighbour's lower halo lands in our upper interior.
        recv = self._sendrecv(
            self._data[regions["lower_halo"]],
            dest=left,
            source=right,
            tag=_EXCHANGE_TAG + 2 * dim,
        )
        if right != MPI.PROC_NULL:
            self._data[regions["upper_interior"]] += recv

        # Upper halo goes right; the left neighbour's upper halo lands in our lower interior.
        recv = self._sendrecv(
            self._data[regions["upper_halo"]],
            dest=right,
            source=left,
            tag=_EXCHANGE_TAG + 2 * dim + 1,
        )
        if left != MPI.PROC_NULL:
            self._data[regions["lower_interior"]] += recv

        self.clear_halos(dim)

    def clear_halos(self, dim: int) -> None:
        """Reset halo cells along ``dim`` to zero."""
        if self.num_ghostpoints == 0 or not self.ghost_axes[dim]:
            return
        regions = self._halo_regions(dim)
        self._data[regions["lower_halo"]] = 0
        self._data[regions["upper_halo"]] = 0

    def exchange_halos(self) -> None:
        """Accumulate halo cells into neighbouring interiors in all dimensions."""
        if self.num_ghostpoints == 0:
            return
        for dim in range(self.ndim):
            self.exchange_halo(dim)

    def fill_halo(self, dim: int) -> None:
        """Fill the ghost cells along ``dim`` from the neighbouring ranks.

        Each rank's ghost cells get the boundary interior values of the
        neighbouring rank.

        This is distinct from ``exchange_halo``, which is a scatter-accumulate
        operation for particle deposition.  Use ``fill_halo`` / ``fill_halos``
        before applying finite-difference stencils.

        At physical domain boundaries (PROC_NULL neighbours) the ghost cells
        are left unchanged (typically 0, which encodes a homogeneous Dirichlet BC).
        """
        if self.num_ghostpoints == 0 or not self.ghost_axes[dim]:
            return

        regions = self._halo_regions(dim)
        left, right = self.neighbour_ranks[dim]

        if not self.is_distributed or self._is_self_neighbour(dim):
            if self.periodic[dim]:
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
            tag=_FILL_TAG + 2 * dim,
        )
        if left != MPI.PROC_NULL:
            self._data[regions["lower_halo"]] = recv

        # Lower interior goes left; the right neighbour's lower interior fills our upper halo.
        recv = self._sendrecv(
            self._data[regions["lower_interior"]],
            dest=left,
            source=right,
            tag=_FILL_TAG + 2 * dim + 1,
        )
        if right != MPI.PROC_NULL:
            self._data[regions["upper_halo"]] = recv

    def fill_halos(self) -> None:
        """Fill ghost cells from neighbouring ranks in all dimensions."""
        if self.num_ghostpoints == 0:
            return
        for dim in range(self.ndim):
            self.fill_halo(dim)

    def global_to_local(
        self, global_idx: int | tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Convert a global index to a local one (including the ghost offset).

        Returns ``None`` if this rank does not own the index.
        """
        if not isinstance(global_idx, tuple):
            global_idx = (global_idx,)

        if len(global_idx) != self.ndim:
            raise ValueError(f"Expected {self.ndim} indices, got {len(global_idx)}")

        local_idx = []
        for i, (gidx, (start, end)) in enumerate(
            zip(global_idx, self.proc_index_bounds, strict=True),
        ):
            if gidx < 0:
                gidx += self.shape[i]
            if not (start <= gidx < end):
                return None  # This global index is not in this rank
            offset = self.num_ghostpoints if self.ghost_axes[i] else 0
            local_idx.append(gidx - start + offset)

        return tuple(local_idx)

    def get_global_value(
        self, index: int | tuple[int, ...], root: int = 0
    ) -> Scalar | None:
        """Return the value at global index ``index`` on ``root`` (collective).

        Non-root ranks return ``None``.
        """
        local_idx = self.global_to_local(index)
        local_value = None if local_idx is None else self.data[local_idx]
        if not self.is_distributed:
            values = [local_value]
        else:
            # Only the owning rank contributes a non-None value.
            values = self._distributed_comm.gather(local_value, root=root)

        if self.mpi_rank != root:
            return None
        assert values is not None
        for value in values:
            if value is not None:
                return value
        raise ValueError(f"Index {index} not found on any rank")

    # -------------------------------------------------- #
    # Local methods

    def _check_compatibility(self, other: DistributedArray) -> None:
        """Validate that another distributed array has the same layout."""
        if self.ndim != other.ndim:
            raise ValueError("Number of dimensions must match")
        if self.shape != other.shape:
            raise ValueError("Shapes must match")
        if self.proc_sizes != other.proc_sizes:
            raise ValueError("MPI process layouts must match")
        if self.num_ghostpoints != other.num_ghostpoints:
            raise ValueError("Halo widths must match")
        if self.ghost_axes != other.ghost_axes:
            raise ValueError("Halo axes must match")

    def _new_like(self, data_local: Array, dtype: type | None = None) -> Self:
        """Create a distributed array with matching layout from local storage data."""
        data = xp.asarray(data_local, dtype=dtype)
        if xp.may_share_memory(data, self._data):
            data = data.copy()
        return self._wrap_local(data)

    def _axis_is_undivided(self, axis: int) -> bool:
        """Return whether local storage spans the full global extent of ``axis``."""
        has_halo = self.ghost_axes[axis] and self.num_ghostpoints > 0
        return self.proc_sizes[axis] == 1 and not has_halo

    def _coerce_other_data(self, other: Any) -> Array | Number:
        """Return local data compatible with this rank for elementwise operations.

        Accepted operands, in order of precedence: distributed arrays with the
        same layout, scalars, arrays with the local storage shape (halos
        included), arrays that broadcast only along undivided axes (applied to
        local storage directly), and arrays that broadcast to the global shape
        (sliced to the owned block, with zero halos).
        """
        if isinstance(other, DistributedArray):
            self._check_compatibility(other)
            return other.data
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

        axes = range(self.ndim - other.ndim, self.ndim)
        if all(
            extent == 1 or self._axis_is_undivided(axis)
            for axis, extent in zip(axes, other.shape, strict=True)
        ):
            # Same result on every rank as broadcasting against the global array.
            try:
                xp.broadcast_shapes(other.shape, self.shape)
            except ValueError as exc:
                raise ValueError(
                    f"operand with shape {other.shape} cannot be broadcast to the "
                    f"global shape {self.shape}",
                ) from exc
            return other

        try:
            global_view = xp.broadcast_to(other, self.shape)
        except ValueError as exc:
            raise ValueError(
                f"operand with shape {other.shape} cannot be broadcast to the "
                f"global shape {self.shape}",
            ) from exc
        data = xp.zeros(self._data.shape, dtype=other.dtype)
        data[self.get_local_slices()] = global_view[self._global_slices()]
        return data

    def _binary_op(self, other: Any, op: Callable) -> Self:
        """Apply an elementwise binary operation."""
        return self._new_like(op(self.data, self._coerce_other_data(other)))

    def _binary_rop(self, other: Any, op: Callable) -> Self:
        """Apply a reflected elementwise binary operation."""
        return self._new_like(op(self._coerce_other_data(other), self.data))

    def _binary_iop(self, other: Any, op: Callable) -> Self:
        """Apply an in-place elementwise binary operation."""
        op(self.data, self._coerce_other_data(other), out=self.data)
        return self

    def _extremum_identity(self, mpi_op: MPI.Op) -> Scalar:
        """Return the neutral element of ``MPI.MIN``/``MPI.MAX`` for this dtype."""
        dtype: np.dtype[Any] = np.dtype(self._dtype)
        is_min = mpi_op == MPI.MIN
        if dtype.kind == "b":
            return np.True_ if is_min else np.False_
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            return dtype.type(info.max if is_min else info.min)
        if dtype.kind == "f":
            return dtype.type(np.inf if is_min else -np.inf)
        raise TypeError(f"min/max are not supported for dtype {dtype}")

    def _allreduce(self, local_value: Scalar, mpi_op: MPI.Op) -> Scalar:
        """Combine a per-rank value across all ranks."""
        if not self.is_distributed:
            return local_value
        if xp.is_gpu(local_value):
            # pickle-based MPI calls need a host value
            local_value = xp.to_numpy(local_value)[()]
        return self._distributed_comm.allreduce(local_value, op=mpi_op)

    def _global_reduction(
        self,
        op: Callable,
        mpi_op: MPI.Op,
        **op_kwargs: Any,
    ) -> Scalar:
        """Reduce all owned interior cells to one scalar on every rank.

        A rank that owns no interior cells contributes the neutral element, so
        decompositions that leave some ranks empty still reduce correctly.
        """
        is_extremum = mpi_op in (MPI.MIN, MPI.MAX)
        if is_extremum and self.size == 0:
            # Raised on every rank alike, so no rank is left waiting in allreduce.
            raise ValueError("zero-size array has no minimum or maximum")

        local_block = self._data[self.get_local_slices()]
        if is_extremum and local_block.size == 0:
            local_value = self._extremum_identity(mpi_op)
        else:
            # sum/prod of an empty block already return their neutral element.
            local_value = op(local_block, **op_kwargs)
        return self._allreduce(local_value, mpi_op)

    def _finish_reduction(self, value: Scalar, keepdims: bool) -> Scalar | Array:
        """Apply ``keepdims`` to a scalar full reduction."""
        if keepdims:
            return xp.full((1,) * self.ndim, value)
        return value

    @staticmethod
    def _reject_out(out: Any) -> None:
        """Raise for NumPy's ``out=`` argument, which is not supported."""
        if out is not None:
            raise TypeError("DistributedArray reductions do not support out=")

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self._dtype

    @property
    def size(self) -> int:
        """Return the total number of global elements."""
        return math.prod(self.shape)

    @property
    def local_size(self) -> int:
        """Return the number of owned local elements, excluding halos."""
        return math.prod(self.shape_local)

    @property
    def shape_with_halos(self) -> tuple[int, ...]:
        """Return the local storage shape including halo cells."""
        return self.data.shape

    @property
    def nbytes(self) -> int:
        """Return the global array byte size."""
        return self.size * self.itemsize

    @property
    def itemsize(self) -> int:
        """Return the element byte size."""
        return xp.dtype(self.dtype).itemsize

    @property
    def T(self) -> Array:
        """Return the transpose of the gathered global array."""
        return self.to_ndarray().T

    @property
    def data(self) -> Array:
        """Return the local storage, including halo cells."""
        return self._data

    @data.setter
    def data(self, value: Array) -> None:
        """Fill from a *global* array (see ``fill``)."""
        self.fill(value)

    @property
    def num_ghostpoints(self) -> int:
        """Return the number of ghost points."""
        return self._num_ghostpoints

    @property
    def ghost_axes(self) -> list[bool]:
        """Return flags indicating which axes carry ghost cells."""
        return self._ghost_axes

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the global array shape."""
        return self._shape

    @property
    def shape_local(self) -> tuple[int, ...]:
        """Return the local array shape."""
        return self._shape_local

    @property
    def proc_index_bounds(self) -> list[tuple[int, int]]:
        """Return this rank global index bounds."""
        return self._proc_index_bounds

    def astype(self, dtype: type, copy: bool = True) -> Self:
        """Return this distributed array with a different dtype."""
        if not copy and xp.dtype(dtype) == self.dtype:
            return self
        return self._wrap_local(self.data.astype(dtype))

    def to_numpy(self) -> np.ndarray:
        """Gather and return the global array as a ``numpy.ndarray``."""
        return xp.to_numpy(self.to_ndarray())

    def to_cupy(self) -> Array:
        """Gather and return the global array as a ``cupy.ndarray``."""
        try:
            import cupy as cp  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "DistributedArray.to_cupy() requires CuPy. "
                "Install cupy-cuda12x (or the CuPy package for your CUDA version).",
            ) from exc

        return cp.asarray(self.to_ndarray())

    def __array__(self, dtype: type | None = None, copy: bool | None = None) -> Array:
        """Return a gathered global ndarray for NumPy interoperability."""
        array = self.to_ndarray()
        if dtype is not None:
            return array.astype(dtype, copy=False if copy is None else copy)
        if copy:
            return array.copy()
        return array

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any):
        """Apply NumPy/CuPy ufuncs elementwise to local storage.

        ``out=`` accepts distributed arrays with the same layout and writes in
        place, so ``np.multiply(a, 2.0, out=a)`` allocates nothing.
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
            kwargs["out"] = tuple(item.data for item in out)
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

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Sum array values globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is None:
            value = self._global_reduction(xp.sum, MPI.SUM, dtype=dtype)
            return self._finish_reduction(value, keepdims)
        return xp.sum(self.to_ndarray(), axis=axis, dtype=dtype, keepdims=keepdims)

    def prod(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Multiply array values globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is None:
            value = self._global_reduction(xp.prod, MPI.PROD, dtype=dtype)
            return self._finish_reduction(value, keepdims)
        return xp.prod(self.to_ndarray(), axis=axis, dtype=dtype, keepdims=keepdims)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the minimum globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is None:
            value = self._global_reduction(xp.min, MPI.MIN)
            return self._finish_reduction(value, keepdims)
        return xp.min(self.to_ndarray(), axis=axis, keepdims=keepdims)

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the maximum globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is None:
            value = self._global_reduction(xp.max, MPI.MAX)
            return self._finish_reduction(value, keepdims)
        return xp.max(self.to_ndarray(), axis=axis, keepdims=keepdims)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the mean globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is None:
            value = self.sum(dtype=dtype) / self.size
            return self._finish_reduction(value, keepdims)
        return xp.mean(self.to_ndarray(), axis=axis, dtype=dtype, keepdims=keepdims)

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the variance globally, or along an axis on the gathered array."""
        self._reject_out(out)
        if axis is not None:
            return xp.var(
                self.to_ndarray(),
                axis=axis,
                dtype=dtype,
                ddof=ddof,
                keepdims=keepdims,
            )
        mean = self.mean(dtype=dtype)
        local_block = self._data[self.get_local_slices()]
        local = xp.sum(xp.abs(local_block - mean) ** 2, dtype=dtype)
        value = self._allreduce(local, MPI.SUM) / (self.size - ddof)
        return self._finish_reduction(value, keepdims)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> Scalar | Array:
        """Return the standard deviation globally, or along an axis."""
        return xp.sqrt(
            self.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims),
        )

    def all(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> bool | Array:
        """Return whether all values are true."""
        self._reject_out(out)
        if axis is None:
            local_value = bool(xp.all(self._data[self.get_local_slices()]))
            value = self._allreduce(local_value, MPI.LAND)
            return self._finish_reduction(value, keepdims)
        return xp.all(self.to_ndarray(), axis=axis, keepdims=keepdims)

    def any(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> bool | Array:
        """Return whether any value is true."""
        self._reject_out(out)
        if axis is None:
            local_value = bool(xp.any(self._data[self.get_local_slices()]))
            value = self._allreduce(local_value, MPI.LOR)
            return self._finish_reduction(value, keepdims)
        return xp.any(self.to_ndarray(), axis=axis, keepdims=keepdims)

    def vdot(self, other: Self) -> Scalar:
        """Return ``sum(conj(self) * other)`` over the whole array, on every rank.

        Like ``numpy.vdot`` on the gathered arrays; halo cells are excluded.
        """
        if not isinstance(other, DistributedArray):
            raise TypeError("vdot needs another DistributedArray")
        self._check_compatibility(other)
        local_value = xp.vdot(self.local, other.local)
        return self._allreduce(local_value, MPI.SUM)

    def norm(self, ord: float = 2) -> float:
        """Return the vector norm of the whole array, on every rank.

        ``ord`` is 2 (Euclidean), 1 or ``numpy.inf``; halo cells are excluded.
        """
        if ord == 2:
            return float(xp.sqrt(xp.real(self.vdot(self))))
        local = xp.abs(self.local)
        if ord == 1:
            return float(self._allreduce(xp.sum(local), MPI.SUM))
        if ord == np.inf:
            local_max = xp.max(local) if local.size else 0.0
            return float(self._allreduce(local_max, MPI.MAX))
        raise ValueError(f"unsupported norm order {ord!r}; use 1, 2 or numpy.inf")

    def reduce_across_ranks(
        self,
        comm: MPI.Comm | None = None,
        op: MPI.Op = MPI.SUM,
    ) -> None:
        """Combine every rank's copy of this array in place, halo cells included.

        Used when each rank holds the whole grid and has deposited only its own
        markers: afterwards every rank holds the sum over all markers. ``comm``
        defaults to the array's communicator; pass the particle partition's
        communicator when the array itself was created without one. This is a
        collective call on ``comm`` and a no-op on a single rank.
        """
        if comm is None:
            comm = self.comm
        if comm is None or comm.Get_size() == 1:
            return
        if self.is_distributed and comm == self.comm:
            raise ValueError(
                "reduce_across_ranks needs every rank to hold the whole array, "
                "but this array is decomposed over the same communicator",
            )
        data = xp.ascontiguousarray(self._data)
        with xp.mpi.mpi_buffer(data, recv=True) as buf:
            comm.Allreduce(MPI.IN_PLACE, buf, op=op)
        if data is not self._data:
            self._data[...] = data

    # -------------------------------------------------- #
    # Magic methods
    def _normalize_basic_index(
        self,
        index: Any,
    ) -> tuple[int | slice, ...] | None:
        """Return ``index`` as one int or slice per axis, or None if it is not basic.

        Basic means integers, slices and at most one Ellipsis; anything else
        (arrays, masks, ``None``) returns None.
        """
        items = index if isinstance(index, tuple) else (index,)

        def is_int(item: Any) -> bool:
            return isinstance(item, (int, np.integer)) and not isinstance(
                item,
                (bool, np.bool_),
            )

        if not all(
            item is Ellipsis or isinstance(item, slice) or is_int(item)
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
        return tuple(int(item) if is_int(item) else item for item in items)

    def _assign_basic(self, index: tuple[int | slice, ...], value: Any) -> None:
        """Write ``value`` into the owned part of the global selection ``index``.

        ``value`` must be the same on every rank and broadcast to the shape of
        the selection. No communication happens, and halo cells are untouched.
        """
        if isinstance(value, DistributedArray):
            value = value.to_ndarray()
        selection_shape = []
        data_index: list[int | slice] = []
        value_index: list[slice] = []
        owned = True
        for axis, (item, (start, end)) in enumerate(
            zip(index, self.proc_index_bounds, strict=True),
        ):
            extent = self.shape[axis]
            offset = self.num_ghostpoints if self.ghost_axes[axis] else 0
            if isinstance(item, int):
                position = item + extent if item < 0 else item
                if not 0 <= position < extent:
                    raise IndexError(
                        f"index {item} is out of bounds for axis {axis} "
                        f"with size {extent}",
                    )
                owned = owned and start <= position < end
                data_index.append(position - start + offset)
                continue

            selected = np.arange(extent)[item]
            selection_shape.append(selected.size)
            # The selection is monotonic, so the owned entries are contiguous.
            positions = np.flatnonzero((selected >= start) & (selected < end))
            if positions.size == 0:
                owned = False
                continue
            first = int(selected[positions[0]]) - start + offset
            last = int(selected[positions[-1]]) - start + offset
            step = item.indices(extent)[2]
            stop = last + step
            data_index.append(slice(first, None if stop < 0 else stop, step))
            value_index.append(slice(int(positions[0]), int(positions[-1]) + 1))

        # Broadcast on every rank, so a bad value raises everywhere alike.
        value = xp.broadcast_to(xp.asarray(value), tuple(selection_shape))
        if owned:
            self._data[tuple(data_index)] = value[tuple(value_index)]

    def __getitem__(self, index: IndexLike) -> Scalar | Array | None:
        """Return a global item, or a gathered global selection.

        An integer per axis returns the value on the owning rank and ``None``
        elsewhere (no communication). Any other index gathers the whole array
        and is therefore collective.
        """
        basic = self._normalize_basic_index(index)
        if basic is not None and all(isinstance(item, int) for item in basic):
            local_idx = self.global_to_local(cast("tuple[int, ...]", basic))
            return None if local_idx is None else self.data[local_idx]
        return self.to_ndarray()[index]

    def __setitem__(self, index: IndexLike, value: Any) -> None:
        """Set a global item or selection.

        Integers, slices and Ellipsis are written locally by each rank into the
        cells it owns, without communication; ``value`` must be identical on
        every rank. Advanced indices (arrays, masks) gather the whole array,
        are collective, and zero the halo cells.
        """
        basic = self._normalize_basic_index(index)
        if basic is not None:
            self._assign_basic(basic, value)
            return
        data = self.to_ndarray()
        data[index] = value
        self.fill(data)

    def __len__(self) -> int:
        """Return the length of the first global axis."""
        return self.shape[0]

    def __add__(self, other: Any) -> Self:
        """Add elementwise."""
        return self._binary_op(other, xp.add)

    def __radd__(self, other: Any) -> Self:
        """Add elementwise with reflected operands."""
        return self._binary_rop(other, xp.add)

    def __iadd__(self, other: Any) -> Self:
        """Add elementwise in place."""
        return self._binary_iop(other, xp.add)

    def __sub__(self, other: Any) -> Self:
        """Subtract elementwise."""
        return self._binary_op(other, xp.subtract)

    def __rsub__(self, other: Any) -> Self:
        """Subtract elementwise with reflected operands."""
        return self._binary_rop(other, xp.subtract)

    def __isub__(self, other: Any) -> Self:
        """Subtract elementwise in place."""
        return self._binary_iop(other, xp.subtract)

    def __mul__(self, other: Any) -> Self:
        """Multiply elementwise."""
        return self._binary_op(other, xp.multiply)

    def __rmul__(self, other: Any) -> Self:
        """Multiply elementwise with reflected operands."""
        return self._binary_rop(other, xp.multiply)

    def __imul__(self, other: Any) -> Self:
        """Multiply elementwise in place."""
        return self._binary_iop(other, xp.multiply)

    def __truediv__(self, other: Any) -> Self:
        """Divide elementwise."""
        return self._binary_op(other, xp.divide)

    def __rtruediv__(self, other: Any) -> Self:
        """Divide elementwise with reflected operands."""
        return self._binary_rop(other, xp.divide)

    def __itruediv__(self, other: Any) -> Self:
        """Divide elementwise in place."""
        return self._binary_iop(other, xp.divide)

    def __floordiv__(self, other: Any) -> Self:
        """Floor-divide elementwise."""
        return self._binary_op(other, xp.floor_divide)

    def __rfloordiv__(self, other: Any) -> Self:
        """Floor-divide elementwise with reflected operands."""
        return self._binary_rop(other, xp.floor_divide)

    def __pow__(self, other: Any) -> Self:
        """Raise values elementwise."""
        return self._binary_op(other, xp.power)

    def __rpow__(self, other: Any) -> Self:
        """Raise values elementwise with reflected operands."""
        return self._binary_rop(other, xp.power)

    def __neg__(self) -> Self:
        """Negate elementwise."""
        return self._wrap_local(-self.data)

    def __pos__(self) -> Self:
        """Return a positive copy."""
        return self.copy()

    def __abs__(self) -> Self:
        """Return absolute values elementwise."""
        return self._wrap_local(xp.abs(self.data))

    def __lt__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.less)

    def __le__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.less_equal)

    def __eq__(self, other: object) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compare elementwise for equality."""
        return self._binary_op(other, xp.equal)

    def __ne__(self, other: object) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compare elementwise for inequality."""
        return self._binary_op(other, xp.not_equal)

    def __gt__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.greater)

    def __ge__(self, other: Any) -> Self:
        """Compare elementwise."""
        return self._binary_op(other, xp.greater_equal)

    def __repr__(self) -> str:
        """Return a compact distributed-array representation."""
        return (
            f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"shape_local={self.shape_local}, num_ghostpoints={self.num_ghostpoints})"
        )
