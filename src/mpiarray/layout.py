"""How a global array is split over the ranks of a communicator.

Most code never builds a `Layout` itself: the creation functions
(`mpiarray.zeros`, `mpiarray.array`, ...) make one from their ``split``,
``halo``, ``periodic`` and ``comm`` arguments, and every array exposes its
own as `DistributedArray.layout`. Build one directly to describe a
decomposition without an array, e.g. for particle ownership or a PETSc DMDA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

from mpiarray._mpi import MPI, Comm, default_comm

#: An int for a 1-D shape, or one int per axis.
ShapeLike: TypeAlias = int | Sequence[int]
#: An axis, several axes, or ``None`` for an array every rank holds whole.
SplitLike: TypeAlias = int | Sequence[int] | None
#: One value for every axis, or one per axis.
HaloLike: TypeAlias = int | Sequence[int]
PeriodicLike: TypeAlias = bool | Sequence[bool]


def chunk_bounds(length: int, nchunks: int, i: int) -> tuple[int, int]:
    """Return the ``(start, end)`` of chunk ``i`` when ``length`` is split ``nchunks`` ways.

    The split is near-even: the first ``length % nchunks`` chunks get one
    element more than the others. This is how a `Layout` splits each axis.

    Args:
        length: Number of elements to split.
        nchunks: Number of chunks.
        i: Which chunk, ``0 .. nchunks - 1``.

    Returns:
        The half-open range of chunk ``i``.
    """
    base, extra = divmod(length, nchunks)
    start = i * base + min(i, extra)
    return start, start + base + (1 if i < extra else 0)


def process_grid(size: int, ndim: int, split: Sequence[int]) -> tuple[int, ...]:
    """Return the number of processes along each axis for ``size`` ranks.

    The ``split`` axes get near-equal shares, in axis order: each one gets
    the largest divisor of the ranks still to place that is not above their
    remaining ``n``-th root, and the last one takes the rest, so the product
    is always ``size``. Every other axis gets 1.

    Args:
        size: Number of ranks.
        ndim: Number of axes.
        split: The axes to split over the ranks.

    Returns:
        One process count per axis, e.g. ``(2, 3)`` for 6 ranks and
        ``split=(0, 1)``.
    """
    grid = [1] * ndim
    remaining = size
    axes = sorted(set(split))
    for position, axis in enumerate(axes):
        remaining_axes = len(axes) - position
        target = round(remaining ** (1 / remaining_axes))
        while remaining % target != 0:
            target -= 1
        grid[axis] = target
        remaining //= target
    return tuple(grid)


# Layout.__init__ has an argument of the same name.
_automatic_process_grid = process_grid


def _per_axis(value: object, ndim: int, name: str) -> tuple:
    """Return ``value`` repeated for every axis, or checked to have one per axis."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = tuple(value)
        if len(values) != ndim:
            raise ValueError(f"{name} has {len(values)} entries, expected {ndim}")
        return values
    return (value,) * ndim


def _normalize_axes(split: SplitLike, ndim: int) -> tuple[int, ...]:
    """Return ``split`` as sorted, non-negative, distinct axes."""
    if split is None:
        return ()
    axes = (split,) if isinstance(split, int) else tuple(split)
    normalized = []
    for axis in axes:
        if not -ndim <= axis < ndim:
            raise ValueError(f"split axis {axis} is out of range for {ndim} dimensions")
        normalized.append(axis % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"split axes {axes} repeat an axis")
    return tuple(sorted(normalized))


class Layout:
    """How a global array of ``shape`` is split over the ranks of ``comm``.

    The ranks are placed row-major on a process grid (the last axis varies
    fastest). Along each split axis the global index range is cut near-evenly
    (see `chunk_bounds`), so every rank owns one block. Each rank stores its
    block with ``halo`` extra cells on both sides of every axis, which the
    arrays fill from, or accumulate into, the neighbouring ranks.

    A layout is immutable and compares by value: arrays with equal layouts
    can be combined elementwise without communication.

    Args:
        shape: The global shape.
        comm: The communicator; default: ``MPI.COMM_WORLD`` (cunumpy's serial
            stand-in when not started by an MPI launcher).
        split: The axis or axes split over the ranks; ``None`` makes every
            rank hold the whole array. A 0-d shape is never split.
        halo: Halo width, for every axis or one per axis.
        periodic: Whether each axis wraps around in halo exchanges.
        process_grid: Explicit process counts per axis, whose product must be
            the number of ranks; it takes precedence over ``split``, which
            becomes the axes with more than one process.

    Raises:
        ValueError: For negative extents or halo widths, bad split axes,
            per-axis values of the wrong length, or a process grid that does
            not match the number of ranks.
    """

    def __init__(
        self,
        shape: ShapeLike,
        *,
        comm: Comm | None = None,
        split: SplitLike = 0,
        halo: HaloLike = 0,
        periodic: PeriodicLike = False,
        process_grid: Sequence[int] | None = None,
    ) -> None:
        """Compute the decomposition; see the class docstring."""
        self._shape = (
            (shape,) if isinstance(shape, int) else tuple(int(n) for n in shape)
        )
        if any(n < 0 for n in self._shape):
            raise ValueError(f"negative dimensions are not allowed: {self._shape}")
        ndim = len(self._shape)
        self._comm = default_comm() if comm is None else comm
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        self._halo = tuple(int(h) for h in _per_axis(halo, ndim, "halo"))
        if any(h < 0 for h in self._halo):
            raise ValueError(f"halo widths must not be negative: {self._halo}")
        self._periodic = tuple(bool(p) for p in _per_axis(periodic, ndim, "periodic"))

        if process_grid is not None:
            grid = tuple(int(n) for n in _per_axis(process_grid, ndim, "process_grid"))
            if any(n < 1 for n in grid) or math.prod(grid) != self._size:
                raise ValueError(
                    f"process_grid {grid} does not multiply to the {self._size} ranks",
                )
            self._split = tuple(axis for axis, n in enumerate(grid) if n > 1)
        else:
            self._split = () if ndim == 0 else _normalize_axes(split, ndim)
            grid = _automatic_process_grid(self._size, ndim, self._split)
        self._grid = grid

        # Without a split axis every rank holds the whole array.
        self._replicated = self._size > 1 and math.prod(grid) == 1
        self._coord = self.coord_of(self._rank)
        if self._replicated:
            self._neighbours = tuple(
                (self._rank, self._rank) if p else (MPI.PROC_NULL, MPI.PROC_NULL)
                for p in self._periodic
            )
        else:
            self._neighbours = tuple(
                (self._shifted(axis, -1), self._shifted(axis, +1))
                for axis in range(ndim)
            )
        self._index_bounds = self.index_bounds_of(self._rank)

    def _shifted(self, axis: int, step: int) -> int:
        """Return the rank one step along ``axis``, or ``PROC_NULL`` past a wall."""
        coord = list(self._coord)
        coord[axis] += step
        if not 0 <= coord[axis] < self._grid[axis]:
            if not self._periodic[axis]:
                return MPI.PROC_NULL
            coord[axis] %= self._grid[axis]
        return self.rank_of(coord)

    # ------------------------------------------------------------------ #
    # Ranks and process coordinates

    def coord_of(self, rank: int) -> tuple[int, ...]:
        """Return the position of ``rank`` on the process grid.

        Args:
            rank: A rank of the communicator.

        Returns:
            One coordinate per axis; all zeros in a replicated layout.
        """
        if self._replicated:
            return (0,) * self.ndim
        coord = []
        for n in reversed(self._grid):
            rank, c = divmod(rank, n)
            coord.append(c)
        return tuple(reversed(coord))

    def rank_of(self, coord: Sequence[int]) -> int:
        """Return the rank at a position of the process grid.

        Args:
            coord: One coordinate per axis.

        Returns:
            The rank, row-major over the process grid.

        Raises:
            ValueError: If ``coord`` does not have one entry per axis.
        """
        if len(coord) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} process coordinates, got {len(coord)}"
            )
        rank = 0
        for c, n in zip(coord, self._grid, strict=True):
            rank = rank * n + int(c)
        return rank

    # ------------------------------------------------------------------ #
    # Index ranges

    def index_bounds_of(self, rank: int) -> tuple[tuple[int, int], ...]:
        """Return the global ``(start, end)`` index range ``rank`` owns, per axis."""
        coord = self.coord_of(rank)
        return tuple(
            chunk_bounds(length, n, c)
            for length, n, c in zip(self._shape, self._grid, coord, strict=True)
        )

    def local_shape_of(self, rank: int) -> tuple[int, ...]:
        """Return the shape of the block ``rank`` owns, without halo cells."""
        return tuple(end - start for start, end in self.index_bounds_of(rank))

    def owner(self, index: int | Sequence[int]) -> int:
        """Return the rank that owns a global index.

        Args:
            index: One integer per axis (an int for 1-D); negative values count
                from the end.

        Returns:
            The owning rank; in a replicated layout, rank 0 (every rank holds
            the element).

        Raises:
            IndexError: If ``index`` has the wrong length or is out of bounds.
        """
        index = (index,) if isinstance(index, int) else tuple(index)
        if len(index) != self.ndim:
            raise IndexError(f"expected {self.ndim} indices, got {len(index)}")
        coord = []
        for axis, (i, length, n) in enumerate(
            zip(index, self._shape, self._grid, strict=True)
        ):
            position = i + length if i < 0 else i
            if not 0 <= position < length:
                raise IndexError(
                    f"index {i} is out of bounds for axis {axis} with size {length}",
                )
            coord.append(
                next(c for c in range(n) if position < chunk_bounds(length, n, c)[1])
            )
        return self.rank_of(coord)

    # ------------------------------------------------------------------ #
    # Properties

    @property
    def comm(self) -> Comm:
        """The communicator."""
        return self._comm

    @property
    def size(self) -> int:
        """The number of ranks of the communicator."""
        return self._size

    @property
    def rank(self) -> int:
        """This process's rank."""
        return self._rank

    @property
    def shape(self) -> tuple[int, ...]:
        """The global shape."""
        return self._shape

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return len(self._shape)

    @property
    def split(self) -> tuple[int, ...]:
        """The axes split over the ranks (empty when every rank holds everything)."""
        return self._split

    @property
    def halo(self) -> tuple[int, ...]:
        """The halo width of each axis."""
        return self._halo

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Whether each axis wraps around in halo exchanges."""
        return self._periodic

    @property
    def process_grid(self) -> tuple[int, ...]:
        """The number of processes along each axis."""
        return self._grid

    @property
    def process_coord(self) -> tuple[int, ...]:
        """This rank's position on the process grid."""
        return self._coord

    @property
    def replicated(self) -> bool:
        """Whether several ranks each hold the whole array (no axis is split)."""
        return self._replicated

    @property
    def distributed(self) -> bool:
        """Whether ranks hold different blocks, so that operations communicate."""
        return self._size > 1 and not self._replicated

    @property
    def neighbours(self) -> tuple[tuple[int, int], ...]:
        """The ``(left, right)`` neighbour ranks along each axis.

        ``MPI.PROC_NULL`` marks a wall (a non-periodic boundary); along a
        periodic axis with a single process, the neighbours are this rank.
        """
        return self._neighbours

    @property
    def index_bounds(self) -> tuple[tuple[int, int], ...]:
        """The global ``(start, end)`` index range this rank owns, per axis."""
        return self._index_bounds

    @property
    def local_shape(self) -> tuple[int, ...]:
        """The shape of this rank's block, without halo cells."""
        return tuple(end - start for start, end in self._index_bounds)

    @property
    def storage_shape(self) -> tuple[int, ...]:
        """The shape of this rank's storage: the block plus its halo cells."""
        return tuple(
            n + 2 * h for n, h in zip(self.local_shape, self._halo, strict=True)
        )

    @property
    def interior(self) -> tuple[slice, ...]:
        """The index into the storage that selects the block, without halos."""
        return tuple(slice(h, -h) if h else slice(None) for h in self._halo)

    def global_slices(self, rank: int | None = None) -> tuple[slice, ...]:
        """Return the index into a global array that selects the block of ``rank``.

        Args:
            rank: Default: this rank.
        """
        bounds = self._index_bounds if rank is None else self.index_bounds_of(rank)
        return tuple(slice(start, end) for start, end in bounds)

    # ------------------------------------------------------------------ #
    # Value semantics

    def _key(self) -> tuple:
        """Return everything that defines the layout, except the communicator."""
        return (self._shape, self._split, self._halo, self._periodic, self._grid)

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` describes the same decomposition."""
        if not isinstance(other, Layout):
            return NotImplemented
        same_comm = self._comm is other._comm or bool(self._comm == other._comm)
        return same_comm and self._key() == other._key()

    def __hash__(self) -> int:
        """Return a hash consistent with ``==``."""
        return hash(self._key())

    def __repr__(self) -> str:
        """Return the defining values; no communication."""
        return (
            f"Layout(shape={self._shape}, split={self._split}, halo={self._halo}, "
            f"periodic={self._periodic}, process_grid={self._grid}, "
            f"rank={self._rank} of {self._size})"
        )
