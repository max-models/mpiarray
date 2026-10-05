"""MPI process layouts, neighbours and subdomain ownership, independent of solvers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import cunumpy as xp
import numpy as np

if TYPE_CHECKING:
    from cunumpy.mpi import SerialComm
    from mpi4py import MPI
else:
    # mpi4py.MPI under an MPI launcher, otherwise cunumpy's serial stand-in
    MPI = xp.mpi.get_mpi()

# An mpi4py communicator, or cunumpy's serial stand-in for one.
Comm: TypeAlias = "MPI.Comm | SerialComm"


def split_array(array_length: int, num_procs: int) -> np.ndarray:
    """Split an array length into near-even process chunks.

    Args:
        array_length: Number of points to split.
        num_procs: Number of chunks.

    Returns:
        The size of each chunk; the first ``array_length % num_procs`` chunks
        are one point longer than the rest.
    """
    base = array_length // num_procs
    remainder = array_length % num_procs
    result = np.full(num_procs, base)
    result[:remainder] += 1
    return result


class DomainDecomposition:
    """Represent the MPI process layout for a decomposed domain.

    The ranks of ``comm`` are placed row-major on a Cartesian process grid
    (the last axis varies fastest). Each rank owns one block of any grid laid
    out over it, and knows its left and right neighbour along every axis.

    Args:
        comm: Communicator to decompose over, or ``None`` for a serial layout
            with a single rank.
        decompose: Whether each axis may be split over ranks; default: all.
        dim_order: For each axis, which of the process counts sorted in
            descending order it gets (``0`` is the largest); default: the
            order of `calculate_proc_sizes`.
        periodic: Whether each axis wraps around; default: none does.
        ndim: Number of axes; needed only when ``decompose`` is not given.
    """

    def _sort_proc_sizes(self, proc_sizes: Sequence[int]) -> list[int]:
        """Sort process counts according to the decomposition order.

        Args:
            proc_sizes: Process counts per axis.

        Returns:
            The same counts, reordered so that axis ``i`` gets the
            ``dim_order[i]``-th largest.
        """
        assert self.dim_order is not None
        assert len(self.dim_order) == self.ndim
        assert set(self.dim_order) == set(
            range(self.ndim),
        ), "dim_order must be a permutation of [0..ndim-1]"

        # Proc sizes in descending order
        sorted_desc = sorted(proc_sizes, reverse=True)

        # Set proc_size to the i:th biggest size
        new_proc_sizes = [sorted_desc[i] for i in self.dim_order]

        return new_proc_sizes

    def __init__(
        self,
        comm: Comm | None,
        decompose: list[bool] | None = None,
        dim_order: list[int] | None = None,
        periodic: Sequence[bool] | None = None,
        ndim: int | None = None,
    ) -> None:
        """Initialize the domain decomposition instance."""
        self._comm = comm
        if comm is None:
            self._mpi_size = 1
            self._mpi_rank = 0
        else:
            self._mpi_size = comm.Get_size()
            self._mpi_rank = comm.Get_rank()

        assert ndim is not None or decompose is not None, (
            "DomainDecomposition must take either ndim or decompose!"
        )

        if ndim is None:
            assert decompose is not None
            ndim = len(decompose)
        self._ndim = ndim

        if decompose is None:
            self._decompose = [True] * self.ndim
        else:
            self._decompose = decompose
        assert len(self.decompose) == self.ndim

        if periodic is None:
            periodic = [False] * self.ndim
        self._periodic = tuple(periodic)
        assert len(self.periodic) == self.ndim

        self._dim_order = dim_order

        # -------------------------------------------------------------- #
        # Calculate the optimal number of processes in each dimension based on the
        # total number of processes and the domain decomposition flags.
        #    proc_sizes: List of integers indicating the number of processes in each
        #                dimension, matching the length of domain_decomposition.
        #                For example: [2, 6, 1] for 12 processes in a 2x6 grid in 3D.

        # Set proc sizes
        proc_sizes = calculate_proc_sizes(self.mpi_size, self.decompose)
        if self.dim_order:
            proc_sizes = self._sort_proc_sizes(proc_sizes)
        self._proc_sizes = proc_sizes

        # With no decomposed axis the process grid has a single slot and every
        # rank holds the whole domain; all ranks then share process coordinate 0.
        self._replicated = self.mpi_size > 1 and math.prod(proc_sizes) == 1

        # Set proc matrix and neighbour ranks
        self._proc_matrix = self.create_proc_matrix()
        if self.replicated:
            self._neighbour_ranks = [
                (
                    (self.mpi_rank, self.mpi_rank)
                    if is_periodic
                    else (MPI.PROC_NULL, MPI.PROC_NULL)
                )
                for is_periodic in self.periodic
            ]
        else:
            self._neighbour_ranks = calculate_neighbor_ranks(
                self.proc_sizes, self.mpi_rank, self.periodic
            )

    def create_proc_matrix(self) -> np.ndarray:
        """Create the process-coordinate lookup matrix.

        Returns:
            An integer array of shape ``(prod(proc_sizes), ndim)`` whose row
            ``r`` holds the process coordinates of rank ``r``.
        """
        ndim = len(self.proc_sizes)
        total_procs = math.prod(self.proc_sizes)

        proc_matrix = np.empty((total_procs, ndim), dtype=int)

        for rank in range(total_procs):
            coords = []
            rem = rank
            for size in reversed(self.proc_sizes):
                coords.append(rem % size)
                rem //= size
            proc_matrix[rank] = list(reversed(coords))

        return proc_matrix

    def _get_proc_bounds(
        self,
        array_length: int,
        num_procs: int,
        rank: int,
        dim: int,
    ) -> tuple[int, int]:
        """Return the ``(start, end)`` indices one rank owns along one axis.

        Args:
            array_length: Global length of the axis.
            num_procs: Number of processes along the axis.
            rank: Flat MPI rank.
            dim: The axis.

        Returns:
            The half-open index range ``[start, end)``.
        """
        return get_proc_bounds(array_length, num_procs, self.get_proc_coord(rank)[dim])

    def get_proc_coord(self, rank: int) -> tuple[int, ...]:
        """Return process coordinates for one rank.

        Args:
            rank: Flat MPI rank.

        Returns:
            The rank's position on the process grid, one entry per axis. In a
            replicated layout every rank is at the origin.
        """
        if self.replicated:
            return self.proc_matrix[0]
        return self.proc_matrix[rank]

    def rank_from_proc_coord(self, proc_coord: tuple[int, ...]) -> int:
        """Return the flat MPI rank for process coordinates.

        Args:
            proc_coord: Position on the process grid, one entry per axis.

        Returns:
            The rank at that position (row-major numbering).

        Raises:
            ValueError: If ``proc_coord`` does not have ``ndim`` entries.
        """
        if len(proc_coord) != self.ndim:
            raise ValueError(f"Expected {self.ndim} process coordinates")
        strides = [math.prod(self.proc_sizes[i + 1 :]) for i in range(self.ndim)]
        return int(sum(proc_coord[i] * strides[i] for i in range(self.ndim)))

    def get_index_bounds(
        self,
        shape: tuple[int, ...],
        rank: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return global index bounds for one rank over a decomposed shape.

        Each axis is split as evenly as possible (see `split_array`).

        Args:
            shape: Global grid shape, one entry per axis.
            rank: Flat MPI rank; default: this rank.

        Returns:
            One half-open ``(start, end)`` index range per axis.

        Raises:
            ValueError: If ``shape`` does not have ``ndim`` entries.
        """
        if rank is None:
            rank = self.mpi_rank
        if len(shape) != self.ndim:
            raise ValueError(f"Expected shape with {self.ndim} dimensions")
        return [
            self._get_proc_bounds(array_length, num_procs, rank, dim)
            for dim, (array_length, num_procs) in enumerate(
                zip(shape, self.proc_sizes, strict=True)
            )
        ]

    def subdomain_edges(self, dim: int, lower: float, upper: float) -> list[float]:
        """Return the ``proc_sizes[dim] + 1`` edges of the equal-width split of one axis.

        Process coordinate ``c`` owns ``[edges[c], edges[c + 1])``; the last one
        also owns ``upper``.  This is the split ``owner_rank_for_position`` uses.

        Args:
            dim: The axis.
            lower: Lower end of the physical domain along ``dim``.
            upper: Upper end of the physical domain along ``dim``.

        Returns:
            The subdomain edges, from ``lower`` to ``upper``.
        """
        num_procs = self.proc_sizes[dim]
        width = (upper - lower) / num_procs
        return [lower + coord * width for coord in range(num_procs)] + [upper]

    def subdomain_bounds(
        self,
        rank: int,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the physical ``(lower, upper)`` corners of one rank's subdomain.

        Args:
            rank: Flat MPI rank.
            lower_bound: Lower corner of the physical domain.
            upper_bound: Upper corner of the physical domain.

        Returns:
            The lower and upper corner of the rank's box, from the equal-width
            split of `subdomain_edges`.
        """
        proc_coord = self.get_proc_coord(rank)
        lower = []
        upper = []
        for dim in range(self.ndim):
            edges = self.subdomain_edges(
                dim, float(lower_bound[dim]), float(upper_bound[dim])
            )
            lower.append(edges[proc_coord[dim]])
            upper.append(edges[proc_coord[dim] + 1])
        return tuple(lower), tuple(upper)

    def owner_rank_for_position(
        self,
        position: Sequence[float] | np.ndarray,
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
        num_gridpoints: tuple[int, ...],
        grid_spacing: Sequence[float] | np.ndarray | None = None,
    ) -> int:
        """Return the rank owning a physical position in a decomposed grid.

        Ownership follows the equal-width split of `subdomain_edges`. A
        position outside the domain belongs to the last rank along each axis
        where it lies outside.

        Args:
            position: Physical coordinates, one per axis.
            lower_bound: Lower corner of the physical domain.
            upper_bound: Upper corner of the physical domain.
            num_gridpoints: Number of grid points per axis.
            grid_spacing: Grid spacing per axis; default: derived from the
                bounds and ``num_gridpoints``.

        Returns:
            The flat rank of the owning process.

        Raises:
            ValueError: If the bounds or ``num_gridpoints`` do not have
                ``ndim`` entries.
        """
        if len(lower_bound) != self.ndim or len(upper_bound) != self.ndim:
            raise ValueError("Bounds must match decomposition dimensions")
        if len(num_gridpoints) != self.ndim:
            raise ValueError("num_gridpoints must match decomposition dimensions")

        if grid_spacing is None:
            grid_spacing = tuple(
                (upper_bound[dim] - lower_bound[dim]) / (num_gridpoints[dim] - 1)
                for dim in range(self.ndim)
            )

        proc_coord = []
        for dim, (lower, upper, num_procs) in enumerate(
            zip(
                lower_bound,
                upper_bound,
                self.proc_sizes,
                strict=True,
            ),
        ):
            coord = num_procs - 1
            value = float(position[dim])
            edges = self.subdomain_edges(dim, float(lower), float(upper))
            for candidate_coord in range(num_procs):
                subdomain_lower = edges[candidate_coord]
                subdomain_upper = edges[candidate_coord + 1]
                if subdomain_lower <= value < subdomain_upper or (
                    candidate_coord == num_procs - 1 and value == subdomain_upper
                ):
                    coord = candidate_coord
                    break
            proc_coord.append(coord)
        return self.rank_from_proc_coord(tuple(proc_coord))

    @property
    def comm(self) -> Comm | None:
        """Return the MPI communicator."""
        return self._comm

    @property
    def dim_order(self) -> list[int] | None:
        """Return the decomposition dimension order."""
        return self._dim_order

    @property
    def mpi_size(self) -> int:
        """Return the MPI size."""
        return self._mpi_size

    @property
    def mpi_rank(self) -> int:
        """Return the MPI rank."""
        return self._mpi_rank

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._ndim

    @property
    def decompose(self) -> list[bool]:
        """Return flags indicating decomposed axes."""
        return self._decompose

    @property
    def replicated(self) -> bool:
        """Return whether every rank holds the whole domain (no decomposed axis)."""
        return self._replicated

    @property
    def proc_sizes(self) -> list[int]:
        """Return process counts along each axis."""
        return self._proc_sizes

    @property
    def proc_matrix(self) -> np.ndarray:
        """Return the process-coordinate matrix."""
        return self._proc_matrix

    @property
    def proc_coord(self) -> tuple[int, ...]:
        """Return this rank process coordinates."""
        return self.get_proc_coord(self.mpi_rank)

    @property
    def neighbour_ranks(self) -> list[tuple[int, int]]:
        """Return the ``(left, right)`` neighbour ranks along each axis.

        At a non-periodic wall the missing neighbour is ``MPI.PROC_NULL``.
        """
        return self._neighbour_ranks

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Return periodicity flags for each axis."""
        return self._periodic


def get_proc_bounds(array_length: int, num_procs: int, rank: int) -> tuple[int, int]:
    """Return the ``(start, end)`` indices of chunk ``rank`` of a split array.

    The chunks are those of `split_array`; ``rank`` is the position along one
    axis of the process grid, not the flat MPI rank.

    Args:
        array_length: Number of points to split.
        num_procs: Number of chunks.
        rank: Index of the chunk, ``0 .. num_procs - 1``.

    Returns:
        The half-open index range ``[start, end)`` of the chunk.
    """
    counts = split_array(array_length, num_procs)
    start = int(counts[:rank].sum())
    return start, start + int(counts[rank])


def calculate_neighbor_ranks(
    proc_sizes: Sequence[int],
    rank: int,
    periodic: Sequence[bool] | None = None,
) -> list[tuple[int, int]]:
    """Return the left and right neighbour ranks of ``rank`` in each dimension.

    Ranks are numbered row-major over the process grid (the last axis varies
    fastest), as in `DomainDecomposition.proc_matrix`.

    Args:
        proc_sizes: Number of processes in each dimension, e.g. ``[px, py, pz]``.
        rank: Flat MPI rank, ``0 .. prod(proc_sizes) - 1``.
        periodic: Whether each dimension wraps around; default: none does.

    Returns:
        For each dimension, a tuple ``(left_rank, right_rank)``. At a
        non-periodic boundary the missing neighbour is ``MPI.PROC_NULL``; at a
        periodic one it wraps around.
    """
    ndim = len(proc_sizes)
    if periodic is None:
        periodic = [False] * ndim
    # flat_rank = sum(coords[i] * strides[i])
    strides = [math.prod(proc_sizes[i + 1 :]) for i in range(ndim)]
    coords = [(rank // strides[i]) % proc_sizes[i] for i in range(ndim)]

    def shifted(dim: int, step: int) -> int:
        """Return the rank ``step`` places along ``dim``, or ``PROC_NULL`` past a wall."""
        coord = coords[dim] + step
        if not 0 <= coord < proc_sizes[dim]:
            if not periodic[dim]:
                return MPI.PROC_NULL
            coord %= proc_sizes[dim]
        return rank + (coord - coords[dim]) * strides[dim]

    return [(shifted(dim, -1), shifted(dim, +1)) for dim in range(ndim)]


def calculate_proc_sizes(size: int, domain_decomposition: Sequence[bool]) -> list[int]:
    """Return the number of processes along each dimension.

    The flagged dimensions get near-equal shares, as close to
    ``size ** (1 / n)`` as divisibility allows, in order; the last flagged
    dimension takes what is left, so the product is always ``size``. Without
    a flagged dimension every count is 1.

    Args:
        size: Total number of MPI processes.
        domain_decomposition: Whether to decompose each dimension, e.g.
            ``[True, True, False]``.

    Returns:
        Process counts per dimension, e.g. ``[2, 6, 1]``.
    """
    proc_sizes = [1] * len(domain_decomposition)
    remaining = size
    remaining_dims = sum(bool(flag) for flag in domain_decomposition)
    for dim, flag in enumerate(domain_decomposition):
        if not flag:
            continue
        # the largest divisor of `remaining` not above its remaining_dims-th root
        target = round(remaining ** (1 / remaining_dims))
        while remaining % target != 0:
            target -= 1
        proc_sizes[dim] = target
        remaining //= target
        remaining_dims -= 1
    return proc_sizes
