"""MPI process layouts, neighbours and subdomain ownership, independent of solvers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np

if TYPE_CHECKING:
    from mpi4py import MPI
else:
    # mpi4py.MPI under an MPI launcher, otherwise cunumpy's serial stand-in
    MPI = xp.mpi.get_mpi()


def split_array(array_length: int, num_procs: int) -> np.ndarray:
    """Split an array length into near-even process chunks."""
    base = array_length // num_procs
    remainder = array_length % num_procs
    result = np.full(num_procs, base)
    result[:remainder] += 1
    return result


class DomainDecomposition:
    """Represent the MPI process layout for a decomposed domain."""

    def _calculate_neighbor_ranks(self) -> list[tuple[int, int]]:
        """Return the left and right neighbour ranks of this rank in each dimension.

        Returns:
            For each dimension, a tuple ``(left_rank, right_rank)``. At a
            non-periodic boundary the missing neighbour is ``MPI.PROC_NULL``;
            at a periodic one it wraps around.
        """
        proc_sizes = self.proc_sizes
        rank = self.mpi_rank

        ndim = len(proc_sizes)
        # compute strides so that flat_rank = sum(coords[i] * strides[i])
        strides = [math.prod(proc_sizes[i + 1 :]) for i in range(ndim)]

        # decode flat rank → multi‐dim coords
        coords = []
        rem = rank
        for i in range(ndim):
            stride = strides[i]
            c = rem // stride
            coords.append(c)
            rem %= stride

        def rank_from_coords(process_coords: list[int]) -> int:
            """Convert process coordinates to a flat MPI rank."""
            return sum(process_coords[j] * strides[j] for j in range(ndim))

        neighbors = []
        for i in range(ndim):
            # LEFT neighbor
            if coords[i] > 0:
                left_coords = coords.copy()
                left_coords[i] -= 1
                left_rank = rank_from_coords(left_coords)
            else:
                if self.periodic[i]:
                    left_coords = coords.copy()
                    left_coords[i] = proc_sizes[i] - 1
                    left_rank = rank_from_coords(left_coords)
                else:
                    left_rank = MPI.PROC_NULL

            # RIGHT neighbor
            if coords[i] < proc_sizes[i] - 1:
                right_coords = coords.copy()
                right_coords[i] += 1
                right_rank = rank_from_coords(right_coords)
            else:
                if self.periodic[i]:
                    right_coords = coords.copy()
                    right_coords[i] = 0
                    right_rank = rank_from_coords(right_coords)
                else:
                    right_rank = MPI.PROC_NULL

            neighbors.append((left_rank, right_rank))

        return neighbors

    def _calculate_proc_sizes(self) -> list[int]:
        """Return the number of processes along each dimension.

        Splits ``mpi_size`` over the dimensions flagged in ``decompose``. If no
        even split over all of them exists, the last flagged dimension is
        dropped and the split retried; in the end everything goes to the first
        flagged dimension.

        Returns:
            Process counts per dimension, e.g. ``[2, 6, 1]`` for 12 processes
            in a 2x6 grid in 3D.
        """
        size = self.mpi_size
        domain_decomposition = self.decompose

        ndim = self.ndim
        proc_sizes = [1] * ndim  # Initialize proc_sizes with 1 in each dimension

        # Priority Decomposition: Start with all requested dimensions
        active_decomposition = domain_decomposition[:]

        while True:
            # Calculate the target number of processes per active dimension
            decompose_dims = sum(active_decomposition)

            if decompose_dims == 0:
                # No decomposable dimensions left, fallback to 1D decomposition
                proc_sizes = [1] * ndim
                for i in range(ndim):
                    if domain_decomposition[i]:
                        proc_sizes[i] = size
                        break
                return proc_sizes

            # Reset proc_sizes and remaining_size for current attempt
            proc_sizes = [1] * ndim
            remaining_size = size

            for i in range(ndim):
                if active_decomposition[i]:
                    # Calculate the approximate target number of processes in this dimension
                    target_procs = round(remaining_size ** (1 / decompose_dims))

                    # Adjust target_procs to be a divisor of the remaining_size
                    while remaining_size % target_procs != 0 and target_procs > 1:
                        target_procs -= 1

                    # Assign the calculated target_procs to this dimension
                    proc_sizes[i] = target_procs
                    remaining_size //= target_procs
                    decompose_dims -= 1

            # Check if the decomposition works
            product_of_procs = math.prod(
                [proc_sizes[i] for i in range(ndim) if active_decomposition[i]],
            )

            if product_of_procs == size:
                return proc_sizes

            # If not, disable the last dimension in active_decomposition and try again
            for j in reversed(range(ndim)):
                if active_decomposition[j]:
                    active_decomposition[j] = False
                    break

    def _sort_proc_sizes(self, proc_sizes: Sequence[int]) -> list[int]:
        """Sort process counts according to the decomposition order."""
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
        comm: MPI.Comm | None,
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
        proc_sizes = self._calculate_proc_sizes()
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
            self._neighbour_ranks = self._calculate_neighbor_ranks()

    def create_proc_matrix(self) -> np.ndarray:
        """Create the process-coordinate lookup matrix."""
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

    def _split_array(self, array_length: int, num_procs: int) -> np.ndarray:
        """Evenly split an array length across processes.

        Returns an array where each entry is the number of elements assigned to a process.
        """
        base = array_length // num_procs
        remainder = array_length % num_procs
        result = np.full(num_procs, base)
        result[:remainder] += 1
        return result

    def _get_proc_bounds(
        self,
        array_length: int,
        num_procs: int,
        rank: int,
        dim: int,
    ) -> tuple[int, int]:
        """Return the local bounds for one process and dimension."""
        counts = split_array(array_length, num_procs)
        proc_coord = self.get_proc_coord(rank)
        start = sum(int(count) for count in counts[: proc_coord[dim]])
        end = start + counts[proc_coord[dim]]
        return int(start), int(end)

    def get_proc_coord(self, rank: int) -> tuple[int, ...]:
        """Return process coordinates for one rank."""
        if self.replicated:
            return self.proc_matrix[0]
        return self.proc_matrix[rank]

    def rank_from_proc_coord(self, proc_coord: tuple[int, ...]) -> int:
        """Return the flat MPI rank for process coordinates."""
        if len(proc_coord) != self.ndim:
            raise ValueError(f"Expected {self.ndim} process coordinates")
        strides = [math.prod(self.proc_sizes[i + 1 :]) for i in range(self.ndim)]
        return int(sum(proc_coord[i] * strides[i] for i in range(self.ndim)))

    def get_index_bounds(
        self,
        shape: tuple[int, ...],
        rank: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return global index bounds for one rank over a decomposed shape."""
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
        """Return the physical ``(lower, upper)`` corners of one rank's subdomain."""
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
        """Return the rank owning a physical position in a decomposed grid."""
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
    def comm(self) -> MPI.Comm | None:
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
        """Return neighboring MPI ranks."""
        return self._neighbour_ranks

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Return periodicity flags for each axis."""
        return self._periodic


def get_proc_bounds(array_length: int, num_procs: int, rank: int) -> tuple[int, int]:
    """Return physical bounds for one process subdomain."""
    counts = split_array(array_length, num_procs)
    start = sum(int(count) for count in counts[:rank])
    end = start + counts[rank]
    return int(start), int(end)


def calculate_neighbor_ranks(
    proc_sizes: list[int],
    rank: int,
) -> list[tuple[int, int]]:
    """Return the left and right neighbour ranks of ``rank`` in each dimension.

    Unlike `DomainDecomposition.neighbour_ranks`, this ignores periodicity.

    Args:
        proc_sizes: Number of processes in each dimension, e.g. ``[px, py, pz]``.
        rank: Flat MPI rank, ``0 .. prod(proc_sizes) - 1``.

    Returns:
        For each dimension, a tuple ``(left_rank, right_rank)``, with
        ``MPI.PROC_NULL`` at the boundaries.
    """
    ndim = len(proc_sizes)
    # compute strides so that flat_rank = sum(coords[i] * strides[i])
    strides = [math.prod(proc_sizes[i + 1 :]) for i in range(ndim)]

    # decode flat rank → multi‐dim coords
    coords = []
    rem = rank
    for i in range(ndim):
        stride = strides[i]
        c = rem // stride
        coords.append(c)
        rem %= stride

    neighbors = []
    for i in range(ndim):
        # LEFT neighbor
        if coords[i] > 0:
            left_coords = coords.copy()
            left_coords[i] -= 1
            left_rank = sum(left_coords[j] * strides[j] for j in range(ndim))
        else:
            left_rank = MPI.PROC_NULL

        # RIGHT neighbor
        if coords[i] < proc_sizes[i] - 1:
            right_coords = coords.copy()
            right_coords[i] += 1
            right_rank = sum(right_coords[j] * strides[j] for j in range(ndim))
        else:
            right_rank = MPI.PROC_NULL

        neighbors.append((left_rank, right_rank))

    return neighbors


def calculate_proc_sizes(size: int, domain_decomposition: list[bool]) -> list[int]:
    """Return the number of processes along each dimension.

    Args:
        size: Total number of MPI processes.
        domain_decomposition: Whether to decompose each dimension, e.g.
            ``[True, True, False]``.

    Returns:
        Process counts per dimension, e.g. ``[2, 6, 1]``.
    """
    ndim = len(domain_decomposition)
    proc_sizes = [1] * ndim  # Initialize proc_sizes with 1 in each dimension

    # Priority Decomposition: Start with all requested dimensions
    active_decomposition = domain_decomposition[:]

    while True:
        # Calculate the target number of processes per active dimension
        decompose_dims = sum(active_decomposition)

        if decompose_dims == 0:
            # No decomposable dimensions left, fallback to 1D decomposition
            proc_sizes = [1] * ndim
            for i in range(ndim):
                if domain_decomposition[i]:
                    proc_sizes[i] = size
                    break
            return proc_sizes

        # Reset proc_sizes and remaining_size for current attempt
        proc_sizes = [1] * ndim
        remaining_size = size

        for i in range(ndim):
            if active_decomposition[i]:
                # Calculate the approximate target number of processes in this dimension
                target_procs = round(remaining_size ** (1 / decompose_dims))

                # Adjust target_procs to be a divisor of the remaining_size
                while remaining_size % target_procs != 0 and target_procs > 1:
                    target_procs -= 1

                # Assign the calculated target_procs to this dimension
                proc_sizes[i] = target_procs
                remaining_size //= target_procs
                decompose_dims -= 1

        # Check if the decomposition works
        product_of_procs = math.prod(
            [proc_sizes[i] for i in range(ndim) if active_decomposition[i]],
        )

        if product_of_procs == size:
            return proc_sizes

        # If not, disable the last dimension in active_decomposition and try again
        for j in reversed(range(ndim)):
            if active_decomposition[j]:
                active_decomposition[j] = False
                break


if __name__ == "__main__":
    # Example usage:
    size = 15
    decompose = [
        True,
        True,
        True,
    ]  # Decompose along x and y only in a 2D grid
    # print(calculate_proc_sizes(size, decompose))  # Output: [1, 3, 5]

    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[0, 1, 2])
    print(f"{ddcomp.proc_sizes =}")
    # print(f"{ddcomp.mpi_rank = }, {ddcomp.neighbour_ranks = }")
