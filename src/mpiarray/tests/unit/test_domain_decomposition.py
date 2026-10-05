"""Tests for domain decomposition."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, cast

import cunumpy as xp
import numpy as np
import pytest

from mpiarray import (
    DomainDecomposition,
    calculate_neighbor_ranks,
    calculate_proc_sizes,
    get_proc_bounds,
    split_array,
)

if TYPE_CHECKING:
    from mpiarray.domain_decomposition import Comm

MPI = xp.mpi.get_mpi()

size = MPI.COMM_WORLD.Get_size()


# mpiexec -n 6 python -m pytest src/mpiarray/tests/unit/test_domain_decomposition.py
@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("decompose", [[True, True, True]])
def test_domain_decomposition_mpi6(
    decompose: list[bool],
    verbose: bool = False,
) -> None:
    """Verify domain decomposition on six MPI ranks."""
    comm = MPI.COMM_WORLD

    # Check the processor size in each direction: 1*2*3 = 6
    ddcomp = DomainDecomposition(comm=comm, decompose=decompose)
    assert ddcomp.proc_sizes == [2, 1, 3]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[0, 1, 2])
    assert ddcomp.proc_sizes == [3, 2, 1]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[0, 2, 1])
    assert ddcomp.proc_sizes == [3, 1, 2]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[1, 0, 2])
    assert ddcomp.proc_sizes == [2, 3, 1]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[1, 2, 0])
    assert ddcomp.proc_sizes == [2, 1, 3]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[2, 0, 1])
    assert ddcomp.proc_sizes == [1, 3, 2]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[2, 1, 0])
    assert ddcomp.proc_sizes == [1, 2, 3]


@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("decompose", [True])
@pytest.mark.parametrize("periodic", [True, False])
def test_domain_decomposition_neighbours_1d(
    decompose: bool,
    periodic: bool,
    verbose: bool = False,
) -> None:
    """Verify one-dimensional domain-decomposition neighbors."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    ddcomp = DomainDecomposition(comm=comm, decompose=[decompose], periodic=[periodic])

    if periodic:
        # Check wrapping in periodic case
        if ddcomp.mpi_rank == 0:
            assert ddcomp.neighbour_ranks[0][0] == size - 1
        elif ddcomp.mpi_rank == size - 1:
            assert ddcomp.neighbour_ranks[0][1] == 0
        else:
            assert ddcomp.neighbour_ranks[0][0] == ddcomp.mpi_rank - 1
            assert ddcomp.neighbour_ranks[0][1] == ddcomp.mpi_rank + 1
    else:
        # Check that neighbours of the edge ranks are null (not periodic)
        if ddcomp.mpi_rank == 0:
            assert ddcomp.neighbour_ranks[0][0] == MPI.PROC_NULL
        elif ddcomp.mpi_rank == size - 1:
            assert ddcomp.neighbour_ranks[0][1] == MPI.PROC_NULL
        else:
            assert ddcomp.neighbour_ranks[0][0] == ddcomp.mpi_rank - 1
            assert ddcomp.neighbour_ranks[0][1] == ddcomp.mpi_rank + 1


@pytest.mark.parametrize("periodic", [(True, False), (False, False)])
def test_undecomposed_layout_is_replicated_on_every_rank(
    periodic: tuple[bool, bool],
) -> None:
    """Without decomposed axes every rank sits at process coordinate 0."""
    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(comm=comm, decompose=[False, False], periodic=periodic)

    assert ddcomp.replicated == (size > 1)
    assert ddcomp.proc_sizes == [1, 1]
    assert list(ddcomp.proc_coord) == [0, 0]
    assert all(list(ddcomp.get_proc_coord(rank)) == [0, 0] for rank in range(size))
    assert ddcomp.get_index_bounds((5, 3)) == [(0, 5), (0, 3)]
    rank = comm.Get_rank()
    for is_periodic, neighbours in zip(periodic, ddcomp.neighbour_ranks, strict=True):
        if is_periodic:
            assert neighbours == (rank, rank)
        else:
            assert neighbours == (MPI.PROC_NULL, MPI.PROC_NULL)


def test_layout_without_communicator() -> None:
    """``comm=None`` is a single-rank layout with periodic self-neighbours."""
    layout = DomainDecomposition(
        comm=None, decompose=[True, False], periodic=(True, False)
    )
    assert layout.get_index_bounds((7, 5)) == [(0, 7), (0, 5)]
    assert layout.neighbour_ranks == [(0, 0), (MPI.PROC_NULL, MPI.PROC_NULL)]


# -------------------------------------------------------------------------- #
# Layouts for any number of ranks, checked serially with a stand-in communicator.


class FakeComm:
    """The two calls DomainDecomposition makes on a communicator of ``size`` ranks."""

    def __init__(self, size: int, rank: int) -> None:
        self.size, self.rank = size, rank

    def Get_size(self) -> int:  # noqa: N802 - mpi4py's name
        return self.size

    def Get_rank(self) -> int:  # noqa: N802 - mpi4py's name
        return self.rank


def fake_comm(size: int, rank: int) -> Comm:
    """Return a `FakeComm`, typed as the communicator it stands in for."""
    return cast("Comm", FakeComm(size, rank))


def _layouts(nprocs: int, **kwargs) -> list[DomainDecomposition]:
    """Return the layout seen by every rank of an ``nprocs``-rank communicator."""
    return [DomainDecomposition(fake_comm(nprocs, r), **kwargs) for r in range(nprocs)]


@pytest.mark.parametrize(
    ("nprocs", "flags", "expected"),
    [
        (1, [True, True], [1, 1]),
        (4, [True, True], [2, 2]),
        (6, [True, True], [2, 3]),
        (7, [True, True], [1, 7]),  # a prime cannot be split over two axes
        (12, [True, True, True], [2, 2, 3]),
        (15, [True, True, True], [1, 3, 5]),
        (12, [True, False, True], [3, 1, 4]),
        (8, [False, False], [1, 1]),
        (8, [False, True], [1, 8]),
    ],
)
def test_calculate_proc_sizes(nprocs: int, flags: list[bool], expected: list[int]):
    assert calculate_proc_sizes(nprocs, flags) == expected


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("nprocs", [1, 2, 6, 12, 16, 30, 64])
def test_proc_sizes_multiply_to_the_rank_count_on_flagged_axes(nprocs, ndim):
    for flags in itertools.product([False, True], repeat=ndim):
        sizes = calculate_proc_sizes(nprocs, flags)
        assert all(n == 1 for n, flag in zip(sizes, flags, strict=True) if not flag)
        assert math.prod(sizes) == (nprocs if any(flags) else 1)


def test_split_array_and_get_proc_bounds_tile_the_axis() -> None:
    assert split_array(10, 3).tolist() == [4, 3, 3]
    assert split_array(2, 4).tolist() == [1, 1, 0, 0]
    for length, nprocs in [(10, 3), (7, 7), (2, 4), (100, 6)]:
        bounds = [get_proc_bounds(length, nprocs, r) for r in range(nprocs)]
        assert bounds[0][0] == 0 and bounds[-1][1] == length
        assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
        assert all(isinstance(i, int) for bound in bounds for i in bound)


def test_calculate_neighbor_ranks_on_a_2x3_grid() -> None:
    null = MPI.PROC_NULL
    # rank 1 sits at coordinates (0, 1): ranks 0 1 2 / 3 4 5
    assert calculate_neighbor_ranks([2, 3], 1) == [(null, 4), (0, 2)]
    assert calculate_neighbor_ranks([2, 3], 1, (True, True)) == [(4, 4), (0, 2)]
    assert calculate_neighbor_ranks([2, 3], 5, (False, True)) == [(2, null), (4, 3)]
    assert calculate_neighbor_ranks([1], 0, (True,)) == [(0, 0)]


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("nprocs", [1, 2, 6, 12])
def test_layouts_agree_across_ranks(nprocs: int, periodic) -> None:
    """Coordinates, ranks, neighbours and index ranges are consistent on all ranks."""
    layouts = _layouts(nprocs, decompose=[True, True], periodic=periodic)
    sizes = layouts[0].proc_sizes
    shape = (13, 7)
    seen = np.zeros(shape, dtype=int)
    for rank, layout in enumerate(layouts):
        assert layout.proc_sizes == sizes and not layout.replicated
        coord = tuple(int(c) for c in layout.proc_coord)
        assert layout.rank_from_proc_coord(coord) == rank
        assert all(0 <= c < n for c, n in zip(coord, sizes, strict=True))
        for dim, (left, right) in enumerate(layout.neighbour_ranks):
            for neighbour, step in ((left, -1), (right, +1)):
                target = coord[dim] + step
                if 0 <= target < sizes[dim]:
                    assert layouts[neighbour].proc_coord[dim] == target
                elif periodic[dim]:
                    assert layouts[neighbour].proc_coord[dim] == target % sizes[dim]
                else:
                    assert neighbour == MPI.PROC_NULL
        (x0, x1), (y0, y1) = layout.get_index_bounds(shape)
        assert layout.get_index_bounds(shape, rank=rank) == [(x0, x1), (y0, y1)]
        seen[x0:x1, y0:y1] += 1
    assert (seen == 1).all()  # every index is owned exactly once


def test_layout_with_dim_order_on_six_ranks() -> None:
    layout = DomainDecomposition(fake_comm(6, 0), decompose=[True, True, True])
    assert layout.proc_sizes == [2, 1, 3]
    assert layout.dim_order is None and layout.comm is not None
    ordered = DomainDecomposition(
        fake_comm(6, 0), decompose=[True, True, True], dim_order=[2, 0, 1]
    )
    assert ordered.proc_sizes == [1, 3, 2]
    assert ordered.dim_order == [2, 0, 1]


def test_layout_from_ndim_decomposes_every_axis() -> None:
    layout = DomainDecomposition(fake_comm(4, 3), ndim=2)
    assert layout.decompose == [True, True]
    assert layout.periodic == (False, False)
    assert layout.proc_sizes == [2, 2]
    assert layout.proc_matrix.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert (layout.mpi_size, layout.mpi_rank, layout.ndim) == (4, 3, 2)


def test_replicated_layout_on_several_ranks() -> None:
    layout = DomainDecomposition(fake_comm(3, 2), decompose=[False], periodic=(True,))
    assert layout.replicated
    assert layout.neighbour_ranks == [(2, 2)]
    assert list(layout.get_proc_coord(1)) == [0]


def test_layout_rejects_bad_arguments() -> None:
    with pytest.raises(AssertionError, match="either ndim or decompose"):
        DomainDecomposition(None)
    with pytest.raises(AssertionError, match="permutation"):
        DomainDecomposition(fake_comm(4, 0), decompose=[True, True], dim_order=[0, 0])
    layout = DomainDecomposition(None, decompose=[True, True])
    with pytest.raises(ValueError, match="2 process coordinates"):
        layout.rank_from_proc_coord((0,))
    with pytest.raises(ValueError, match="shape with 2 dimensions"):
        layout.get_index_bounds((4,))


def test_subdomains_and_owner_rank_for_position() -> None:
    lower, upper, gridpoints = (0.0, 0.0), (1.0, 2.0), (11, 21)
    layouts = _layouts(4, decompose=[True, True])
    assert layouts[0].subdomain_edges(0, 0.0, 1.0) == [0.0, 0.5, 1.0]
    assert layouts[3].subdomain_bounds(3, lower, upper) == ((0.5, 1.0), (1.0, 2.0))
    owner = layouts[0].owner_rank_for_position
    assert owner((0.6, 1.9), lower, upper, gridpoints) == 3
    assert owner((0.1, 0.2), lower, upper, gridpoints) == 0
    assert owner((1.0, 2.0), lower, upper, gridpoints) == 3  # the upper edge
    # outside the domain, on either side, a position goes to the last subdomain
    assert owner((5.0, -1.0), lower, upper, gridpoints) == 3
    assert owner(np.array([0.6, 0.2]), lower, upper, gridpoints, (0.1, 0.1)) == 2
    for rank, layout in enumerate(layouts):  # each subdomain's centre is its own
        lo, hi = layout.subdomain_bounds(rank, lower, upper)
        centre = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
        assert owner(centre, lower, upper, gridpoints) == rank
    with pytest.raises(ValueError, match="Bounds"):
        owner((0.0, 0.0), (0.0,), upper, gridpoints)
    with pytest.raises(ValueError, match="num_gridpoints"):
        owner((0.0, 0.0), lower, upper, (11,))
