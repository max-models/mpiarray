"""Tests for Layout, process_grid and chunk_bounds."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, cast

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout, chunk_bounds, process_grid

if TYPE_CHECKING:
    from mpiarray._mpi import Comm

MPI = xp.mpi.get_mpi()


class FakeComm:
    """The two calls a Layout makes on a communicator of ``size`` ranks."""

    def __init__(self, size: int, rank: int) -> None:
        self.size, self.rank = size, rank

    def Get_size(self) -> int:  # noqa: N802 - mpi4py's name
        return self.size

    def Get_rank(self) -> int:  # noqa: N802 - mpi4py's name
        return self.rank


def fake_comm(size: int, rank: int) -> Comm:
    """Return a `FakeComm`, typed as the communicator it stands in for."""
    return cast("Comm", FakeComm(size, rank))


def _layouts(nprocs: int, shape, **kwargs) -> list[Layout]:
    """Return the layout every rank of an ``nprocs``-rank communicator computes."""
    return [Layout(shape, comm=fake_comm(nprocs, r), **kwargs) for r in range(nprocs)]


@pytest.mark.parametrize(
    ("size", "ndim", "split", "expected"),
    [
        (1, 2, (0, 1), (1, 1)),
        (4, 2, (0, 1), (2, 2)),
        (6, 2, (0, 1), (2, 3)),
        (7, 2, (0, 1), (1, 7)),  # a prime cannot be split over two axes
        (12, 3, (0, 1, 2), (2, 2, 3)),
        (15, 3, (0, 1, 2), (1, 3, 5)),
        (12, 3, (2, 0), (3, 1, 4)),
        (8, 2, (), (1, 1)),
        (8, 2, (1,), (1, 8)),
    ],
)
def test_process_grid(size: int, ndim: int, split, expected) -> None:
    assert process_grid(size, ndim, split) == expected


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("size", [1, 2, 6, 12, 16, 30, 64])
def test_process_grid_multiplies_to_the_rank_count_on_split_axes(size, ndim) -> None:
    for count in range(ndim + 1):
        for split in itertools.combinations(range(ndim), count):
            grid = process_grid(size, ndim, split)
            assert all(n == 1 for axis, n in enumerate(grid) if axis not in split)
            assert math.prod(grid) == (size if split else 1)


def test_chunk_bounds_tile_the_axis_near_evenly() -> None:
    assert [chunk_bounds(10, 3, i) for i in range(3)] == [(0, 4), (4, 7), (7, 10)]
    assert [chunk_bounds(2, 4, i) for i in range(4)] == [(0, 1), (1, 2), (2, 2), (2, 2)]
    for length, nchunks in itertools.product([0, 1, 7, 10, 100], [1, 3, 4, 7]):
        bounds = [chunk_bounds(length, nchunks, i) for i in range(nchunks)]
        assert bounds[0][0] == 0 and bounds[-1][1] == length
        assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
        sizes = [end - start for start, end in bounds]
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("nprocs", [1, 2, 6, 12])
def test_layouts_agree_across_ranks(nprocs: int, periodic) -> None:
    """Coordinates, ranks, neighbours, bounds and owners are consistent everywhere."""
    shape = (13, 7)
    layouts = _layouts(nprocs, shape, split=(0, 1), periodic=periodic)
    grid = layouts[0].process_grid
    assert math.prod(grid) == nprocs
    seen = np.zeros(shape, dtype=int)
    for rank, layout in enumerate(layouts):
        assert layout.process_grid == grid and not layout.replicated
        assert layout.distributed == (nprocs > 1)
        coord = layout.process_coord
        assert layout.coord_of(rank) == coord and layout.rank_of(coord) == rank
        for axis, (left, right) in enumerate(layout.neighbours):
            for neighbour, step in ((left, -1), (right, +1)):
                target = coord[axis] + step
                if 0 <= target < grid[axis]:
                    assert layouts[neighbour].process_coord[axis] == target
                elif periodic[axis]:
                    assert layouts[neighbour].process_coord[axis] == target % grid[axis]
                else:
                    assert neighbour == MPI.PROC_NULL
        assert layout.index_bounds == layout.index_bounds_of(rank)
        assert layout.local_shape == layout.local_shape_of(rank)
        seen[layout.global_slices()] += 1
        (x0, x1), (y0, y1) = layout.index_bounds
        for index in itertools.product(range(x0, x1), range(y0, y1)):
            assert layouts[0].owner(index) == rank
    assert (seen == 1).all()  # every index is owned exactly once


def test_storage_with_halos_per_axis() -> None:
    layout = Layout((10, 6, 3), comm=fake_comm(4, 1), split=(0, 1), halo=(2, 1, 0))
    assert layout.process_grid == (2, 2, 1)
    assert layout.index_bounds == ((0, 5), (3, 6), (0, 3))
    assert layout.local_shape == (5, 3, 3)
    assert layout.storage_shape == (9, 5, 3)
    assert layout.interior == (slice(2, -2), slice(1, -1), slice(None))
    assert layout.global_slices(3) == (slice(5, 10), slice(3, 6), slice(0, 3))
    assert Layout(5, comm=fake_comm(1, 0), halo=3).storage_shape == (11,)


def test_replicated_and_single_rank_layouts() -> None:
    replicated = Layout(
        (4, 3), comm=fake_comm(3, 2), split=None, periodic=(True, False)
    )
    assert replicated.replicated and not replicated.distributed
    assert replicated.split == ()
    assert replicated.process_coord == (0, 0)
    assert replicated.coord_of(1) == (0, 0)
    assert replicated.index_bounds == ((0, 4), (0, 3))
    assert replicated.neighbours == ((2, 2), (MPI.PROC_NULL, MPI.PROC_NULL))
    assert replicated.owner((3, 2)) == 0

    single = Layout((4, 3), comm=fake_comm(1, 0), periodic=(True, False))
    assert not single.replicated and not single.distributed
    assert single.neighbours == ((0, 0), (MPI.PROC_NULL, MPI.PROC_NULL))


def test_explicit_process_grid_sets_the_split() -> None:
    layout = Layout((12, 6), comm=fake_comm(6, 4), process_grid=(3, 2))
    assert layout.split == (0, 1)
    assert layout.process_grid == (3, 2)
    assert layout.process_coord == (2, 0)
    assert Layout((12, 6), comm=fake_comm(6, 0), process_grid=(1, 6)).split == (1,)
    with pytest.raises(ValueError, match="multiply to the 6 ranks"):
        Layout((12, 6), comm=fake_comm(6, 0), process_grid=(2, 2))
    with pytest.raises(ValueError, match="multiply"):
        Layout((12, 6), comm=fake_comm(6, 0), process_grid=(0, 6))
    with pytest.raises(ValueError, match="process_grid has 1 entries"):
        Layout((12, 6), comm=fake_comm(6, 0), process_grid=(6,))


def test_split_and_argument_normalization() -> None:
    comm = fake_comm(4, 0)
    assert Layout((4, 4, 4), comm=comm, split=-1).split == (2,)
    assert Layout((4, 4, 4), comm=comm, split=(2, 0)).split == (0, 2)
    assert Layout((4, 4), comm=comm, split=None).split == ()
    assert Layout((), comm=comm).split == ()  # a 0-d array is never split
    assert Layout(7, comm=comm).shape == (7,)
    layout = Layout((4, 4), comm=comm, halo=1, periodic=True)
    assert layout.halo == (1, 1) and layout.periodic == (True, True)
    assert (layout.comm, layout.size, layout.rank, layout.ndim) == (comm, 4, 0, 2)
    with pytest.raises(ValueError, match="out of range"):
        Layout((4, 4), comm=comm, split=2)
    with pytest.raises(ValueError, match="repeat"):
        Layout((4, 4), comm=comm, split=(0, -2))
    with pytest.raises(ValueError, match="negative dimensions"):
        Layout((4, -1), comm=comm)
    with pytest.raises(ValueError, match="must not be negative"):
        Layout((4, 4), comm=comm, halo=(1, -1))
    with pytest.raises(ValueError, match="periodic has 3 entries"):
        Layout((4, 4), comm=comm, periodic=(True, True, True))


def test_owner_and_rank_of_reject_bad_indices() -> None:
    layout = Layout((10, 4), comm=fake_comm(2, 0))
    assert layout.owner((-1, 0)) == 1
    assert Layout(10, comm=fake_comm(2, 0)).owner(7) == 1
    with pytest.raises(IndexError, match="expected 2 indices"):
        layout.owner(3)
    with pytest.raises(IndexError, match="out of bounds for axis 0"):
        layout.owner((10, 0))
    with pytest.raises(ValueError, match="expected 2 process coordinates"):
        layout.rank_of((0,))


def test_layouts_compare_by_value() -> None:
    comm = fake_comm(2, 0)
    a = Layout((8, 4), comm=comm, halo=1)
    assert a == Layout((8, 4), comm=comm, halo=1)
    assert hash(a) == hash(Layout((8, 4), comm=comm, halo=1))
    assert a != Layout((8, 4), comm=comm, halo=2)
    assert a != Layout((8, 4), comm=comm, halo=1, periodic=True)
    assert a != Layout((8, 4), comm=fake_comm(2, 0), halo=1)  # another communicator
    assert a != "a layout"
    assert repr(a) == (
        "Layout(shape=(8, 4), split=(0,), halo=(1, 1), periodic=(False, False), "
        "process_grid=(2, 1), rank=0 of 2)"
    )


def test_default_communicator_is_comm_world() -> None:
    layout = Layout(4)
    assert layout.comm is MPI.COMM_WORLD
    assert layout == mpa.zeros(4).layout
