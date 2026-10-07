"""Tests for explicit bounds, aligned layouts and weighted (load-balanced) layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout

if TYPE_CHECKING:
    from mpiarray._mpi import Comm


class FakeComm:
    def __init__(self, size: int, rank: int) -> None:
        self.size, self.rank = size, rank

    def Get_size(self) -> int:  # noqa: N802 - mpi4py's name
        return self.size

    def Get_rank(self) -> int:  # noqa: N802 - mpi4py's name
        return self.rank


def fake(size: int, rank: int = 0) -> Comm:
    return cast("Comm", FakeComm(size, rank))


def test_explicit_bounds_set_the_blocks() -> None:
    layouts = [
        Layout((10, 6), comm=fake(4, r), bounds=[(0, 2, 10), (0, 1, 6)])
        for r in range(4)
    ]
    assert layouts[0].process_grid == (2, 2) and layouts[0].split == (0, 1)
    assert [lay.index_bounds for lay in layouts] == [
        ((0, 2), (0, 1)),
        ((0, 2), (1, 6)),
        ((2, 10), (0, 1)),
        ((2, 10), (1, 6)),
    ]
    assert layouts[0].bounds == ((0, 2, 10), (0, 1, 6))
    assert "bounds=((0, 2, 10), (0, 1, 6))" in repr(layouts[0])
    assert "bounds" not in repr(Layout((10, 6), comm=fake(4)))
    cells = np.array([[1, 0], [1, 5], [2, 0], [9, 5]])
    assert layouts[0].owners(cells).tolist() == [0, 1, 2, 3]
    assert [layouts[0].owner(tuple(c)) for c in cells] == [0, 1, 2, 3]
    assert layouts[0] != Layout((10, 6), comm=layouts[0].comm, process_grid=(2, 2))


def test_bounds_with_unsplit_axes_and_errors() -> None:
    layout = Layout((10, 6), comm=fake(3), bounds=[(0, 5, 8, 10), None])
    assert layout.process_grid == (3, 1) and layout.bounds[1] == (0, 6)
    with pytest.raises(ValueError, match="must increase from 0 to 10"):
        Layout(10, comm=fake(2), bounds=[(0, 5, 9)])
    with pytest.raises(ValueError, match="must increase"):
        Layout(10, comm=fake(2), bounds=[(0, 5, 5, 10)])
    with pytest.raises(ValueError, match="bounds imply the process grid"):
        Layout(10, comm=fake(2), bounds=[(0, 5, 10)], process_grid=(1,))
    with pytest.raises(ValueError, match="does not multiply"):
        Layout(10, comm=fake(3), bounds=[(0, 5, 10)])
    with pytest.raises(
        ValueError, match="halo 2 along axis 0 is wider .* \\(1 elements\\)"
    ):
        Layout(10, comm=fake(2), bounds=[(0, 1, 10)], halo=2)


def test_aligned_layouts_share_the_block_starts() -> None:
    cells = [Layout((8, 5), comm=fake(4, r), split=(0, 1), halo=1) for r in range(4)]
    nodes = [c.aligned((9, 6)) for c in cells]
    for c, n in zip(cells, nodes, strict=True):
        assert [b[0] for b in n.index_bounds] == [b[0] for b in c.index_bounds]
        assert n.halo == (1, 1) and n.process_grid == c.process_grid
    assert nodes[-1].index_bounds == ((4, 9), (3, 6))
    assert cells[0].aligned((8, 5), halo=0, periodic=True).periodic == (True, True)
    with pytest.raises(ValueError, match="leaves the last block empty"):
        cells[0].aligned((4, 5))
    with pytest.raises(ValueError, match="does not have 2 axes"):
        cells[0].aligned(9)
    whole = Layout(6, comm=fake(2), split=None)
    assert whole.aligned(7).replicated


def test_weighted_layouts_balance_the_weight() -> None:
    weights = np.ones((12, 4))
    weights[:3] = 10
    layout = Layout.weighted((12, 4), weights, comm=fake(3))
    assert layout.bounds == ((0, 1, 3, 12), (0, 4))
    profiles = Layout.weighted(
        (12, 4), [np.r_[[10.0] * 3, [1.0] * 9], None], comm=fake(3)
    )
    assert profiles.bounds == layout.bounds
    flat = Layout.weighted((12, 4), np.zeros((12, 4)), comm=fake(3))
    assert flat.bounds == Layout((12, 4), comm=fake(3)).bounds  # no weight: even
    halos = Layout.weighted(12, np.r_[[100.0] * 2, [1.0] * 10], comm=fake(3), halo=2)
    assert (
        min(b - a for a, b in zip(halos.bounds[0], halos.bounds[0][1:], strict=False))
        >= 2
    )
    replicated = Layout.weighted(12, np.ones(12), comm=fake(3), split=None)
    assert replicated.replicated
    unprofiled = Layout.weighted(
        (12, 6), [None, np.ones(6)], comm=fake(4), split=(0, 1)
    )
    assert unprofiled.bounds[0] == Layout((12, 6), comm=fake(4), split=(0, 1)).bounds[0]


def test_weighted_layout_errors() -> None:
    with pytest.raises(ValueError, match="do not match"):
        Layout.weighted((12, 4), np.ones((3, 3)), comm=fake(3))
    with pytest.raises(ValueError, match="must not be negative"):
        Layout.weighted(12, -np.ones(12), comm=fake(3))
    from mpiarray.layout import _balanced_cuts

    with pytest.raises(ValueError, match="cannot be cut into 3 chunks of at least 5"):
        _balanced_cuts(np.ones(12), 3, 5)
    with pytest.raises(ValueError, match="profile of length 5 along axis 0"):
        Layout.weighted((12, 4), [np.ones(5), None], comm=fake(3))


def test_weights_from_a_distributed_array() -> None:
    size = mpa.default_comm().Get_size()
    n = 4 * size + 4
    weights = mpa.fromfunction(lambda i: (i < n // 4) * 9.0 + 1.0, (n,))
    layout = Layout.weighted((n,), weights)
    sums = [
        np.sum(xp.to_numpy(weights.gather())[a:b])
        for a, b in zip(layout.bounds[0], layout.bounds[0][1:], strict=False)
    ]
    assert max(sums) <= 2 * min(sums) + 10
