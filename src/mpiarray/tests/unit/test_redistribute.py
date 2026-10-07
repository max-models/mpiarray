"""Tests for redistribute, rebalance and arrays moved between layouts."""

from __future__ import annotations

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout

MPI = maybempi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
W = 2 * size + 3

LAYOUTS = [
    {"split": 0},
    {"split": 1, "halo": 1},
    {"split": (0, 1), "halo": (2, 1), "periodic": (True, False)},
    {"split": None},
    {"split": (0, 1), "reorder": True},
]


@pytest.mark.parametrize("source", LAYOUTS, ids=str)
@pytest.mark.parametrize("target", LAYOUTS, ids=str)
def test_redistribute_between_layouts(source, target) -> None:
    data = np.arange(W * (W + 1), dtype=float).reshape(W, W + 1)
    a = mpa.array(data, layout=Layout(data.shape, **source))
    b = a.redistribute(Layout(data.shape, **target))
    assert b.layout == Layout(data.shape, **target)
    np.testing.assert_array_equal(xp.to_numpy(b.local), data[b.layout.global_slices()])
    halo = np.ones(b.layout.storage_shape, dtype=bool)
    halo[b.layout.interior] = False
    assert not xp.to_numpy(b.local_with_halos)[halo].any()


def test_redistribute_to_uneven_bounds_and_back() -> None:
    data = np.arange(float(W * 3)).reshape(W, 3)
    a = mpa.array(data)
    cuts = (0, *range(1, size), W) if size > 1 else (0, W)
    uneven = a.redistribute(Layout(data.shape, bounds=[cuts, None]))
    assert uneven.layout.bounds[0] == cuts
    np.testing.assert_array_equal(xp.to_numpy(uneven.gather()), data)
    back = mpa.array(uneven, layout=a.layout)
    np.testing.assert_array_equal(xp.to_numpy(back.local), xp.to_numpy(a.local))
    assert a.redistribute(a.layout).local_with_halos is not a.local_with_halos


def test_redistribute_errors() -> None:
    a = mpa.zeros(W)
    with pytest.raises(ValueError, match="cannot redistribute shape"):
        a.redistribute(Layout(W + 1))
    if size > 1:
        with pytest.raises(ValueError, match="another communicator"):
            a.redistribute(Layout(W, comm=MPI.COMM_SELF))
    if size >= 4 and size % 2 == 0:  # same size, other processes
        world = MPI.COMM_WORLD
        rank = world.Get_rank()
        pairs, halves = (
            world.Split(rank % 2, rank),
            world.Split(rank // (size // 2), rank),
        )
        b = mpa.zeros(W, comm=pairs)
        with pytest.raises(ValueError, match="another communicator"):
            b.redistribute(Layout(W, comm=halves))


def test_rebalance_evens_out_the_work() -> None:
    n = 6 * size + 6
    cost = mpa.fromfunction(lambda i: np.where(i < n // 3, 20.0, 1.0), (n,))
    balanced = cost.rebalance(cost)
    np.testing.assert_array_equal(
        xp.to_numpy(balanced.gather()), xp.to_numpy(cost.gather())
    )
    work = MPI.COMM_WORLD.allgather(float(np.sum(xp.to_numpy(balanced.local))))
    before = MPI.COMM_WORLD.allgather(float(np.sum(xp.to_numpy(cost.local))))
    assert max(work) <= max(before)
    assert balanced.layout.bounds == Layout.weighted((n,), cost).bounds
