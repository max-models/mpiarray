"""Tests for mpa.from_local: arrays made of the blocks the ranks hold."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def _stack(lengths: list[int], columns: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return the global array and this rank's rows, for the given row counts."""
    rows = sum(lengths)
    data = np.arange(rows * columns, dtype=float).reshape(rows, columns)
    start = sum(lengths[:rank])
    return data, data[start : start + lengths[rank]]


@pytest.mark.parametrize(
    "lengths",
    [
        [r + 1 for r in range(size)],  # uneven: redistributed
        [0] + [2] * (size - 1) if size > 1 else [3],  # rank 0 has nothing
        [
            mpa.chunk_bounds(4 * size, size, r)[1]
            - mpa.chunk_bounds(4 * size, size, r)[0]
            for r in range(size)
        ],
    ],
)
@pytest.mark.parametrize("halo", [0, 1])
def test_blocks_are_stacked_in_rank_order(lengths, halo) -> None:
    data, block = _stack(lengths)
    a = mpa.from_local(block, halo=halo)
    assert a.shape == data.shape and a.layout.halo == (halo, halo)
    np.testing.assert_array_equal(xp.to_numpy(a.gather()), data)
    np.testing.assert_array_equal(xp.to_numpy(a.local), data[a.layout.global_slices()])


def test_blocks_along_another_axis() -> None:
    data, block = _stack([r + 2 for r in range(size)], columns=4)
    a = mpa.from_local(np.ascontiguousarray(block.T), split=-1)
    assert a.layout.split == (1,)
    np.testing.assert_array_equal(xp.to_numpy(a.gather()), data.T)


def test_whole_arrays_and_given_layouts() -> None:
    data = np.arange(12.0).reshape(4, 3)
    replicated = mpa.from_local(data, split=None)
    assert replicated.layout.replicated == (size > 1)
    np.testing.assert_array_equal(xp.to_numpy(replicated.local), data)

    a = mpa.arange(float(size + 3), halo=1, periodic=True)
    a.update_halos()
    same = mpa.from_local(a.local, layout=a.layout)
    np.testing.assert_array_equal(xp.to_numpy(same.gather()), xp.to_numpy(a.gather()))
    with_halos = mpa.from_local(a.local_with_halos, layout=a.layout, with_halos=True)
    assert with_halos.local_with_halos is a.local_with_halos  # used as it is


def test_blocks_that_do_not_fit_raise_on_every_rank() -> None:
    layout = Layout((size + 3, 2))
    with pytest.raises(ValueError, match="do not match the layout's local shapes"):
        mpa.from_local(np.zeros((1, 5)), layout=layout)
    with pytest.raises(ValueError, match="with_halos=True needs a layout"):
        mpa.from_local(np.zeros(3), with_halos=True)
    with pytest.raises(ValueError, match="cannot be stacked along axis 2"):
        mpa.from_local(np.zeros((2, 3)), split=2)
    if size > 1:
        with pytest.raises(ValueError, match="cannot be stacked"):
            mpa.from_local(np.zeros((2, 3 + rank)))
        with pytest.raises(ValueError, match="cannot be stacked"):
            mpa.from_local(np.zeros(2, dtype=np.float32 if rank else np.float64))
