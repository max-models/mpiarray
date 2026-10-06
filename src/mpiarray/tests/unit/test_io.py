"""Tests for mpa.save and mpa.load: .npy files written and read in parallel."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray.io import _header

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
W = 2 * size + 3


@pytest.fixture
def path(tmp_path):
    """Return one file path, the same on every rank (rank 0's temporary directory)."""
    return comm.bcast(str(tmp_path / "array.npy"), root=0)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, np.bool_])
@pytest.mark.parametrize("split", [0, (0, 1), None])
def test_save_writes_a_numpy_file(path, split, dtype) -> None:
    data = (np.arange(W * (W + 1) * 2).reshape(W, W + 1, 2) % 7).astype(dtype)
    a = mpa.array(data, split=split, halo=1)
    mpa.save(path, a)
    loaded = np.load(path)
    assert loaded.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(loaded, data)


@pytest.mark.parametrize("split", [0, 1, (0, 1), None])
def test_load_reads_each_block(path, split) -> None:
    data = np.arange(W * (W + 2), dtype=float).reshape(W, W + 2)
    if rank == 0:
        np.save(path, data)
    comm.Barrier()
    a = mpa.load(path, split=split, halo=1, periodic=True)
    assert a.layout.halo == (1, 1) and a.layout.periodic == (True, True)
    np.testing.assert_array_equal(xp.to_numpy(a.local), data[a.layout.global_slices()])
    same = mpa.load(path, layout=a.layout)
    assert same.layout is a.layout


def test_round_trip_through_another_layout(path) -> None:
    data = np.arange(W * W, dtype=np.float64).reshape(W, W)
    mpa.save(path, mpa.array(data, split=(0, 1)))
    a = mpa.load(path)  # default: split along the first axis
    np.testing.assert_array_equal(xp.to_numpy(a.gather()), data)


def test_fortran_ordered_and_empty_files(path) -> None:
    data = np.arange(W * 3, dtype=float).reshape(W, 3)
    if rank == 0:
        np.save(path, np.asfortranarray(data))
    comm.Barrier()
    np.testing.assert_array_equal(xp.to_numpy(mpa.load(path).gather()), data)
    empty = mpa.zeros((W, 0))
    mpa.save(path, empty)
    assert np.load(path).shape == (W, 0)
    assert mpa.load(path).shape == (W, 0)


def test_long_headers_use_format_version_2() -> None:
    many_fields = np.dtype([(f"field{i}", "u1") for i in range(6000)])
    header = _header((2,), many_fields)
    assert header.startswith(b"\x93NUMPY\x02\x00")
    assert _header((2, 3), np.dtype(float)).startswith(b"\x93NUMPY\x01\x00")


def test_load_reads_format_version_2(path) -> None:
    data = np.arange(W * 4, dtype=float).reshape(W, 4)
    if rank == 0:
        with open(path, "wb") as handle:
            np.lib.format.write_array(handle, data, version=(2, 0))
    comm.Barrier()
    np.testing.assert_array_equal(xp.to_numpy(mpa.load(path).gather()), data)
