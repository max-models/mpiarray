"""Tests for mpa.save_hdf5 and mpa.load_hdf5."""

from __future__ import annotations

import sys

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa

h5py = pytest.importorskip("h5py")
MPI = maybempi.get_mpi()
comm = MPI.COMM_WORLD
size = comm.Get_size()
W = 2 * size + 3


@pytest.fixture
def path(tmp_path):
    return comm.bcast(str(tmp_path / "data.h5"), root=0)


def test_round_trip_with_attributes(path) -> None:
    rho = mpa.fromfunction(lambda i, j: i * 10.0 + j, (W, W + 1), split=(0, 1), halo=1)
    ids = mpa.arange(W, dtype=np.int32)
    mpa.save_hdf5(path, {"rho": rho, "ids": ids}, {"step": 7, "name": "run", "dt": 0.5})
    arrays, attrs = mpa.load_hdf5(path, halo=1)
    assert attrs == {"step": 7, "name": "run", "dt": 0.5}
    assert sorted(arrays) == ["ids", "rho"]
    np.testing.assert_array_equal(
        xp.to_numpy(arrays["rho"].gather()), xp.to_numpy(rho.gather())
    )
    assert arrays["ids"].dtype == np.int32 and arrays["ids"].layout.halo == (1,)
    with h5py.File(path, "r", locking=False) as handle:
        np.testing.assert_array_equal(handle["ids"][...], np.arange(W))
    only, _ = mpa.load_hdf5(path, ["ids"], split=None)
    assert list(only) == ["ids"] and only["ids"].layout.split == ()


def test_errors(path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        mpa.save_hdf5(path, {})
    if size > 1:
        with pytest.raises(ValueError, match="same communicator"):
            mpa.save_hdf5(
                path, {"a": mpa.zeros(W), "b": mpa.zeros(W, comm=MPI.COMM_SELF)}
            )
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="mpiarray\\[hdf5\\]"):
        mpa.load_hdf5(path)
