"""Tests for boundary conditions in update_halos, against numpy.pad of the global array."""

from __future__ import annotations

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray

MPI = maybempi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
L = 3 * size + 3  # blocks of at least 3, enough for halo 2 with "reflect"


def _padded(data: np.ndarray, a: DistributedArray, boundary) -> np.ndarray:
    """Return the global array padded axis by axis, as the halo update does."""
    padded = data
    for axis, (h, periodic) in enumerate(
        zip(a.layout.halo, a.layout.periodic, strict=True)
    ):
        width = [(0, 0)] * data.ndim
        width[axis] = (h, h)
        if periodic:
            padded = np.pad(padded, width, mode="wrap")
        elif boundary is None or boundary == "zero" or not isinstance(boundary, str):
            value = 0 if boundary in (None, "zero") else boundary
            padded = np.pad(padded, width, mode="constant", constant_values=value)
        else:
            padded = np.pad(padded, width, mode=boundary)
    return padded


def _expected_storage(data: np.ndarray, a: DistributedArray, boundary) -> np.ndarray:
    """Return what this rank's storage holds after the halo update."""
    padded = _padded(data, a, boundary)
    index = tuple(
        slice(start, end + 2 * h)
        for (start, end), h in zip(a.layout.index_bounds, a.layout.halo, strict=True)
    )
    return padded[index]


@pytest.mark.parametrize(
    "boundary", [None, "zero", 2.5, "edge", "symmetric", "reflect"]
)
@pytest.mark.parametrize("halo", [1, 2])
@pytest.mark.parametrize(
    ("shape", "split", "periodic"),
    [
        ((L,), 0, False),
        ((L, L - 1), (0, 1), False),
        ((L, L - 1), (0, 1), (True, False)),
        ((L, 5), None, (False, True)),
    ],
)
def test_boundaries_match_numpy_pad(shape, split, periodic, halo, boundary) -> None:
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) ** 1.5 + 1
    a = mpa.array(data, split=split, halo=halo, periodic=periodic)
    a.update_halos(boundary=boundary)
    np.testing.assert_allclose(
        xp.to_numpy(a.local_with_halos), _expected_storage(data, a, boundary)
    )


@pytest.mark.parametrize("boundary", ["edge", 7.0])
def test_boundaries_without_waiting_fill_the_faces(boundary) -> None:
    data = np.arange(L * (L - 1), dtype=float).reshape(L, L - 1)
    a = mpa.array(data, split=(0, 1), halo=1)
    pending = a.update_halos(wait=False, boundary=boundary)
    assert pending is not None
    pending.wait()
    expected = _expected_storage(data, a, boundary)
    got = xp.to_numpy(a.local_with_halos)
    np.testing.assert_allclose(got[1:-1, :], expected[1:-1, :])  # faces along axis 1
    np.testing.assert_allclose(got[:, 1:-1], expected[:, 1:-1])  # faces along axis 0


def test_boundaries_on_one_axis_only() -> None:
    data = np.arange(L * 4, dtype=float).reshape(L, 4)
    a = mpa.array(data, halo=1)
    a.update_halos(axis=1, boundary="edge")
    got = xp.to_numpy(a.local_with_halos)
    np.testing.assert_array_equal(got[1:-1, 0], got[1:-1, 1])
    assert not got[0].any() and not got[-1].any()  # axis 0 untouched


def test_bad_boundaries_raise_on_every_rank() -> None:
    a = mpa.zeros(L, halo=1)
    with pytest.raises(ValueError, match="unknown boundary 'wall'"):
        a.update_halos(boundary="wall")
    with pytest.raises(ValueError, match="unknown boundary True"):
        a.update_halos(boundary=True)
    narrow = mpa.zeros(2 * size, halo=2)  # blocks of exactly 2: too narrow to reflect
    with pytest.raises(ValueError, match="needs blocks of at least 3 elements"):
        narrow.update_halos(boundary="reflect")
