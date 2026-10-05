"""Tests for the creation functions: array, zeros, arange, linspace, fromfunction, ..."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout, chunk_bounds
from mpiarray.creation import _check_identical

if TYPE_CHECKING:
    from mpiarray._mpi import Comm

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def _np(data) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    return xp.to_numpy(data)


def test_array_splits_the_first_axis_near_evenly() -> None:
    a = mpa.array([1, 2, 3, 4])
    start, end = chunk_bounds(4, size, rank)
    np.testing.assert_array_equal(_np(a.local), np.arange(1, 5)[start:end])
    np.testing.assert_array_equal(_np(a.gather()), [1, 2, 3, 4])
    assert a.layout.split == (0,) and a.layout.comm is comm
    assert a.dtype == np.dtype(int)


@pytest.mark.parametrize("split", [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize("halo", [0, 1, (2, 0)])
def test_array_round_trips_for_any_layout(split, halo) -> None:
    data = np.arange(7 * 5, dtype=float).reshape(7, 5)
    a = mpa.array(data, split=split, halo=halo, periodic=(True, False))
    np.testing.assert_array_equal(_np(a.gather()), data)
    np.testing.assert_array_equal(_np(a.local), data[a.layout.global_slices()])
    halo_mask = np.ones(a.layout.storage_shape, dtype=bool)
    halo_mask[a.layout.interior] = False
    assert not _np(a.local_with_halos)[halo_mask].any()  # halo cells start at zero


def test_array_of_a_distributed_array_copies_or_redistributes() -> None:
    a = mpa.array(np.arange(6.0), halo=1)
    copied = mpa.array(a)
    assert copied.layout == a.layout
    assert copied.local_with_halos is not a.local_with_halos
    as_int = mpa.array(a, dtype=int)
    assert as_int.dtype == np.dtype(int)

    replicated = mpa.array(a, layout=Layout(6, split=None))
    assert replicated.layout.split == ()
    np.testing.assert_array_equal(_np(replicated.local), np.arange(6.0))


def test_asarray_avoids_copies() -> None:
    a = mpa.arange(6)
    assert mpa.asarray(a) is a
    assert mpa.asarray(a, dtype=a.dtype, layout=a.layout) is a
    assert mpa.asarray(a, dtype=float).dtype == np.dtype(float)
    other = mpa.asarray(a, layout=Layout(6, split=None))
    assert other.layout.split == ()
    from_list = mpa.asarray([1.0, 2.0], halo=1)
    assert from_list.layout.halo == (1,)


def test_debug_check_accepts_identical_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPIARRAY_DEBUG", "1")
    np.testing.assert_array_equal(_np(mpa.array([1, 2, 3]).gather()), [1, 2, 3])


@pytest.mark.skipif(size < 2, reason="needs ranks that disagree")
def test_debug_check_rejects_different_data_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPIARRAY_DEBUG", "1")
    with pytest.raises(ValueError, match="different data on ranks"):
        mpa.array([rank, 1, 2])


def test_debug_check_names_the_ranks_that_differ() -> None:
    class Disagreeing:
        def allgather(self, value: str) -> list[str]:
            return [value, "something else", value]

    with pytest.raises(ValueError, match=r"ranks \[1\] than on rank 0"):
        _check_identical(xp.arange(3), cast("Comm", Disagreeing()))


@pytest.mark.parametrize(
    ("function", "value"), [(mpa.zeros, 0), (mpa.ones, 1), (mpa.empty, None)]
)
def test_constant_constructors(function, value) -> None:
    a = function((5, 3), dtype=np.int32, halo=1)
    assert a.shape == (5, 3) and a.dtype == np.int32
    assert a.local_with_halos.shape == a.layout.storage_shape
    if value is not None:
        assert (_np(a.local_with_halos) == value).all()
    b = function(layout=a.layout)
    assert b.layout is a.layout and b.dtype == np.dtype(float)


def test_full_and_like_constructors() -> None:
    a = mpa.full((4, 2), 7)
    assert a.dtype == xp.asarray(7).dtype
    assert a.sum() == 7 * 8
    assert mpa.full((4, 2), 0.5, dtype=np.float32).dtype == np.float32

    for like, value in [
        (mpa.zeros_like(a), 0),
        (mpa.ones_like(a), 1),
        (mpa.full_like(a, 3), 3),
        (mpa.empty_like(a), None),
    ]:
        assert like.layout is a.layout and like.dtype == a.dtype
        if value is not None:
            assert (_np(like.local_with_halos) == value).all()
    assert mpa.zeros_like(a, dtype=bool).dtype == np.bool_
    assert mpa.full_like(a, 2.5, dtype=float).sum() == 2.5 * 8


def test_shape_and_layout_must_agree() -> None:
    layout = Layout((4, 2))
    assert mpa.zeros((4, 2), layout=layout).layout is layout
    with pytest.raises(ValueError, match="does not match the layout"):
        mpa.zeros((2, 4), layout=layout)
    with pytest.raises(TypeError, match="a shape or a layout"):
        mpa.zeros()


@pytest.mark.parametrize(
    "args", [(10,), (2, 11, 3), (5, -7, -2), (0.0, 1.0, 0.1), (3, 3), (-2.5, 4)]
)
def test_arange_matches_numpy(args) -> None:
    a = mpa.arange(*args, halo=1)
    expected = np.arange(*args)
    assert a.dtype == expected.dtype
    np.testing.assert_allclose(_np(a.gather()), expected)


def test_arange_options() -> None:
    assert mpa.arange(4, dtype=np.float32).dtype == np.float32
    assert mpa.arange(6, split=None).layout.replicated == (size > 1)
    with pytest.raises(ValueError, match="step must not be zero"):
        mpa.arange(0, 5, 0)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0, 1, 5), {}),
        ((0, 1, 5), {"endpoint": False}),
        ((-3.0, 7.0, 11), {}),
        ((2, 2, 4), {}),
        ((0, 1, 1), {}),
        ((0, 1, 0), {}),
        ((1, 0, 1), {"endpoint": False}),
        ((0, 10, 6), {"dtype": int}),
    ],
)
def test_linspace_matches_numpy(args, kwargs) -> None:
    a = mpa.linspace(*args, **kwargs)
    expected = np.linspace(*args, **kwargs)
    assert a.dtype == expected.dtype
    np.testing.assert_array_equal(_np(a.gather()), expected)


def test_linspace_rejects_negative_num() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        mpa.linspace(0, 1, -1)


def test_fromfunction_matches_numpy() -> None:
    def f(i, j):
        return 10 * i + j

    a = mpa.fromfunction(f, (5, 4), split=(0, 1), halo=1)
    np.testing.assert_array_equal(_np(a.gather()), np.fromfunction(f, (5, 4)))
    ints = mpa.fromfunction(f, (3, 2), dtype=int)
    assert ints.dtype == np.dtype(int)
    constant = mpa.fromfunction(lambda i, j: 1.5, (3, 2))
    np.testing.assert_array_equal(_np(constant.gather()), np.full((3, 2), 1.5))
    scalar = mpa.fromfunction(lambda: 5.0, ())
    assert scalar.shape == () and scalar.get(()) == 5.0
