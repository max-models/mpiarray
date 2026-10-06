"""Tests for the creation functions: array, zeros, arange, linspace, fromfunction, ...

Lengths scale with the number of ranks, since a split axis needs at least one
element per rank; every test runs serially and under ``mpiexec -n N``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout, chunk_bounds
from mpiarray.creation import _check_identical

if TYPE_CHECKING:
    from mpiarray._mpi import Comm

MPI = maybempi.get_mpi()
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
N = size + 3  # an axis length that every rank count up to N can split
W = 2 * size + 3  # long enough for halo width 2 on every rank


def _np(data) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    return xp.to_numpy(data)


def test_array_splits_the_first_axis_near_evenly() -> None:
    data = list(range(1, N + 1))
    a = mpa.array(data)
    start, end = chunk_bounds(N, size, rank)
    np.testing.assert_array_equal(_np(a.local), data[start:end])
    np.testing.assert_array_equal(_np(a.gather()), data)
    assert a.layout.split == (0,) and a.layout.comm is comm
    assert a.dtype == np.dtype(int)


@pytest.mark.skipif(size < 2, reason="needs more ranks than elements")
def test_too_many_ranks_for_the_array_raise_on_every_rank() -> None:
    with pytest.raises(ValueError, match="split over"):
        mpa.array(list(range(size - 1)))
    with pytest.raises(ValueError, match="split over"):
        mpa.zeros((0, 3))
    with pytest.raises(ValueError, match="wider than the smallest block"):
        mpa.zeros(size, halo=2)
    assert mpa.array(list(range(size - 1)), split=None).gather().size == size - 1


@pytest.mark.parametrize("split", [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize("halo", [0, 1, (2, 0)])
def test_array_round_trips_for_any_layout(split, halo) -> None:
    data = np.arange(W * (W - 1), dtype=float).reshape(W, W - 1)
    a = mpa.array(data, split=split, halo=halo, periodic=(True, False))
    np.testing.assert_array_equal(_np(a.gather()), data)
    np.testing.assert_array_equal(_np(a.local), data[a.layout.global_slices()])
    halo_mask = np.ones(a.layout.storage_shape, dtype=bool)
    halo_mask[a.layout.interior] = False
    assert not _np(a.local_with_halos)[halo_mask].any()  # halo cells start at zero


def test_array_of_a_distributed_array_keeps_or_changes_its_layout() -> None:
    data = np.arange(float(W))
    a = mpa.array(data, halo=1, periodic=True)
    copied = mpa.array(a)
    assert copied.layout == a.layout
    assert copied.local_with_halos is not a.local_with_halos
    assert mpa.array(a, dtype=int).dtype == np.dtype(int)

    wider = mpa.array(a, halo=2)  # redistributed, the other options kept
    assert wider.layout.halo == (2,) and wider.layout.periodic == (True,)
    np.testing.assert_array_equal(_np(wider.gather()), data)
    replicated = mpa.array(a, split=None)
    assert replicated.layout.process_grid == (1,) and replicated.layout.halo == (1,)
    assert replicated.layout.replicated == (size > 1)
    np.testing.assert_array_equal(_np(replicated.local), data)
    again = mpa.array(replicated)  # keeps split=None
    assert again.layout == replicated.layout
    explicit = mpa.array(a, layout=Layout(W, split=None))
    assert explicit.layout.halo == (0,)
    other_comm = mpa.array(a, comm=MPI.COMM_SELF, periodic=False)
    assert other_comm.layout.size == 1 and other_comm.layout.periodic == (False,)
    np.testing.assert_array_equal(_np(other_comm.local), data)
    grid = mpa.array(a, process_grid=(size,))
    assert grid.layout == a.layout


def test_asarray_avoids_copies() -> None:
    a = mpa.arange(N, halo=1)
    assert mpa.asarray(a) is a
    assert mpa.asarray(a, dtype=a.dtype, layout=a.layout) is a
    assert mpa.asarray(a, halo=1, split=0) is a
    assert mpa.asarray(a, dtype=float).dtype == np.dtype(float)
    assert mpa.asarray(a, halo=0).layout.halo == (0,)
    other = mpa.asarray(a, layout=Layout(N, split=None))
    assert other.layout.split == ()
    from_list = mpa.asarray([1.0] * N, halo=1)
    assert from_list.layout.halo == (1,)


def test_process_grid_and_split_must_agree() -> None:
    grid = mpa.zeros((N, 3), process_grid=(size, 1))
    assert grid.layout.split == ((0,) if size > 1 else ())
    assert mpa.zeros((N, 3), process_grid=(size, 1), split=(0, 1)).layout == grid.layout
    if size > 1:
        with pytest.raises(ValueError, match="but split is"):
            mpa.zeros((N, 3), process_grid=(size, 1), split=1)


def test_debug_check_accepts_identical_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPIARRAY_DEBUG", "1")
    np.testing.assert_array_equal(_np(mpa.array(list(range(N))).gather()), range(N))


@pytest.mark.skipif(size < 2, reason="needs ranks that disagree")
def test_debug_check_rejects_different_data_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPIARRAY_DEBUG", "1")
    with pytest.raises(ValueError, match="different data on ranks"):
        mpa.array([rank] + list(range(N)))


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
    a = function((N, 3), dtype=np.int32, halo=1)
    assert a.shape == (N, 3) and a.dtype == np.int32
    assert a.local_with_halos.shape == a.layout.storage_shape
    if value is not None:
        assert (_np(a.local_with_halos) == value).all()
    b = function(layout=a.layout)
    assert b.layout is a.layout and b.dtype == np.dtype(float)


def test_full_and_like_constructors() -> None:
    a = mpa.full((N, 2), 7)
    assert a.dtype == xp.asarray(7).dtype
    assert a.sum() == 7 * 2 * N
    assert mpa.full((N, 2), 0.5, dtype=np.float32).dtype == np.float32

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
    assert mpa.full_like(a, 2.5, dtype=float).sum() == 2.5 * 2 * N


def test_shape_and_layout_must_agree() -> None:
    layout = Layout((N, 2))
    assert mpa.zeros((N, 2), layout=layout).layout is layout
    with pytest.raises(ValueError, match="does not match the layout"):
        mpa.zeros((N, 3), layout=layout)
    with pytest.raises(TypeError, match="a shape or a layout"):
        mpa.zeros()


@pytest.mark.parametrize(
    "args",
    [
        (W,),
        (2, 2 + 3 * W, 3),
        (5, 5 - 2 * W, -2),
        (0.0, 0.1 * W, 0.1),
        (-2.5, W),
    ],
)
def test_arange_matches_numpy(args) -> None:
    a = mpa.arange(*args, halo=1)
    expected = np.arange(*args)
    assert a.dtype == expected.dtype
    np.testing.assert_allclose(_np(a.gather()), expected)


def test_arange_options() -> None:
    assert mpa.arange(N, dtype=np.float32).dtype == np.float32
    assert mpa.arange(N, split=None).layout.replicated == (size > 1)
    assert mpa.arange(3, 3, split=None).gather().size == 0
    with pytest.raises(ValueError, match="step must not be zero"):
        mpa.arange(0, 5, 0)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0, 1, N), {}),
        ((0, 1, N), {"endpoint": False}),
        ((-3.0, 7.0, 2 * N + 1), {}),
        ((2, 2, N), {}),
        ((0, 1, 1), {"split": None}),
        ((0, 1, 0), {"split": None}),
        ((1, 0, 1), {"endpoint": False, "split": None}),
        ((0, 10, N + 1), {"dtype": int}),
    ],
)
def test_linspace_matches_numpy(args, kwargs) -> None:
    a = mpa.linspace(*args, **kwargs)
    kwargs.pop("split", None)
    expected = np.linspace(*args, **kwargs)
    assert a.dtype == expected.dtype
    np.testing.assert_array_equal(_np(a.gather()), expected)


def test_linspace_rejects_negative_num() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        mpa.linspace(0, 1, -1)


def test_fromfunction_matches_numpy() -> None:
    def f(i, j):
        return 10 * i + j

    shape = (2 * N, N)
    a = mpa.fromfunction(f, shape, split=(0, 1), halo=1)
    np.testing.assert_array_equal(_np(a.gather()), np.fromfunction(f, shape))
    ints = mpa.fromfunction(f, (N, 2), dtype=int)
    assert ints.dtype == np.dtype(int)
    constant = mpa.fromfunction(lambda i, j: 1.5, (N, 2))
    np.testing.assert_array_equal(_np(constant.gather()), np.full((N, 2), 1.5))
    scalar = mpa.fromfunction(lambda: 5.0, ())
    assert scalar.shape == () and scalar.get(()) == 5.0
