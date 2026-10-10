"""Tests for NumPy functions on distributed arrays (__array_function__)."""

from __future__ import annotations

from typing import Any, cast

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray

MPI = maybempi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
N = size + 3


def _g(a) -> np.ndarray:
    return xp.to_numpy(a.gather()) if isinstance(a, DistributedArray) else np.asarray(a)


def test_elementwise_functions() -> None:
    data = np.linspace(-2.0, 3.0, 2 * N).reshape(N, 2) + 0.123
    a = mpa.array(data, halo=1)
    np.testing.assert_array_equal(
        _g(np.where(a > 0, a, -1.0)), np.where(data > 0, data, -1)
    )
    np.testing.assert_array_equal(_g(np.clip(a, -1, 1)), np.clip(data, -1, 1))
    np.testing.assert_array_equal(_g(np.clip(a, None, 0.5)), np.clip(data, None, 0.5))
    np.testing.assert_array_equal(_g(np.clip(a, min=0.0)), np.clip(data, 0.0, None))
    np.testing.assert_array_equal(_g(np.round(a, 1)), np.round(data, 1))
    np.testing.assert_array_equal(_g(np.around(a)), np.around(data))
    c = mpa.array(data + 2j * data)
    np.testing.assert_array_equal(_g(np.real(c)), data)
    np.testing.assert_array_equal(_g(np.imag(c)), 2 * data)
    with pytest.raises(TypeError, match="nonzero"):
        np.where(a > 0)
    with pytest.raises(TypeError, match="does not take"):
        np.clip(a, 0, 1, casting="unsafe")


def test_comparison_functions() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a, b = mpa.array(data), mpa.array(data + 1e-9)
    assert (
        _g(np.isclose(a, b)).all() and np.allclose(a, b) and not np.allclose(a, b + 1)
    )
    assert np.array_equal(a, a.copy()) and not np.array_equal(a, b)
    assert np.array_equal(a, mpa.array(data, split=None))  # other layout: moved first
    assert np.array_equal(data, a) and not np.array_equal(a, data[:2])
    nan = data.copy()
    nan[0, 0] = np.nan
    assert np.array_equal(mpa.array(nan), mpa.array(nan), equal_nan=True)
    assert not np.array_equal(mpa.array(nan), mpa.array(nan))


def test_reductions_norms_and_shape_functions() -> None:
    data = np.arange(1.0, 2 * N + 1).reshape(N, 2)
    a = mpa.array(data)
    assert np.amin(a) == data.min() and np.amax(a) == data.max()
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(data))
    vector = mpa.array(data[:, 0])
    for ord in (None, 1, 2, np.inf):
        assert np.linalg.norm(vector, ord) == pytest.approx(
            np.linalg.norm(data[:, 0], ord)
        )
    with pytest.raises(TypeError, match="ord=1"):
        np.linalg.norm(a, 1)
    with pytest.raises(TypeError, match="axis"):
        np.linalg.norm(a, axis=0)
    assert np.vdot(a, a) == pytest.approx(np.vdot(data, data))
    vdata = data[:, 0]
    v, w = mpa.array(vdata), mpa.array(vdata + 1.0)
    assert v.dot(w) == pytest.approx(np.dot(vdata, vdata + 1.0))
    assert np.dot(v, w) == pytest.approx(np.dot(vdata, vdata + 1.0))
    assert np.matmul(v, w) == pytest.approx(np.dot(vdata, vdata + 1.0))
    with pytest.raises(TypeError, match="1-D"):
        np.dot(a, a)
    with pytest.raises(TypeError, match="DistributedArray"):
        np.dot(v, vdata)
    assert np.shape(a) == (N, 2) and np.ndim(a) == 2 and np.size(a) == 2 * N
    assert np.size(a, 1) == 2
    assert isinstance(np.copy(a), DistributedArray)
    if hasattr(np, "astype"):
        assert np.astype(cast("Any", a), np.float32).dtype == np.float32


def test_like_functions_and_unsupported_functions() -> None:
    a = mpa.arange(float(N), halo=1)
    for function, value in [(np.zeros_like, 0), (np.ones_like, 1), (np.full_like, 4)]:
        args = (a, 4) if function is np.full_like else (a,)
        like = cast("Any", function)(*args)  # NumPy's stubs do not know the dispatch
        assert isinstance(like, DistributedArray) and like.layout is a.layout
        assert (_g(like) == value).all()
    assert cast("Any", np.empty_like(a)).layout is a.layout
    with pytest.raises(TypeError, match="no shape or subok"):
        np.zeros_like(a, shape=(3,))
    with pytest.raises(TypeError, match="no implementation found"):
        np.median(a)


@pytest.mark.parametrize("split", [0, 1, (0, 1), None])
def test_cumsum_and_cumprod(split) -> None:
    data = (np.arange(2 * N * (N + 1)).reshape(2 * N, N + 1) % 3 + 1).astype(float)
    a = mpa.array(data, split=split, halo=1)
    for axis in (0, 1, -1):
        np.testing.assert_allclose(
            _g(np.cumsum(a, axis=axis)), np.cumsum(data, axis=axis)
        )
        np.testing.assert_allclose(
            _g(a.cumprod(axis=axis)), np.cumprod(data, axis=axis)
        )
    assert a.cumsum(axis=0, dtype=np.float32).dtype == np.float32
    with pytest.raises(TypeError, match="flattens"):
        a.cumsum()
    line = mpa.array(data[:, 0])
    np.testing.assert_allclose(_g(line.cumsum()), np.cumsum(data[:, 0]))


@pytest.mark.parametrize("split", [0, 1, None])
def test_concatenate_and_stack(split) -> None:
    data = np.arange(2.0 * N * N).reshape(N, N, 2)
    a = mpa.array(data, split=split, halo=1)
    b = mpa.array(data + 100.0, split=split, halo=1)
    for axis in range(3):
        if size > 1 and split is not None and axis == split:
            with pytest.raises(TypeError, match="split axes"):
                np.concatenate([a, b], axis=axis)
            continue
        np.testing.assert_array_equal(
            _g(np.concatenate([a, b], axis=axis)),
            np.concatenate([data, data + 100.0], axis=axis),
        )
    for axis in (0, 1, -1):
        np.testing.assert_array_equal(
            _g(np.stack([a, b], axis=axis)), np.stack([data, data + 100.0], axis=axis)
        )
    with pytest.raises(ValueError, match="need at least one"):
        np.concatenate([])
    with pytest.raises(TypeError, match="every array to be one"):
        np.concatenate([a, data])
    mismatched_axis = next(axis for axis in range(3) if split is None or axis != split)
    with pytest.raises(ValueError, match="same layout"):
        np.concatenate([a, mpa.array(data, split=split, halo=0)], axis=mismatched_axis)
    with pytest.raises(ValueError, match="same shape"):
        np.stack([a, mpa.array(data[:-1])])
