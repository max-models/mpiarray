"""Tests for reductions along axes, which combine partial results instead of gathering."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa

MPI = xp.mpi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
N = size + 2
SHAPE = (N + 1, N, 3)
REDUCTIONS = ["sum", "prod", "min", "max", "mean", "all", "any"]


def _data(dtype) -> np.ndarray:
    values = np.arange(np.prod(SHAPE)).reshape(SHAPE) % 5 + 1
    if np.dtype(dtype).kind == "c":
        return (values - 2.5j * (values % 3)).astype(dtype)
    if np.dtype(dtype).kind == "b":
        return values % 4 != 0
    return values.astype(dtype)


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, (0, 1), (0, 2), (1, 2), (0, 1, 2), ()])
@pytest.mark.parametrize("split", [(0, 1), 1, None])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_, np.complex128])
def test_axis_reductions_match_numpy(dtype, split, axis, keepdims) -> None:
    data = _data(dtype)
    a = mpa.array(data, split=split, halo=1)
    for name in REDUCTIONS:
        if name == "prod" and dtype is np.bool_:
            continue
        result = getattr(a, name)(axis=axis, keepdims=keepdims)
        expected = getattr(np, name)(data, axis=axis, keepdims=keepdims)
        assert result.shape == expected.shape, name
        np.testing.assert_allclose(xp.to_numpy(result), expected, err_msg=name)


def test_axis_reductions_with_dtype_and_empty_axes() -> None:
    data = _data(np.int64)
    a = mpa.array(data, split=(0, 1))
    assert a.sum(axis=0, dtype=np.float32).dtype == np.float32
    np.testing.assert_allclose(xp.to_numpy(a.mean(axis=1)), data.mean(axis=1))
    empty = mpa.zeros((0, 3), split=None)
    np.testing.assert_array_equal(xp.to_numpy(empty.sum(axis=0)), np.zeros(3))
    with pytest.raises(ValueError, match="zero-size array to reduction operation min"):
        empty.min(axis=0)


def test_bad_axes_raise_like_numpy() -> None:
    a = mpa.zeros(SHAPE)
    with pytest.raises(np.exceptions.AxisError):
        a.sum(axis=3)
    with pytest.raises(ValueError, match="duplicate value in 'axis'"):
        a.max(axis=(0, -3))
