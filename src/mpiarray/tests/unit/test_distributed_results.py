"""Results that stay distributed: reductions along axes, selections, argmin/argmax."""

from __future__ import annotations

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray

MPI = maybempi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
N = size + 2
SHAPE = (N + 1, N, 3)


def _data(dtype=float) -> np.ndarray:
    values = (np.arange(np.prod(SHAPE)).reshape(SHAPE) * 7 % 11).astype(float) - 3
    if np.dtype(dtype).kind == "c":
        return values + 1j * (values % 4)
    return values.astype(dtype)


@pytest.mark.parametrize("split", [(0, 1), 0, 1, None])
def test_reductions_along_axes_are_distributed_arrays(split) -> None:
    data = _data()
    a = mpa.array(data, split=split, halo=1)
    for axis in (0, 1, (0, 2), -1):
        result = a.sum(axis=axis)
        assert isinstance(result, DistributedArray)
        np.testing.assert_allclose(xp.to_numpy(result.gather()), data.sum(axis=axis))
        axes = [k % 3 for k in np.atleast_1d(axis)]
        unsplit = all(a.layout.process_grid[k] == 1 for k in axes)
        if unsplit and a.layout.distributed:  # nothing moves: the blocks stay put
            assert result.layout.size == a.layout.size and result.layout.distributed


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("dtype", [float, np.int64, complex])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), (1, 2)])
def test_var_and_std_along_axes(axis, dtype, ddof, keepdims) -> None:
    data = _data(dtype)
    a = mpa.array(data, split=(0, 1))
    for name in ("var", "std"):
        result = getattr(a, name)(axis=axis, ddof=ddof, keepdims=keepdims)
        expected = getattr(np, name)(data, axis=axis, ddof=ddof, keepdims=keepdims)
        assert isinstance(result, DistributedArray)
        np.testing.assert_allclose(xp.to_numpy(result.gather()), expected, rtol=1e-10)
    assert a.var(axis=(0, 1, 2)) == pytest.approx(data.var())


@pytest.mark.parametrize("dtype", [float, np.int64, bool])
@pytest.mark.parametrize("split", [(0, 1), 1, None])
def test_argmin_and_argmax(split, dtype) -> None:
    data = _data().astype(dtype) if dtype is not bool else _data() > 2
    a = mpa.array(data, split=split, halo=1)
    for name in ("argmin", "argmax"):
        assert getattr(a, name)() == getattr(np, name)(data)
        assert getattr(np, name)(a) == getattr(np, name)(data)
        for axis in (0, 1, 2, -2):
            for keepdims in (False, True):
                result = getattr(a, name)(axis=axis, keepdims=keepdims)
                expected = getattr(np, name)(data, axis=axis, keepdims=keepdims)
                np.testing.assert_array_equal(xp.to_numpy(result.gather()), expected)


def test_argmin_and_argmax_with_nan_and_errors() -> None:
    data = np.arange(float(3 * N)).reshape(N, 3)
    data[N - 1, 1] = np.nan
    data[1, 2] = np.nan
    a = mpa.array(data, split=(0, 1) if size > 1 else 0)
    assert a.argmin() == np.argmin(data) and a.argmax() == np.argmax(data)
    for axis in (0, 1):
        np.testing.assert_array_equal(
            xp.to_numpy(a.argmax(axis=axis).gather()), np.argmax(data, axis=axis)
        )
    with pytest.raises(TypeError, match="argmin is not supported for dtype complex128"):
        mpa.zeros(N, dtype=complex).argmin()
    with pytest.raises(ValueError, match="empty sequence"):
        mpa.zeros((0,), split=None).argmax()
    with pytest.raises(TypeError, match="out="):
        a.argmin(out=np.zeros(1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("split", [(0, 1), 1, None])
def test_selections_are_distributed_arrays(split) -> None:
    data = _data()
    a = mpa.array(data, split=split, halo=1)
    for index in [
        (slice(1, None, 2), slice(None), 1),
        (Ellipsis, 0),
        (2,),
        (slice(None, None, -1), slice(1, 3)),
        (slice(3, 3),),
    ]:
        result = a[index]
        assert isinstance(result, DistributedArray)
        assert result.shape == data[index].shape
        np.testing.assert_array_equal(xp.to_numpy(result.gather()), data[index])
    masked = a[xp.asarray(data > 4)]
    assert isinstance(masked, DistributedArray)
    np.testing.assert_array_equal(xp.to_numpy(masked.gather()), data[data > 4])
    assert a[1, 2, 0] == data[1, 2, 0]
