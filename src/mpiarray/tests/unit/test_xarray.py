"""Tests for the xarray interoperability (mpiarray.xarray)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import mpiarray as mpa

comm = mpa.default_comm()
size = comm.Get_size()
N = size + 3


@pytest.mark.parametrize("split", [0, 1, None])
@pytest.mark.parametrize("root", [None, 0])
def test_to_xarray(split, root) -> None:
    data = np.arange(float(N * (N + 1))).reshape(N, N + 1)
    a = mpa.array(data, split=split, halo=1)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 2.0, N + 1)
    result = a.to_xarray(
        ("x", "y"),
        coords={"x": x, "y": y},
        name="field",
        attrs={"units": "eV"},
        root=root,
    )
    if root is not None and comm.Get_rank() != root:
        assert result is None
        return
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x", "y")
    assert result.name == "field"
    assert result.attrs == {"units": "eV"}
    np.testing.assert_array_equal(result.coords["x"], x)
    np.testing.assert_array_equal(result.coords["y"], y)
    np.testing.assert_array_equal(result.to_numpy(), data)


def test_to_xarray_without_coords() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a = mpa.array(data)
    result = a.to_xarray(("x", "y"))
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x", "y")
    assert not result.coords
    np.testing.assert_array_equal(result.to_numpy(), data)


def test_to_xarray_module_function_matches_method() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a = mpa.array(data)
    np.testing.assert_array_equal(
        mpa.to_xarray(a, ("x", "y")).to_numpy(), a.to_xarray(("x", "y")).to_numpy()
    )


def test_to_xarray_wrong_number_of_dims() -> None:
    a = mpa.array(np.arange(2.0 * N).reshape(N, 2))
    with pytest.raises(ValueError, match="dims has 1 names for a 2-D array"):
        a.to_xarray(("x",))
    with pytest.raises(ValueError, match="dims has 3 names for a 2-D array"):
        a.to_xarray(("x", "y", "z"))
