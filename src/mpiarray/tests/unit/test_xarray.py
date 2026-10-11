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
        ("x", "y"), x=x, y=y, name="field", attrs={"units": "eV"}, root=root
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


def test_to_xarray_coords_dict_merges_with_axis_coords() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a = mpa.array(data)
    x = np.arange(float(N))
    result = a.to_xarray(("x", "y"), x=x, coords={"extra": ("x", x * 0)})
    assert set(result.coords) == {"x", "extra"}
    np.testing.assert_array_equal(result.coords["x"], x)


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


def test_labeled_sets_dims_and_returns_self() -> None:
    a = mpa.array(np.arange(2.0 * N).reshape(N, 2))
    assert a.dims is None
    labeled = a.labeled("x", "y")
    assert labeled is a
    assert a.dims == ("x", "y")


def test_labeled_rejects_the_wrong_number_of_names() -> None:
    a = mpa.array(np.arange(2.0 * N).reshape(N, 2))
    with pytest.raises(ValueError, match="one name per axis; got 1 for a 2-D array"):
        a.labeled("x")
    with pytest.raises(ValueError, match="one name per axis; got 3 for a 2-D array"):
        a.labeled("x", "y", "z")


def test_to_xarray_uses_labeled_dims_by_default() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a = mpa.array(data).labeled("x", "y")
    result = a.to_xarray()
    assert result.dims == ("x", "y")
    np.testing.assert_array_equal(result.to_numpy(), data)
    # an explicit dims= still overrides it
    assert a.to_xarray(("a", "b")).dims == ("a", "b")


def test_to_xarray_without_dims_or_labeled_raises() -> None:
    a = mpa.array(np.arange(2.0 * N).reshape(N, 2))
    with pytest.raises(ValueError, match="needs dims, or a.labeled"):
        a.to_xarray()
    with pytest.raises(ValueError, match="needs dims, or a.labeled"):
        mpa.to_xarray(a)


@pytest.mark.parametrize("split", [0, 1, None])
def test_labeled_dims_propagate_through_shape_preserving_operations(split) -> None:
    data = np.arange(float(N * N)).reshape(N, N)
    a = mpa.array(data, split=split, halo=1).labeled("x", "y")
    for result in (
        a.copy(),
        a.astype(np.float32),
        a + 1.0,
        -a,
    ):
        assert result.dims == ("x", "y")
    import mpiarray.layout as layout_mod

    other_split = 1 if split != 1 else 0
    redistributed = a.redistribute(
        layout_mod.Layout(data.shape, comm=comm, split=other_split, halo=1)
    )
    assert redistributed.dims == ("x", "y")
    rebalanced = a.rebalance(np.ones(data.shape))
    assert rebalanced.dims == ("x", "y")


def test_dims_do_not_propagate_through_shape_changing_operations() -> None:
    data = np.arange(2.0 * N).reshape(N, 2)
    a = mpa.array(data).labeled("x", "y")
    assert a.sum(axis=0).dims is None
    assert a[1:].dims is None
