"""Tests for mpiarray.sparse.StencilOperator."""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray, Layout, StencilOperator

comm = mpa.default_comm()
size = comm.Get_size()

LAPLACIAN_2D = {
    "offsets": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    "coefficients": [4.0, -1.0, -1.0, -1.0, -1.0],
}


def dense_laplacian(shape: tuple[int, int]) -> np.ndarray:
    """The dense 5-point Dirichlet Laplacian (unit spacing), natural (C) order."""
    nx, ny = shape
    n = nx * ny
    matrix = np.zeros((n, n))
    index = np.arange(n).reshape(shape)
    for i, j in itertools.product(range(nx), range(ny)):
        matrix[index[i, j], index[i, j]] = 4.0
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= i + di < nx and 0 <= j + dj < ny:
                matrix[index[i, j], index[i + di, j + dj]] = -1.0
    return matrix


def dirichlet_apply(a: np.ndarray) -> np.ndarray:
    """``A @ a`` for the dense Laplacian above, without building the matrix."""
    out = 4 * a.copy()
    out[1:, :] -= a[:-1, :]
    out[:-1, :] -= a[1:, :]
    out[:, 1:] -= a[:, :-1]
    out[:, :-1] -= a[:, 1:]
    return out


@pytest.fixture
def grid() -> np.ndarray:
    nx, ny = 3 * size + 3, 5
    return np.arange(nx * ny, dtype=float).reshape(nx, ny)


def filled(layout: Layout, grid: np.ndarray) -> DistributedArray:
    a = mpa.zeros(layout=layout)
    a.local[...] = grid[layout.global_slices()]
    return a


# ---------------------------------------------------------------------- #
# Construction errors


def test_construction_errors() -> None:
    layout = Layout((3 * size + 3, 5), comm=comm, halo=1)
    with pytest.raises(ValueError, match="offsets but 1 coefficients"):
        StencilOperator(layout, offsets=[(0, 0), (1, 0)], coefficients=[1.0])
    with pytest.raises(ValueError, match="every offset must have 2 entries"):
        StencilOperator(layout, offsets=[(0, 0, 0)], coefficients=[1.0])
    with pytest.raises(ValueError, match="duplicate offset"):
        StencilOperator(
            layout,
            offsets=[(0, 0), (0, 0)],
            coefficients=[1.0, 2.0],
        )
    with pytest.raises(ValueError, match="the stencil needs halo widths"):
        StencilOperator(
            layout,
            offsets=[(-2, 0)],
            coefficients=[1.0],
        )
    with pytest.raises(ValueError, match="the stencil needs halo widths"):
        StencilOperator(
            layout,
            offsets=[(0, 0)],
            coefficients=[1.0],
            overrides={(0, 0): [((0, 2), 1.0)]},
        )
    with pytest.raises(ValueError, match="must have shape layout.local_shape"):
        StencilOperator(
            layout,
            offsets=[(0, 0)],
            coefficients=[np.zeros((1, 1))],
        )


def test_apply_rejects_mismatched_layout() -> None:
    a = StencilOperator(Layout((4, 4), comm=comm, halo=1), **LAPLACIAN_2D)
    u = mpa.zeros((4, 5), halo=1)
    with pytest.raises(ValueError, match="does not match the operator's layout"):
        a.apply(u)


def test_properties_and_repr() -> None:
    layout = Layout((4, 4), comm=comm, halo=1)
    operator = StencilOperator(
        layout,
        **LAPLACIAN_2D,
        overrides={(0, 0): [((0, 0), 1.0)]},
    )
    assert operator.layout == layout
    assert operator.offsets == tuple(LAPLACIAN_2D["offsets"])
    assert operator.coefficients == tuple(LAPLACIAN_2D["coefficients"])
    assert "5 offsets" in repr(operator)
    assert "1 overrides" in repr(operator)


# ---------------------------------------------------------------------- #
# apply(): periodic, Dirichlet walls, spatially varying coefficients, overrides


def test_apply_periodic(grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1, periodic=True)
    u = filled(layout, grid)
    operator = StencilOperator(layout, **LAPLACIAN_2D)
    result = operator.apply(u).to_numpy()
    nx, ny = grid.shape
    expected = (
        4 * grid
        - np.roll(grid, 1, axis=0)
        - np.roll(grid, -1, axis=0)
        - np.roll(grid, 1, axis=1)
        - np.roll(grid, -1, axis=1)
    )
    np.testing.assert_allclose(result, expected)


def test_apply_dirichlet_wall(grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    u = filled(layout, grid)
    operator = StencilOperator(layout, **LAPLACIAN_2D)
    result = operator.apply(u, boundary=0.0).to_numpy()
    np.testing.assert_allclose(result, dirichlet_apply(grid))


def test_apply_into_out(grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    u = filled(layout, grid)
    operator = StencilOperator(layout, **LAPLACIAN_2D)
    out = mpa.zeros(layout=layout)
    returned = operator.apply(u, out, boundary=0.0)
    assert returned is out
    np.testing.assert_allclose(out.to_numpy(), dirichlet_apply(grid))


def test_apply_with_varying_coefficient(grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    u = filled(layout, grid)
    weight = np.full(layout.local_shape, 2.0)
    operator = StencilOperator(
        layout,
        offsets=[(0, 0)],
        coefficients=[weight],
    )
    result = operator.apply(u).to_numpy()
    np.testing.assert_allclose(result, 2.0 * grid)


def test_apply_with_overrides(grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    u = filled(layout, grid)
    nx, ny = grid.shape
    corner = (nx - 1, ny - 1)
    operator = StencilOperator(
        layout,
        **LAPLACIAN_2D,
        overrides={(0, 0): [((0, 0), 1.0)], corner: [((0, 0), -2.0)]},
    )
    result = operator.apply(u, boundary=0.0).to_numpy()
    expected = dirichlet_apply(grid)
    expected[0, 0] = grid[0, 0]
    expected[corner] = -2.0 * grid[corner]
    np.testing.assert_allclose(result, expected)


def test_apply_1d() -> None:
    n = 4 * size + 3
    a = np.arange(n, dtype=float)
    layout = Layout(n, comm=comm, halo=1, periodic=True)
    u = filled(layout, a)
    operator = StencilOperator(
        layout,
        offsets=[(0,), (-1,), (1,)],
        coefficients=[2.0, -1.0, -1.0],
    )
    result = operator.apply(u).to_numpy()
    expected = 2 * a - np.roll(a, 1) - np.roll(a, -1)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------- #
# to_petsc_mat: skips without petsc4py


@pytest.fixture
def PETSc() -> Any:  # noqa: N802 - PETSc's own name
    return pytest.importorskip("petsc4py.PETSc", reason="needs petsc4py")


def test_to_petsc_mat_without_petsc4py(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    monkeypatch.setitem(sys.modules, "petsc4py.PETSc", None)
    layout = Layout((3 * size + 3, 5), comm=comm, halo=1)
    operator = StencilOperator(layout, **LAPLACIAN_2D)
    with pytest.raises(ImportError, match="pip install petsc4py"):
        operator.to_petsc_mat()


def test_to_petsc_mat_matches_dense(PETSc: Any, grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    operator = StencilOperator(layout, **LAPLACIAN_2D)
    matrix = operator.to_petsc_mat()
    u = filled(layout, grid)
    vec = u.to_petsc()
    av = matrix.createVecLeft()
    matrix.mult(vec, av)
    from_mat = mpa.from_petsc(av, layout).to_numpy()
    np.testing.assert_allclose(from_mat, dirichlet_apply(grid))
    expected = dense_laplacian(grid.shape)
    np.testing.assert_allclose(matrix.convert("dense").getDenseArray(), expected)
    matrix.destroy()


def test_to_petsc_mat_with_overrides(PETSc: Any, grid: np.ndarray) -> None:
    layout = Layout(grid.shape, comm=comm, halo=1)
    nx, ny = grid.shape
    operator = StencilOperator(
        layout,
        **LAPLACIAN_2D,
        overrides={(0, 0): [((0, 0), 1.0)]},
    )
    matrix = operator.to_petsc_mat()
    u = filled(layout, grid)
    vec = u.to_petsc()
    av = matrix.createVecLeft()
    matrix.mult(vec, av)
    result = mpa.from_petsc(av, layout).to_numpy()
    expected = dirichlet_apply(grid)
    expected[0, 0] = grid[0, 0]
    np.testing.assert_allclose(result, expected)
    matrix.destroy()
