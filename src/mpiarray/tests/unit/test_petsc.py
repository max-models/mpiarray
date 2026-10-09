"""Tests for the PETSc interoperability (mpiarray.petsc).

The tests that need petsc4py skip without it (as in CI); run them with an
environment that has it, serially and under ``mpiexec -n 2/3/4``.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import Layout

comm = mpa.default_comm()
size = comm.Get_size()


@pytest.fixture
def PETSc() -> Any:  # noqa: N802 - PETSc's own name
    return pytest.importorskip("petsc4py.PETSc", reason="needs petsc4py")


# ---------------------------------------------------------------------- #
# block_numbering: no petsc4py needed


def reference_numbering(shape: tuple[int, ...], cuts: list[tuple[int, ...]]) -> Any:
    """Number the elements by storing the blocks one after another, in loops."""
    numbering = np.full(shape, -1, dtype=np.int64)
    row = 0
    grid = [range(len(c) - 1) for c in cuts]
    for coord in itertools.product(*grid):
        block = tuple(slice(c[k], c[k + 1]) for c, k in zip(cuts, coord, strict=True))
        n = numbering[block].size
        numbering[block] = np.arange(row, row + n).reshape(numbering[block].shape)
        row += n
    return numbering


@pytest.mark.parametrize(
    ("shape", "cuts"),
    [
        ((5,), [(0, 3, 3, 5)]),
        ((5,), [(0, 0, 5, 5)]),
        ((6, 4), [(0, 2, 2, 6), (0, 1, 4)]),
        ((6, 4), [(0, 6), (0, 0, 4, 4)]),
        ((4, 3, 5), [(0, 1, 4), (0, 3, 3), (0, 2, 2, 5)]),
        ((0, 3), [(0, 0, 0), (0, 1, 3)]),
    ],
)
def test_block_numbering_allows_empty_blocks(
    shape: tuple[int, ...], cuts: list[tuple[int, ...]]
) -> None:
    numbering = mpa.block_numbering(shape, cuts)
    assert numbering.dtype == np.int64 and numbering.shape == shape
    np.testing.assert_array_equal(numbering, reference_numbering(shape, cuts))
    assert sorted(numbering.ravel().tolist()) == list(range(numbering.size))


def test_block_numbering_of_a_layout_and_errors() -> None:
    layout = Layout((7, 5), comm=comm, split=None)
    np.testing.assert_array_equal(
        mpa.block_numbering(layout.shape, layout.bounds), np.arange(35).reshape(7, 5)
    )
    np.testing.assert_array_equal(mpa.block_numbering(4, [(0, 2, 4)]), np.arange(4))
    with pytest.raises(ValueError, match="cuts has 1 entries, expected 2"):
        mpa.block_numbering((4, 2), [(0, 4)])
    with pytest.raises(ValueError, match="must go from 0 to 4 without decreasing"):
        mpa.block_numbering(4, [(0, 3, 2, 4)])
    with pytest.raises(ValueError, match="along axis 0"):
        mpa.block_numbering(4, [(0, 3)])
    with pytest.raises(ValueError, match="along axis 1"):
        mpa.block_numbering((4, 2), [(0, 4), (1, 2)])
    with pytest.raises(ValueError, match="must go from"):
        mpa.block_numbering(4, [(4,)])


# ---------------------------------------------------------------------- #
# Without petsc4py: the methods raise a clear ImportError


class FakeVec:
    """Just enough of a global PETSc.Vec for the transfer functions: a host buffer."""

    def __init__(self, local_size: int, global_size: int) -> None:
        self.array = np.zeros(local_size)
        self.global_size = global_size

    def getLocalSize(self) -> int:  # noqa: N802 - PETSc's name
        return self.array.size

    def getSize(self) -> int:  # noqa: N802 - PETSc's name
        return self.global_size

    def getArray(self, readonly: bool = False) -> Any:  # noqa: N802 - PETSc's name
        return self.array

    def __enter__(self) -> Any:
        return self.array

    def __exit__(self, *args: object) -> None:
        pass


def test_methods_without_petsc4py(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    monkeypatch.setitem(sys.modules, "petsc4py.PETSc", None)
    a = mpa.zeros((size * 2, 3), halo=1)
    a.local[...] = np.arange(a.size).reshape(a.shape)[a.layout.global_slices()]
    with pytest.raises(ImportError, match=r"mpiarray\[petsc\]"):
        a.layout.dmda()
    with pytest.raises(ImportError, match=r"mpiarray\[petsc\]"):
        a.to_petsc()
    # a given vector needs no PETSc: the copies are local
    vec = FakeVec(a.local.size, a.size)
    assert a.to_petsc(vec) is vec
    np.testing.assert_array_equal(vec.array, np.asarray(a.local).ravel())
    b = mpa.zeros(layout=a.layout)
    b.copy_from_petsc(vec)
    np.testing.assert_array_equal(b.to_numpy(), a.to_numpy())
    with pytest.raises(ValueError, match="does not hold the block"):
        a.to_petsc(FakeVec(a.local.size + 1, a.size))
    with pytest.raises(ValueError, match="does not hold the block"):
        a.copy_from_petsc(FakeVec(a.local.size, a.size + 1))


# ---------------------------------------------------------------------- #
# Layouts to map onto DMDAs, for the number of ranks of this run


def uneven_cuts(length: int, n: int) -> tuple[int, ...]:
    """Return cuts with blocks of 1, 2, 3, ... elements and the rest in the last."""
    cuts = [0]
    for i in range(n - 1):
        cuts.append(cuts[-1] + i + 1)
    return (*cuts, length)


LAYOUTS: dict[str, Callable[[], Layout]] = {
    "1d": lambda: Layout(4 * size + 3, comm=comm),
    "1d-halo-periodic": lambda: Layout(4 * size + 1, comm=comm, halo=2, periodic=True),
    "1d-bounds": lambda: Layout(
        size * (size + 1) // 2 + 5,
        comm=comm,
        bounds=[uneven_cuts(size * (size + 1) // 2 + 5, size)],
    ),
    "2d-axis0": lambda: Layout((3 * size + 2, 5), comm=comm, halo=1),
    "2d-axis1": lambda: Layout((4, 3 * size + 1), comm=comm, split=1, halo=(0, 1)),
    "2d-grid": lambda: Layout((9, 11), comm=comm, split=(0, 1), periodic=(False, True)),
    "2d-bounds": lambda: Layout(
        (12, 10),
        comm=comm,
        bounds=[
            uneven_cuts(12, size // 2 if size % 2 == 0 else size),
            uneven_cuts(10, 2 if size % 2 == 0 else 1),
        ],
    ),
    "2d-reorder": lambda: Layout((10, 7), comm=comm, split=(0, 1), reorder=True),
    "3d-split01": lambda: Layout((7, 6, 5), comm=comm, split=(0, 1), halo=1),
    "3d-grid": lambda: Layout(
        (5, 6, 7), comm=comm, process_grid=(1, 1, size), periodic=(True, False, True)
    ),
    "replicated": lambda: Layout((6, 4), comm=comm, split=None),
}


@pytest.fixture(params=sorted(LAYOUTS))
def layout(request: pytest.FixtureRequest) -> Layout:
    return LAYOUTS[request.param]()


def test_dmda_has_the_layout_decomposition(PETSc: Any, layout: Layout) -> None:
    da = layout.dmda()
    assert da.getDim() == layout.ndim
    assert da.getSizes() == layout.shape[::-1]
    assert tuple(da.getRanges())[::-1] == layout.index_bounds
    if layout.replicated or layout.size == 1:
        assert da.getComm().getSize() == 1
    else:
        assert da.getComm().getSize() == layout.size
        assert da.getComm().getRank() == layout.rank
        assert da.getProcSizes() == layout.process_grid[::-1]
    periodic = PETSc.DM.BoundaryType.PERIODIC
    assert [b == periodic for b in da.getBoundaryType()][::-1] == list(layout.periodic)
    assert da.getStencilWidth() == max(1, *layout.halo)
    assert da.getStencilType() == PETSc.DMDA.StencilType.STAR
    # cached per layout: an equal layout gets the same DMDA
    assert layout.dmda() is da
    assert mpa.zeros(layout=layout).layout.dmda() is da


def test_petsc_numbering_matches_the_ao(PETSc: Any, layout: Layout) -> None:
    numbering = mpa.petsc_numbering(layout)
    assert numbering.shape == layout.shape and numbering.dtype == np.int64
    ao = layout.dmda().getAO()
    natural = np.arange(numbering.size, dtype=PETSc.IntType)
    np.testing.assert_array_equal(ao.app2petsc(natural), numbering.ravel())
    # this rank's rows are its block, C-ordered
    start, end = layout.dmda().createGlobalVec().getOwnershipRange()
    np.testing.assert_array_equal(
        numbering[layout.global_slices()].ravel(), np.arange(start, end)
    )


def test_round_trip(PETSc: Any, layout: Layout) -> None:
    a = mpa.zeros(layout=layout)
    a.local[...] = (
        np.arange(a.size, dtype=float).reshape(a.shape)[layout.global_slices()] + 0.5
    )
    vec = a.to_petsc()
    np.testing.assert_array_equal(vec.getArray(readonly=True), a.local.ravel())
    # the vector's natural order is mpiarray's global C order
    da = layout.dmda()
    natural = da.createNaturalVec()
    da.globalToNatural(vec, natural)
    lo, hi = natural.getOwnershipRange()
    np.testing.assert_array_equal(
        natural.getArray(readonly=True), np.arange(a.size)[lo:hi] + 0.5
    )

    b = mpa.from_petsc(vec, layout)
    assert b.layout == layout and b.dtype == PETSc.ScalarType
    np.testing.assert_array_equal(b.to_numpy(), a.to_numpy())
    assert b.local_with_halos.sum() == b.local.sum()  # zero halo cells

    # copy_from_petsc leaves the halo cells alone
    c = mpa.full(None, -1.0, layout=layout)
    vec.scale(2.0)
    c.copy_from_petsc(vec)
    np.testing.assert_array_equal(c.local, 2 * a.local)
    halo_cells = c.local_with_halos.size - c.local.size
    assert (c.local_with_halos == -1).sum() == halo_cells

    # into a given vector, and the functions
    other = da.createGlobalVec()
    assert a.to_petsc(other) is other
    mpa.copy_to_petsc(a, vec)
    assert vec.equal(other)
    d = mpa.zeros(layout=layout)
    mpa.copy_from_petsc(vec, d)
    np.testing.assert_array_equal(d.local, a.local)


def test_vector_of_another_layout_is_refused(PETSc: Any) -> None:
    a = mpa.zeros(2 * size + 1)
    vec = mpa.zeros(2 * size + 2).to_petsc()
    with pytest.raises(ValueError, match="does not hold the block"):
        a.copy_from_petsc(vec)
    with pytest.raises(ValueError, match="does not hold the block"):
        mpa.from_petsc(vec, a.layout)


def test_dmda_options_and_errors(PETSc: Any) -> None:
    layout = Layout((3 * size, 4), comm=comm, periodic=(True, False))
    da = layout.dmda(stencil_type="box", stencil_width=2, boundary_type="ghosted")
    assert da is not layout.dmda()
    assert da.getStencilType() == PETSc.DMDA.StencilType.BOX
    assert da.getStencilWidth() == 2
    ghosted = PETSc.DM.BoundaryType.GHOSTED
    assert da.getBoundaryType() == (ghosted, ghosted)
    per_axis = layout.dmda(
        boundary_type=(PETSc.DM.BoundaryType.PERIODIC, PETSc.DM.BoundaryType.MIRROR)
    )
    # given in the layout's axis order, reversed for PETSc
    assert per_axis.getBoundaryType() == (
        PETSc.DM.BoundaryType.MIRROR,
        PETSc.DM.BoundaryType.PERIODIC,
    )
    with pytest.raises(ValueError, match="1 to 3 axes, the layout has 4"):
        Layout((2 * size, 2, 2, 2), comm=comm).dmda()
    with pytest.raises(ValueError, match="1 to 3 axes, the layout has 0"):
        Layout((), comm=comm).dmda()
    with pytest.raises(ValueError, match="stencil_type must be 'star' or 'box'"):
        layout.dmda(stencil_type="cross")
    with pytest.raises(ValueError, match="boundary_type has 3 entries, expected 2"):
        layout.dmda(boundary_type=("none",) * 3)


def test_clear_dmda_cache(PETSc: Any) -> None:
    from mpiarray import petsc

    layout = Layout(2 * size + 3, comm=comm)
    da = layout.dmda()
    vec = da.createGlobalVec()
    petsc.clear_dmda_cache()
    assert layout.dmda() is not da
    vec.set(1.0)
    assert vec.sum() == 2 * size + 3


# ---------------------------------------------------------------------- #
# A Poisson solve with KSP


def laplacian(shape: tuple[int, ...]) -> Any:
    """Return the dense 5-point Dirichlet Laplacian (unit spacing), natural order."""
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


@pytest.mark.parametrize(
    "make",
    [
        lambda: Layout((13, 10), comm=comm, halo=1),
        lambda: Layout((13, 10), comm=comm, split=(0, 1)),
        lambda: Layout((12, 9), comm=comm, bounds=[uneven_cuts(12, size), None]),
    ],
)
def test_poisson_solve(PETSc: Any, make: Callable[[], Layout]) -> None:
    layout = make()
    da = layout.dmda()
    numbering = mpa.petsc_numbering(layout)
    nx, ny = layout.shape

    matrix = da.createMatrix()
    (x0, x1), (y0, y1) = layout.index_bounds
    for i, j in itertools.product(range(x0, x1), range(y0, y1)):
        columns = [numbering[i, j]]
        values = [4.0]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= i + di < nx and 0 <= j + dj < ny:
                columns.append(numbering[i + di, j + dj])
                values.append(-1.0)
        matrix.setValues([numbering[i, j]], columns, values)
    matrix.assemble()

    i, j = np.indices(layout.shape)
    f = mpa.zeros(layout=layout)
    f.local[...] = (np.sin(0.7 * i) * np.cos(0.3 * j) + 0.1 * i)[layout.global_slices()]
    rhs = f.to_petsc()
    solution = da.createGlobalVec()
    ksp = PETSc.KSP().create(comm=da.getComm())
    ksp.setOperators(matrix)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-13, atol=0.0, max_it=2000)
    ksp.solve(rhs, solution)
    assert ksp.getConvergedReason() > 0

    phi = mpa.from_petsc(solution, layout)
    expected = np.linalg.solve(laplacian(layout.shape), f.to_numpy().ravel())
    np.testing.assert_allclose(phi.to_numpy().ravel(), expected, rtol=1e-9, atol=1e-11)
    for obj in (ksp, matrix, rhs, solution):
        obj.destroy()
