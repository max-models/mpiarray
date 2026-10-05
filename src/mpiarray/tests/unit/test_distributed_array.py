"""Tests for distributed array."""

from __future__ import annotations

import math
from typing import Any

import cunumpy as xp
import numpy as np
import pytest

from mpiarray import DistributedArray

MPI = xp.mpi.get_mpi()
xp.set_printoptions(precision=2, suppress=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()


def _as_numpy(data: xp.ndarray) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    if hasattr(data, "get"):
        return data.get()
    return np.asarray(data)


@pytest.mark.skipif(size != 2, reason="this test must run on exactly 2 MPI ranks")
@pytest.mark.parametrize("arr_len", [10, 100, 1000])
@pytest.mark.parametrize("num_ghost", [0, 1, 2, 3, 4])
def test_darray_fill(arr_len: int, num_ghost: int, verbose: bool = False) -> None:
    """Verify distributed array filling across MPI ranks."""
    comm = MPI.COMM_WORLD
    fill_data = xp.arange(arr_len, dtype=float)
    darray = DistributedArray(
        data=fill_data,
        shape=(arr_len,),
        comm=comm,
        decompose=[True],
        periodic=[True],
        num_ghostpoints=num_ghost,
        dtype=float,
    )

    # Check the data
    ndarray = darray.to_ndarray()
    assert xp.allclose(ndarray, fill_data)

    # Fill and check
    fill_data2 = fill_data * 2.0
    darray.fill(data=fill_data2)
    ndarray = darray.to_ndarray()
    assert xp.allclose(ndarray, fill_data2)

    # Check the copy method
    darray2 = darray.copy()
    assert darray != darray2
    assert xp.allclose(darray.to_ndarray(), darray2.to_ndarray())


@pytest.mark.skipif(size != 2, reason="this test must run on exactly 2 MPI ranks")
@pytest.mark.parametrize("arr_len", [10, 100, 1000])
@pytest.mark.parametrize("num_ghost", [1, 2, 3, 4])
def test_darray_halo_exchange(
    arr_len: int,
    num_ghost: int,
    verbose: bool = False,
) -> None:
    """Verify distributed array halo exchange."""
    comm = MPI.COMM_WORLD

    LEFT_FILL = 1.0
    RIGHT_FILL = 2.0
    LEFT_GHOST = 10.0
    RIGHT_GHOST = 100.0

    # Setup fill array
    fill_data = xp.ones(arr_len, dtype=float) * LEFT_FILL
    fill_data[arr_len // 2 :] = RIGHT_FILL

    # Create distributed array
    dist_array = DistributedArray(
        data=fill_data,
        shape=(arr_len,),
        comm=comm,
        decompose=[True],
        periodic=[False],
        num_ghostpoints=num_ghost,
        dtype=float,
    )

    # Check the data
    np_array = dist_array.to_ndarray()
    assert xp.allclose(np_array, fill_data)

    val = dist_array.get_global_value(0, root=0)
    print(f"{dist_array.mpi_rank =}, {val =}")
    if dist_array.mpi_rank == 0:
        assert val == LEFT_FILL
    else:
        assert val is None

    val = dist_array.get_global_value(arr_len // 2 + 1, root=0)
    print(f"{dist_array.mpi_rank =}, {val =}")
    if dist_array.mpi_rank == 0:
        assert val == RIGHT_FILL
    else:
        assert val is None

    # Set the right ghost point value for rank 0
    if dist_array.mpi_rank == 0:
        dist_array.data[-1] = LEFT_GHOST
    elif dist_array.mpi_rank == 1:
        dist_array.data[0] = RIGHT_GHOST

    # Exchange the halos
    dist_array.exchange_halos()

    # Check the exchanged halo values (rank 0)
    i1 = arr_len // 2 - num_ghost
    v1 = dist_array[i1]
    if v1 is not None:
        assert v1 == LEFT_FILL + RIGHT_GHOST, (
            f"{v1 =} should be {LEFT_FILL + RIGHT_GHOST}"
        )

    # Check the exchanged halo values (rank 1)
    i2 = arr_len // 2 + num_ghost - 1
    v2 = dist_array[i2]
    if v2 is not None:
        assert v2 == RIGHT_FILL + LEFT_GHOST, (
            f"{v2 =} should be {RIGHT_FILL + LEFT_GHOST}"
        )


@pytest.mark.skipif(size != 2, reason="this test must run on exactly 2 MPI ranks")
@pytest.mark.parametrize("arr_len", [10, 100, 1000])
@pytest.mark.parametrize("num_ghost", [1, 2, 3, 4])
def test_darray_magic_methods(
    arr_len: int,
    num_ghost: int,
    verbose: bool = False,
) -> None:
    """Verify distributed array arithmetic operations."""
    comm = MPI.COMM_WORLD

    FILL_VALUE1 = 1.0
    FILL_VALUE2 = 3.0
    ONES = xp.ones(arr_len, dtype=float)
    darray1 = DistributedArray(
        data=ONES * FILL_VALUE1,
        shape=(arr_len,),
        comm=comm,
        decompose=[True],
        periodic=[True],
        num_ghostpoints=num_ghost,
        dtype=float,
    )

    darray2 = DistributedArray(
        data=ONES * FILL_VALUE2,
        shape=(arr_len,),
        comm=comm,
        decompose=[True],
        periodic=[True],
        num_ghostpoints=num_ghost,
        dtype=float,
    )
    if verbose:
        print(f"{darray1 =}")
        print(f"{darray2 =}")

    # Check +
    darray3 = darray1 + FILL_VALUE2
    assert xp.allclose(darray3.to_ndarray(), FILL_VALUE1 + FILL_VALUE2)

    # Check -
    darray3 = darray1 - FILL_VALUE2
    assert xp.allclose(darray3.to_ndarray(), FILL_VALUE1 - FILL_VALUE2)

    # Check *
    darray3 = darray1 * FILL_VALUE2
    assert xp.allclose(darray3.to_ndarray(), FILL_VALUE1 * FILL_VALUE2)

    # Check /
    darray3 = darray1 / FILL_VALUE2
    assert xp.allclose(darray3.to_ndarray(), FILL_VALUE1 / FILL_VALUE2)


@pytest.mark.skipif(size < 2, reason="this test must run on at least 2 MPI ranks")
@pytest.mark.parametrize("Nx", [2, 4, 8, 16])
@pytest.mark.parametrize("Ny", [2, 4, 8, 16])
@pytest.mark.parametrize("Ncomp", [1, 2, 3, 4])
@pytest.mark.parametrize("num_ghost", [0, 1, 2, 3, 4])
def test_darray_multiple_components(
    Nx: int,
    Ny: int,
    Ncomp: int,
    num_ghost: int,
    verbose: bool = False,
) -> None:
    """Verify distributed arrays with multiple components."""
    dist_array = DistributedArray(
        # data=fill_data,
        shape=(Nx, Ny, Ncomp),
        comm=comm,
        decompose=[True, True, False],
        periodic=[False, False, False],
        num_ghostpoints=num_ghost,
        dtype=float,
    )

    dist_array._data[:] = dist_array.mpi_rank + 1

    # Gather all the data
    ndarray = dist_array.to_ndarray()
    if verbose:
        print(f"{ndarray =}")

    # if comm.Get_rank() == 0:
    #     pprint(ndarray.tolist())


@pytest.mark.skipif(size < 2, reason="this test must run on at least 2 MPI ranks")
def test_darray_periodic_fill_halos_2d() -> None:
    """Verify periodic ghost fills in both decomposed spatial directions."""
    nx, ny, ncomp = 8, 6, 3
    global_data = xp.arange(nx * ny * ncomp, dtype=float).reshape(nx, ny, ncomp)
    dist_array = DistributedArray(
        data=global_data,
        shape=global_data.shape,
        comm=comm,
        decompose=[True, True, False],
        periodic=[True, True, False],
        num_ghostpoints=1,
        ghost_axes=[True, True, False],
        dtype=float,
    )

    dist_array.fill_halos()
    data = dist_array.data
    xs, xe = dist_array.proc_index_bounds[0]
    ys, ye = dist_array.proc_index_bounds[1]

    expected_left = global_data[(xs - 1) % nx, ys:ye, :]
    expected_right = global_data[xe % nx, ys:ye, :]
    expected_bottom = global_data[xs:xe, (ys - 1) % ny, :]
    expected_top = global_data[xs:xe, ye % ny, :]

    assert data.shape[-1] == ncomp
    assert xp.allclose(data[0, 1:-1, :], expected_left)
    assert xp.allclose(data[-1, 1:-1, :], expected_right)
    assert xp.allclose(data[1:-1, 0, :], expected_bottom)
    assert xp.allclose(data[1:-1, -1, :], expected_top)


def test_darray_numpy_like_constructors_and_properties() -> None:
    """Verify convenience constructors and array-like metadata."""
    data = xp.arange(12, dtype=float).reshape(3, 4)
    dist_array = DistributedArray.from_array(
        data,
        comm=comm,
        decompose=[True, False],
        periodic=[False, False],
        num_ghostpoints=1,
        ghost_axes=[True, False],
    )

    assert dist_array.shape == data.shape
    assert dist_array.size == data.size
    assert dist_array.local_size == math.prod(dist_array.shape_local)
    assert dist_array.shape_with_halos == dist_array.data.shape
    assert xp.allclose(dist_array.to_ndarray(), data)
    numpy_data = dist_array.to_numpy()
    assert isinstance(numpy_data, np.ndarray)
    assert np.allclose(numpy_data, _as_numpy(data))
    assert xp.allclose(dist_array.T, data.T)

    ones = DistributedArray.ones(
        shape=data.shape,
        comm=comm,
        decompose=[True, False],
        periodic=[False, False],
        dtype=float,
    )
    assert xp.allclose(ones.to_ndarray(), xp.ones_like(data))

    full = DistributedArray.full(
        shape=data.shape,
        fill_value=7,
        comm=comm,
        decompose=[True, False],
        periodic=[False, False],
        dtype=int,
    )
    assert xp.allclose(full.to_ndarray(), xp.full(data.shape, 7))

    inferred = DistributedArray.zeros(shape=data.shape, comm=comm, dtype=float)
    assert inferred.ndim == len(data.shape)
    assert xp.allclose(inferred.to_ndarray(), xp.zeros_like(data))


def test_darray_numpy_like_arithmetic_and_slicing() -> None:
    """Verify scalar, ndarray, reflected, and in-place elementwise operations."""
    data = xp.arange(12, dtype=float).reshape(3, 4)
    dist_array = DistributedArray.from_array(
        data,
        comm=comm,
        decompose=[True, False],
        periodic=[False, False],
    )

    assert xp.allclose((dist_array + 2).to_ndarray(), data + 2)
    assert xp.allclose((2 + dist_array).to_ndarray(), 2 + data)
    assert xp.allclose((10 - dist_array).to_ndarray(), 10 - data)
    assert xp.allclose((dist_array * data).to_ndarray(), data * data)
    assert xp.allclose((dist_array**2).to_ndarray(), data**2)
    assert xp.allclose(abs(-dist_array).to_ndarray(), data)

    updated = dist_array.copy()
    updated += 3
    assert xp.allclose(updated.to_ndarray(), data + 3)

    mask = dist_array >= 5
    assert xp.allclose(mask.to_ndarray(), data >= 5)

    assert xp.allclose(dist_array[:, 1], data[:, 1])
    edited = dist_array.copy()
    replacement = xp.asarray([20, 21, 22], dtype=float)
    edited[:, 1] = replacement
    expected = data.copy()
    expected[:, 1] = replacement
    assert xp.allclose(edited.to_ndarray(), expected)


def test_darray_numpy_like_reductions_and_ufuncs() -> None:
    """Verify common reductions and ufunc interoperability."""
    data = xp.arange(1, 13, dtype=float).reshape(3, 4)
    dist_array = DistributedArray.from_array(
        data,
        comm=comm,
        decompose=[True, False],
        periodic=[False, False],
    )

    assert xp.allclose(dist_array.sum(), xp.sum(data))
    assert xp.allclose(dist_array.mean(), xp.mean(data))
    assert xp.allclose(dist_array.min(), xp.min(data))
    assert xp.allclose(dist_array.max(), xp.max(data))
    assert xp.allclose(dist_array.std(), xp.std(data))
    assert xp.allclose(dist_array.sum(axis=0), xp.sum(data, axis=0))
    assert xp.allclose((dist_array**0.5).to_ndarray(), xp.sqrt(data))

    as_int = dist_array.astype(int)
    assert as_int.dtype == xp.dtype(int)
    assert xp.allclose(as_int.to_ndarray(), data.astype(int))


# -------------------------------------------------------------------------- #
# Rank-count agnostic tests: these run serially and under any ``mpirun -n N``.

# (shape, decompose, ghost_axes)
HALO_LAYOUTS = [
    ((16,), [True], [True]),
    ((12, 10), [True, True], [True, True]),
    ((12, 10, 3), [True, True, False], [True, True, False]),
    ((8, 6, 6), [True, True, True], [True, True, True]),
]


def _periodic_flags(ghost_axes: list[bool], mode: str) -> list[bool]:
    """Return periodicity flags: all, none, or only the first ghost axis."""
    if mode == "all":
        return list(ghost_axes)
    if mode == "none":
        return [False] * len(ghost_axes)
    return [i == 0 for i in range(len(ghost_axes))]


def _comm(kind: str) -> MPI.Comm | None:
    """Return the communicator for a parametrized test."""
    return None if kind == "none" else comm


def _padded_global_indices(
    darray: DistributedArray,
    rank: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return, per axis, the global index of every local storage cell of ``rank``.

    Periodic axes are wrapped; the second list flags indices inside the domain.
    """
    g = darray.num_ghostpoints
    indices, valid = [], []
    for axis, (start, end) in enumerate(darray.get_index_bounds(rank)):
        pad = g if darray.ghost_axes[axis] else 0
        idx = np.arange(start - pad, end + pad)
        if darray.periodic[axis]:
            idx %= darray.shape[axis]
        indices.append(idx)
        valid.append((idx >= 0) & (idx < darray.shape[axis]))
    return indices, valid


def _rank_block(darray: DistributedArray, rank: int, dtype: type) -> np.ndarray:
    """Return deterministic local storage values (halos included) for ``rank``."""
    indices, _ = _padded_global_indices(darray, rank)
    rng = np.random.default_rng(1234 + rank)
    return rng.integers(1, 10, size=[len(i) for i in indices]).astype(dtype)


@pytest.mark.parametrize("comm_kind", ["world", "none"])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("periodic_mode", ["all", "none", "first"])
@pytest.mark.parametrize("num_ghost", [1, 2])
@pytest.mark.parametrize(("shape", "decompose", "ghost_axes"), HALO_LAYOUTS)
def test_exchange_halos_matches_serial_reference(
    shape: tuple[int, ...],
    decompose: list[bool],
    ghost_axes: list[bool],
    num_ghost: int,
    periodic_mode: str,
    dtype: type,
    comm_kind: str,
) -> None:
    """Halo accumulation equals folding every rank's storage into the global array."""
    darray = DistributedArray.zeros(
        shape=shape,
        comm=_comm(comm_kind),
        num_ghostpoints=num_ghost,
        ghost_axes=ghost_axes,
        decompose=decompose,
        periodic=tuple(_periodic_flags(ghost_axes, periodic_mode)),
        dtype=dtype,
    )
    darray.local_with_halos[...] = xp.asarray(
        _rank_block(darray, darray.mpi_rank, dtype),
    )

    expected = np.zeros(shape, dtype=dtype)
    for rank in range(darray.mpi_size):
        indices, valid = _padded_global_indices(darray, rank)
        block = _rank_block(darray, rank, dtype)
        np.add.at(
            expected,
            np.ix_(*[idx[mask] for idx, mask in zip(indices, valid, strict=True)]),
            block[np.ix_(*valid)],
        )

    darray.exchange_halos()

    assert darray.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(darray.to_numpy(), expected)
    halo_mask = np.ones(darray.shape_with_halos, dtype=bool)
    halo_mask[darray.get_local_slices()] = False
    assert not _as_numpy(darray.local_with_halos)[halo_mask].any()


@pytest.mark.parametrize("comm_kind", ["world", "none"])
@pytest.mark.parametrize("periodic_mode", ["all", "none", "first"])
@pytest.mark.parametrize("num_ghost", [1, 2])
@pytest.mark.parametrize(("shape", "decompose", "ghost_axes"), HALO_LAYOUTS)
def test_fill_halos_matches_serial_reference(
    shape: tuple[int, ...],
    decompose: list[bool],
    ghost_axes: list[bool],
    num_ghost: int,
    periodic_mode: str,
    comm_kind: str,
) -> None:
    """Ghost fills (corners included) equal the global neighbours of each cell."""
    global_data = np.arange(math.prod(shape), dtype=float).reshape(shape) + 1.0
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=_comm(comm_kind),
        num_ghostpoints=num_ghost,
        ghost_axes=ghost_axes,
        decompose=decompose,
        periodic=tuple(_periodic_flags(ghost_axes, periodic_mode)),
    )

    darray.fill_halos()

    indices, valid = _padded_global_indices(darray, darray.mpi_rank)
    inside = np.ix_(*valid)
    expected = np.zeros(darray.shape_with_halos)
    expected[inside] = global_data[
        np.ix_(*[idx[mask] for idx, mask in zip(indices, valid, strict=True)])
    ]
    np.testing.assert_array_equal(_as_numpy(darray.local_with_halos), expected)


def test_local_shapes_and_bounds_for_every_rank() -> None:
    """Every rank reports the same, consistent bounds and shapes for all ranks."""
    darray = DistributedArray.zeros(
        shape=(9, 7, 2),
        comm=comm,
        decompose=[True, True, False],
        num_ghostpoints=1,
    )
    shapes = [darray.get_shape_local(rank) for rank in range(size)]
    bounds = [darray.get_index_bounds(rank) for rank in range(size)]

    assert shapes == comm.allgather(darray.shape_local)
    assert bounds == comm.allgather(darray.proc_index_bounds)
    assert darray.get_shape_local() == darray.shape_local
    assert sum(math.prod(shape) for shape in shapes) == darray.size
    for rank_shape, rank_bounds in zip(shapes, bounds, strict=True):
        assert rank_shape == tuple(end - start for start, end in rank_bounds)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.bool_, np.complex128])
@pytest.mark.parametrize("num_ghost", [0, 2])
def test_to_ndarray_gathers_uneven_blocks(dtype: type, num_ghost: int) -> None:
    """Gathering reassembles uneven N-D blocks of any dtype on every rank."""
    global_data = (np.arange(7 * 5 * 2).reshape(7, 5, 2) % 3).astype(dtype)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        decompose=[True, True, False],
        num_ghostpoints=num_ghost,
        ghost_axes=[True, True, False],
    )

    gathered = darray.to_numpy()

    assert gathered.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(gathered, global_data)
    # The gathered array must not alias local storage.
    darray.local_with_halos[...] = 0
    np.testing.assert_array_equal(gathered, global_data)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("shape", [(1,), (1, 3), (2, 1)])
def test_reductions_with_ranks_owning_no_cells(
    dtype: type,
    shape: tuple[int, ...],
) -> None:
    """Reductions stay correct and collective when some ranks own no cells."""
    global_data = (np.arange(math.prod(shape)).reshape(shape) + 2).astype(dtype)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        num_ghostpoints=1,
    )

    assert darray.sum() == global_data.sum()
    assert darray.prod() == global_data.prod()
    assert darray.min() == global_data.min()
    assert darray.max() == global_data.max()
    assert np.isclose(darray.mean(), global_data.mean())
    assert darray.all() == global_data.all()
    assert darray.any() == global_data.any()


def test_min_max_of_zero_size_array_raise_on_every_rank() -> None:
    """A zero-size array raises (collectively) for min/max and sums to zero."""
    darray = DistributedArray.zeros(shape=(0, 3), comm=comm)

    with pytest.raises(ValueError, match="zero-size"):
        darray.min()
    with pytest.raises(ValueError, match="zero-size"):
        darray.max()
    assert darray.sum() == 0


def test_numpy_reduction_functions_dispatch_to_distributed_array() -> None:
    """``np.sum`` and friends work on distributed arrays, including keywords."""
    global_data = np.arange(1, 25, dtype=float).reshape(6, 4)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        num_ghostpoints=1,
    )

    assert np.isclose(np.sum(darray), global_data.sum())
    assert np.isclose(np.prod(darray / 10), np.prod(global_data / 10))
    assert np.min(darray) == global_data.min()
    assert np.max(darray) == global_data.max()
    assert np.isclose(np.mean(darray), global_data.mean())
    assert np.isclose(np.var(darray), global_data.var())
    assert np.isclose(np.var(darray, ddof=1), global_data.var(ddof=1))
    assert np.isclose(np.std(darray, ddof=1), global_data.std(ddof=1))
    assert np.all(darray) == global_data.all()
    assert np.any(darray > 23) == (global_data > 23).any()
    assert np.sum(darray, dtype=np.float32).dtype == np.float32

    kept = np.sum(darray, keepdims=True)
    assert np.shape(kept) == (1, 1)
    assert np.isclose(float(_as_numpy(kept)[0, 0]), global_data.sum())
    np.testing.assert_allclose(
        _as_numpy(np.mean(darray, axis=0)),
        global_data.mean(axis=0),
    )
    np.testing.assert_allclose(
        _as_numpy(np.max(darray, axis=1, keepdims=True)),
        global_data.max(axis=1, keepdims=True),
    )

    with pytest.raises(TypeError, match="out="):
        np.sum(darray, out=np.empty(()))


@pytest.mark.parametrize(
    "operand_shape",
    [(3,), (1, 1, 3), (6, 1), (8, 1, 1), (8, 6, 1), (8, 6, 3), (1,)],
)
def test_broadcast_operands_match_global_numpy_semantics(
    operand_shape: tuple[int, ...],
) -> None:
    """Array operands broadcast like NumPy against the global shape on any layout."""
    shape = (8, 6, 3)
    global_data = np.arange(math.prod(shape), dtype=float).reshape(shape)
    operand = np.arange(math.prod(operand_shape), dtype=float).reshape(operand_shape)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        decompose=[True, True, False],
        num_ghostpoints=1,
        ghost_axes=[True, True, False],
    )
    xp_operand = xp.asarray(operand)

    np.testing.assert_allclose((darray + xp_operand).to_numpy(), global_data + operand)
    np.testing.assert_allclose((xp_operand - darray).to_numpy(), operand - global_data)
    np.testing.assert_allclose(
        np.multiply(darray, xp_operand).to_numpy(),
        global_data * operand,
    )
    darray += xp_operand
    np.testing.assert_allclose(darray.to_numpy(), global_data + operand)


@pytest.mark.parametrize("operand_shape", [(5,), (7, 1, 1), (2, 8, 6, 3)])
def test_incompatible_operands_raise_on_every_rank(
    operand_shape: tuple[int, ...],
) -> None:
    """Operands that cannot broadcast to the global shape raise a clear error."""
    darray = DistributedArray.zeros(
        shape=(8, 6, 3),
        comm=comm,
        decompose=[True, True, False],
    )
    with pytest.raises(ValueError, match="global shape|more dimensions"):
        darray + xp.ones(operand_shape)


class _TaggedArray(DistributedArray):
    """Subclass used to check that results keep their type."""


def test_results_keep_subclass_and_do_not_alias() -> None:
    """Copies and elementwise results keep the subclass and own their data."""
    darray = _TaggedArray.from_array(
        xp.arange(10, dtype=float),
        comm=comm,
        num_ghostpoints=1,
    )

    results = [
        darray.copy(),
        darray + 1,
        -darray,
        abs(darray),
        darray > 3,
        np.sqrt(darray),
        darray.astype(np.float32),
        darray._new_like(darray.data),
    ]
    for result in results:
        assert type(result) is _TaggedArray
        assert result.shape_with_halos == darray.shape_with_halos
        assert result.data is not darray.data
        assert not np.may_share_memory(_as_numpy(result.data), _as_numpy(darray.data))
    assert "_TaggedArray" in repr(darray)
    assert darray.astype(float, copy=False) is darray

    copied = darray.copy()
    copied += 100
    np.testing.assert_array_equal(darray.to_numpy(), np.arange(10, dtype=float))


def test_empty_constructor_layout() -> None:
    """``empty`` allocates storage with the requested layout and dtype."""
    darray = DistributedArray.empty(
        shape=(6, 4),
        comm=comm,
        num_ghostpoints=2,
        ghost_axes=[True, False],
        dtype=np.int32,
    )
    expected_shape = (darray.shape_local[0] + 4, darray.shape_local[1])
    assert darray.shape_with_halos == expected_shape
    assert darray.dtype == np.int32


def test_invalid_inputs_raise_value_error() -> None:
    """Invalid shapes and dtypes raise ``ValueError`` instead of asserting."""
    darray = DistributedArray.zeros(shape=(6, 4), comm=comm, num_ghostpoints=1)

    with pytest.raises(ValueError, match="global data shape"):
        darray.fill(xp.zeros((5, 4)))
    with pytest.raises(ValueError, match="dtype"):
        darray.fill(xp.zeros((6, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="local data shape"):
        darray.fill_local(xp.zeros((1, 1)))
    with pytest.raises(ValueError, match="ghost_axes"):
        DistributedArray.zeros(shape=(6, 4), comm=comm, ghost_axes=[True])
    with pytest.raises(ValueError, match="dimensions"):
        DistributedArray.zeros(shape=(6, 4), comm=comm, ndim=3)


def test_get_global_value_on_any_rank_count() -> None:
    """``get_global_value`` returns the owner's value on root and ``None`` elsewhere."""
    global_data = np.arange(12, dtype=float).reshape(4, 3)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        num_ghostpoints=1,
    )
    for index in [(0, 0), (3, 2), (-1, 1)]:
        value = darray.get_global_value(index, root=0)
        if darray.mpi_rank == 0:
            assert value == global_data[index]
        else:
            assert value is None


@pytest.mark.parametrize("periodic", [True, False])
def test_replicated_layout_without_decomposed_axes(periodic: bool) -> None:
    """Without decomposed axes every rank holds, and works on, the whole array."""
    global_data = np.arange(1, 9, dtype=float).reshape(4, 2)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        decompose=[False, False],
        periodic=(periodic, False),
        num_ghostpoints=1,
        ghost_axes=[True, False],
    )

    assert darray.shape_local == global_data.shape
    assert darray.proc_index_bounds == [(0, 4), (0, 2)]
    assert not darray.is_distributed
    np.testing.assert_array_equal(darray.to_numpy(), global_data)
    # Reductions must not count the array once per rank.
    assert darray.sum() == global_data.sum()
    assert np.isclose(darray.var(), global_data.var())
    assert darray.get_global_value((3, 1)) == (
        global_data[3, 1] if darray.mpi_rank == 0 else None
    )

    darray.fill_halos()
    halos = _as_numpy(darray.local_with_halos)
    expected_lower = global_data[-1] if periodic else np.zeros(2)
    expected_upper = global_data[0] if periodic else np.zeros(2)
    np.testing.assert_array_equal(halos[0], expected_lower)
    np.testing.assert_array_equal(halos[-1], expected_upper)

    darray.exchange_halos()
    expected = global_data.copy()
    if periodic:
        expected[0] += global_data[0]
        expected[-1] += global_data[-1]
    np.testing.assert_array_equal(darray.to_numpy(), expected)


# (shape, decompose, ghost_axes): decomposed, with ghosts, and replicated.
DOT_LAYOUTS = [
    ((9, 7, 2), [True, True, False], [True, True, False]),
    ((1,), [True], [True]),
    ((6, 4), [False, False], [True, True]),
]


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.int64])
@pytest.mark.parametrize(("shape", "decompose", "ghost_axes"), DOT_LAYOUTS)
def test_vdot_and_norm_match_numpy_on_the_gathered_array(
    shape: tuple[int, ...],
    decompose: list[bool],
    ghost_axes: list[bool],
    dtype: type,
) -> None:
    """Dot products and norms cover every cell exactly once, halos excluded."""
    rng = np.random.default_rng(3)
    a = rng.integers(-5, 6, size=shape).astype(dtype)
    b = rng.integers(-5, 6, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.integers(-5, 6, size=shape)
        b = b + 1j * rng.integers(-5, 6, size=shape)
    layout = {
        "comm": comm,
        "num_ghostpoints": 1,
        "decompose": decompose,
        "ghost_axes": ghost_axes,
    }
    da = DistributedArray.from_array(xp.asarray(a), **layout)
    db = DistributedArray.from_array(xp.asarray(b), **layout)
    # Halo contents must not leak into the result.
    halo_mask = np.ones(da.shape_with_halos, dtype=bool)
    halo_mask[da.get_local_slices()] = False
    da.local_with_halos[xp.asarray(halo_mask)] = 1000
    db.local_with_halos[xp.asarray(halo_mask)] = 1000

    assert da.vdot(db) == pytest.approx(np.vdot(a, b))
    assert db.vdot(da) == pytest.approx(np.vdot(b, a))
    assert da.norm() == pytest.approx(np.linalg.norm(a.ravel()))
    assert da.norm(1) == pytest.approx(np.linalg.norm(a.ravel(), 1))
    assert da.norm(np.inf) == pytest.approx(np.linalg.norm(a.ravel(), np.inf))


def test_vdot_and_norm_reject_bad_arguments() -> None:
    """Mismatched layouts, non-distributed operands and unknown orders raise."""
    darray = DistributedArray.zeros(shape=(6, 4), comm=comm)
    other = DistributedArray.zeros(shape=(4, 6), comm=comm)

    with pytest.raises(ValueError, match="Shapes must match"):
        darray.vdot(other)
    with pytest.raises(TypeError, match="DistributedArray"):
        darray.vdot(np.zeros((6, 4)))
    with pytest.raises(ValueError, match="norm order"):
        darray.norm(3)


@pytest.mark.parametrize("contiguous", [True, False])
def test_reduce_across_ranks_sums_per_rank_copies(contiguous: bool) -> None:
    """Each rank's whole-grid copy is summed over ``comm``, halos included."""
    rank = comm.Get_rank()
    darray = DistributedArray.zeros(shape=(5, 3), comm=None, num_ghostpoints=1)
    values = xp.full(darray.shape_with_halos, float(rank + 1))
    if not contiguous:
        values = xp.asfortranarray(values)
    darray = darray._new_like(values)
    assert darray.data.flags.c_contiguous == contiguous

    darray.reduce_across_ranks(comm)

    expected = size * (size + 1) / 2
    np.testing.assert_array_equal(
        _as_numpy(darray.local_with_halos),
        np.full(darray.shape_with_halos, expected),
    )


def test_reduce_across_ranks_uses_own_comm_for_replicated_layout() -> None:
    """A replicated array reduces over its own communicator by default."""
    darray = DistributedArray.zeros(
        shape=(4,),
        comm=comm,
        decompose=[False],
        num_ghostpoints=1,
    )
    darray.local_with_halos[...] = 2.0

    darray.reduce_across_ranks()

    expected = 2.0 * size
    np.testing.assert_array_equal(_as_numpy(darray.local_with_halos), expected)


def test_reduce_across_ranks_without_communicator_is_a_no_op() -> None:
    """Serial arrays (no communicator anywhere) are left unchanged."""
    darray = DistributedArray.from_array(xp.arange(4.0), comm=None)

    darray.reduce_across_ranks()

    np.testing.assert_array_equal(darray.to_numpy(), np.arange(4.0))


@pytest.mark.skipif(size < 2, reason="needs a decomposed array on 2+ ranks")
def test_reduce_across_ranks_rejects_a_decomposed_array() -> None:
    """Summing different blocks of a decomposed array would be meaningless."""
    darray = DistributedArray.zeros(shape=(8,), comm=comm, decompose=[True])

    with pytest.raises(ValueError, match="decomposed"):
        darray.reduce_across_ranks()


def _index_layout_arrays() -> tuple[np.ndarray, DistributedArray]:
    """Return a global array and a decomposed copy with halos on two axes."""
    global_data = np.arange(9 * 7 * 3, dtype=float).reshape(9, 7, 3)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        decompose=[True, True, False],
        num_ghostpoints=2,
        ghost_axes=[True, True, False],
    )
    return global_data, darray


BASIC_INDICES = [
    (slice(None), 1),
    (Ellipsis, 2),
    (slice(1, 8, 3), slice(None), 0),
    (slice(None, None, -1), slice(5, 0, -2)),
    (slice(-3, None), -1),
    (4,),
    (4, 3),
    (-1, 0, slice(None)),
    (slice(2, 2),),
    Ellipsis,
    slice(None, None, 2),
]


@pytest.mark.parametrize("index", BASIC_INDICES)
def test_basic_index_assignment_matches_numpy_without_touching_halos(
    index: Any,
) -> None:
    """Slices, ints and Ellipsis write locally, exactly like NumPy on the global array."""
    global_data, darray = _index_layout_arrays()
    halo_mask = np.ones(darray.shape_with_halos, dtype=bool)
    halo_mask[darray.get_local_slices()] = False
    darray.local_with_halos[xp.asarray(halo_mask)] = -7.0

    expected = global_data.copy()
    target_shape = expected[index].shape
    value = np.arange(math.prod(target_shape), dtype=float).reshape(target_shape) + 0.5
    expected[index] = value
    darray[index] = xp.asarray(value)

    np.testing.assert_array_equal(darray.to_numpy(), expected)
    assert (_as_numpy(darray.local_with_halos)[halo_mask] == -7.0).all()


def test_basic_index_assignment_broadcasts_values() -> None:
    """Scalars and broadcastable arrays fill the selection like NumPy."""
    global_data, darray = _index_layout_arrays()
    expected = global_data.copy()

    darray[2:7, :, 1] = 5.0
    expected[2:7, :, 1] = 5.0
    darray[:, 3] = xp.asarray([10.0, 20.0, 30.0])
    expected[:, 3] = [10.0, 20.0, 30.0]
    column = DistributedArray.from_array(
        xp.asarray(global_data[:, :, :1] * 0 + 1.0),
        comm=comm,
    )
    darray[..., 2:3] = column
    expected[..., 2:3] = 1.0

    np.testing.assert_array_equal(darray.to_numpy(), expected)


def test_basic_index_assignment_errors_raise_on_every_rank() -> None:
    """Out-of-range integers and non-broadcastable values raise everywhere."""
    _, darray = _index_layout_arrays()

    with pytest.raises(IndexError, match="out of bounds"):
        darray[9, 0] = 1.0
    with pytest.raises(IndexError, match="too many indices"):
        darray[0, 0, 0, 0] = 1.0
    with pytest.raises(IndexError, match="single ellipsis"):
        darray[..., 0, ...] = 1.0
    with pytest.raises(ValueError):
        darray[:, 0] = xp.ones(4)


def test_advanced_index_assignment_falls_back_to_gathering() -> None:
    """Boolean masks and index arrays still work (collectively)."""
    global_data, darray = _index_layout_arrays()
    expected = global_data.copy()

    darray[xp.asarray(global_data > 100)] = 0.0
    expected[global_data > 100] = 0.0
    darray[xp.asarray([0, 8]), 1, 2] = -1.0
    expected[[0, 8], 1, 2] = -1.0

    np.testing.assert_array_equal(darray.to_numpy(), expected)


def test_getitem_elements_and_selections() -> None:
    """Full integer indices stay local; partial ones gather like NumPy."""
    global_data, darray = _index_layout_arrays()

    value = darray[4, 3, 1]
    if darray.global_to_local((4, 3, 1)) is None:
        assert value is None
    else:
        assert value == global_data[4, 3, 1]
    np.testing.assert_array_equal(_as_numpy(darray[4]), global_data[4])
    np.testing.assert_array_equal(_as_numpy(darray[-1, 2]), global_data[-1, 2])
    np.testing.assert_array_equal(
        _as_numpy(darray[::-2, 1:4]),
        global_data[::-2, 1:4],
    )


def test_ufunc_out_writes_in_place() -> None:
    """``out=`` with distributed arrays reuses their storage."""
    global_data = np.arange(1, 25, dtype=float).reshape(6, 4)
    darray = DistributedArray.from_array(
        xp.asarray(global_data),
        comm=comm,
        num_ghostpoints=1,
    )
    storage = darray.data

    result = np.multiply(darray, 2.0, out=darray)
    assert result is darray
    assert darray.data is storage
    np.testing.assert_array_equal(darray.to_numpy(), global_data * 2)

    target = DistributedArray.zeros(shape=(6, 4), comm=comm, num_ghostpoints=1)
    assert np.add(darray, 1.0, out=target) is target
    np.testing.assert_array_equal(target.to_numpy(), global_data * 2 + 1)

    quotient = DistributedArray.zeros(shape=(6, 4), comm=comm, num_ghostpoints=1)
    remainder = DistributedArray.zeros(shape=(6, 4), comm=comm, num_ghostpoints=1)
    outputs = np.divmod(darray, 5.0, out=(quotient, remainder))
    assert outputs[0] is quotient
    assert outputs[1] is remainder
    np.testing.assert_array_equal(quotient.to_numpy(), (global_data * 2) // 5)
    np.testing.assert_array_equal(remainder.to_numpy(), (global_data * 2) % 5)


def test_ufunc_where_and_bad_out_arguments() -> None:
    """``where=`` accepts distributed masks; incompatible ``out=`` is rejected."""
    global_data = np.arange(12, dtype=float).reshape(3, 4)
    darray = DistributedArray.from_array(xp.asarray(global_data), comm=comm)
    target = DistributedArray.zeros(shape=(3, 4), comm=comm)

    np.negative(darray, out=target, where=darray > 5)
    np.testing.assert_array_equal(
        target.to_numpy(),
        np.where(global_data > 5, -global_data, 0.0),
    )

    with pytest.raises(ValueError, match="Shapes must match"):
        np.add(darray, 1.0, out=DistributedArray.zeros(shape=(4, 3), comm=comm))
    with pytest.raises(TypeError):
        np.add(darray, 1.0, out=np.zeros((3, 4)))


# -------------------------------------------------------------------------- #
# The rest of the API, on any number of ranks.


def test_constructors_fill_storage_and_report_sizes() -> None:
    """``data_local=``, ``full`` without a dtype, the ``data`` setter and byte sizes."""
    probe = DistributedArray.zeros(shape=(5, 3), comm=comm, num_ghostpoints=1)
    local = xp.full(probe.shape_with_halos, 2.5)
    darray = DistributedArray(
        shape=(5, 3), comm=comm, num_ghostpoints=1, data_local=local
    )
    np.testing.assert_array_equal(_as_numpy(darray.local_with_halos), _as_numpy(local))

    ints = DistributedArray.full(shape=(5, 3), fill_value=7, comm=comm)
    assert ints.dtype == xp.asarray(7).dtype
    assert int(ints.sum()) == 7 * 15

    global_data = np.arange(15.0).reshape(5, 3)
    darray.data = xp.asarray(global_data)  # a global array, as in fill()
    np.testing.assert_array_equal(darray.to_numpy(), global_data)
    assert darray.itemsize == 8
    assert darray.nbytes == 15 * 8


def test_storage_errors() -> None:
    darray = DistributedArray.zeros(shape=(4, 3), comm=comm, num_ghostpoints=1)
    with pytest.raises(ValueError, match="dtype"):
        darray.fill_local(xp.zeros(darray.shape_with_halos, dtype=np.int64))
    with pytest.raises(ValueError, match="local data shape"):
        darray._wrap_local(xp.zeros((1, 1)))
    with pytest.raises(ValueError, match="Expected 2 indices, got 1"):
        darray.global_to_local(1)
    # out of range, so no rank owns it: the root raises after the gather
    if darray.mpi_rank == 0:
        with pytest.raises(ValueError, match="not found on any rank"):
            darray.get_global_value((4, 0), root=0)
    else:
        assert darray.get_global_value((4, 0), root=0) is None


def test_halo_methods_without_ghost_cells_change_nothing() -> None:
    darray = DistributedArray.from_array(xp.arange(6.0), comm=comm, periodic=(True,))
    before = _as_numpy(darray.local_with_halos).copy()
    darray.fill_halos()
    darray.exchange_halos()
    darray.clear_halos(0)
    np.testing.assert_array_equal(_as_numpy(darray.local_with_halos), before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (4, 3, 1), "ndim": 3}, "Number of dimensions"),
        ({"shape": (4, 3), "decompose": [False, False]}, "process layouts"),
        ({"shape": (4, 3), "num_ghostpoints": 2}, "Halo widths"),
        (
            {"shape": (4, 3), "num_ghostpoints": 1, "ghost_axes": [True, False]},
            "Halo axes",
        ),
    ],
)
def test_operands_with_another_layout_are_rejected(kwargs, message) -> None:
    darray = DistributedArray.zeros(shape=(4, 3), comm=comm, num_ghostpoints=1)
    other = DistributedArray.zeros(comm=comm, **{"num_ghostpoints": 1, **kwargs})
    if message == "process layouts" and darray.proc_sizes == other.proc_sizes:
        pytest.skip("on one rank every layout has the same process grid")
    with pytest.raises(ValueError, match=message):
        darray + other


def test_min_and_max_reject_complex_values_on_every_rank() -> None:
    """With more ranks than cells some ranks own nothing; all of them must raise.

    Raising only on the empty ranks would leave the others waiting in allreduce.
    """
    darray = DistributedArray.zeros(shape=(1,), comm=comm, dtype=np.complex128)
    for reduction in (darray.min, darray.max):
        with pytest.raises(TypeError, match="min/max are not supported"):
            reduction()


@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_reductions_along_an_axis_match_numpy(axis) -> None:
    global_data = np.arange(1.0, 13.0).reshape(4, 3)
    darray = DistributedArray.from_array(xp.asarray(global_data), comm=comm)
    for name in ("sum", "prod", "min", "max", "mean", "var", "std", "all", "any"):
        expected = getattr(np, name)(
            global_data > 6 if name in ("all", "any") else global_data, axis=axis
        )
        operand = darray > 6 if name in ("all", "any") else darray
        np.testing.assert_allclose(
            _as_numpy(xp.asarray(getattr(operand, name)(axis=axis))), expected
        )


@pytest.mark.parametrize("ddof", [0, 1])
def test_var_and_std_along_an_axis_pass_ddof(ddof: int) -> None:
    global_data = np.arange(1.0, 13.0).reshape(4, 3) ** 2
    darray = DistributedArray.from_array(xp.asarray(global_data), comm=comm)
    for name in ("var", "std"):
        result = getattr(darray, name)(axis=0, ddof=ddof)
        expected = getattr(np, name)(global_data, axis=0, ddof=ddof)
        np.testing.assert_allclose(_as_numpy(xp.asarray(result)), expected)


def test_reflected_and_in_place_operators_match_numpy() -> None:
    a = np.arange(1, 13).reshape(4, 3)
    b = np.arange(12, 0, -1).reshape(4, 3)
    da = DistributedArray.from_array(xp.asarray(a), comm=comm)
    db = DistributedArray.from_array(xp.asarray(b), comm=comm)
    cases = {
        "a - b": (da - db, a - b),
        "2 - a": (2 - da, 2 - a),
        "3 * a": (3 * da, 3 * a),
        "a / b": (da / db, a / b),
        "1 / a": (1 / da, 1 / a),
        "a // 4": (da // 4, a // 4),
        "50 // a": (50 // da, 50 // a),
        "a ** 2": (da**2, a**2),
        "2 ** a": (2**da, 2**a),
        "+a": (+da, +a),
        "abs(-a)": (abs(-da), abs(-a)),
        "a < b": (da < db, a < b),
        "a <= b": (da <= db, a <= b),
        "a == b": (da == db, a == b),
        "a != b": (da != db, a != b),
    }
    for label, (result, expected) in cases.items():
        assert isinstance(result, DistributedArray), label
        np.testing.assert_allclose(result.to_numpy(), expected, err_msg=label)

    c = da.copy()
    c -= 1
    c *= 2
    np.testing.assert_array_equal(c.to_numpy(), (a - 1) * 2)
    f = da.astype(float)
    f /= 4
    np.testing.assert_allclose(f.to_numpy(), a / 4)


def test_array_protocol_and_ufunc_methods() -> None:
    global_data = np.arange(6.0).reshape(2, 3)
    darray = DistributedArray.from_array(xp.asarray(global_data), comm=comm)

    as_int = darray.__array__(dtype=np.int32)
    assert as_int.dtype == np.int32
    np.testing.assert_array_equal(_as_numpy(as_int), global_data.astype(np.int32))
    copied = darray.__array__(copy=True)
    np.testing.assert_array_equal(_as_numpy(copied), global_data)
    np.testing.assert_array_equal(_as_numpy(darray.__array__()), global_data)

    quotient, remainder = np.divmod(darray, 4.0)  # a ufunc with two outputs
    assert isinstance(quotient, DistributedArray)
    np.testing.assert_array_equal(quotient.to_numpy(), global_data // 4.0)
    np.testing.assert_array_equal(remainder.to_numpy(), global_data % 4.0)

    with pytest.raises(TypeError):
        np.add.reduce(darray)  # only plain ufunc calls are distributed


def test_to_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    darray = DistributedArray.from_array(xp.arange(4.0), comm=comm)
    monkeypatch.setitem(sys.modules, "cupy", None)  # not installed
    with pytest.raises(ImportError, match="requires CuPy"):
        darray.to_cupy()
    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(asarray=lambda a: ("cupy", a))
    )
    tag, gathered = darray.to_cupy()
    assert tag == "cupy"
    np.testing.assert_array_equal(_as_numpy(gathered), np.arange(4.0))


_STAGING_SCRIPT = """
import sys

import numpy as np

import cunumpy as xp

sys.path.insert(0, sys.argv[2])
from mpiarray import DistributedArray

MPI = xp.mpi.get_mpi()
assert not isinstance(MPI, xp.mpi.SerialMPI), "expected a real MPI run"
comm = MPI.COMM_WORLD
xp.mpi.set_mpi_cuda_aware(False)
global_data = np.arange(8 * 6 * 2, dtype=float).reshape(8, 6, 2)
layout = {
    "comm": comm,
    "num_ghostpoints": 1,
    "ghost_axes": [True, True, False],
    "decompose": [True, True, False],
    "periodic": (True, False, False),
}
filled = DistributedArray.from_array(xp.asarray(global_data), **layout)
filled.fill_halos()
exchanged = filled.copy()
exchanged.exchange_halos()
replicated = DistributedArray.from_array(xp.asarray(global_data), comm=None)
replicated.reduce_across_ranks(comm)
np.savez(
    f"{sys.argv[1]}_{comm.rank}.npz",
    backend=xp.get_backend(),
    filled=xp.to_numpy(filled.local_with_halos),
    exchanged=xp.to_numpy(exchanged.to_ndarray()),
    replicated=xp.to_numpy(replicated.to_ndarray()),
    total=xp.to_numpy(xp.asarray(filled.sum())),
)
"""


@pytest.mark.skipif(size != 1, reason="starts its own 2-rank MPI job")
def test_host_staged_communication_matches_direct(tmp_path) -> None:
    """CuPy arrays staged through the host (no CUDA-aware MPI) give NumPy's results.

    Runs on cunumpy's fake CuPy, so it needs no GPU, only an MPI launcher.
    """
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[3])
    pytest.importorskip("mpi4py", reason="needs mpi4py (the mpi extra)")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("needs mpiexec to start a 2-rank job")
    script = tmp_path / "staging.py"
    script.write_text(_STAGING_SCRIPT)

    def run(backend: str) -> list[Any]:
        env = {**os.environ, "CUNUMPY_BACKEND": backend}
        env.pop("CUNUMPY_FAKE_CUPY", None)
        if backend == "cupy":
            env["CUNUMPY_FAKE_CUPY"] = "1"
        prefix = str(tmp_path / backend)
        subprocess.run(
            [launcher, "-n", "2", sys.executable, str(script), prefix, src_dir],
            env=env,
            check=True,
            timeout=120,
        )
        return [np.load(f"{prefix}_{rank}.npz") for rank in range(2)]

    direct = run("numpy")
    staged = run("cupy")
    for expected, actual in zip(direct, staged, strict=True):
        assert str(expected["backend"]) == "numpy"
        assert str(actual["backend"]) == "cupy"
        for key in ("filled", "exchanged", "replicated", "total"):
            np.testing.assert_array_equal(actual[key], expected[key])


if __name__ == "__main__":
    test_darray_fill(arr_len=10, num_ghost=1)
    test_darray_halo_exchange(arr_len=10, num_ghost=1)
    test_darray_magic_methods(arr_len=10, num_ghost=1)
    test_darray_multiple_components(Nx=13, Ny=8, Ncomp=1, num_ghost=0)
