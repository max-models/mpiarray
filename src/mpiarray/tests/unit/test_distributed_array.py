"""Tests for DistributedArray: gathers, indexing, halos, operators and reductions.

Every test runs serially and under ``mpiexec -n N`` for any N.
"""

from __future__ import annotations

import math
from typing import Any

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray, Layout

MPI = xp.mpi.get_mpi()
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def _np(data: Any) -> np.ndarray:
    """Return backend data as a NumPy array for assertions."""
    return xp.to_numpy(data)


def _halo_mask(a: DistributedArray) -> np.ndarray:
    """Return a mask of the halo cells of ``a``'s storage."""
    mask = np.ones(a.layout.storage_shape, dtype=bool)
    mask[a.layout.interior] = False
    return mask


# -------------------------------------------------------------------------- #
# Construction, properties, gathering


def test_constructor_checks_the_storage_shape() -> None:
    layout = Layout((4, 3), halo=1)
    a = DistributedArray(layout, xp.zeros(layout.storage_shape))
    assert a.layout is layout
    with pytest.raises(ValueError, match="storage shape"):
        DistributedArray(layout, xp.zeros((1, 1)))


def test_properties() -> None:
    a = mpa.zeros((5, 3), dtype=np.int32, halo=1)
    assert (a.shape, a.ndim, a.size) == ((5, 3), 2, 15)
    assert (a.dtype, a.itemsize, a.nbytes) == (np.dtype(np.int32), 4, 60)
    assert a.local.shape == a.layout.local_shape
    assert a.local_with_halos.shape == a.layout.storage_shape
    assert len(a) == 5
    with pytest.raises(TypeError, match="unsized"):
        len(mpa.zeros(()))


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.bool_, np.complex128])
@pytest.mark.parametrize("halo", [0, 2])
@pytest.mark.parametrize("split", [0, (0, 1), None])
def test_gather_reassembles_uneven_blocks(dtype: type, halo: int, split) -> None:
    data = (np.arange(7 * 5 * 2).reshape(7, 5, 2) % 3).astype(dtype)
    a = mpa.array(data, split=split, halo=(halo, halo, 0))
    gathered = a.gather()
    assert gathered.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(_np(gathered), data)
    a.local_with_halos[...] = 0  # the gathered array must not alias the storage
    np.testing.assert_array_equal(_np(gathered), data)


@pytest.mark.parametrize("split", [0, None])
def test_gather_to_a_root(split) -> None:
    data = np.arange(10.0)
    a = mpa.array(data, split=split)
    root = size - 1
    for result in (a.gather(root=root), a.to_numpy(root=root)):
        if rank == root:
            np.testing.assert_array_equal(_np(result), data)
        else:
            assert result is None
    np.testing.assert_array_equal(a.to_numpy(), data)


def test_copy_astype_and_subclasses() -> None:
    class Tagged(DistributedArray):
        pass

    base = mpa.arange(10.0, halo=1)
    a = Tagged(base.layout, base.local_with_halos)
    results = [
        a.copy(),
        a + 1,
        -a,
        +a,
        abs(a),
        a > 3,
        np.sqrt(a),
        a.astype(np.float32),
        mpa.array(a),
    ]
    for result in results:
        assert type(result) is Tagged
        assert result.layout == a.layout
        assert not np.may_share_memory(
            _np(result.local_with_halos), _np(a.local_with_halos)
        )
    assert a.astype(float, copy=False) is a
    copied = a.copy()
    copied += 100
    np.testing.assert_array_equal(a.to_numpy(), np.arange(10.0))
    with pytest.raises(TypeError, match="unhashable"):
        hash(a)


def test_repr_shows_the_layout_and_the_local_block_only() -> None:
    a = mpa.array(np.arange(4))
    start, end = a.layout.index_bounds[0]
    text = repr(a)
    assert text.startswith("DistributedArray(shape=(4,), dtype=int64, split=(0,)")
    assert f"rank {rank} of {size} holds [{start}:{end}]" in text
    assert str(list(range(start, end))) in text


# -------------------------------------------------------------------------- #
# Global indexing


def _index_arrays() -> tuple[np.ndarray, DistributedArray]:
    """Return a global array and a copy split over two axes with halos."""
    data = np.arange(9 * 7 * 3, dtype=float).reshape(9, 7, 3)
    return data, mpa.array(data, split=(0, 1), halo=(2, 2, 0))


def test_get_and_getitem_return_the_same_value_on_every_rank() -> None:
    data, a = _index_arrays()
    for index in [(0, 0, 0), (8, 6, 2), (-1, 3, -2), (4, 3, 1)]:
        assert a.get(index) == data[index]
        assert a[index] == data[index]
        assert comm.allgather(a[index]) == [data[index]] * size
    assert mpa.arange(5).get(-2) == 3
    np.testing.assert_array_equal(_np(a[4]), data[4])
    np.testing.assert_array_equal(_np(a[-1, 2]), data[-1, 2])
    np.testing.assert_array_equal(_np(a[::-2, 1:4]), data[::-2, 1:4])
    np.testing.assert_array_equal(_np(a[xp.asarray([0, 8]), 1]), data[[0, 8], 1])
    with pytest.raises(IndexError, match="out of bounds"):
        a.get((9, 0, 0))
    with pytest.raises(IndexError, match="expected 3 indices"):
        a.get((1, 1))


def test_local_index_points_into_the_storage_of_the_owner() -> None:
    data, a = _index_arrays()
    owned = 0
    for index in [(0, 0, 0), (8, 6, 2), (4, 3, 1)]:
        local = a.local_index(index)
        if local is None:
            assert a.layout.owner(index) != rank
        else:
            owned += 1
            assert a.layout.owner(index) in (rank, 0)
            assert _np(a.local_with_halos)[local] == data[index]
    assert comm.allreduce(owned) >= 3


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
    np.int64(3),
]


@pytest.mark.parametrize("index", BASIC_INDICES)
def test_basic_assignment_matches_numpy_without_touching_halos(index: Any) -> None:
    data, a = _index_arrays()
    a.local_with_halos[xp.asarray(_halo_mask(a))] = -7.0
    expected = data.copy()
    target = expected[index].shape
    value = np.arange(math.prod(target), dtype=float).reshape(target) + 0.5
    expected[index] = value
    a[index] = xp.asarray(value)
    np.testing.assert_array_equal(a.to_numpy(), expected)
    assert (_np(a.local_with_halos)[_halo_mask(a)] == -7.0).all()


def test_basic_assignment_broadcasts_values() -> None:
    data, a = _index_arrays()
    expected = data.copy()
    a[2:7, :, 1] = 5.0
    expected[2:7, :, 1] = 5.0
    a[:, 3] = xp.asarray([10.0, 20.0, 30.0])
    expected[:, 3] = [10.0, 20.0, 30.0]
    a[..., 2:3] = mpa.ones((9, 7, 1))
    expected[..., 2:3] = 1.0
    np.testing.assert_array_equal(a.to_numpy(), expected)


def test_basic_assignment_errors_raise_on_every_rank() -> None:
    _, a = _index_arrays()
    with pytest.raises(IndexError, match="out of bounds"):
        a[9, 0] = 1.0
    with pytest.raises(IndexError, match="too many indices"):
        a[0, 0, 0, 0] = 1.0
    with pytest.raises(IndexError, match="single ellipsis"):
        a[..., 0, ...] = 1.0
    with pytest.raises(ValueError):
        a[:, 0] = xp.ones(4)


def test_advanced_assignment_gathers() -> None:
    data, a = _index_arrays()
    expected = data.copy()
    a[xp.asarray(data > 100)] = 0.0
    expected[data > 100] = 0.0
    a[xp.asarray([0, 8]), 1, 2] = -1.0
    expected[[0, 8], 1, 2] = -1.0
    np.testing.assert_array_equal(a.to_numpy(), expected)


def test_iteration_and_truth_value() -> None:
    data = np.arange(6.0).reshape(3, 2)
    rows = list(mpa.array(data))
    assert len(rows) == 3
    np.testing.assert_array_equal(_np(rows[1]), data[1])
    assert bool(mpa.array([3.0]))
    assert not mpa.zeros(1)
    assert bool(mpa.full((), True))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(mpa.zeros(2))


# -------------------------------------------------------------------------- #
# Halo cells

# (shape, split, halo per unit width)
HALO_LAYOUTS = [
    ((16,), 0, (1,)),
    ((12, 10), (0, 1), (1, 1)),
    ((12, 10, 3), (0, 1), (1, 1, 0)),
    ((8, 6, 6), (0, 1, 2), (1, 1, 1)),
    ((9, 4), None, (1, 0)),
]


def _periodic(halo_axes: tuple[int, ...], mode: str) -> tuple[bool, ...]:
    """Return periodicity flags: on every halo axis, none, or only the first axis."""
    if mode == "all":
        return tuple(bool(h) for h in halo_axes)
    if mode == "none":
        return (False,) * len(halo_axes)
    return tuple(axis == 0 for axis in range(len(halo_axes)))


def _comm(kind: str):
    """Return the communicator of a parametrized test."""
    return MPI.COMM_SELF if kind == "self" else comm


def _storage_indices(
    a: DistributedArray, r: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return, per axis, the global index of every storage cell of rank ``r``.

    Periodic axes are wrapped; the second list flags indices inside the domain.
    """
    indices, valid = [], []
    layout = a.layout
    for axis, (start, end) in enumerate(layout.index_bounds_of(r)):
        h = layout.halo[axis]
        index = np.arange(start - h, end + h)
        if layout.periodic[axis]:
            index %= layout.shape[axis]
        indices.append(index)
        valid.append((index >= 0) & (index < layout.shape[axis]))
    return indices, valid


def _rank_storage(a: DistributedArray, r: int, dtype: type) -> np.ndarray:
    """Return deterministic storage values (halos included) for rank ``r``."""
    indices, _ = _storage_indices(a, r)
    rng = np.random.default_rng(1234 + r)
    return rng.integers(1, 10, size=[len(i) for i in indices]).astype(dtype)


@pytest.mark.parametrize("comm_kind", ["world", "self"])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("periodic_mode", ["all", "none", "first"])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize(("shape", "split", "halo"), HALO_LAYOUTS)
def test_accumulate_halos_matches_a_serial_reference(
    shape, split, halo, width, periodic_mode, dtype, comm_kind
) -> None:
    """Accumulating equals folding every rank's storage into the global array."""
    halo = tuple(width * h for h in halo)
    a = mpa.zeros(
        shape,
        dtype=dtype,
        split=split,
        halo=halo,
        periodic=_periodic(halo, periodic_mode),
        comm=_comm(comm_kind),
    )
    a.local_with_halos[...] = xp.asarray(_rank_storage(a, a.layout.rank, dtype))
    if a.layout.replicated:  # every rank holds the whole array: no folding over ranks
        ranks = [a.layout.rank]
    else:
        ranks = range(a.layout.size)
    expected = np.zeros(shape, dtype=dtype)
    for r in ranks:
        indices, valid = _storage_indices(a, r)
        np.add.at(
            expected,
            np.ix_(*[i[m] for i, m in zip(indices, valid, strict=True)]),
            _rank_storage(a, r, dtype)[np.ix_(*valid)],
        )

    a.accumulate_halos()

    assert a.dtype == np.dtype(dtype)
    if a.layout.replicated:
        np.testing.assert_array_equal(_np(a.local), expected)
    else:
        np.testing.assert_array_equal(a.to_numpy(), expected)
    assert not _np(a.local_with_halos)[_halo_mask(a)].any()


@pytest.mark.parametrize("comm_kind", ["world", "self"])
@pytest.mark.parametrize("periodic_mode", ["all", "none", "first"])
@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize(("shape", "split", "halo"), HALO_LAYOUTS)
def test_update_halos_matches_the_global_neighbours(
    shape, split, halo, width, periodic_mode, comm_kind
) -> None:
    """Halo updates (corners included) give each cell its global neighbours."""
    halo = tuple(width * h for h in halo)
    data = np.arange(math.prod(shape), dtype=float).reshape(shape) + 1.0
    a = mpa.array(
        data,
        split=split,
        halo=halo,
        periodic=_periodic(halo, periodic_mode),
        comm=_comm(comm_kind),
    )
    a.update_halos()
    indices, valid = _storage_indices(a, a.layout.rank)
    expected = np.zeros(a.layout.storage_shape)
    expected[np.ix_(*valid)] = data[
        np.ix_(*[i[m] for i, m in zip(indices, valid, strict=True)])
    ]
    np.testing.assert_array_equal(_np(a.local_with_halos), expected)


def test_halo_methods_work_per_axis() -> None:
    data = np.arange(12.0).reshape(4, 3) + 1
    a = mpa.array(data, split=None, halo=1, periodic=True)
    a.update_halos(axis=1)
    storage = _np(a.local_with_halos)
    np.testing.assert_array_equal(storage[1:-1, 0], data[:, -1])
    assert not storage[0].any()  # axis 0 untouched
    a.clear_halos(axis=1)
    assert not _np(a.local_with_halos)[_halo_mask(a)].any()
    a.local_with_halos[0, 1:-1] = 1.0  # deposit below the first row
    a.accumulate_halos(axis=0)
    np.testing.assert_array_equal(_np(a.local)[-1], data[-1] + 1)


def test_halo_methods_without_halo_cells_change_nothing() -> None:
    a = mpa.arange(6.0, periodic=True)
    before = _np(a.local_with_halos).copy()
    a.update_halos()
    a.accumulate_halos()
    a.clear_halos()
    np.testing.assert_array_equal(_np(a.local_with_halos), before)


# -------------------------------------------------------------------------- #
# Elementwise operations


@pytest.mark.parametrize(
    "operand_shape",
    [(3,), (1, 1, 3), (6, 1), (8, 1, 1), (8, 6, 1), (8, 6, 3), (1,), ()],
)
def test_array_operands_broadcast_like_numpy_on_the_global_shape(operand_shape) -> None:
    shape = (8, 6, 3)
    data = np.arange(math.prod(shape), dtype=float).reshape(shape)
    operand = np.arange(math.prod(operand_shape), dtype=float).reshape(operand_shape)
    a = mpa.array(data, split=(0, 1), halo=(1, 1, 0))
    b = xp.asarray(operand)
    np.testing.assert_allclose((a + b).to_numpy(), data + operand)
    difference = b - a  # ndarray - DistributedArray defers to __array_ufunc__
    assert isinstance(difference, DistributedArray)
    np.testing.assert_allclose(difference.to_numpy(), operand - data)
    np.testing.assert_allclose(np.multiply(a, b).to_numpy(), data * operand)
    a += b
    np.testing.assert_allclose(a.to_numpy(), data + operand)


def test_storage_shaped_operands_are_used_as_they_are() -> None:
    a = mpa.ones((6, 2), halo=1)
    b = a + xp.full(a.layout.storage_shape, 2.0)
    assert (_np(b.local_with_halos) == 3.0).all()


@pytest.mark.parametrize("operand_shape", [(5,), (7, 1, 1), (2, 8, 6, 3)])
def test_operands_that_do_not_broadcast_raise(operand_shape) -> None:
    a = mpa.zeros((8, 6, 3), split=(0, 1))
    with pytest.raises(ValueError, match="global shape|more dimensions"):
        a + xp.ones(operand_shape)


def test_operands_with_another_layout_raise() -> None:
    a = mpa.zeros((4, 3), halo=1)
    with pytest.raises(ValueError, match="Shapes must match"):
        a + mpa.zeros((3, 4), halo=1)
    with pytest.raises(ValueError, match="Layouts must match"):
        a + mpa.zeros((4, 3), halo=2)
    with pytest.raises(ValueError, match="Layouts must match"):
        a + mpa.zeros((4, 3), halo=1, comm=MPI.COMM_SELF)


def test_operators_match_numpy() -> None:
    x = np.arange(1, 13).reshape(4, 3)
    y = np.arange(12, 0, -1).reshape(4, 3)
    a, b = mpa.array(x), mpa.array(y)
    cases = {
        "a + b": (a + b, x + y),
        "1 + a": (1 + a, 1 + x),
        "a - b": (a - b, x - y),
        "2 - a": (2 - a, 2 - x),
        "a * b": (a * b, x * y),
        "3 * a": (3 * a, 3 * x),
        "a / b": (a / b, x / y),
        "1 / a": (1 / a, 1 / x),
        "a // 4": (a // 4, x // 4),
        "50 // a": (50 // a, 50 // x),
        "a ** 2": (a**2, x**2),
        "2 ** a": (2**a, 2**x),
        "-a": (-a, -x),
        "+a": (+a, +x),
        "abs(-a)": (abs(-a), abs(-x)),
        "a < b": (a < b, x < y),
        "a <= b": (a <= b, x <= y),
        "a == b": (a == b, x == y),
        "a != b": (a != b, x != y),
        "a > b": (a > b, x > y),
        "a >= b": (a >= b, x >= y),
    }
    for label, (result, expected) in cases.items():
        assert isinstance(result, DistributedArray), label
        np.testing.assert_allclose(result.to_numpy(), expected, err_msg=label)

    c = a.copy()
    c -= 1
    c *= 2
    np.testing.assert_array_equal(c.to_numpy(), (x - 1) * 2)
    f = a.astype(float)
    f /= 4
    np.testing.assert_allclose(f.to_numpy(), x / 4)


def test_ufuncs_with_out_where_and_several_outputs() -> None:
    data = np.arange(1, 25, dtype=float).reshape(6, 4)
    a = mpa.array(data, halo=1)
    storage = a.local_with_halos
    assert np.multiply(a, 2.0, out=a) is a  # ty: ignore[no-matching-overload]
    assert a.local_with_halos is storage
    np.testing.assert_array_equal(a.to_numpy(), data * 2)

    target = mpa.zeros_like(a)
    assert np.add(a, 1.0, out=target) is target  # ty: ignore[no-matching-overload]
    np.testing.assert_array_equal(target.to_numpy(), data * 2 + 1)

    quotient, remainder = mpa.zeros_like(a), mpa.zeros_like(a)
    outputs = np.divmod(a, 5.0, out=(quotient, remainder))  # ty: ignore[no-matching-overload]
    assert outputs[0] is quotient and outputs[1] is remainder
    np.testing.assert_array_equal(quotient.to_numpy(), (data * 2) // 5)
    q, r = np.divmod(a, 5.0)
    np.testing.assert_array_equal(r.to_numpy(), (data * 2) % 5)

    masked = mpa.zeros_like(a)
    np.negative(a, out=masked, where=a > 20)  # ty: ignore[no-matching-overload]
    np.testing.assert_array_equal(
        masked.to_numpy(), np.where(data * 2 > 20, -data * 2, 0)
    )

    with pytest.raises(ValueError, match="Shapes must match"):
        np.add(a, 1.0, out=mpa.zeros((4, 6), halo=1))  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError):
        np.add(a, 1.0, out=np.zeros((6, 4)))
    with pytest.raises(TypeError):
        np.add.reduce(a)  # ty: ignore[no-matching-overload]  # only plain ufunc calls are distributed


def test_array_protocol_gathers() -> None:
    data = np.arange(6.0).reshape(2, 3)
    a = mpa.array(data)
    np.testing.assert_array_equal(_np(a.__array__()), data)
    np.testing.assert_array_equal(_np(a.__array__(copy=True)), data)
    as_int = a.__array__(dtype=np.int32)
    assert as_int.dtype == np.int32


# -------------------------------------------------------------------------- #
# Reductions


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
@pytest.mark.parametrize("shape", [(1,), (1, 3), (2, 1)])
def test_reductions_with_ranks_owning_no_cells(dtype, shape) -> None:
    data = (np.arange(math.prod(shape)).reshape(shape) + 2).astype(dtype)
    a = mpa.array(data, split=tuple(range(len(shape))), halo=1)
    assert a.sum() == data.sum()
    assert a.prod() == data.prod()
    assert a.min() == data.min()
    assert a.max() == data.max()
    assert np.isclose(a.mean(), data.mean())
    assert a.all() == data.all()
    assert a.any() == data.any()


def test_whole_array_reductions_return_host_scalars() -> None:
    a = mpa.arange(10.0)
    for value in (a.sum(), a.max(), a.mean(), a.var(), a.std(), a.vdot(a)):
        assert not xp.is_gpu(value)
        assert np.ndim(value) == 0
    assert isinstance(a.all(), bool)


def test_zero_size_and_complex_min_max_raise_on_every_rank() -> None:
    empty = mpa.zeros((0, 3))
    with pytest.raises(ValueError, match="zero-size"):
        empty.min()
    with pytest.raises(ValueError, match="zero-size"):
        empty.max()
    assert empty.sum() == 0
    # with more ranks than cells some ranks own nothing; all of them must raise
    complex_array = mpa.zeros(1, dtype=np.complex128)
    for reduction in (complex_array.min, complex_array.max):
        with pytest.raises(TypeError, match="min/max are not supported"):
            reduction()


def test_numpy_reduction_functions_dispatch_to_the_methods() -> None:
    data = np.arange(1, 25, dtype=float).reshape(6, 4)
    a = mpa.array(data, halo=1)
    assert np.isclose(np.sum(a), data.sum())
    assert np.isclose(np.prod(a / 10), np.prod(data / 10))
    assert np.min(a) == np.min(data) and np.max(a) == np.max(data)
    assert np.isclose(np.mean(a), data.mean())
    assert np.isclose(np.var(a, ddof=1), data.var(ddof=1))
    assert np.isclose(np.std(a, ddof=1), data.std(ddof=1))
    assert np.all(a) == data.all() and np.any(a > 23) == (data > 23).any()
    assert np.sum(a, dtype=np.float32).dtype == np.float32
    kept = np.sum(a, keepdims=True)
    assert np.shape(kept) == (1, 1)
    assert np.isclose(float(_np(kept)[0, 0]), data.sum())
    assert np.shape(a.std(keepdims=True)) == (1, 1)
    with pytest.raises(TypeError, match="out="):
        np.sum(a, out=np.empty(()))


@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_reductions_along_an_axis_match_numpy(axis) -> None:
    data = np.arange(1.0, 13.0).reshape(4, 3)
    a = mpa.array(data, split=(0, 1))
    for name in ("sum", "prod", "min", "max", "mean", "var", "std", "all", "any"):
        operand, reference = (a > 6, data > 6) if name in ("all", "any") else (a, data)
        np.testing.assert_allclose(
            _np(xp.asarray(getattr(operand, name)(axis=axis))),
            getattr(np, name)(reference, axis=axis),
            err_msg=name,
        )
    for ddof in (0, 1):
        np.testing.assert_allclose(
            _np(a.var(axis=0, ddof=ddof)), data.var(axis=0, ddof=ddof)
        )


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.int64])
@pytest.mark.parametrize(
    ("shape", "split"), [((9, 7, 2), (0, 1)), ((1,), 0), ((6, 4), None)]
)
def test_vdot_and_norm_cover_every_cell_once(shape, split, dtype) -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(-5, 6, size=shape).astype(dtype)
    y = rng.integers(-5, 6, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.integers(-5, 6, size=shape)
        y = y + 1j * rng.integers(-5, 6, size=shape)
    a, b = mpa.array(x, split=split, halo=1), mpa.array(y, split=split, halo=1)
    a.local_with_halos[xp.asarray(_halo_mask(a))] = 1000  # must not leak in
    b.local_with_halos[xp.asarray(_halo_mask(b))] = 1000
    assert a.vdot(b) == pytest.approx(np.vdot(x, y))
    assert b.vdot(a) == pytest.approx(np.vdot(y, x))
    assert a.norm() == pytest.approx(np.linalg.norm(x.ravel()))
    assert a.norm(1) == pytest.approx(np.linalg.norm(x.ravel(), 1))
    assert a.norm(np.inf) == pytest.approx(np.linalg.norm(x.ravel(), np.inf))


def test_vdot_and_norm_reject_bad_arguments() -> None:
    a = mpa.zeros((6, 4))
    with pytest.raises(ValueError, match="Shapes must match"):
        a.vdot(mpa.zeros((4, 6)))
    with pytest.raises(TypeError, match="DistributedArray"):
        a.vdot(np.zeros((6, 4)))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="norm order"):
        a.norm(3)


@pytest.mark.parametrize("contiguous", [True, False])
def test_allreduce_replicated_sums_every_rank_copy(contiguous: bool) -> None:
    layout = Layout((5, 3), split=None, halo=1)
    storage = xp.full(layout.storage_shape, float(rank + 1))
    if not contiguous:
        storage = xp.asfortranarray(storage)
    a = DistributedArray(layout, storage)
    a.allreduce_replicated()
    expected = size * (size + 1) / 2
    np.testing.assert_array_equal(_np(a.local_with_halos), expected)
    a.allreduce_replicated(MPI.MAX)
    np.testing.assert_array_equal(_np(a.local_with_halos), expected)


def test_allreduce_replicated_on_one_rank_changes_nothing() -> None:
    a = mpa.arange(4.0, comm=MPI.COMM_SELF)
    a.allreduce_replicated()
    np.testing.assert_array_equal(a.to_numpy(), np.arange(4.0))


@pytest.mark.skipif(size < 2, reason="needs an array split over 2+ ranks")
def test_allreduce_replicated_rejects_a_split_array() -> None:
    with pytest.raises(ValueError, match="split over the ranks"):
        mpa.zeros(8).allreduce_replicated()


# -------------------------------------------------------------------------- #
# GPU buffers staged through the host, on cunumpy's fake CuPy

_STAGING_SCRIPT = """
import sys

import numpy as np

import cunumpy as xp

sys.path.insert(0, sys.argv[2])
import mpiarray as mpa

MPI = xp.mpi.get_mpi()
assert not isinstance(MPI, xp.mpi.SerialMPI), "expected a real MPI run"
comm = MPI.COMM_WORLD
xp.mpi.set_mpi_cuda_aware(False)
data = np.arange(8 * 6 * 2, dtype=float).reshape(8, 6, 2)
updated = mpa.array(
    xp.asarray(data), split=(0, 1), halo=(1, 1, 0), periodic=(True, False, False)
)
updated.update_halos()
accumulated = updated.copy()
accumulated.accumulate_halos()
replicated = mpa.array(xp.asarray(data), split=None)
replicated.allreduce_replicated()
np.savez(
    f"{sys.argv[1]}_{comm.rank}.npz",
    backend=xp.get_backend(),
    updated=xp.to_numpy(updated.local_with_halos),
    accumulated=xp.to_numpy(accumulated.gather()),
    rooted=xp.to_numpy(rooted) if (rooted := accumulated.gather(root=0)) is not None else 0,
    replicated=xp.to_numpy(replicated.gather()),
    total=np.asarray(updated.sum()),
    element=np.asarray(updated.get((5, 4, 1))),
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

    pytest.importorskip("mpi4py", reason="needs mpi4py (the mpi extra)")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("needs mpiexec to start a 2-rank job")
    src_dir = str(Path(__file__).resolve().parents[3])
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
        return [np.load(f"{prefix}_{r}.npz") for r in range(2)]

    direct, staged = run("numpy"), run("cupy")
    for expected, actual in zip(direct, staged, strict=True):
        assert str(expected["backend"]) == "numpy"
        assert str(actual["backend"]) == "cupy"
        for key in (
            "updated",
            "accumulated",
            "rooted",
            "replicated",
            "total",
            "element",
        ):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
