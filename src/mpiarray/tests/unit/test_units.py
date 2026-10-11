"""Physical unit algebra, with plain MPI storage on every rank."""

import astropy.units as u
import numpy as np
import pytest

import mpiarray as mpa

N = mpa.default_comm().Get_size() + 6
VALUES = np.arange(1.0, N + 1)


def array(unit=None, values=VALUES, **kwargs):
    return mpa.array(values, **kwargs).set_units(unit)


def check(result, values, unit):
    assert result.units == unit
    assert not isinstance(result.local_with_halos, u.Quantity)
    np.testing.assert_allclose(result.to_numpy(), values)


def test_set_units():
    a = mpa.ones(N)
    assert a.units is None
    assert a.set_units("eV") is a
    assert a.units == u.eV
    assert a.set_units(u.m).units == u.m
    assert a.set_units(None).units is None
    with pytest.raises(TypeError, match="units must"):
        a.set_units(12)
    with pytest.raises(ValueError):
        a.set_units("not_a_unit")
    with pytest.raises(ValueError, match="offset"):
        a.set_units(u.deg_C)


def test_arithmetic_and_scalars():
    a, b = array(u.eV), array(u.m)
    check(a * b, VALUES**2, u.eV * u.m)
    check(a / b, np.ones(N), u.eV / u.m)
    check(a * 2, VALUES * 2, u.eV)
    check(2 * a, VALUES * 2, u.eV)
    check(a / 2, VALUES / 2, u.eV)
    check(2 / a, 2 / VALUES, 1 / u.eV)
    check(a + a, VALUES * 2, u.eV)
    check(a - a.mean(), VALUES - VALUES.mean(), u.eV)
    check(a + 2 * u.eV, VALUES + 2, u.eV)
    check(a * (2 * u.m), VALUES * 2, u.eV * u.m)
    check(array() + 2 * u.percent, VALUES + 0.02, None)
    check(array(u.percent) + 1, VALUES + 100, u.percent)
    for op in [np.add, np.subtract, np.less]:
        with pytest.raises(ValueError, match="compatible"):
            op(a, b)
        with pytest.raises(ValueError, match="compatible"):
            op(a, 1)
    check(array(u.one) + 1, VALUES + 1, u.one)


@pytest.mark.parametrize(
    "op",
    [
        np.add,
        np.subtract,
        np.maximum,
        np.minimum,
        np.hypot,
        np.remainder,
        np.fmod,
        np.fmax,
        np.fmin,
        np.copysign,
        np.nextafter,
    ],
)
def test_compatible_rescaling(op):
    a, b = array(u.m), array(u.cm)
    check(op(a, b), op(VALUES, VALUES / 100), u.m)
    check(op(b, a), op(VALUES, VALUES * 100), u.cm)
    check(array(u.eV) + array(u.J), VALUES + VALUES * u.J.to(u.eV), u.eV)


@pytest.mark.parametrize(
    "op",
    [
        np.less,
        np.less_equal,
        np.equal,
        np.not_equal,
        np.greater,
        np.greater_equal,
        np.isclose,
    ],
)
def test_comparisons(op):
    a, b = array(u.m), array(u.cm, VALUES * 100)
    check(op(a, b), op(VALUES, VALUES), None)
    assert np.allclose(a, b)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "op",
    [
        np.negative,
        np.positive,
        np.absolute,
        np.floor,
        np.ceil,
        np.trunc,
        np.rint,
        np.conjugate,
        np.spacing,
        np.sign,
        np.round,
    ],
)
def test_preserving(op):
    check(op(array(u.m)), op(VALUES), u.m)
    check(np.round(array(u.m), 2), np.round(VALUES, 2), u.m)


@pytest.mark.parametrize(
    "op,power", [(np.square, 2), (np.sqrt, 0.5), (np.cbrt, 1 / 3), (np.reciprocal, -1)]
)
def test_fixed_power(op, power):
    check(op(array(u.m)), op(VALUES), u.m**power)


@pytest.mark.parametrize("op", [np.power, np.float_power])
def test_power(op):
    check(op(array(u.m), 2), VALUES**2, u.m**2)
    for exponent in [array(), np.ones(N), 2 * u.one]:
        with pytest.raises(TypeError, match="plain scalar"):
            op(array(u.m), exponent)


@pytest.mark.parametrize(
    "op",
    [
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
    ],
)
def test_dimensionless(op):
    with pytest.raises(ValueError, match="dimensionless"):
        op(array(u.m))
    with np.errstate(invalid="ignore", divide="ignore"):
        check(op(array(u.one)), op(VALUES), None)
        check(op(array(u.percent)), op(VALUES / 100), None)


@pytest.mark.parametrize("op", [np.logaddexp, np.logaddexp2])
def test_binary_dimensionless(op):
    with pytest.raises(ValueError, match="dimensionless"):
        op(array(u.one), array(u.m))
    check(op(array(u.percent), array(u.percent)), op(VALUES / 100, VALUES / 100), None)


@pytest.mark.parametrize("op", [np.isnan, np.isinf, np.isfinite, np.signbit])
def test_boolean(op):
    check(op(array(u.m)), op(VALUES), None)


@pytest.mark.parametrize("split", [0, None])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize(
    "name,power",
    [("sum", 1), ("mean", 1), ("min", 1), ("max", 1), ("var", 2), ("std", 1)],
)
def test_reductions(split, keepdims, name, power):
    result = getattr(array(u.m, split=split), name)(keepdims=keepdims)
    assert isinstance(result, u.Quantity)
    assert result.unit == u.m**power
    np.testing.assert_allclose(
        result.value, getattr(np, name)(VALUES, keepdims=keepdims)
    )
    plain = getattr(array(split=split), name)(keepdims=keepdims)
    assert not isinstance(plain, u.Quantity)
    np.testing.assert_allclose(plain, result.value)


@pytest.mark.parametrize("split", [0, None])
def test_inner_products_and_norms(split):
    a, b = array(u.m, split=split), array(u.s, split=split)
    for result in [a.dot(b), a.vdot(b), np.dot(a, b), np.vdot(a, b), np.matmul(a, b)]:
        assert result.unit == u.m * u.s
        assert result.value == np.dot(VALUES, VALUES)
    assert a.dot(array(split=split)).unit == u.m
    for order in [1, 2, np.inf]:
        result = a.norm(order)
        assert result.unit == u.m
        assert result.value == pytest.approx(np.linalg.norm(VALUES, order))
    assert np.linalg.norm(a).unit == u.m


@pytest.mark.parametrize("split", [0, None])
def test_copies_and_layouts(split):
    a = array(u.m, split=split)
    for result in [
        a.copy(),
        a.astype("float32"),
        a.astype(a.dtype, copy=False),
        a.redistribute(mpa.Layout((N,), split=None)),
        a.rebalance(np.ones(N)),
    ]:
        check(result, VALUES, u.m)
    a.update_halos()
    assert a.units == u.m


def test_out_and_inplace():
    a = array(u.m)
    a += array(u.cm)
    check(a, VALUES * 1.01, u.m)
    a *= array(u.s)
    check(a, VALUES**2 * 1.01, u.m * u.s)
    out = array()
    assert np.square(a, out=out) is out
    check(out, (VALUES**2 * 1.01) ** 2, (u.m * u.s) ** 2)
    original = a.to_numpy()
    with pytest.raises(ValueError):
        a += array(u.kg)
    check(a, original, u.m * u.s)


def test_unsupported():
    a = array(u.m)
    for op in [np.gcd, np.floor_divide, np.logical_and, np.divmod]:
        with pytest.raises(TypeError, match="units are not supported"):
            op(a, a)
    for op in [np.argmin, np.argmax]:
        with pytest.raises(TypeError, match="units are not supported"):
            op(a)
    for op in [np.sum, np.var]:
        with pytest.raises(TypeError, match="units are not supported"):
            op(a, axis=0)
    with pytest.raises(TypeError, match="prod"):
        a.prod()
    plain = array(values=VALUES.astype(int))
    check(np.gcd(plain, plain), VALUES, None)
    assert isinstance(a.all(), (bool, np.bool_))
    assert isinstance(a.any(), (bool, np.bool_))


def test_xarray():
    a = array(u.eV).labeled("x")
    attrs = {"note": "test"}
    assert a.to_xarray(attrs=attrs).attrs == {"note": "test", "units": "eV"}
    assert attrs == {"note": "test"}
    assert a.to_xarray(attrs={"units": "custom"}).attrs["units"] == "custom"


def test_quantity_left_operands():
    a = array(u.m)
    check(2 * u.m + a, VALUES + 2, u.m)
    check(2 * u.m - a, 2 - VALUES, u.m)
    check(2 * u.s * a, 2 * VALUES, u.s * u.m)
    check(2 * u.s / a, 2 / VALUES, u.s / u.m)


def test_masked_out_requires_consistent_units():
    a, out = array(u.m), array(u.m)
    mask = mpa.array(VALUES % 2 == 0)
    np.add(a, a, out=out, where=mask)
    check(out, np.where(VALUES % 2 == 0, VALUES * 2, VALUES), u.m)
    with pytest.raises(ValueError, match="masked out"):
        np.square(a, out=out, where=mask)
    check(out, np.where(VALUES % 2 == 0, VALUES * 2, VALUES), u.m)


# Compare physical results against Astropy, not just the unit labels.
@pytest.mark.parametrize(
    "left_unit,right_unit",
    [(u.m, u.cm), (u.eV, u.J), (u.m / u.s, u.km / u.h), (u.percent, u.one)],
)
@pytest.mark.parametrize(
    "operation",
    [
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
        np.maximum,
        np.minimum,
        np.hypot,
        np.less,
        np.greater_equal,
    ],
)
@pytest.mark.parametrize("halo", [0, 1])
def test_binary_arithmetic_matches_quantity(operation, left_unit, right_unit, halo):
    rng = np.random.default_rng(71)
    left = rng.uniform(0.1, 20, N)
    right = rng.uniform(0.1, 20, N)
    a, b = array(left_unit, left, halo=halo), array(right_unit, right, halo=halo)
    expected = operation(left * left_unit, right * right_unit)
    actual = operation(a, b)
    if isinstance(expected, u.Quantity):
        np.testing.assert_allclose(actual.to_numpy() * actual.units, expected)
    else:
        check(actual, expected, None)
    check(a, left, left_unit)
    check(b, right, right_unit)


ASSIGN_INDICES = [
    0,
    -1,
    slice(None),
    slice(1, None, 2),
    slice(None, None, -1),
    slice(0, 0),
    np.array([0, N - 1, 0]),
    VALUES % 2 == 0,
    Ellipsis,
]


@pytest.mark.parametrize("index", ASSIGN_INDICES)
@pytest.mark.parametrize("kind", ["scalar", "quantity", "distributed"])
@pytest.mark.parametrize("split", [0, None])
def test_assignment_converts_like_quantity(index, kind, split):
    a = array(u.m, halo=1, split=split)
    a.local_with_halos[...] = -77
    a.local[...] = VALUES[a.layout.global_slices()]
    expected = VALUES * u.m
    shape = np.shape(VALUES[index])
    raw = np.full(shape, 250.0)
    if kind == "scalar":
        value = 250 * u.cm
    elif kind == "quantity":
        value = raw * u.cm
    else:
        value = array(u.cm, raw, split=None)
    expected[index] = raw * u.cm
    a[index] = value
    check(a, expected.value, expected.unit)
    if isinstance(index, (int, slice)) or index is Ellipsis:
        assert a.local_with_halos[0] == a.local_with_halos[-1] == -77
    else:
        assert a.local_with_halos[0] == a.local_with_halos[-1] == 0
    if kind == "distributed":
        check(value, raw, u.cm)


@pytest.mark.parametrize(
    "index", [0, slice(None), slice(0, 0), np.array([0, N - 1]), VALUES % 2 == 0]
)
@pytest.mark.parametrize("kind", ["scalar", "quantity", "distributed", "bare"])
def test_assignment_rejects_incompatible_without_mutation(index, kind):
    a = array(u.m, halo=1)
    before = a.local_with_halos.copy()
    if kind == "scalar":
        value = 2 * u.s
    elif kind == "quantity":
        value = np.ones(np.shape(VALUES[index])) * u.s
    elif kind == "distributed":
        value = array(u.s, np.ones(np.shape(VALUES[index])), split=None)
    else:
        value = 2
    with pytest.raises(ValueError, match="assignment needs compatible"):
        a[index] = value
    np.testing.assert_array_equal(a.local_with_halos, before)
    assert a.units == u.m


def test_assignment_differently_distributed_rhs_and_broadcast():
    shape = (N, 3)
    a = array(u.m, np.zeros(shape))
    rhs = array(u.cm, np.arange(3.0) * 100, split=None)
    a[...] = rhs
    check(a, np.broadcast_to(np.arange(3.0), shape), u.m)
    source = array(
        u.mm,
        np.full(shape, 500.0),
        split=1 if mpa.default_comm().Get_size() <= 3 else None,
    )
    a[:] = source
    check(a, np.full(shape, 0.5), u.m)


@pytest.mark.parametrize(
    "target,source,expected",
    [
        (None, u.percent, 0.25),
        (u.percent, None, 2500),
        (u.one, u.percent, 0.25),
        (u.cm, u.m, 2500),
        (u.m, u.m, 25),
    ],
)
def test_assignment_scaled_dimensionless_and_same_unit(target, source, expected):
    a = array(target)
    value = 25 if source is None else 25 * source
    a[:] = value
    check(a, np.full(N, expected), target)


def test_assignment_dimensional_to_unitless_rejected():
    a = array()
    with pytest.raises(ValueError, match="compatible"):
        a[:] = 2 * u.m
    check(a, VALUES, None)


@pytest.mark.parametrize("operation", ["SUM", "MIN", "MAX", "PROD", None])
@pytest.mark.parametrize("unit", [None, u.m, u.percent])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_replicated_reduction_values_units_and_halos(operation, unit, noncontiguous):
    from mpiarray._mpi import MPI

    comm = mpa.default_comm()
    rank, size = comm.Get_rank(), comm.Get_size()
    a = array(unit, np.full((N, 2), rank + 1.0), split=None, halo=1)
    a.local_with_halos[...] = rank + 1
    if noncontiguous:
        a._data = np.asfortranarray(a.local_with_halos)
    expected = {
        None: size * (size + 1) / 2,
        "SUM": size * (size + 1) / 2,
        "MIN": 1,
        "MAX": size,
        "PROD": float(np.prod(np.arange(1, size + 1))),
    }[operation]
    a.allreduce_replicated(None if operation is None else getattr(MPI, operation))
    expected_unit = unit**size if unit is not None and operation == "PROD" else unit
    assert a.units == expected_unit
    np.testing.assert_allclose(a.local_with_halos, expected)
    assert not isinstance(a.local_with_halos, u.Quantity)


@pytest.mark.parametrize("units", [(u.m, u.s), (u.m, u.cm), (u.m, None)])
def test_replicated_inconsistent_units_rejected_on_every_rank(units):
    from mpiarray._mpi import MPI

    comm = mpa.default_comm()
    if comm.Get_size() < 2:
        pytest.skip("needs rank-dependent units")
    a = array(units[comm.Get_rank() % 2], split=None, halo=1)
    before = a.local_with_halos.copy()
    for op in [MPI.SUM, MPI.PROD]:
        with pytest.raises(ValueError, match="same unit on every rank"):
            a.allreduce_replicated(op)
        np.testing.assert_array_equal(a.local_with_halos, before)
        assert a.units == units[comm.Get_rank() % 2]


def test_replicated_unsupported_operation_rejected():
    from mpiarray._mpi import MPI

    if mpa.default_comm().Get_size() < 2:
        pytest.skip("single rank is a no-op")
    a = array(u.m, VALUES.astype(int), split=None)
    with pytest.raises(TypeError, match="units are not supported"):
        a.allreduce_replicated(MPI.BAND)
    check(a, VALUES, u.m)


@pytest.mark.parametrize("first_unit,second_unit", [(u.m, u.cm), (u.cm, u.m)])
@pytest.mark.parametrize("relative", [0, 2 * u.percent])
@pytest.mark.parametrize("tolerance_kind", ["scalar", "global", "distributed"])
def test_isclose_physical_tolerances(first_unit, second_unit, relative, tolerance_kind):
    left = VALUES * u.m
    right = (VALUES + np.linspace(0.001, 0.1, N)) * u.m
    a = array(first_unit, left.to_value(first_unit), halo=1)
    b = array(second_unit, right.to_value(second_unit), halo=1)
    tolerance = 3 * u.cm
    if tolerance_kind != "scalar":
        tolerance = np.linspace(0.5, 5, N) * u.cm
    if tolerance_kind == "distributed":
        atol = array(u.cm, tolerance.value, halo=1)
    else:
        atol = tolerance
    expected = u.isclose(left, right, rtol=relative, atol=tolerance)
    check(np.isclose(a, b, rtol=relative, atol=atol), expected, None)
    assert np.allclose(a, b, rtol=relative, atol=atol) == np.all(expected)


@pytest.mark.parametrize("kwargs", [{"atol": 2 * u.s}, {"rtol": 2 * u.m}])
def test_invalid_tolerances_rejected(kwargs):
    with pytest.raises(ValueError, match="compatible"):
        np.isclose(array(u.m), array(u.cm), **kwargs)


def test_scaled_dimensionless_tolerances_and_quantity_operand():
    a = array(u.m)
    expected = u.isclose(VALUES * u.m, (VALUES + 0.01) * u.m, atol=2 * u.cm)
    check(
        np.isclose(array(u.cm, (VALUES + 0.01) * 100), VALUES * u.m, atol=2 * u.cm),
        expected,
        None,
    )
    check(
        np.isclose(array(), array(values=VALUES + 0.01), atol=2 * u.percent),
        np.ones(N, dtype=bool),
        None,
    )
    check(np.isclose(a, a, atol=0 * u.m), np.ones(N, dtype=bool), None)


@pytest.mark.parametrize("name", ["sum", "mean", "min", "max", "var", "std"])
def test_complex_reductions_match_quantity(name):
    values = VALUES + 1j * VALUES[::-1]
    a = array(u.cm, values, halo=1)
    expected = getattr(values * u.cm, name)()
    actual = getattr(a, name)()
    assert actual.unit == expected.unit
    np.testing.assert_allclose(actual.value, expected.value)


def test_complex_inner_products_match_quantity():
    left, right = VALUES + 1j, VALUES[::-1] - 2j
    a, b = array(u.m, left), array(u.s, right)
    for op in [np.dot, np.vdot]:
        actual = op(a, b)
        expected = op(left * u.m, right * u.s)
        assert actual.unit == expected.unit
        np.testing.assert_allclose(actual.value, expected.value)
    assert a.norm().to_value(u.m) == pytest.approx(np.linalg.norm(left))


@pytest.mark.parametrize("ddof", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
def test_variance_ddof_scaled_units(ddof, keepdims):
    a = array(u.percent)
    for name in ["var", "std"]:
        expected = getattr(VALUES * u.percent, name)(ddof=ddof, keepdims=keepdims)
        result = getattr(a, name)(ddof=ddof, keepdims=keepdims)
        assert result.unit == expected.unit
        np.testing.assert_allclose(result.value, expected.value)


@pytest.mark.parametrize("exponent", [-2, -0.5, 0, 0.5, 3])
def test_fractional_negative_zero_powers_match_quantity(exponent):
    a = array(u.cm)
    result = a**exponent
    expected = (VALUES * u.cm) ** exponent
    check(result, expected.value, expected.unit)


def test_failed_inplace_cast_preserves_units_and_values():
    a = array(u.m, VALUES.astype(int))
    with pytest.raises(TypeError):
        a += array(u.cm)
    check(a, VALUES, u.m)
    with pytest.raises(TypeError):
        np.multiply(a, 0.5, out=a)
    check(a, VALUES, u.m)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_comparisons_nonfinite_and_unit_conversion(equal_nan):
    values = VALUES.copy()
    values[:3] = [np.nan, np.inf, -np.inf]
    a, b = array(u.m, values), array(u.cm, values * 100)
    expected = np.isclose(values, values, equal_nan=equal_nan)
    check(np.isclose(a, b, equal_nan=equal_nan), expected, None)
    assert np.array_equal(a, b, equal_nan=equal_nan) == equal_nan


def test_assignment_aliasing_and_failed_broadcast():
    a = array(u.m)
    a[::-1] = a
    check(a, VALUES[::-1], u.m)
    a[:] = a
    check(a, VALUES[::-1], u.m)
    with pytest.raises(ValueError):
        a[:] = np.ones((N, 2)) * u.cm
    check(a, VALUES[::-1], u.m)


def test_unit_metadata_cannot_change_on_failed_parse():
    a = array(u.m)
    for invalid in [123, "invalid-unit", u.deg_C]:
        with pytest.raises((ValueError, TypeError)):
            a.set_units(invalid)
        check(a, VALUES, u.m)


def test_replicated_product_then_sum_retains_squared_unit():
    from mpiarray._mpi import MPI

    size = mpa.default_comm().Get_size()
    a = array(u.m, np.full(N, 2.0), split=None)
    a.allreduce_replicated(MPI.PROD)
    a.allreduce_replicated(MPI.SUM)
    check(a, np.full(N, 2.0**size * size), u.m**size)


def test_replicated_single_rank_unitful_operation_is_noop():
    from mpiarray._mpi import MPI

    a = array(u.m, comm=MPI.COMM_SELF)
    for op in [MPI.SUM, MPI.PROD, MPI.MAX]:
        a.allreduce_replicated(op)
        check(a, VALUES, u.m)


def test_unitless_replicated_bitwise_operation_still_works():
    from mpiarray._mpi import MPI

    a = array(values=np.full(N, 7), split=None)
    a.allreduce_replicated(MPI.BAND)
    check(a, np.full(N, 7), None)


def test_tolerance_broadcast_and_distributed_relative_tolerance():
    values = np.broadcast_to(VALUES[:, None], (N, 3)).copy()
    a, b = array(u.m, values), array(u.cm, (values + 0.01) * 100)
    relative = array(u.percent, np.full((N, 3), 1.0))
    atol = np.array([0, 1, 2]) * u.cm
    expected = u.isclose(values * u.m, (values + 0.01) * u.m, rtol=0.01, atol=atol)
    check(np.isclose(a, b, rtol=relative, atol=atol), expected, None)


def test_dimensionful_tolerance_on_unitless_array_rejected():
    with pytest.raises(ValueError, match="atol needs compatible"):
        np.isclose(array(), array(), atol=1 * u.m)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("int64", 50 * u.cm),
        ("int8", 1000 * u.m),
        ("uint8", -1 * u.m),
        ("int64", np.inf * u.m),
        ("int64", np.nan * u.m),
        ("int64", (1 + 1j) * u.m),
        ("int64", float(2**63) * u.m),
    ],
)
@pytest.mark.parametrize("index", [slice(None), np.array([0, N - 1])])
def test_assignment_rejects_integer_precision_loss(dtype, value, index):
    a = array(u.m, np.ones(N, dtype=dtype), halo=1)
    original = a.local_with_halos.copy()
    with pytest.raises(TypeError, match="lose precision"):
        a[index] = value
    np.testing.assert_array_equal(a.local_with_halos, original)
    assert a.units == u.m


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int64"])
def test_exact_integer_assignment_matches_quantity(dtype):
    expected = u.Quantity(np.ones(N, dtype=dtype), u.m, dtype=dtype)
    expected[:] = 200 * u.cm
    a = array(u.m, np.ones(N, dtype=dtype))
    a[:] = 200 * u.cm
    check(a, expected.value, u.m)
    assert a.dtype == np.dtype(dtype)


def test_integer_dimensionless_assignment_checks_conversion():
    a = array(values=np.ones(N, dtype=int))
    with pytest.raises(TypeError, match="lose precision"):
        a[:] = 50 * u.percent
    a[:] = 200 * u.percent
    check(a, np.full(N, 2), None)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_integer_assignment_exact_boundaries(dtype):
    a = array(u.m, np.ones(N, dtype=dtype))
    for value in [np.iinfo(dtype).min, np.iinfo(dtype).max]:
        a[:] = u.Quantity(value, u.m, dtype=dtype)
        np.testing.assert_array_equal(a.to_numpy(), np.full(N, value, dtype=dtype))
