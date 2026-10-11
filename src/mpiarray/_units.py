"""Unit algebra for `DistributedArray`'s optional ``_unit`` attribute.

Backed by `astropy.units <https://docs.astropy.org/en/stable/units/>`_,
imported lazily: `DistributedArray` can be imported and used with no units
attached, and ``import mpiarray`` itself, without ever importing astropy.
Only paths that use a non-``None`` unit import astropy. Install it separately (``pip install
astropy``, or the ``units`` extra).

mpiarray never wraps its own storage (a NumPy/CuPy array) in an
``astropy.units.Quantity``: MPI sends and receives that storage directly via
buffer protocols, and cunumpy dispatches it to NumPy or CuPy, neither of
which tolerate a ``Quantity`` wrapper. A unit is tracked separately, as
``DistributedArray._unit`` (an ``astropy.units.UnitBase`` or ``None``), and
the functions here compute what an operation's *result* unit should be
(raising if the operands are not compatible), touching the bulk array data
only for a plain scalar rescale when combining differently-scaled but
equivalent units (e.g. ``eV`` and ``J``) — never wrapping it.

Every ufunc/function this module does not recognize (by ``__name__``) raises
if a unit is attached to any operand, and is completely unaffected (today's
behaviour, unchanged) when none is: fail closed, not fail silent. Known gaps
left for later: `numpy.clip`/`where`/`divmod`, angle conversions (`deg2rad`,
...), index and axis reductions (rejected), and selections (which drop units,
like `dims`).
"""

from __future__ import annotations

from typing import Any

#: unit becomes ``unit ** exponent`` for these fixed exponents.
_POWER: dict[str, float] = {"square": 2, "sqrt": 0.5, "cbrt": 1 / 3, "reciprocal": -1}

#: unit becomes ``unit ** exponent``, with ``exponent`` read from the second
#: (plain scalar) operand.
_VARIABLE_POWER = frozenset({"power", "float_power"})

#: unit becomes ``unit_a * unit_b`` or ``unit_a / unit_b``; a unit-less
#: operand is dimensionless (it just scales the result).
_MULTIPLICATIVE = frozenset({"multiply", "divide", "true_divide"})

#: both operands' units must be equivalent (rescaled to the first's if not
#: identical); the result keeps the first operand's unit.
_ADDITIVE = frozenset(
    {
        "add",
        "subtract",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "mod",
        "fmod",
        "remainder",
        "hypot",
        "copysign",
        "nextafter",
    }
)

#: like `_ADDITIVE` (operands must be equivalent, rescaled the same way), but
#: the boolean result has no unit.
_COMPARISON = frozenset(
    {"less", "less_equal", "equal", "not_equal", "greater", "greater_equal", "isclose"}
)

#: unary; the result keeps the (single) input's unit unchanged.
_UNIT_PRESERVING = frozenset(
    {
        "round",
        "negative",
        "positive",
        "absolute",
        "floor",
        "ceil",
        "trunc",
        "rint",
        "conj",
        "conjugate",
        "spacing",
        "sign",
    }
)

#: needs a dimensionless input; the result has none.
_DIMENSIONLESS_REQUIRED = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "logaddexp",
        "logaddexp2",
    }
)

#: ignore units entirely: always allowed, result has none (boolean).
_UNIT_AGNOSTIC = frozenset({"isnan", "isinf", "isfinite", "signbit"})


def dimensionless() -> Any:
    """Return astropy's dimensionless unit (imports astropy)."""
    import astropy.units as u

    return u.dimensionless_unscaled


def parse(unit: Any) -> Any:
    """Return ``unit`` as an ``astropy.units.UnitBase``, parsing a string."""
    if unit is None:
        return None
    import astropy.units as u

    if not isinstance(unit, (str, u.UnitBase)):
        raise TypeError("units must be a string, an astropy Unit, or None")
    result = u.Unit(unit) if isinstance(unit, str) else unit
    if result == u.deg_C:
        raise ValueError("offset temperature units are not supported; use K")
    return result


def combine_multiplicative(unit_a: Any, unit_b: Any) -> Any:
    """Return the unit of ``a * b`` (``None`` if both ``a`` and ``b`` are unit-less)."""
    if unit_a is None and unit_b is None:
        return None
    dimless = dimensionless()
    a = dimless if unit_a is None else unit_a
    b = dimless if unit_b is None else unit_b
    return a * b


def wrap_scalar(value: Any, unit: Any) -> Any:
    """Return ``value`` as an ``astropy.units.Quantity`` of ``unit``."""
    return value * unit


def unit_of(value: Any) -> Any:
    """Read a distributed unit or a host Quantity's unit without importing astropy."""
    return getattr(value, "_unit", getattr(value, "unit", None))


def raw(value: Any) -> Any:
    """Unwrap a host Quantity before passing it to the array backend."""
    return value.value if hasattr(value, "unit") else value


def conversion_factor(source: Any, target: Any, name: str) -> float:
    """Return a numeric scale, rejecting incompatible units before touching values."""
    if source is None and target is None:
        return 1.0
    dimless = dimensionless()
    source = dimless if source is None else source
    target = dimless if target is None else target
    if not source.is_equivalent(target):
        raise ValueError(
            f"{name} needs compatible units; got {source!r} and {target!r}"
        )
    return source.to(target)


def plan(op: Any, inputs: tuple[Any, ...]) -> tuple[Any, dict[int, float]]:
    """Compute the result unit and scalar rescalings before executing an operation."""
    name = getattr(op, "__name__", None)
    units = [unit_of(x) for x in inputs]
    dimless = dimensionless()
    normalized = [dimless if unit is None else unit for unit in units]
    if name in _UNIT_AGNOSTIC:
        return None, {}
    if name in _UNIT_PRESERVING:
        return units[0], {}
    if name in _DIMENSIONLESS_REQUIRED:
        scales = {}
        for index, unit in enumerate(normalized):
            if not unit.is_equivalent(dimless):
                raise ValueError(f"{name}() needs dimensionless inputs; got {unit!r}")
            if unit != dimless:
                scales[index] = unit.to(dimless)
        return None, scales
    if name in _POWER:
        return normalized[0] ** _POWER[name], {}
    if name in _VARIABLE_POWER:
        import numpy as np

        if units[1] is not None or np.ndim(inputs[1]) != 0:
            raise TypeError(
                f"{name}(): the exponent must be a plain scalar without units"
            )
        return normalized[0] ** float(inputs[1]), {}
    if name in _MULTIPLICATIVE:
        a, b = normalized
        return (a / b if name in ("divide", "true_divide") else a * b), {}
    if name in _ADDITIVE or name in _COMPARISON:
        a, b = normalized
        if not a.is_equivalent(b):
            raise ValueError(f"{name}() needs compatible units; got {a!r} and {b!r}")
        scales = {} if a == b else {1: b.to(a)}
        return (None if name in _COMPARISON else units[0]), scales
    raise TypeError(f"units are not supported for {name or op!r} yet")
