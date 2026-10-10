"""Module-level aliases for NumPy's ufuncs, for code that should not import NumPy.

Each name here is NumPy's own ufunc object, unchanged: calling it on a
`mpiarray.DistributedArray` dispatches through
`DistributedArray.__array_ufunc__`, exactly as calling ``numpy.<name>``
would; on a plain NumPy or CuPy array it behaves as NumPy's ufunc always
has. Arithmetic (``+``, ``-``, ``*``, ...), comparisons, ``abs``, unary
``-``/``+`` and the methods (`DistributedArray.round`, `DistributedArray.dot`,
...) already cover their own ufuncs and are not repeated here.
"""

import numpy as np

# exponentials and logarithms
exp = np.exp
exp2 = np.exp2
expm1 = np.expm1
log = np.log
log2 = np.log2
log10 = np.log10
log1p = np.log1p

# powers and roots
sqrt = np.sqrt
cbrt = np.cbrt
square = np.square
reciprocal = np.reciprocal
float_power = np.float_power

# trigonometric
sin = np.sin
cos = np.cos
tan = np.tan
arcsin = np.arcsin
arccos = np.arccos
arctan = np.arctan
arctan2 = np.arctan2
hypot = np.hypot
deg2rad = np.deg2rad
rad2deg = np.rad2deg
degrees = np.degrees
radians = np.radians

# hyperbolic
sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
arcsinh = np.arcsinh
arccosh = np.arccosh
arctanh = np.arctanh

# rounding
floor = np.floor
ceil = np.ceil
trunc = np.trunc
rint = np.rint

# sign, extrema and remainder
sign = np.sign
signbit = np.signbit
copysign = np.copysign
maximum = np.maximum
minimum = np.minimum
fmax = np.fmax
fmin = np.fmin
mod = np.mod
fmod = np.fmod
remainder = np.remainder
divmod = np.divmod  # noqa: A001 - matches NumPy's own name
gcd = np.gcd
lcm = np.lcm
spacing = np.spacing
nextafter = np.nextafter
heaviside = np.heaviside
logaddexp = np.logaddexp
logaddexp2 = np.logaddexp2

# tests
isnan = np.isnan
isinf = np.isinf
isfinite = np.isfinite

# logical and bitwise
logical_and = np.logical_and
logical_or = np.logical_or
logical_xor = np.logical_xor
logical_not = np.logical_not
bitwise_and = np.bitwise_and
bitwise_or = np.bitwise_or
bitwise_xor = np.bitwise_xor
invert = np.invert
left_shift = np.left_shift
right_shift = np.right_shift

# complex
conj = np.conj
conjugate = np.conjugate

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divmod",
    "exp",
    "exp2",
    "expm1",
    "float_power",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "heaviside",
    "hypot",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "lcm",
    "left_shift",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "nextafter",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]
