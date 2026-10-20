"""xarray interoperability: a gathered array as a labeled ``xarray.DataArray``.

mpiarray has no notion of what an axis means, so the names default to none
unless set with `DistributedArray.labeled`::

    import mpiarray as mpa

    rho = mpa.zeros((64, 48)).labeled("x", "y")
    data = rho.to_xarray(x=x_values, y=y_values, name="rho")
    # or give the names at the call site instead:
    data = mpa.zeros((64, 48)).to_xarray(("x", "y"), x=x_values, y=y_values)

``name`` and general ``attrs`` are given at each call. ``attrs["units"]`` defaults
to the array's physical unit, if one was attached with ``set_units``; an explicit
attribute overrides this default.

Each coordinate's values are passed as a keyword argument named after its axis
(``x=``, ``y=``, ...; also any auxiliary coordinate over the named axes, e.g.
physical ``X=``/``Y=``/``Z=`` grids over logical ``eta1``/``eta2``). An axis
named like one of `to_xarray`'s own parameters (``dims``, ``name``, ``attrs``,
``root`` or ``coords``) cannot be passed this way; give it through ``coords=``
instead (a dict, merged with the keyword coordinates).

This needs xarray, installed separately (``pip install xarray``, or the
``xarray`` extra); imported lazily, so ``import mpiarray`` works without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    import xarray as xr

    from mpiarray.distributed_array import DistributedArray

__all__ = ["to_xarray"]


@overload
def to_xarray(
    array: DistributedArray,
    dims: Sequence[str] | None = None,
    *,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: None = None,
    coords: dict[str, Any] | None = None,
    **axis_coords: Any,
) -> xr.DataArray: ...
@overload
def to_xarray(
    array: DistributedArray,
    dims: Sequence[str] | None = None,
    *,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: int,
    coords: dict[str, Any] | None = None,
    **axis_coords: Any,
) -> xr.DataArray | None: ...
def to_xarray(
    array: DistributedArray,
    dims: Sequence[str] | None = None,
    *,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: int | None = None,
    coords: dict[str, Any] | None = None,
    **axis_coords: Any,
) -> xr.DataArray | None:
    """Return ``array`` gathered and labeled as an ``xarray.DataArray``.

    Collective: call on every rank, as `DistributedArray.gather` (which this
    uses) requires.

    Args:
        array: The distributed array to gather.
        dims: One name per axis, in order; default: ``array.dims``, set by
            `DistributedArray.labeled`.
        name: The result's name.
        attrs: Attributes attached to the result (e.g. ``{"units": "eV"}``).
        root: The rank that receives the array; default: every rank (see
            `DistributedArray.gather`).
        coords: Coordinate values, as ``xarray.DataArray`` accepts them (e.g.
            a dict of dim name to a 1-D array); merged with ``**axis_coords``,
            for an axis named like one of this function's own parameters.
        **axis_coords: Coordinate values keyed by axis name, e.g.
            ``x=x_values`` for a `labeled` ``"x"`` axis.

    Returns:
        The labeled array, or ``None`` on a rank that `gather` left out.

    Raises:
        ValueError: If ``dims`` is not given and the array was not labeled
            with `DistributedArray.labeled`, or it does not have one name per
            axis.
    """
    import xarray as xr

    if dims is None:
        dims = array.dims
        if dims is None:
            raise ValueError(
                "to_xarray needs dims, or a.labeled(...) to set them beforehand",
            )
    if len(dims) != array.ndim:
        raise ValueError(f"dims has {len(dims)} names for a {array.ndim}-D array")
    data = array.to_numpy(root=root)
    if data is None:
        return None
    attrs = dict(attrs or {})
    if array.units is not None:
        attrs.setdefault("units", str(array.units))
    return xr.DataArray(
        data,
        dims=tuple(dims),
        coords={**(coords or {}), **axis_coords},
        name=name,
        attrs=attrs,
    )
