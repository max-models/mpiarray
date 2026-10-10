"""xarray interoperability: a gathered array as a labeled ``xarray.DataArray``.

mpiarray has no notion of what an axis means (unlike a domain-specific field
object, which knows its axes are ``x``/``y``/``z``), so the names and
coordinate values are given at the call site::

    import mpiarray as mpa

    rho = mpa.zeros((64, 48))
    data = rho.to_xarray(("x", "y"), coords={"x": x_values, "y": y_values})

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
    dims: Sequence[str],
    *,
    coords: Any = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: None = None,
) -> xr.DataArray: ...
@overload
def to_xarray(
    array: DistributedArray,
    dims: Sequence[str],
    *,
    coords: Any = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: int,
) -> xr.DataArray | None: ...
def to_xarray(
    array: DistributedArray,
    dims: Sequence[str],
    *,
    coords: Any = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    root: int | None = None,
) -> xr.DataArray | None:
    """Return ``array`` gathered and labeled as an ``xarray.DataArray``.

    Collective: call on every rank, as `DistributedArray.gather` (which this
    uses) requires.

    Args:
        array: The distributed array to gather.
        dims: One name per axis, in order.
        coords: Coordinate values, as ``xarray.DataArray`` accepts them (e.g.
            a dict of dim name to a 1-D array); default: none attached.
        name: The result's name.
        attrs: Attributes attached to the result (e.g. ``{"units": "eV"}``).
        root: The rank that receives the array; default: every rank (see
            `DistributedArray.gather`).

    Returns:
        The labeled array, or ``None`` on a rank that `gather` left out.

    Raises:
        ValueError: If ``dims`` does not have one name per axis.
    """
    import xarray as xr

    if len(dims) != array.ndim:
        raise ValueError(f"dims has {len(dims)} names for a {array.ndim}-D array")
    data = array.to_numpy(root=root)
    if data is None:
        return None
    return xr.DataArray(data, dims=tuple(dims), coords=coords, name=name, attrs=attrs)
