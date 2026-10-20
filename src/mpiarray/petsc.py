"""PETSc interoperability: DMDAs with a layout's decomposition, and vector transfer.

Solve on mpiarray data with PETSc (KSP, SNES, TS) without building a second
decomposition by hand::

    import mpiarray as mpa

    rho = mpa.zeros((64, 48), halo=1)
    da = rho.layout.dmda()          # a PETSc.DMDA with exactly rho's blocks
    b = rho.to_petsc()              # a global PETSc.Vec of da, holding rho's blocks
    x = da.createGlobalVec()
    ...                             # assemble a matrix (see `petsc_numbering`), solve
    phi = mpa.from_petsc(x, rho.layout)

**Axis order.** PETSc numbers ranks with x varying fastest, mpiarray row-major
with the last axis fastest. The DMDA therefore gets the layout's axes
reversed: PETSc's x is the layout's last axis. The rank numbers then agree, and
a rank's block of a PETSc global vector (x fastest) has exactly the memory
order of its C-ordered mpiarray block, so the transfer is one local copy.
PETSc's *natural* ordering (x fastest over the reversed axes) is mpiarray's
global C order.

This module needs petsc4py, installed separately (``pip install petsc4py``,
or with conda, which has binaries); it is imported
lazily, so ``import mpiarray`` works without it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cunumpy as xp
import numpy as np

from mpiarray.layout import Layout, _per_axis, block_numbering

if TYPE_CHECKING:
    from mpiarray.distributed_array import DistributedArray

#: A PETSc boundary type: ``PETSc.DM.BoundaryType`` member, its int or its name.
BoundaryLike = Any

__all__ = [
    "block_numbering",
    "clear_dmda_cache",
    "copy_from_petsc",
    "copy_to_petsc",
    "dmda",
    "from_petsc",
    "petsc_numbering",
]

# DMDAs made by `dmda`, per layout and options; see `dmda` and `clear_dmda_cache`.
_DMDA_CACHE: dict[tuple, Any] = {}


def _petsc() -> Any:
    """Return ``petsc4py.PETSc``, or raise an ImportError saying how to install it."""
    try:
        from petsc4py import PETSc  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(
            "PETSc interoperability needs petsc4py; install it with "
            "`pip install petsc4py` (or `conda install -c conda-forge petsc4py`)",
        ) from error
    return PETSc


def _petsc_comm(layout: Layout) -> Any:
    """Return the communicator of the layout's DMDA.

    ``COMM_SELF`` when one rank holds everything: on a single rank (possibly
    maybempi's serial stand-in, which PETSc does not know), and in a
    replicated layout, where every rank holds its own copy of the whole array.
    """
    PETSc = _petsc()  # noqa: N806 - PETSc's own name
    if layout.size == 1 or layout.replicated:
        return PETSc.COMM_SELF
    return PETSc.Comm(layout.comm)


def dmda(
    layout: Layout,
    *,
    stencil_width: int | None = None,
    stencil_type: str = "star",
    boundary_type: BoundaryLike | Sequence[BoundaryLike] | None = None,
) -> Any:
    """Return a ``PETSc.DMDA`` with exactly the decomposition of ``layout``.

    The DMDA has one degree of freedom per element and the layout's axes in
    reverse order (sizes ``layout.shape[::-1]``, process grid
    ``layout.process_grid[::-1]``, ownership ranges from the layout's cuts),
    so that its rank numbering, and each rank's block of a global vector,
    match the layout's (see the module docstring). Its communicator is
    ``layout.comm``; on one rank, and in a replicated layout (several ranks,
    nothing split), it is ``COMM_SELF`` and holds the whole array on every
    rank. Also available as `Layout.dmda`.

    The DMDA is cached per layout and options (layouts compare by value), so
    repeated calls, and `DistributedArray.to_petsc`, return the same object
    without rebuilding it. Do not destroy it or change its settings (such as
    with ``setFromOptions``): take a ``da.clone()`` for that. The cache keeps
    the DMDAs alive until `clear_dmda_cache`.

    Collective over ``layout.comm`` the first time it is called for a layout
    and options.

    Args:
        layout: The decomposition, with 1 to 3 axes.
        stencil_width: The width of PETSc's ghost region (local vectors and
            matrix preallocation). DMDA has one width for all axes; default:
            the largest of ``layout.halo``, but at least 1. PETSc requires it
            to be at most the smallest block along every split axis.
        stencil_type: ``"star"`` or ``"box"``.
        boundary_type: The PETSc boundary type of every axis, or one per axis
            in the layout's axis order (``PETSc.DM.BoundaryType`` members or
            names such as ``"periodic"``, ``"ghosted"``, ``"none"``). Default:
            ``PERIODIC`` along the layout's periodic axes, ``NONE`` along the
            others.

    Returns:
        The (shared, cached) ``petsc4py.PETSc.DMDA``.

    Raises:
        ImportError: Without petsc4py.
        ValueError: For a layout with no or more than 3 axes, an unknown
            stencil type, or a ``boundary_type`` of the wrong length.
    """
    PETSc = _petsc()  # noqa: N806 - PETSc's own name
    if not 1 <= layout.ndim <= 3:
        raise ValueError(
            f"a PETSc DMDA has 1 to 3 axes, the layout has {layout.ndim}",
        )
    stencils = {"star": PETSc.DMDA.StencilType.STAR, "box": PETSc.DMDA.StencilType.BOX}
    if stencil_type not in stencils:
        raise ValueError(
            f"stencil_type must be 'star' or 'box', not {stencil_type!r}",
        )
    width = max(1, *layout.halo) if stencil_width is None else int(stencil_width)
    if boundary_type is None:
        boundaries = tuple(
            PETSc.DM.BoundaryType.PERIODIC if p else PETSc.DM.BoundaryType.NONE
            for p in layout.periodic
        )
    else:
        boundaries = _per_axis(boundary_type, layout.ndim, "boundary_type")

    key = (layout, width, stencil_type, boundaries)
    cached = _DMDA_CACHE.get(key)
    if cached is not None:
        return cached
    da = PETSc.DMDA().create(
        dim=layout.ndim,
        dof=1,
        sizes=layout.shape[::-1],
        proc_sizes=layout.process_grid[::-1],
        boundary_type=boundaries[::-1],
        stencil_type=stencils[stencil_type],
        stencil_width=width,
        ownership_ranges=tuple(
            tuple(int(n) for n in np.diff(cuts)) for cuts in layout.bounds[::-1]
        ),
        comm=_petsc_comm(layout),
        setup=True,
    )
    _DMDA_CACHE[key] = da
    return da


def clear_dmda_cache() -> None:
    """Destroy the DMDAs cached by `dmda` and empty the cache.

    Collective: destroying a DMDA is collective over its communicator, so
    every rank must call this. Vectors made from the DMDAs stay usable.
    """
    for da in _DMDA_CACHE.values():
        da.destroy()
    _DMDA_CACHE.clear()


def petsc_numbering(layout: Layout) -> np.ndarray:
    """Return the row of every global element in the DMDA's global ordering.

    PETSc numbers the rows of a DMDA's global vectors (and matrices) rank by
    rank: each rank's block is a consecutive range, in rank order, and is
    C-ordered inside (see the module docstring). With this array a matrix can
    be assembled from natural grid indices, e.g. the row of element
    ``(i, j)`` is ``numbering[i, j]``.

    Computed locally from the layout's cuts (`Layout.bounds`) by
    `block_numbering`, without communication or petsc4py; every rank gets the
    whole array. In a replicated layout, and on
    one rank, it is the C order ``arange(size).reshape(shape)``.

    Args:
        layout: The decomposition.

    Returns:
        An int64 NumPy array of ``layout.shape``.
    """
    return block_numbering(layout.shape, layout.bounds)


def _check_vec(vec: Any, layout: Layout) -> None:
    """Raise unless ``vec`` holds exactly this rank's block of ``layout``."""
    local, total = vec.getLocalSize(), vec.getSize()
    if local != math.prod(layout.local_shape) or total != math.prod(layout.shape):
        raise ValueError(
            f"a PETSc vector of local size {local} (global {total}) does not hold "
            f"the block of shape {layout.local_shape} of an array of shape "
            f"{layout.shape}; make it with layout.dmda().createGlobalVec()",
        )


def copy_to_petsc(a: DistributedArray, vec: Any) -> None:
    """Copy this rank's block of ``a``, without halo cells, into a global PETSc vector.

    ``vec`` must be a global vector of ``a.layout.dmda()`` (or have its
    parallel layout); the copy is local, without communication. Values are
    converted to PETSc's scalar type. CuPy blocks are copied through the
    host; handing device memory to PETSc's CUDA vectors is future work.

    Args:
        a: The array to copy.
        vec: The ``PETSc.Vec`` to overwrite.

    Raises:
        ValueError: If ``vec`` does not hold this rank's block of ``a``.
    """
    _check_vec(vec, a.layout)
    block = xp.to_numpy(a.local)
    with vec as array:  # get and restore the array, so PETSc sees the change
        array.reshape(a.layout.local_shape)[...] = block


def copy_from_petsc(vec: Any, a: DistributedArray) -> None:
    """Copy a global PETSc vector into this rank's block of ``a``.

    The reverse of `copy_to_petsc`: local, without communication; the halo
    cells of ``a`` are left untouched (call ``a.update_halos()`` if needed).
    CuPy blocks are filled through the host.

    Args:
        vec: A global vector of ``a.layout.dmda()``.
        a: The array to overwrite.

    Raises:
        ValueError: If ``vec`` does not hold this rank's block of ``a``.
    """
    _check_vec(vec, a.layout)
    block = vec.getArray(readonly=True).reshape(a.layout.local_shape)
    local = a.local
    local[...] = xp.asarray(block) if xp.is_gpu(local) else block


def from_petsc(vec: Any, layout: Layout) -> DistributedArray:
    """Return a new distributed array holding a global PETSc vector.

    Args:
        vec: A global vector of ``layout.dmda()``.
        layout: The layout of the new array; its halo cells are zero.

    Returns:
        A `DistributedArray` of PETSc's scalar type.

    Raises:
        ValueError: If ``vec`` does not hold this rank's block of ``layout``.
    """
    from mpiarray.creation import zeros

    _check_vec(vec, layout)
    a = zeros(layout=layout, dtype=vec.getArray(readonly=True).dtype)
    copy_from_petsc(vec, a)
    return a
