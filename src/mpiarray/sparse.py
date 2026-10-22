"""Stencil operators: fixed-sparsity-pattern linear operators on distributed arrays.

::

    import mpiarray as mpa
    from mpiarray import StencilOperator

    u = mpa.zeros((64, 48), halo=1, periodic=True)
    # the 5-point 2D Laplacian (unit spacing)
    laplacian = StencilOperator(
        u.layout,
        offsets=[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
        coefficients=[4.0, -1.0, -1.0, -1.0, -1.0],
    )
    lu = laplacian.apply(u)        # a new DistributedArray, same layout as u
    matrix = laplacian.to_petsc_mat()  # the same operator as a PETSc Mat

A `StencilOperator` stores only its ``offsets`` (per-axis neighbour shifts)
and ``coefficients`` (one scalar, or one array of `~Layout.local_shape`, per
offset) -- never a full sparse matrix -- which is enough for the constant (or
spatially varying but fixed-pattern) finite-difference operators used by
stencil solvers: `apply` halo-exchanges and sums the shifted terms directly on
a `DistributedArray`; `to_petsc_mat` assembles the identical pattern as a
``PETSc.Mat`` for use with PETSc's Krylov and multigrid solvers.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import cunumpy as xp
import scope_profiler as sp

from mpiarray._profiling import PROFILE_TAGS
from mpiarray.creation import zeros_like
from mpiarray.distributed_array import DistributedArray
from mpiarray.layout import Layout

#: A per-axis integer shift, e.g. ``(-1, 0)`` for "one cell back along axis 0".
Offset = tuple[int, ...]

#: A global grid index, as for `DistributedArray.__getitem__`.
Index = tuple[int, ...]

__all__ = ["StencilOperator"]


def _is_scalar(coefficient: Any) -> bool:
    """Whether ``coefficient`` is a single number rather than a per-cell array."""
    return not hasattr(coefficient, "shape")


class StencilOperator:
    """A linear operator given by a fixed pattern of neighbour offsets.

    Represents operators of the form::

        (A u)[i] = sum_k coefficients[k][i] * u[i + offsets[k]]

    the shape finite-difference stencils have: a small, fixed set of
    neighbour offsets shared by (almost) every grid point, for example the
    5-point 2D Laplacian's offsets ``(0, 0)``, ``(-1, 0)``, ``(1, 0)``,
    ``(0, -1)``, ``(0, 1)``.

    A handful of rows that differ from the regular pattern (Dirichlet rows,
    cut cells near an embedded boundary, ...) can be given their own terms
    through ``overrides``: a mapping from a global grid index (as for
    `DistributedArray.__getitem__`) to its own list of ``(offset,
    coefficient)`` pairs, replacing the regular stencil at that row only, on
    whichever rank owns it.

    Operators of the same `layout` combine with ``+``, ``-``, unary ``-`` and
    multiplication by a number, merging offsets (and overridden rows) term by
    term, e.g. ``laplacian + mass_matrix`` or ``-0.5 * laplacian``.

    Args:
        layout: The decomposition the operator applies to; also fixes how
            many axes an offset has.
        offsets: The neighbour shifts, one per axis each, with no duplicates.
        coefficients: One entry per offset: a number, or an array of
            `~Layout.local_shape` for a coefficient that varies by grid point
            (the same on every rank; each rank indexes its own block).
        overrides: Exceptions to the regular stencil, by global index.

    Raises:
        ValueError: If ``offsets`` and ``coefficients`` have different
            lengths, an offset's length does not match ``layout.ndim``,
            offsets are duplicated, or ``layout.halo`` is too narrow for the
            offsets used (regular or in ``overrides``).
    """

    def __init__(
        self,
        layout: Layout,
        offsets: Sequence[Offset],
        coefficients: Sequence[Any],
        *,
        overrides: Mapping[Index, Sequence[tuple[Offset, Any]]] | None = None,
    ) -> None:
        """Validate and store the stencil; see the class docstring."""
        offsets = tuple(tuple(int(o) for o in offset) for offset in offsets)
        if len(offsets) != len(coefficients):
            raise ValueError(
                f"{len(offsets)} offsets but {len(coefficients)} coefficients",
            )
        if any(len(offset) != layout.ndim for offset in offsets):
            raise ValueError(
                f"every offset must have {layout.ndim} entries "
                f"(the layout has {layout.ndim} axes), got {offsets}",
            )
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"duplicate offset in {offsets}")
        for coefficient in coefficients:
            if not _is_scalar(coefficient) and tuple(coefficient.shape) != tuple(
                layout.local_shape,
            ):
                raise ValueError(
                    "a per-cell coefficient must have shape layout.local_shape "
                    f"{layout.local_shape}, got {tuple(coefficient.shape)}",
                )
        overrides = {
            tuple(int(i) for i in index): [
                (tuple(int(o) for o in offset), coefficient)
                for offset, coefficient in terms
            ]
            for index, terms in (overrides or {}).items()
        }
        needed = [0] * layout.ndim
        for offset in itertools.chain(
            offsets, (offset for terms in overrides.values() for offset, _ in terms)
        ):
            for axis, o in enumerate(offset):
                needed[axis] = max(needed[axis], abs(o))
        if any(n > h for n, h in zip(needed, layout.halo, strict=True)):
            raise ValueError(
                f"the stencil needs halo widths {tuple(needed)}, "
                f"the layout has {layout.halo}",
            )
        self._layout = layout
        self._offsets = offsets
        self._coefficients = tuple(coefficients)
        self._overrides = overrides

    @property
    def layout(self) -> Layout:
        """The decomposition this operator applies to."""
        return self._layout

    @property
    def offsets(self) -> tuple[Offset, ...]:
        """The regular stencil's neighbour offsets."""
        return self._offsets

    @property
    def coefficients(self) -> tuple[Any, ...]:
        """The regular stencil's coefficients, one per `offsets` entry."""
        return self._coefficients

    @sp.profile("mpiarray.StencilOperator.apply", tags=PROFILE_TAGS)
    def apply(
        self,
        u: DistributedArray,
        out: DistributedArray | None = None,
        *,
        boundary: Any = None,
    ) -> DistributedArray:
        """Return ``A @ u``, this operator applied to ``u``.

        Collective: calls ``u.update_halos()`` first, so the shifted terms
        read the neighbours' current values.

        Args:
            u: The vector to apply the operator to; must share this
                operator's layout.
            out: Where to write the result; default: a new array like ``u``.
            boundary: Forwarded to `DistributedArray.update_halos`, filling
                ``u``'s halo cells at walls (non-periodic boundaries) before
                the stencil is applied.

        Returns:
            ``out`` (or the new array), holding the result.

        Raises:
            ValueError: If ``u``'s layout is not this operator's layout.
        """
        if u.layout != self._layout:
            raise ValueError("u's layout does not match the operator's layout")
        u.update_halos(boundary=boundary)
        halo = self._layout.halo
        local_shape = self._layout.local_shape
        storage = u.local_with_halos
        result = xp.zeros(local_shape, dtype=storage.dtype)
        for offset, coefficient in zip(self._offsets, self._coefficients, strict=True):
            window = tuple(
                slice(h + o, h + o + n)
                for h, o, n in zip(halo, offset, local_shape, strict=True)
            )
            result += coefficient * storage[window]
        for index, terms in self._overrides.items():
            local = u.local_index(index)
            if local is None:  # another rank owns this row
                continue
            value = 0.0
            for offset, coefficient in terms:
                position = tuple(p + o for p, o in zip(local, offset, strict=True))
                value += coefficient * storage[position]
            result[tuple(p - h for p, h in zip(local, halo, strict=True))] = value
        if out is None:
            out = zeros_like(u)
        out.local[...] = result
        return out

    def _local_position(self, index: Index) -> tuple[int, ...] | None:
        """``index`` relative to this rank's block, or ``None`` if not owned."""
        position = tuple(
            i - start
            for i, (start, _) in zip(index, self._layout.index_bounds, strict=True)
        )
        if any(
            p < 0 or p >= n
            for p, n in zip(position, self._layout.local_shape, strict=True)
        ):
            return None
        return position

    def _terms_at(self, index: Index) -> list[tuple[Offset, Any]] | None:
        """The row at global ``index``, with any per-cell coefficient resolved to a number.

        Returns ``None`` if this rank does not own ``index`` (its per-cell
        coefficients, if any, are not available locally).
        """
        position = self._local_position(index)
        if position is None:
            return None
        terms = self._overrides.get(index)
        if terms is not None:
            return list(terms)
        return [
            (offset, coefficient if _is_scalar(coefficient) else coefficient[position])
            for offset, coefficient in zip(
                self._offsets, self._coefficients, strict=True
            )
        ]

    def __add__(self, other: StencilOperator) -> StencilOperator:
        """Return the operator ``self + other``, on their shared layout.

        Regular offsets are merged, adding coefficients where both operators
        use the same offset. Overridden rows are merged the same way, row by
        row; a row overridden in only one operand is combined with the
        other's regular stencil at that row.

        Args:
            other: An operator of the same `layout`.

        Returns:
            The combined `StencilOperator`.

        Raises:
            ValueError: If ``other``'s layout differs from this operator's.
        """
        if not isinstance(other, StencilOperator):
            return NotImplemented
        if self._layout != other._layout:
            raise ValueError("can only add StencilOperators of the same layout")
        coefficients = dict(zip(self._offsets, self._coefficients, strict=True))
        for offset, coefficient in zip(
            other._offsets, other._coefficients, strict=True
        ):
            coefficients[offset] = coefficients.get(offset, 0.0) + coefficient
        offsets = tuple(coefficients)
        overrides = {}
        for index in self._overrides.keys() | other._overrides.keys():
            a, b = self._terms_at(index), other._terms_at(index)
            if a is None or b is None:  # not owned by this rank
                continue
            merged: dict[Offset, Any] = {}
            for offset, coefficient in itertools.chain(a, b):
                merged[offset] = merged.get(offset, 0.0) + coefficient
            overrides[index] = list(merged.items())
        return StencilOperator(
            self._layout,
            offsets,
            [coefficients[offset] for offset in offsets],
            overrides=overrides,
        )

    def __sub__(self, other: StencilOperator) -> StencilOperator:
        """Return ``self - other``; see `__add__`."""
        if not isinstance(other, StencilOperator):
            return NotImplemented
        return self + (-1.0) * other

    def __neg__(self) -> StencilOperator:
        """Return ``-self``."""
        return -1.0 * self

    def __mul__(self, scalar: Any) -> StencilOperator:
        """Return this operator scaled by ``scalar`` (a number)."""
        if not _is_scalar(scalar):
            return NotImplemented
        return StencilOperator(
            self._layout,
            self._offsets,
            [scalar * coefficient for coefficient in self._coefficients],
            overrides={
                index: [(offset, scalar * coefficient) for offset, coefficient in terms]
                for index, terms in self._overrides.items()
            },
        )

    __rmul__ = __mul__

    def to_petsc_mat(self) -> Any:  # pragma: no cover - needs petsc4py, see petsc.py
        """Assemble this operator's pattern as a ``PETSc.Mat``.

        Rows are this operator's `layout`'s decomposition (via
        `Layout.dmda`, whose ``stencil_width`` must cover the offsets used --
        the default does). An offset that steps outside a non-periodic axis
        is simply omitted from that row (an implicit Dirichlet zero); give it
        its own `overrides` entry for any other boundary condition.

        Collective over ``layout.comm``.

        Returns:
            An assembled ``petsc4py.PETSc.Mat`` with the layout's DMDA
            decomposition.

        Raises:
            ImportError: Without petsc4py.
        """
        from mpiarray.petsc import petsc_numbering

        layout = self._layout
        stencil_width = max(1, *layout.halo)
        da = layout.dmda(stencil_width=stencil_width)
        matrix = da.createMatrix()
        numbering = petsc_numbering(layout)
        shape = layout.shape
        periodic = layout.periodic
        bounds = layout.index_bounds

        for index in itertools.product(*(range(start, end) for start, end in bounds)):
            terms = self._terms_at(index)
            assert terms is not None  # index is within this rank's own bounds
            columns = []
            values = []
            for offset, coefficient in terms:
                neighbour = []
                for i, o, n, p in zip(index, offset, shape, periodic, strict=True):
                    j = i + o
                    if p:
                        j %= n
                    elif not 0 <= j < n:
                        neighbour = None
                        break
                    neighbour.append(j)
                if neighbour is None:
                    continue
                columns.append(int(numbering[tuple(neighbour)]))
                values.append(float(coefficient))
            matrix.setValues([int(numbering[index])], columns, values)
        matrix.assemble()
        return matrix

    def __repr__(self) -> str:
        """Return the number of offsets, the layout, and any overrides."""
        n = len(self._offsets)
        overrides = f", {len(self._overrides)} overrides" if self._overrides else ""
        return f"StencilOperator({n} offsets, layout={self._layout!r}{overrides})"
