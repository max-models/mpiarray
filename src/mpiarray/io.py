"""Saving and loading distributed arrays as ``.npy`` files, in parallel.

The files are ordinary NumPy files: `numpy.load` reads what `save` writes,
and `load` reads what `numpy.save` writes. With several ranks, every rank
writes and reads only its own block, through MPI-IO; nothing global is
built. Without, NumPy does the work.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from typing import Any, cast

import cunumpy as xp
import numpy as np

from mpiarray._mpi import MPI, Comm, check_collective
from mpiarray.creation import _HaloArg, _make_layout, _PeriodicArg, array
from mpiarray.distributed_array import DistributedArray, _subarray
from mpiarray.layout import DEFAULT, Layout, SplitArg

PathLike = str | os.PathLike[str]


def _header(shape: tuple[int, ...], dtype: np.dtype) -> bytes:
    """Return the ``.npy`` header (magic string included) of a C-ordered array."""
    header = io.BytesIO()
    fields = {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": shape,
    }
    try:
        np.lib.format.write_array_header_1_0(header, fields)
    except ValueError:  # a header longer than version 1.0 allows
        header = io.BytesIO()
        np.lib.format.write_array_header_2_0(header, fields)
    return header.getvalue()


def _block_view(layout: Layout, dtype: np.dtype) -> tuple[Any, Any]:
    """Return the MPI element type and the file type of this rank's block."""
    from mpi4py.util import dtlib

    starts = tuple(start for start, _ in layout.index_bounds)
    filetype = _subarray(layout.shape, starts, layout.local_shape, np.dtype(dtype).str)
    return dtlib.from_numpy_dtype(np.dtype(dtype)), filetype


def save(path: PathLike, a: DistributedArray) -> None:
    """Write ``a`` to a ``.npy`` file, each rank writing its own block.

    Collective. Halo cells are not saved.

    Args:
        path: The file to write; it is created or overwritten.
        a: The array.
    """
    layout = a.layout
    path = os.fspath(path)
    if not layout.distributed:
        if layout.rank == 0:
            np.save(path, xp.to_numpy(a.local))
        if layout.size > 1:  # replicated: the others wait for the file
            check_collective(layout.comm, "save")
            cast("MPI.Comm", layout.comm).Barrier()
        return

    comm = cast("MPI.Intracomm", layout.comm)
    check_collective(comm, "save")
    header = _header(a.shape, a.dtype)
    handle = MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    try:
        handle.Set_size(0)
        if layout.rank == 0:
            handle.Write_at(0, header)
        if a.size:
            etype, filetype = _block_view(layout, a.dtype)
            handle.Set_view(len(header), etype, filetype)
            handle.Write_all(np.ascontiguousarray(xp.to_numpy(a.local)))
    finally:
        handle.Close()


def load(
    path: PathLike,
    *,
    split: SplitArg = DEFAULT,
    halo: _HaloArg = DEFAULT,
    periodic: _PeriodicArg = DEFAULT,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
    layout: Layout | None = None,
) -> DistributedArray:
    """Read a ``.npy`` file into a distributed array, each rank reading its own block.

    Collective. The layout options are those of `mpiarray.array`; the shape
    and dtype come from the file. Fortran-ordered files are read whole and
    then split.

    Args:
        path: The file, as written by `save` or `numpy.save`.
        split: The split axis or axes.
        halo: The halo width; the halo cells start at zero.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.
        layout: An existing layout, whose shape must be the file's.

    Returns:
        The array.
    """
    path = os.fspath(path)
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        offset = handle.tell()
    options: dict[str, Any] = {
        "split": split,
        "halo": halo,
        "periodic": periodic,
        "comm": comm,
        "process_grid": process_grid,
        "layout": layout,
    }
    if fortran_order:
        return array(np.load(path), **options)
    layout = _make_layout(
        shape,
        layout,
        split,
        0 if halo is DEFAULT else halo,
        False if periodic is DEFAULT else periodic,
        comm,
        process_grid,
    )
    if not layout.distributed:
        block = np.load(path, mmap_mode="r")[layout.global_slices()]
    else:
        mpi_comm = cast("MPI.Intracomm", layout.comm)
        check_collective(mpi_comm, "load")
        block = np.empty(layout.local_shape, dtype=dtype)
        handle = MPI.File.Open(mpi_comm, path, MPI.MODE_RDONLY)
        try:
            if block.size:
                etype, filetype = _block_view(layout, dtype)
                handle.Set_view(offset, etype, filetype)
                handle.Read_all(block)
        finally:
            handle.Close()
    storage = xp.zeros(layout.storage_shape, dtype=dtype)
    storage[layout.interior] = xp.asarray(np.ascontiguousarray(block))
    return DistributedArray(layout, storage)
