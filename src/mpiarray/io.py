"""Saving and loading distributed arrays: ``.npy`` files, and HDF5 files.

`save` and `load` handle one array in an ordinary NumPy file: `numpy.load`
reads what `save` writes, and `load` reads what `numpy.save` writes. With
several ranks, every rank writes and reads only its own block, through
MPI-IO; nothing global is built.

`save_hdf5` and `load_hdf5` keep several arrays and attributes in one HDF5
file (needs h5py: ``pip install "mpiarray[hdf5]"``). With an MPI-enabled h5py
build every rank writes and reads its own block; with an ordinary build,
rank 0 writes each array after gathering it, and every rank reads its block.
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping, Sequence
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
            layout.comm.Barrier()
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


def _h5py() -> Any:
    """Import h5py, with a hint if it is missing."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            'save_hdf5 and load_hdf5 need h5py: pip install "mpiarray[hdf5]"'
        ) from error
    return h5py


def save_hdf5(
    path: PathLike,
    arrays: Mapping[str, DistributedArray],
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write several arrays, and attributes, to one HDF5 file.

    Collective. Each array becomes a dataset of its global shape (halo cells
    are not saved), and ``attrs`` become attributes of the file. With an
    MPI-enabled h5py build all ranks write their blocks into the file at
    once; with an ordinary build each array is gathered on rank 0, which
    writes the file.

    Args:
        path: The file to write; it is created or overwritten.
        arrays: The arrays, by dataset name; all on the same communicator.
        attrs: Attributes for the file: numbers, strings or small arrays.

    Raises:
        ValueError: Without arrays, or for arrays on different communicators.
        ImportError: Without h5py.
    """
    if not arrays:
        raise ValueError("save_hdf5 needs at least one array")
    layouts = [a.layout for a in arrays.values()]
    comm = layouts[0].comm
    if any(not (lay.comm is comm or bool(lay.comm == comm)) for lay in layouts):
        raise ValueError("save_hdf5 needs every array on the same communicator")
    h5py = _h5py()
    path = os.fspath(path)
    parallel = comm.Get_size() > 1 and bool(h5py.get_config().mpi)
    if parallel:  # pragma: no cover - needs an MPI-enabled h5py build
        check_collective(comm, "save_hdf5")
        with h5py.File(path, "w", driver="mpio", comm=comm) as handle:
            handle.attrs.update(dict(attrs or {}))
            for name, a in arrays.items():
                dataset = handle.create_dataset(name, shape=a.shape, dtype=a.dtype)
                with dataset.collective:
                    dataset[a.layout.global_slices()] = xp.to_numpy(a.local)
        return
    gathered = {name: a.to_numpy(root=0) for name, a in arrays.items()}
    if comm.Get_rank() == 0:
        with h5py.File(path, "w") as handle:
            handle.attrs.update(dict(attrs or {}))
            for name, data in gathered.items():
                handle.create_dataset(name, data=data)
    if comm.Get_size() > 1:  # the others wait for the file
        check_collective(comm, "save_hdf5")
        comm.Barrier()


def load_hdf5(
    path: PathLike,
    names: Sequence[str] | None = None,
    *,
    split: SplitArg = DEFAULT,
    halo: _HaloArg = DEFAULT,
    periodic: _PeriodicArg = DEFAULT,
    comm: Comm | None = None,
    process_grid: Sequence[int] | None = None,
) -> tuple[dict[str, DistributedArray], dict[str, Any]]:
    """Read datasets of an HDF5 file into distributed arrays, each rank its own block.

    Collective. The layout options apply to every dataset (each gets a layout
    for its own shape).

    Args:
        path: The file, e.g. written by `save_hdf5`.
        names: The datasets to read; default: every dataset at the top level.
        split: The split axis or axes.
        halo: The halo width; the halo cells start at zero.
        periodic: Whether each axis wraps around.
        comm: The communicator.
        process_grid: Explicit process counts per axis.

    Returns:
        The arrays by name, and the file's attributes.

    Raises:
        ImportError: Without h5py.
    """
    h5py = _h5py()
    path = os.fspath(path)
    arrays: dict[str, DistributedArray] = {}
    with h5py.File(path, "r", locking=False) as handle:
        attrs = {key: _plain(value) for key, value in handle.attrs.items()}
        if names is None:
            names = [
                key for key, item in handle.items() if isinstance(item, h5py.Dataset)
            ]
        for name in names:
            dataset = handle[name]
            layout = _make_layout(
                dataset.shape,
                None,
                split,
                0 if halo is DEFAULT else halo,
                False if periodic is DEFAULT else periodic,
                comm,
                process_grid,
            )
            block = (
                dataset[layout.global_slices()]
                if dataset.size
                else np.empty(layout.local_shape, dtype=dataset.dtype)
            )
            storage = xp.zeros(layout.storage_shape, dtype=dataset.dtype)
            storage[layout.interior] = xp.asarray(np.ascontiguousarray(block))
            arrays[name] = DistributedArray(layout, storage)
    return arrays, attrs


def _plain(value: Any) -> Any:
    """Return an HDF5 attribute as a Python value where it is a scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value
