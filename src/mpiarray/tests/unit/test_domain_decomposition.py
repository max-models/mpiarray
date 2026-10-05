"""Tests for domain decomposition."""

from __future__ import annotations

import cunumpy as xp
import pytest

from mpiarray import DomainDecomposition

MPI = xp.mpi.get_mpi()

size = MPI.COMM_WORLD.Get_size()


# mpirun -n 6 pytest tests/distributed/test_domain_decomposition.py
@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("decompose", [[True, True, True]])
def test_domain_decomposition_mpi6(
    decompose: list[bool],
    verbose: bool = False,
) -> None:
    """Verify domain decomposition on six MPI ranks."""
    comm = MPI.COMM_WORLD

    # Check the processor size in each direction: 1*2*3 = 6
    ddcomp = DomainDecomposition(comm=comm, decompose=decompose)
    assert ddcomp.proc_sizes == [2, 1, 3]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[0, 1, 2])
    assert ddcomp.proc_sizes == [3, 2, 1]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[0, 2, 1])
    assert ddcomp.proc_sizes == [3, 1, 2]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[1, 0, 2])
    assert ddcomp.proc_sizes == [2, 3, 1]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[1, 2, 0])
    assert ddcomp.proc_sizes == [2, 1, 3]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[2, 0, 1])
    assert ddcomp.proc_sizes == [1, 3, 2]

    ddcomp = DomainDecomposition(comm=comm, decompose=decompose, dim_order=[2, 1, 0])
    assert ddcomp.proc_sizes == [1, 2, 3]


@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("decompose", [True])
@pytest.mark.parametrize("periodic", [True, False])
def test_domain_decomposition_neighbours_1d(
    decompose: bool,
    periodic: bool,
    verbose: bool = False,
) -> None:
    """Verify one-dimensional domain-decomposition neighbors."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    ddcomp = DomainDecomposition(comm=comm, decompose=[decompose], periodic=[periodic])

    if periodic:
        # Check wrapping in periodic case
        if ddcomp.mpi_rank == 0:
            assert ddcomp.neighbour_ranks[0][0] == size - 1
        elif ddcomp.mpi_rank == size - 1:
            assert ddcomp.neighbour_ranks[0][1] == 0
        else:
            assert ddcomp.neighbour_ranks[0][0] == ddcomp.mpi_rank - 1
            assert ddcomp.neighbour_ranks[0][1] == ddcomp.mpi_rank + 1
    else:
        # Check that neighbours of the edge ranks are null (not periodic)
        if ddcomp.mpi_rank == 0:
            assert ddcomp.neighbour_ranks[0][0] == MPI.PROC_NULL
        elif ddcomp.mpi_rank == size - 1:
            assert ddcomp.neighbour_ranks[0][1] == MPI.PROC_NULL
        else:
            assert ddcomp.neighbour_ranks[0][0] == ddcomp.mpi_rank - 1
            assert ddcomp.neighbour_ranks[0][1] == ddcomp.mpi_rank + 1


@pytest.mark.skipif(size != 6, reason="this test must run on exactly 6 MPI ranks")
@pytest.mark.parametrize("decompose", [True])
@pytest.mark.parametrize("periodic", [True, False])
def test_split_array(
    decompose: bool,
    periodic: bool,
    verbose: bool = False,
) -> None:
    """Verify process chunk splitting."""
    comm = MPI.COMM_WORLD

    ddcomp = DomainDecomposition(comm=comm, decompose=[decompose], periodic=[periodic])

    if comm.Get_rank() == 0:
        print(ddcomp._split_array.__doc__)
    local_shape = ddcomp._split_array(100, num_procs=1)

    print(f"{local_shape =}")


if __name__ == "__main__":
    # test_domain_decomposition_mpi6(decompose=[True, True, True])
    # test_domain_decomposition_neighbours_1d(decompose=True, periodic=False)
    # test_domain_decomposition_neighbours_1d(decompose=True, periodic=True)
    test_split_array(decompose=True, periodic=True)


@pytest.mark.parametrize("periodic", [(True, False), (False, False)])
def test_undecomposed_layout_is_replicated_on_every_rank(
    periodic: tuple[bool, bool],
) -> None:
    """Without decomposed axes every rank sits at process coordinate 0."""
    comm = MPI.COMM_WORLD
    ddcomp = DomainDecomposition(comm=comm, decompose=[False, False], periodic=periodic)

    assert ddcomp.replicated == (size > 1)
    assert ddcomp.proc_sizes == [1, 1]
    assert list(ddcomp.proc_coord) == [0, 0]
    assert all(list(ddcomp.get_proc_coord(rank)) == [0, 0] for rank in range(size))
    assert ddcomp.get_index_bounds((5, 3)) == [(0, 5), (0, 3)]
    rank = comm.Get_rank()
    for is_periodic, neighbours in zip(periodic, ddcomp.neighbour_ranks, strict=True):
        if is_periodic:
            assert neighbours == (rank, rank)
        else:
            assert neighbours == (MPI.PROC_NULL, MPI.PROC_NULL)


def test_layout_without_communicator() -> None:
    """``comm=None`` is a single-rank layout with periodic self-neighbours."""
    layout = DomainDecomposition(
        comm=None, decompose=[True, False], periodic=(True, False)
    )
    assert layout.get_index_bounds((7, 5)) == [(0, 7), (0, 5)]
    assert layout.neighbour_ranks == [(0, 0), (MPI.PROC_NULL, MPI.PROC_NULL)]
