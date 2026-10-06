"""Tests for MPIARRAY_DEBUG: mismatched collectives raise instead of hanging."""

from __future__ import annotations

import textwrap

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import _mpi
from mpiarray.tests.unit._mpi_jobs import SERIAL_RUN, run_job

MPI = xp.mpi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
N = size + 3


def test_matching_collectives_pass_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPIARRAY_DEBUG", "1")
    data = np.arange(float(N))
    a = mpa.array(data, halo=1, periodic=True)
    a.update_halos()
    a.clear_halos()
    a.accumulate_halos()  # zero halos: nothing changes
    np.testing.assert_array_equal(xp.to_numpy(a.gather()), data)
    rooted = a.gather(root=0)
    assert (rooted is None) == (a.layout.rank != 0)
    assert a.sum() == data.sum() and a.var() == pytest.approx(data.var())
    assert a.norm(1) == np.abs(data).sum() and a.vdot(a) == data @ data
    assert a.get(1) == 1.0 and a[1:3].tolist() == [1.0, 2.0]
    assert a.sum(axis=0) == data.sum()
    mpa.zeros(N, split=None).allreduce_replicated()


def test_checks_are_off_by_default_and_on_one_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MPIARRAY_DEBUG", raising=False)
    assert not _mpi.debug_checks()
    _mpi.check_collective(MPI.COMM_SELF, "nothing")  # never waits
    monkeypatch.setenv("MPIARRAY_DEBUG", "yes")
    assert _mpi.debug_checks()
    _mpi.check_collective(MPI.COMM_SELF, "one rank")


_MISMATCH = """
import mpiarray as mpa
a = mpa.arange(10.0)
if a.layout.rank == 0:
    a.sum()
else:
    a.gather()
"""

_ONE_RANK_ONLY = """
import mpiarray as mpa
a = mpa.arange(10.0)
if a.layout.rank == 0:
    print(a.gather())  # the classic mistake
"""


@pytest.mark.skipif(not SERIAL_RUN, reason="starts its own MPI jobs")
@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            _MISMATCH,
            "the ranks are in different collective calls: rank 0: sum (call 1)",
        ),
        (_ONE_RANK_ONLY, "rank 0 waited 2 s in gather (collective call 1)"),
    ],
)
def test_mistakes_raise_instead_of_hanging(script: str, message: str, tmp_path) -> None:
    path = tmp_path / "mistake.py"
    path.write_text(textwrap.dedent(script))
    env = {"MPIARRAY_DEBUG": "1", "MPIARRAY_DEBUG_TIMEOUT": "2"}
    output = run_job(2, [str(path)], env=env, expect_failure=True)
    assert message in output
