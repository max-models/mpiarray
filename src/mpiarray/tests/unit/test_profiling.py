"""Tests for mpiarray's scope-profiler regions: off unless the "mpiarray" tag is enabled."""

from __future__ import annotations

import maybempi
import pytest
import scope_profiler as sp

import mpiarray as mpa

MPI = maybempi.get_mpi()
size = MPI.COMM_WORLD.Get_size()
N = size + 3


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch: pytest.MonkeyPatch):
    """Start and end every test without profiling state, which is global."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _regions_of_a_halo_update_and_a_sum(**setup_options) -> set[str]:
    sp.setup(output=None, **setup_options)
    a = mpa.arange(float(N), halo=1, periodic=True)
    a.update_halos()
    assert a.sum() == sum(range(N))
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default() -> None:
    regions = _regions_of_a_halo_update_and_a_sum()
    assert not {name for name in regions if name.startswith("mpiarray.")}


def test_enable_tags_records_the_regions() -> None:
    regions = _regions_of_a_halo_update_and_a_sum(enable_tags=["mpiarray"])
    assert {"mpiarray.update_halos", "mpiarray.reduce"} <= regions


def test_a_region_calling_itself_is_recorded_once() -> None:
    """Unsplit ``update_halos(wait=False)`` and 1-d ``var(axis=0)`` record one call each."""
    sp.setup(output=None, enable_tags=["mpiarray"])
    a = mpa.arange(float(N), halo=1, split=None)
    update = a.update_halos(wait=False)
    assert update is not None
    update.wait()
    a.var(axis=0)
    results = sp.finalize(verbose=False, return_results=True)
    for name in ("mpiarray.update_halos", "mpiarray.var"):
        assert set(results.get_region(name).num_calls_per_rank().values()) == {1}
