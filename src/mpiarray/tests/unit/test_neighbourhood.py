"""wait=False halo updates and accumulation (direct exchange with every neighbour)."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import DistributedArray

size = mpa.default_comm().Get_size()
L = 2 * size + 3

CASES = [
    ((L, L - 2), (0, 1), (1, 1), (True, False), "edge"),
    ((L, L - 1, 5), (0, 1), (1, 2, 1), (True, True, False), 3.0),
    ((L + 4,), 0, (2,), True, None),
    ((L, 10), (0, 1), (2, 0), (False, False), "symmetric"),
    ((L, 6), None, (1, 1), (True, False), None),
]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(("shape", "split", "halo", "periodic", "boundary"), CASES)
def test_without_waiting_equals_waiting(
    shape, split, halo, periodic, boundary, order
) -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=shape)
    a = mpa.array(data, split=split, halo=halo, periodic=periodic)
    b = DistributedArray(
        a.layout, xp.asarray(np.asarray(xp.to_numpy(a.local_with_halos), order=order))
    )
    a.update_halos(boundary=boundary)
    pending = b.update_halos(wait=False, boundary=boundary)
    assert pending is not None
    pending.wait()
    np.testing.assert_array_equal(
        xp.to_numpy(b.local_with_halos), xp.to_numpy(a.local_with_halos)
    )

    storage = rng.normal(size=a.layout.storage_shape) * (a.layout.rank + 1)
    a.local_with_halos[...] = xp.asarray(storage)
    b.local_with_halos[...] = xp.asarray(storage)
    a.accumulate_halos()
    pending = b.accumulate_halos(wait=False)
    assert pending is not None
    pending.wait()
    np.testing.assert_allclose(
        xp.to_numpy(b.local_with_halos), xp.to_numpy(a.local_with_halos)
    )


def test_one_axis_without_waiting() -> None:
    data = np.arange(float(L * 4)).reshape(L, 4)
    a = mpa.array(data, split=0, halo=1)
    b = a.copy()
    a.update_halos(axis=0, boundary="edge")
    pending = b.update_halos(axis=0, wait=False, boundary="edge")
    assert pending is not None
    pending.wait()
    np.testing.assert_array_equal(
        xp.to_numpy(b.local_with_halos), xp.to_numpy(a.local_with_halos)
    )
    assert b.accumulate_halos(axis=0, wait=False) is not None


@pytest.mark.parametrize("wait", [True, False])
def test_kept_buffers_are_reused(wait: bool) -> None:
    """Fortran-ordered storage goes through buffers, which later exchanges reuse."""
    data = np.arange(float(L * 6)).reshape(L, 6)
    reference = mpa.array(data, split=(0, 1) if size > 1 else 0, halo=1, periodic=True)
    a = DistributedArray(
        reference.layout, xp.asfortranarray(reference.local_with_halos.copy())
    )
    for _ in range(2):
        reference.update_halos()
        pending = a.update_halos(wait=wait)
        if pending is not None:
            pending.wait()
        a.update_halos(axis=1)
        reference.update_halos(axis=1)
    np.testing.assert_array_equal(
        xp.to_numpy(a.local_with_halos), xp.to_numpy(reference.local_with_halos)
    )
