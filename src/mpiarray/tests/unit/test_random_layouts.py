"""Randomized checks against NumPy, over many layouts and every operation.

The configurations come from a fixed seed, so every rank draws the same ones.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa

MPI = xp.mpi.get_mpi()
size = MPI.COMM_WORLD.Get_size()


def _configurations(count: int) -> list[dict]:
    rng = np.random.default_rng(20261006)
    configurations = []
    for _ in range(count):
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(n) for n in rng.integers(3 * size + 3, 3 * size + 8, ndim))
        axes = [a for a in range(ndim) if rng.random() < 0.6]
        configurations.append(
            {
                "shape": shape,
                "split": tuple(axes) if axes else None,
                "halo": tuple(int(h) for h in rng.integers(0, 3, ndim)),
                "periodic": tuple(bool(p) for p in rng.random(ndim) < 0.5),
                "seed": int(rng.integers(1 << 30)),
            }
        )
    return configurations


def _reference_halos(data: np.ndarray, layout) -> np.ndarray:
    """This rank's storage after update_halos: the global array padded axis by axis."""
    padded = data
    for axis, (h, periodic) in enumerate(
        zip(layout.halo, layout.periodic, strict=True)
    ):
        width = [(0, 0)] * data.ndim
        width[axis] = (h, h)
        padded = np.pad(padded, width, mode="wrap" if periodic else "constant")
    return padded[
        tuple(
            slice(start, end + 2 * h)
            for (start, end), h in zip(layout.index_bounds, layout.halo, strict=True)
        )
    ]


@pytest.mark.parametrize("config", _configurations(30), ids=lambda c: str(c["shape"]))
def test_random_layout_against_numpy(config: dict) -> None:
    rng = np.random.default_rng(config["seed"])
    data = rng.normal(size=config["shape"])
    other = rng.normal(size=config["shape"])
    options = {k: config[k] for k in ("split", "halo", "periodic")}
    a, b = mpa.array(data, **options), mpa.array(other, **options)

    np.testing.assert_array_equal(xp.to_numpy(a.gather()), data)
    result = 2 * a - np.sin(b)
    np.testing.assert_allclose(xp.to_numpy(result.gather()), 2 * data - np.sin(other))

    assert a.sum() == pytest.approx(data.sum())
    assert a.max() == data.max() and a.min() == data.min()
    assert a.var() == pytest.approx(data.var())
    assert a.vdot(b) == pytest.approx(np.vdot(data, other))
    axis = int(rng.integers(data.ndim))
    np.testing.assert_allclose(xp.to_numpy(a.sum(axis=axis)), data.sum(axis=axis))

    index = tuple(
        slice(
            int(rng.integers(n // 2)),
            int(rng.integers(n // 2, n + 1)),
            int(rng.integers(1, 3)),
        )
        for n in data.shape
    )
    np.testing.assert_array_equal(xp.to_numpy(a[index]), data[index])
    element = tuple(int(rng.integers(n)) for n in data.shape)
    assert a[element] == data[element]

    a.update_halos()
    np.testing.assert_array_equal(
        xp.to_numpy(a.local_with_halos), _reference_halos(data, a.layout)
    )
