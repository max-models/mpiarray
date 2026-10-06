"""Tests for mpa.migrate: rows of arrays sent to destination ranks."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

import mpiarray as mpa

comm = mpa.default_comm()
rank, size = comm.Get_rank(), comm.Get_size()


def test_rows_reach_their_destinations_in_order() -> None:
    rng = np.random.default_rng(rank)
    count = 20 + 3 * rank
    ids = rank * 1000 + np.arange(count)
    destinations = rng.integers(0, size, count)
    positions = np.stack([ids, -ids], axis=1).astype(float)
    flags = (ids % 2).astype(np.int8)
    got_ids, got_positions, got_flags = mpa.migrate(destinations, ids, positions, flags)
    sent = comm.allgather((ids, destinations))
    expected = np.concatenate([i[d == rank] for i, d in sent])
    np.testing.assert_array_equal(xp.to_numpy(got_ids), expected)
    np.testing.assert_array_equal(
        xp.to_numpy(got_positions), np.stack([expected, -expected], 1)
    )
    np.testing.assert_array_equal(xp.to_numpy(got_flags), expected % 2)


def test_owners_then_migrate() -> None:
    layout = mpa.Layout((4 * size, 3))
    cells = np.column_stack([np.arange(4 * size), np.zeros(4 * size, dtype=int)])
    (moved,) = mpa.migrate(layout.owners(cells), cells)
    assert (layout.owners(xp.to_numpy(moved)) == rank).all()
    assert len(moved) == 4 * size  # every rank sent its 4 cells to this one


def test_empty_and_bad_inputs() -> None:
    (empty,) = mpa.migrate(np.zeros(0, dtype=int), np.zeros((0, 2)))
    assert empty.shape == (0, 2)
    with pytest.raises(ValueError, match="between 0 and"):
        mpa.migrate(np.array([size]), np.zeros(1))
    with pytest.raises(ValueError, match="one row per destination"):
        mpa.migrate(np.zeros(2, dtype=int), np.zeros(3))
    with pytest.raises(ValueError, match="1-D array of ranks"):
        mpa.migrate(np.zeros((2, 1), dtype=int), np.zeros(2))
