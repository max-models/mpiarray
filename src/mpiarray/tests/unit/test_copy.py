"""Tests for CopyPlan: boxes copied between arrays on different communicators.

Run serially and under ``mpiexec -n N``: the groups of ranks the arrays live
on are made from the ranks there are.
"""

from __future__ import annotations

import cunumpy as xp
import maybempi
import numpy as np
import pytest

import mpiarray as mpa
from mpiarray import CopyPlan, Layout, copy_boxes

MPI = maybempi.get_mpi()
WORLD = MPI.COMM_WORLD
SIZE = WORLD.Get_size()
RANK = WORLD.Get_rank()

#: Groups of ranks: the first half, the second half (they share a rank for an
#: odd count), the last rank alone, and every rank.
GROUPS = {
    "first": tuple(range((SIZE + 1) // 2)),
    "second": tuple(range(SIZE // 2, SIZE)),
    "last": (SIZE - 1,),
    "all": tuple(range(SIZE)),
}

SHAPE = (2 * SIZE + 5, SIZE + 4)


def group(name: str):
    """The communicator of a group on its ranks, ``None`` elsewhere (collective)."""
    ranks = GROUPS[name]
    member = RANK in ranks
    comm = WORLD.Split(0 if member else MPI.UNDEFINED, RANK)
    return comm if member else None


def pattern(offset: float, shape: tuple[int, ...]) -> np.ndarray:
    """Distinct values over a global index range."""
    return offset + np.arange(float(np.prod(shape))).reshape(shape)


def make(comm, data: np.ndarray, **layout) -> mpa.DistributedArray | None:
    """``data`` as an array on ``comm``; ``None`` on ranks outside it."""
    if comm is None:
        return None
    return mpa.array(data, layout=Layout(data.shape, comm=comm, **layout))


def fill_storage(array: mpa.DistributedArray, value) -> np.ndarray:
    """Fill the whole storage, halos included, by global index; return the indices."""
    layout = array.layout
    starts = [
        start - h
        for (start, _), h in zip(layout.index_bounds, layout.halo, strict=True)
    ]
    if not layout.distributed:
        starts = [-h for h in layout.halo]
    index = np.indices(layout.storage_shape) + np.reshape(starts, (-1, 1, 1))
    array.local_with_halos[...] = xp.asarray(value(index))
    return index


def owned(array: mpa.DistributedArray, index: np.ndarray) -> np.ndarray:
    """The storage cells this rank writes: its block, and the halo at the walls."""
    layout = array.layout
    mask = np.ones(index.shape[1:], dtype=bool)
    for axis, ((start, end), n) in enumerate(
        zip(layout.index_bounds, layout.shape, strict=True)
    ):
        if not layout.distributed:
            start, end = 0, n
        g = index[axis]
        mask &= (
            ((g >= start) & (g < end))
            | ((start == 0) & (g < 0))
            | ((end == n) & (g >= n))
        )
    return mask


def target_value(index: np.ndarray) -> np.ndarray:
    """What the target holds before the copy: a negative pattern, halos included."""
    return -1.0 - 100.0 * index[0] - index[1]


SOURCES = [{"split": 0}, {"split": 1, "halo": 1}, {"split": None}]
TARGETS = [
    {"split": 0, "halo": 2},
    {"split": (0, 1), "halo": 1},
    {"split": None, "halo": 1},
]


@pytest.mark.parametrize("source_layout", SOURCES, ids=str)
@pytest.mark.parametrize("target_layout", TARGETS, ids=str)
@pytest.mark.parametrize(
    ("source_group", "target_group"),
    [("first", "second"), ("last", "all"), ("all", "last")],
)
def test_boxes_go_from_one_group_of_ranks_to_another(
    source_layout, target_layout, source_group, target_group
) -> None:
    data = pattern(1000.0, SHAPE)
    source = make(group(source_group), data, **source_layout)
    target_comm = group(target_group)
    target = make(target_comm, np.zeros(SHAPE), **target_layout)
    n, m = SHAPE
    # into the middle, into the halo of the lower x wall, and of the upper y wall
    copies = [
        (((1, 4), (0, 3)), ((2, 5), (1, 4))),
        (((0, 1), (0, m)), ((-1, 0), (0, m))),  # a row into the lower x wall's halo
        (((3, 5), (0, 1)), ((3, 5), (m, m + 1))),  # a column into the upper y wall's
    ]
    index = fill_storage(target, target_value) if target is not None else None

    plan = CopyPlan([(source, s, target, t) for s, t in copies], comm=WORLD)
    plan.execute()

    assert plan.num_copies == 3
    if target is None:
        return
    expected = target_value(index)
    mine = owned(target, index)
    for source_box, target_box in copies:
        inside = np.ones(mine.shape, dtype=bool)
        moved = []
        for axis, ((s0, _), (t0, t1)) in enumerate(
            zip(source_box, target_box, strict=True)
        ):
            inside &= (index[axis] >= t0) & (index[axis] < t1)
            moved.append(index[axis] - t0 + s0)
        hit = inside & mine
        expected[hit] = data[moved[0][hit], moved[1][hit]]
    np.testing.assert_array_equal(xp.to_numpy(target.local_with_halos), expected)


def test_a_wall_halo_filled_by_the_plan_reaches_every_rank_with_update_halos() -> None:
    """The plan writes the wall halo on the ranks at the wall; update_halos the rest."""
    comm = group("all")
    data = pattern(0.0, SHAPE)
    n, m = SHAPE
    target = make(comm, np.zeros(SHAPE), split=(0, 1), halo=1)
    copy_boxes(
        [(make(comm, data, split=0), ((0, n), (0, 1)), target, ((0, n), (-1, 0)))],
        comm=WORLD,
    )
    target.update_halos()

    layout = target.layout
    (i0, i1), (j0, j1) = layout.index_bounds
    storage = xp.to_numpy(target.local_with_halos)
    if j0 == 0:
        # the column under the wall, corners at seams included: the source column
        rows = np.arange(i0 - 1, i1 + 1)
        inside = (rows >= 0) & (rows < n)
        np.testing.assert_array_equal(storage[1:-1, 0], data[i0:i1, 0])
        np.testing.assert_array_equal(storage[inside, 0], data[rows[inside], 0])


def test_an_array_copies_into_its_own_halo_as_a_periodic_axis_would() -> None:
    comm = group("all")
    n, m = SHAPE
    array = make(comm, pattern(0.0, SHAPE), split=0, halo=1)
    data = pattern(0.0, SHAPE)
    copy_boxes(
        [
            (array, ((0, n), (m - 1, m)), array, ((0, n), (-1, 0))),
            (array, ((0, n), (0, 1)), array, ((0, n), (m, m + 1))),
        ],
        comm=WORLD,
    )
    (i0, i1), _ = array.layout.index_bounds
    storage = xp.to_numpy(array.local_with_halos)
    np.testing.assert_array_equal(storage[1:-1, 0], data[i0:i1, m - 1])
    np.testing.assert_array_equal(storage[1:-1, -1], data[i0:i1, 0])
    np.testing.assert_array_equal(storage[1:-1, 1:-1], data[i0:i1])


def test_where_target_boxes_overlap_the_last_copy_wins() -> None:
    comm = group("all")
    first = make(comm, pattern(0.0, SHAPE), split=0)
    second = make(comm, pattern(500.0, SHAPE), split=1)
    target = make(group("second"), np.zeros(SHAPE), split=None)
    box = ((0, 3), (0, 3))
    plan = CopyPlan([(first, box, target, box), (second, box, target, box)], comm=WORLD)
    plan.execute()

    if target is not None:
        np.testing.assert_array_equal(
            xp.to_numpy(target.local)[:3, :3], pattern(500.0, SHAPE)[:3, :3]
        )
    assert plan.num_sent >= 0


def test_an_empty_plan_does_nothing() -> None:
    plan = CopyPlan([], comm=WORLD)
    plan.execute()
    assert plan.num_copies == 0 and plan.num_sent == 0


def test_bad_boxes_raise_on_every_rank() -> None:
    comm = group("last")
    array = make(comm, pattern(0.0, SHAPE), halo=1)
    n, m = SHAPE
    bad = {
        "differ in shape": (((0, 2), (0, 2)), ((0, 3), (0, 2))),
        "is empty": (((1, 1), (0, 2)), ((1, 1), (0, 2))),
        "not inside the array's shape": (((0, n + 1), (0, 1)), ((0, n + 1), (0, 1))),
        "and its halo": (((0, 2), (0, 1)), ((0, 2), (-2, -1))),
    }
    for message, (source_box, target_box) in bad.items():
        with pytest.raises(ValueError, match=message):
            CopyPlan([(array, source_box, array, target_box)], comm=WORLD)


def test_arrays_of_different_dtypes_are_refused() -> None:
    comm = group("all")
    floats = make(comm, pattern(0.0, SHAPE))
    integers = make(comm, np.zeros(SHAPE, dtype=np.int64))
    with pytest.raises(ValueError, match="differ in dtype"):
        CopyPlan([(floats, ((0, 1), (0, 1)), integers, ((0, 1), (0, 1)))], comm=WORLD)


@pytest.mark.skipif(SIZE == 1, reason="needs ranks that disagree")
def test_ranks_with_different_copies_are_refused() -> None:
    comm = group("all")
    array = make(comm, pattern(0.0, SHAPE))
    copies = [] if RANK == 0 else [(array, ((0, 1), (0, 1)), array, ((1, 2), (0, 1)))]
    with pytest.raises(ValueError, match="different numbers of copies"):
        CopyPlan(copies, comm=WORLD)
