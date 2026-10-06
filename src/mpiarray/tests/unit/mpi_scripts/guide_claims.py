"""Check the numbers quoted in docs/src/content/docs/guides/*.md.

Run by test_examples.py on 4 ranks.
"""

import numpy as np

import mpiarray as mpa

MPI = mpa.layout.MPI
null = MPI.PROC_NULL
comm = MPI.COMM_WORLD
assert comm.size == 4, "run on 4 ranks"
rank = comm.rank

# layouts.md: process grids
assert mpa.process_grid(4, 2, (0, 1)) == (2, 2)
assert mpa.process_grid(6, 2, (0, 1)) == (2, 3)
assert mpa.process_grid(12, 2, (0, 1)) == (3, 4)
assert mpa.process_grid(12, 3, (0, 1, 2)) == (2, 2, 3)
assert mpa.process_grid(15, 3, (0, 1, 2)) == (1, 3, 5)
assert mpa.process_grid(7, 2, (0, 1)) == (1, 7)

# layouts.md: neighbour table and index split on 4 ranks
a = mpa.zeros((10, 7), split=(0, 1), halo=1, periodic=(True, False))
table = {
    0: ((0, 0), ((2, 2), (null, 1))),
    1: ((0, 1), ((3, 3), (0, null))),
    2: ((1, 0), ((0, 0), (null, 3))),
    3: ((1, 1), ((1, 1), (2, null))),
}
assert (a.layout.process_coord, a.layout.neighbours) == table[rank], rank
rows = [(0, 5), (5, 10)][a.layout.process_coord[0]]
cols = [(0, 4), (4, 7)][a.layout.process_coord[1]]
assert a.layout.index_bounds == (rows, cols)
assert a.layout.owner((6, 2)) == 2
assert a.layout.owners(np.array([[6, 2], [0, 0], [9, 6]])).tolist() == [2, 0, 3]
assert a.layout.owners(np.array([[-1, 9]]), clip=True).tolist() == [1]

# halo-exchange.md: a halo wider than the smallest block raises
try:
    mpa.zeros(10, halo=3)
    raise AssertionError("expected a ValueError")
except ValueError as error:
    assert "wider than the smallest block" in str(error)
assert mpa.chunk_bounds(7, 2, 1) == (4, 7)

# halo-exchange.md: 8-element periodic axis, block = rank + 1, one halo cell
u = mpa.zeros(8, halo=1, periodic=True)
u.local[...] = rank + 1
u.update_halos()
expected = {0: [4, 1, 1, 2], 1: [1, 2, 2, 3], 2: [2, 3, 3, 4], 3: [3, 4, 4, 1]}
assert u.local_with_halos.tolist() == expected[rank], u.local_with_halos

# halo-exchange.md: the Laplacian example
n = 64
h = 2 * np.pi / n
u = mpa.fromfunction(lambda i: np.sin(i * h), (n,), halo=1, periodic=True)
lap = mpa.zeros_like(u)
u.update_halos()
v = u.local_with_halos
lap.local[...] = (v[2:] - 2 * v[1:-1] + v[:-2]) / h**2
assert (lap + u).norm(np.inf) < 1e-2

# distributed-arrays.md: array splits; repr is local
b = mpa.array([1, 2, 3, 4])
assert b.local.tolist() == [[1], [2], [3], [4]][rank]
assert "rank" in repr(b)
if rank == 0:
    print("all guide claims hold")
