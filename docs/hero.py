"""The figure on the docs home page: a wave crossing the blocks of an array split over four ranks.

A 28 x 16 :mod:`mpiarray` array, split over a 2 x 2 grid of MPI ranks with one layer of halo cells
and periodic in both directions, carries the wave equation: every step ``update_halos()`` fills
the halo cells from the neighbouring ranks, and the leapfrog stencil runs on each rank's own
block. The cells of the gathered snapshots (shading: the amplitude), the rank borders and the
halo cells, flashing at every exchange, are drawn with ``plasma_plots.hero`` into
``docs/src/assets/hero.svg``; the animation plays the run forward and back. Rerun with::

    mpiexec -n 4 python docs/hero.py   # or: make docs-hero
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

import mpiarray as mpa

shape, courant, steps, every = (28, 16), 0.5, 55, 5
x0, x1, y0, y1 = 3.2, 8.0, 0.0, 3.0  # where the array sits in the picture
kwargs = {"split": (0, 1), "halo": 1, "periodic": (True, True)}


def pulse(i, j):
    """A Gaussian bump, as a function of the global cell indices."""
    return np.exp(-((i - 9.0) ** 2 + (j - 6.5) ** 2) / 4.0)


u = mpa.fromfunction(pulse, shape, **kwargs)
previous = u.copy()
frames = [u.gather()]
for step in range(1, steps + 1):
    u.update_halos()  # halo cells <- the neighbours' boundary cells
    h = u.local_with_halos
    laplacian = (
        h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2] - 4 * h[1:-1, 1:-1]
    )
    new = 2 * u.local - previous.local + courant**2 * laplacian
    previous.local[...] = u.local
    u.local[...] = new
    if step % every == 0:
        frames.append(u.gather())  # collective: every rank calls it

if u.layout.rank == 0:
    frames = np.array(frames)
    frames = np.concatenate(
        [frames, frames[-2:0:-1]]
    )  # forward and back: a seamless loop
    shade = np.clip(0.06 + 1.6 * np.abs(frames), 0.06, 0.95)
    dx, dy = (x1 - x0) / shape[0], (y1 - y0) / shape[1]
    duration = 12.0
    fig = HeroFigure(xlim=(0, 8), ylim=(0, 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = fig.rect(
                x0 + i * dx,
                y0 + j * dy,
                x0 + (i + 1) * dx,
                y0 + (j + 1) * dy,
                tone="accent",
                opacity=shade[0, i, j],
                outline="bg",
            )
            if np.ptp(np.round(shade[:, i, j], 1)) > 0:
                cell.animate("fill-opacity", np.round(shade[:, i, j], 1), duration)
    cuts = u.layout.bounds  # where the blocks start along each axis, and the length
    flash = [0.9, 0.15, 0.15, 0.9]
    for c in cuts[0][1:-1]:
        for side in (c - 1, c):  # the boundary cells that become the neighbours' halos
            fig.rect(
                x0 + side * dx,
                y0,
                x0 + (side + 1) * dx,
                y1,
                tone="high",
                opacity=0.0,
                outline="high",
                title="halo cells",
            ).animate("stroke-opacity", flash, 1.5)
        fig.line([x0 + c * dx] * 2, [y0, y1], tone="high", width=2.5)
    for c in cuts[1][1:-1]:
        for side in (c - 1, c):
            fig.rect(
                x0,
                y0 + side * dy,
                x1,
                y0 + (side + 1) * dy,
                tone="high",
                opacity=0.0,
                outline="high",
                title="halo cells",
            ).animate("stroke-opacity", flash, 1.5)
        fig.line([x0, x1], [y0 + c * dy] * 2, tone="high", width=2.5)
    fig.save(Path(__file__).parent / "src" / "assets" / "hero.svg")
