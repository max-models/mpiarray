---
title: xarray
description: Gathering a distributed array as a labeled xarray.DataArray, and marking rank boundaries on a plot.
sidebar:
  order: 7
---

`a.to_xarray(dims, ...)` gathers an array (like `gather`/`to_numpy`) and labels the result as
an `xarray.DataArray`, for plotting or further analysis with xarray-based tools. It needs
xarray, which is not a dependency of mpiarray; install it separately:

```bash
pip install xarray  # or: pip install mpiarray[xarray]
```

xarray is imported on first use only; `import mpiarray` works without it.

## A gathered, labeled array

mpiarray has no notion of what an axis means (unlike a domain-specific field object, which
might hardcode its axes as `x`/`y`/`z`), so the dimension names and coordinate values are
given at the call site:

```python
import numpy as np
import mpiarray as mpa

x = np.linspace(0.0, 1.0, 64)
y = np.linspace(0.0, 1.0, 48)
rho = mpa.zeros((64, 48), split=0)
data = rho.to_xarray(("x", "y"), coords={"x": x, "y": y}, name="rho")
# also: mpa.to_xarray(rho, ("x", "y"), coords={"x": x, "y": y}, name="rho")
```

This is collective, like `gather`: call it on every rank. `root=` gathers to one rank only
(`None`, the default: every rank), which matters when handing the result to a plotting
library whose functions only draw on rank 0, such as plasma-plots:

```python
data = rho.to_xarray(("x", "y"), coords={"x": x, "y": y}, root=0)  # every rank calls this
# ... then, only on rank 0 (plasma-plots does this for you):
# plot_slice(data)
```

Gather *before* calling into a rank-0-only plotting function, not inside it: those functions
skip their body entirely on the other ranks, so a `gather`/`to_xarray` call made only inside
one would hang, waiting for ranks that never call it.

## Marking rank boundaries on a plot

`Layout.boundary_values(axis, coordinate)` turns a layout's block cuts (`Layout.bounds`, index
positions) into the coordinate values where one rank's block ends and the next begins, given
the axis's own coordinate array:

```python
>>> rho.layout.bounds[0]  # index cuts, under mpiexec -n 4
(0, 16, 32, 48, 64)
>>> rho.layout.boundary_values(0, x)  # the same cuts, in x's own values
[0.25396825396825395, 0.5079365079365079, 0.7619047619047619]
```

[plasma-plots](https://github.com/max-models/plasma-plots)'s `plot_slice` has a matching
`"rank_boundaries"` overlay (plasma-plots 0.1.3 and later) that draws these as dashed lines,
so they can share an axis with ordinary `"coordinate_lines"` without colliding:

```python
from plasma_plots.plotting import plot_slice

plot_slice(
    data,
    overlays={"rank_boundaries": {"x": rho.layout.boundary_values(0, x)}},
)
```
