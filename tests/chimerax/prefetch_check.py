"""ChimeraX checks for the v0.2 defaults and the prefetcher (no display needed).

Run:
  NGFF_PREFETCH_STORE=<store with >= 4 time points> \\
  chimerax --nogui --exit --script tests/chimerax/prefetch_check.py

Opens the store, checks the rendering defaults, lets the prefetcher run (in nogui there is
no redraw loop, so the 'new frame' trigger is fired by hand), and checks that blocks arrive
in ChimeraX's data cache bit-exact, that the histogram matches numpy, and that closing the
model stops it. Exits 1 on failure.
"""

import os
import time

import numpy as np
from chimerax.core.commands import run
from chimerax.map.volume import MultiChannelSeries
from chimerax.map_series import MapSeries

store = os.environ["NGFF_PREFETCH_STORE"]
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))


def fire(seconds, until=lambda: False):
    """Fire 'new frame' every 10 ms for up to `seconds`, or until `until()`."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and not until():
        session.triggers.activate_trigger("new frame", session.update_loop)
        time.sleep(0.01)


models = run(session, f"open {store} format ngff")
m = models[0]
series = m.map_series if isinstance(m, MultiChannelSeries) else [m]
check("opened a time series",
      all(isinstance(s, MapSeries) for s in series) and len(series[0].maps) >= 4)

v0 = series[0].maps[0]
ro = v0.rendering_options
check("maximum-intensity projection by default", ro.maximum_intensity_projection)
check("colour map on the GPU by default", ro.colormap_on_gpu)
check("3-D texture projection", ro.projection_mode == "3d", ro.projection_mode)
check("linear ramp: two image levels", len(v0.image_levels) == 2
      and v0.image_levels[0][1] == 0 and v0.image_levels[1][1] == 1, str(v0.image_levels))

pfs = getattr(session, "ngff_prefetchers", [])
check("a prefetcher is running", len(pfs) == 1 and not pfs[0].closed)
pf = pfs[0]
fire(20, until=lambda: pf.reads >= 4 * len(series) and not pf._inflight)
check("blocks reached the data cache", pf.reads >= 4 * len(series), pf.status())

# A prefetched frame is read from the cache, bit-exact against a direct read.
t = 2
v = series[0].maps[t]
before = pf.cache.used
a = v.matrix()
check("matrix() of a prefetched frame adds nothing to the cache", pf.cache.used == before)
grid = v.data.available_subsamplings[
    v.data.choose_subsampling(v.step_aligned_region(v.region)[2])[0]]
b = grid.image.read_block(grid.level, grid.t, grid.c, (0, 0, 0), a.shape[::-1], (1, 1, 1))
check("prefetched block is bit-exact", np.array_equal(a, b[:a.shape[0], :a.shape[1], :a.shape[2]]))

# The histogram comes from stats.py and agrees with numpy on the extremes.
ms = v.matrix_value_statistics()
check("histogram extremes match numpy", (ms.minimum, ms.maximum) == (a.min(), a.max())
      or (ms.minimum, ms.maximum) == (float(a.min()), float(a.max())), f"{ms.minimum} {ms.maximum}")
check("histogram counts every voxel", abs(ms.counts.sum() - a.size) <= 1e-6 * a.size)

# Showing the next frame moves the window and the series keeps playing.
series[0].show_time(1)
fire(2)
check("prefetcher follows the shown time point", pf._cur == 1, pf.status())

run(session, "close session")
check("closing the model stops the prefetcher", pf.closed and not pf.cache.data)

print(f"PREFETCH CHECK: {sum(results)} passed, {len(results) - sum(results)} failed")
if not all(results):
    os._exit(1)
