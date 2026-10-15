"""Save one time point as a PNG, to compare how two versions of the bundle look.

Run with a display (see run_gui.sh):
  NGFF_LOOK_STORE=<store> NGFF_LOOK_OUT=<file.png> [NGFF_LOOK_T=30] [NGFF_LOOK_STEP=2] \\
  tests/chimerax/run_gui.sh --exit --script tests/chimerax/look.py
Same camera for every version: the view right after open, looking down z.
"""

import os

from chimerax.core.commands import run
from chimerax.map.volume import MultiChannelSeries
from chimerax.map_series.play import Play_Series

store = os.environ["NGFF_LOOK_STORE"]
out = os.environ["NGFF_LOOK_OUT"]
t = int(os.environ.get("NGFF_LOOK_T", "30"))
step = os.environ.get("NGFF_LOOK_STEP", "2")

run(session, "windowsize 900 900")
run(session, "set bgColor black")
m = run(session, f"open {store} format ngff")[0]
series = m.map_series if isinstance(m, MultiChannelSeries) else [m]
run(session, f"volume {' '.join('#' + s.maps[0].id_string for s in series)} step {step}")
run(session, "view orient")
run(session, "turn x 180")
run(session, "view")
Play_Series(series, session).change_time(t)
for _ in range(3):
    run(session, "wait 1")
run(session, f"save {out} width 1200 supersample 2")
print(f"LOOK saved {out}")
os._exit(0)
