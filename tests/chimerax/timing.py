"""Time opening a store, the first image, and stepping frames, with the installed bundle.

Run (nogui measures the data path; with a display, `wait 1` also renders each frame):
  NGFF_TIMING_STORE=<store> [NGFF_TIMING_FRAMES=20] [NGFF_TIMING_START=1]
  [NGFF_TIMING_STEP=1|2|4] [NGFF_TIMING_NO_DEFER=1] \\
  chimerax [--nogui] --exit --script tests/chimerax/timing.py

NGFF_TIMING_START picks the first frame stepped to (frames not read before are cold),
and NGFF_TIMING_STEP sets the display step before stepping (1 reads level 0, 2 level 1).

NGFF_TIMING_NO_DEFER=1 opens without the bundle's deferred volume updates, to show
what scripts would cost if ChimeraX read every hidden frame while opening.
Prints one TIMING line per measurement (seconds, time.perf_counter).
"""

import os
import statistics
import sys
import time

from chimerax.core.commands import run
from chimerax.map.volume import MultiChannelSeries
from chimerax.map_series.play import Play_Series

store = os.environ["NGFF_TIMING_STORE"]
nframes = int(os.environ.get("NGFF_TIMING_FRAMES", "20"))
start = int(os.environ.get("NGFF_TIMING_START", "1"))
step = os.environ.get("NGFF_TIMING_STEP")
gui = session.ui.is_gui


def report(msg):
    # In the GUI print() goes to the ChimeraX log; write to the terminal as well.
    print(msg)
    if gui:
        sys.__stdout__.write(msg + "\n")
        sys.__stdout__.flush()


if os.environ.get("NGFF_TIMING_NO_DEFER") == "1":
    from chimerax.ngff import open as ngff_open

    class _NoDefer:
        def __init__(self, session):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ngff_open._deferred_volume_updates = _NoDefer

from chimerax.ngff import ngff  # noqa: E402

reads = []
_read_block = ngff.Image.read_block


def counted(self, *args):
    reads.append(args[:3])
    return _read_block(self, *args)


ngff.Image.read_block = counted

t0 = time.perf_counter()
models = run(session, f"open {store}")
t_open = time.perf_counter() - t0
reads_open = len(reads)
if gui:
    run(session, "wait 1")  # draws one frame: builds and renders the first images
t_first = time.perf_counter() - t0
report(f"TIMING open_command {t_open:.3f} s ({reads_open} region reads)")
report(f"TIMING first_image {t_first:.3f} s ({'rendered' if gui else 'data read, nogui'})")

m = models[0]
series = m.map_series if isinstance(m, MultiChannelSeries) else [m]
if step:
    # Only the shown frames: in scripts, `volume #1 step N` on the whole series reads
    # every frame. The player copies the step to each new frame.
    t2 = time.perf_counter()
    shown = " ".join("#" + s.maps[0].id_string for s in series)
    run(session, f"volume {shown} step {step}")
    for s in series:
        s.maps[0].matrix()
    if gui:
        run(session, "wait 1")
    report(f"TIMING set_step_{step} {time.perf_counter() - t2:.3f} s")
player = Play_Series(series, session)
steps = []
last = min(start + nframes, len(series[0].maps))
for t in range(start, last):
    t1 = time.perf_counter()
    player.change_time(t)
    for s in series:
        s.maps[t].matrix()  # the shown region of the new frame (already read in scripts)
    if gui:
        run(session, "wait 1")
    steps.append(time.perf_counter() - t1)
if steps:
    report(f"TIMING frame_step frames {start}-{last - 1} step {step or 'default'} "
          f"n={len(steps)} median {statistics.median(steps):.3f} s, min {min(steps):.3f}, "
          f"max {max(steps):.3f}, first {steps[0]:.3f}")
    report(f"TIMING frame_step_reads {len(reads) - reads_open} region reads for {len(steps)} "
          f"frames x {len(series)} channel(s)")
