"""GUI benchmark: how long a timepoint takes to appear, cold and warm.

Run on a real GL context (see run_gui.sh):
  NGFF_BENCH_STORE=<store> [NGFF_BENCH_STEP=1] [NGFF_BENCH_FRAMES=12] [NGFF_BENCH_START=1]
  [NGFF_BENCH_SCENARIO=step|scrub|revisit] [NGFF_BENCH_IDLE=<seconds idle after open>]
  [NGFF_BENCH_PROFILE=1] [NGFF_BENCH_GPU=1] \\
  tests/chimerax/run_gui.sh --exit --script tests/chimerax/bench.py

Opens the store, sets the display step, then changes the time point as the series player
does and waits for the frame to be drawn. Per frame it prints the wall time to the drawn
frame, the time spent reading data, filling textures and uploading them, and the GPU and
process memory. Lines start with BENCH.
"""

import os
import random
import statistics
import subprocess
import sys
import threading
import time

from chimerax.core.commands import run
from chimerax.map.volume import MultiChannelSeries
from chimerax.map_series.play import Play_Series

store = os.environ["NGFF_BENCH_STORE"]
step = os.environ.get("NGFF_BENCH_STEP", "1")
nframes = int(os.environ.get("NGFF_BENCH_FRAMES", "12"))
start = int(os.environ.get("NGFF_BENCH_START", "1"))
scenario = os.environ.get("NGFF_BENCH_SCENARIO", "step")
idle = float(os.environ.get("NGFF_BENCH_IDLE", "0"))
gpu = os.environ.get("NGFF_BENCH_GPU", "1")
profile = os.environ.get("NGFF_BENCH_PROFILE") == "1"
profile_at = int(os.environ.get("NGFF_BENCH_PROFILE_AT", "0"))


def _excepthook_report():
    import traceback
    report("BENCH ERROR\n" + traceback.format_exc())


def report(msg):
    print(msg)
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def draw_frame():
    """Draw one frame now and wait for the GPU to finish it."""
    session.ui.processEvents()
    f0 = session.main_view.frame_number
    session.update_loop.draw_new_frame()
    render.make_current()
    GL.glFinish()
    return session.main_view.frame_number - f0


def _pf():
    ps = getattr(session, "ngff_prefetchers", [])
    return f"{len(ps[0]._inflight)}/{len(ps[0]._todo)}/{ps[0].built}" if ps else "-"


def gpu_mb():
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.strip())
    except Exception:
        return -1


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2**20
    return -1.0


# Time spent in the stages of a new frame, accumulated by wrapping the functions.
spent = dict.fromkeys(("read_block", "plane_fetch", "texdata", "color", "blend", "upload"), 0.0)


def timed(owner, name, key):
    orig = getattr(owner, name)

    def wrapper(*a, **kw):
        if threading.current_thread() is not threading.main_thread():
            return orig(*a, **kw)  # prefetch threads: not time the user waits
        t = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            spent[key] += time.perf_counter() - t

    setattr(owner, name, wrapper)


from chimerax.graphics import opengl  # noqa: E402
from chimerax.map import image3d  # noqa: E402
from chimerax.ngff import ngff, render  # noqa: E402

render.install()  # before timing wrappers go on top
timed(image3d.Texture3dPlanes, "_texture_3d_data", "texdata")
timed(ngff.Image, "read_block", "read_block")
if os.environ.get("NGFF_BENCH_TRACE_GL"):
    _it = opengl.Texture.initialize_texture

    def _init_traced(self, size, *a, **kw):
        t_ = time.perf_counter()
        try:
            return _it(self, size, *a, **kw)
        finally:
            report(f"BENCH   initialize_texture dim {self.dimension} size {size} "
                   f"{time.perf_counter() - t_:.3f} s")

    opengl.Texture.initialize_texture = _init_traced
if os.environ.get("NGFF_BENCH_TRACE"):
    _rb = ngff.Image.read_block

    def _traced(self, *a, **kw):
        if threading.current_thread() is threading.main_thread():
            report(f"BENCH   main-thread read_block {a}")
        return _rb(self, *a, **kw)

    ngff.Image.read_block = _traced
timed(image3d.Image3d, "_matrix_plane", "plane_fetch")  # includes the data read
timed(image3d.Image3d, "_color_plane", "color")  # CPU colormap of one plane
timed(image3d.BlendedImage, "_color_plane", "blend")  # CPU colour + blend (includes color)
timed(opengl.Texture, "reload_texture", "upload")

try:
    for _level in ("warning", "error", "bug"):
        _orig = getattr(session.logger, _level)
        setattr(session.logger, _level, lambda m, *a, _o=_orig, _l=_level, **k: (
            report(f"BENCH log {_l}: {str(m)[:600]}"), _o(m, *a, **k))[1])
    render = session.main_view.render
    render.make_current()
    from OpenGL import GL  # noqa: E402

    report(f"BENCH gl {GL.glGetString(GL.GL_RENDERER).decode()} / "
           f"{GL.glGetString(GL.GL_VERSION).decode()}")
    base_gpu = gpu_mb()

    t0 = time.perf_counter()
    models = run(session, f"open {store} format ngff")
    draw_frame()
    report(f"BENCH open+first_draw {time.perf_counter() - t0:.3f} s")
    m = models[0]
    series = m.map_series if isinstance(m, MultiChannelSeries) else [m]
    nt = len(series[0].maps)
    for s_ in series:
        v_ = s_.maps[0]
        ro_ = v_.rendering_options
        im_ = v_.image_model()
        report(f"BENCH config {v_.id_string}: p_mode {im_._p_mode} c_mode {im_._c_mode} "
               f"mip {ro_.maximum_intensity_projection} cmap_gpu {ro_.colormap_on_gpu} "
               f"blend_gpu {ro_.blend_on_gpu} projection {ro_.projection_mode} "
               f"region {v_.region}")
    if step != "default":
        shown = " ".join("#" + s.maps[0].id_string for s in series)
        t1 = time.perf_counter()
        run(session, f"volume {shown} step {step}")
        draw_frame()
        report(f"BENCH set_step_{step} {time.perf_counter() - t1:.3f} s")
    if idle:
        report(f"BENCH idle {idle} s")
        t_end = time.perf_counter() + idle
        while time.perf_counter() < t_end:
            draw_frame()
            time.sleep(0.05)

    for p_ in getattr(session, "ngff_prefetchers", []):
        report(f"BENCH prefetch {p_.status()}")
        report(f"BENCH prefetch timings (t, read s, stats s): "
               f"{[(t, round(r, 2), round(h, 2)) for t, r, h in p_.timings]}")
    player = Play_Series(series, session, rendering_cache_size=nt * len(series))
    if scenario == "scrub":
        rng = random.Random(1)
        times = [rng.randrange(1, nt) for _ in range(nframes)]
    elif scenario == "revisit":
        times = list(range(start, start + nframes // 2)) * 2
    else:
        times = list(range(start, min(start + nframes, nt)))

    walls = []
    for i, t in enumerate(times):
        for k in spent:
            spent[k] = 0.0
        prof = None
        if profile and profile_at <= i < profile_at + 2:
            import cProfile
            prof = cProfile.Profile()
            prof.enable()
        t1 = time.perf_counter()
        player.change_time(t)
        drawn = draw_frame()
        wall = time.perf_counter() - t1
        if prof:
            prof.disable()
            import pstats
            pstats.Stats(prof, stream=sys.__stdout__).sort_stats("cumulative").print_stats(28)
        walls.append(wall)
        report(f"BENCH frame t={t} wall {wall:.3f} s  read {spent['read_block']:.3f}  "
               f"texdata {spent['texdata']:.3f}  color {spent['color']:.3f}  "
               f"blend {spent['blend']:.3f}  upload {spent['upload']:.3f}  "
               f"drawn {drawn}  pf {_pf()}  gpu +{gpu_mb() - base_gpu} MB  rss {rss_gb():.1f} GB")

    s = sorted(walls)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    report(f"BENCH summary scenario={scenario} step={step} n={len(walls)} "
           f"median {statistics.median(walls):.3f} s  p95 {p95:.3f}  min {s[0]:.3f}  "
           f"max {s[-1]:.3f}")
except Exception:
    _excepthook_report()
    os._exit(1)
