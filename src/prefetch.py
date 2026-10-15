"""Read time points ahead of the one on screen, so stepping and playing find them in memory.

ChimeraX reads a time point the first time it is shown. At full resolution that is
882 MB per channel, and the read, the histogram and the texture all wait on the user
interface thread. A Prefetcher watches which time point a series shows and, on its own
threads, reads the ones around it (ahead in the direction of travel first) and hands the
blocks to ChimeraX's data cache, where `Volume.matrix()` finds them. It also gives each
block's value histogram to the Volume Viewer, so that is not recomputed on the main thread.

All ChimeraX objects (the cache, volumes, the session) are touched on the main thread only;
workers just read and count, then queue the result for the next 'new frame'.
"""

import os
import queue
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

from chimerax.map_data.datacache import Data_Cache

from . import schedule

# Channel-frames being read at once. More reads in flight do not read faster (the decode
# threads are shared) but their Python code competes with the main thread for the GIL:
# with 24 in flight, drawing a frame took 0.4 s instead of 0.025 s.
IN_FLIGHT = 4
BIG_BLOCK_BYTES = 256 * 2**20  # blocks this big are mostly I/O and decoding, not Python...
IN_FLIGHT_BIG = 8  # ...so more of them can be read at once
IDLE_FRAMES = 24  # time points kept ready around the shown one until it starts moving
RESERVE_BYTES = 48 * 2**30  # the cache never takes memory below this much staying available
GPU_UNKNOWN_BYTES = 8 * 2**30  # texture budget where the free video memory cannot be read
GPU_RESERVE_BYTES = 24 * 2**30  # video memory left free for other programs (the decon server)
IDLE_S = 0.3  # nothing changing for this long: draw ahead
EMPTY_RETRY_S = 3.0  # how long before looking again for a time point not in the store yet
DEFAULT_FRACTION = 0.75  # of the memory available (plus what the cache holds already)


def available_bytes():
    """MemAvailable from /proc/meminfo, or None where it is not there."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def budget_bytes():
    """Memory the prefetch cache may use; 0 turns prefetch off.

    CHIMERAX_NGFF_PREFETCH_GB sets it; otherwise DEFAULT_FRACTION of the memory that is
    available (and re-read whenever the shown time point or region changes, so it follows
    what other programs on the machine are using).
    """
    value = os.environ.get("CHIMERAX_NGFF_PREFETCH_GB", "").strip()
    if value:
        try:
            return max(0, int(float(value) * 2**30))
        except ValueError:
            pass
    avail = available_bytes()
    return int(DEFAULT_FRACTION * avail) if avail else 16 * 2**30


def gpu_cap_bytes():
    """CHIMERAX_NGFF_GPU_GB (0 turns off drawing ahead and bounding); None: no fixed cap."""
    value = os.environ.get("CHIMERAX_NGFF_GPU_GB", "").strip()
    if value:
        try:
            return max(0, int(float(value) * 2**30))
        except ValueError:
            pass
    return None


def free_video_memory(render):
    """Free video memory in bytes (NVIDIA's GL_NVX_gpu_memory_info), or None."""
    try:
        from OpenGL import GL
        render.make_current()
        kb = GL.glGetIntegerv(0x9049)  # GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        return int(kb) * 1024
    except Exception:
        try:
            from OpenGL import GL
            GL.glGetError()
        except Exception:
            pass
        return None


def has_texture(volume):
    im = volume._image
    planes = getattr(im, "_planes_3d", None) if im is not None else None
    return planes is not None and getattr(planes, "texture", None) is not None


class FrameCache(Data_Cache):
    """ChimeraX's data cache, evicting the time points furthest from the one shown first.

    The stock cache drops the least recently used block, which is the wrong one when a
    series plays: blocks read ahead are the newest but the ones needed last.
    """

    def __init__(self, size, score):
        Data_Cache.__init__(self, size)
        self.score = score

    def reduce_use(self):
        if self.used <= self.size:
            return
        def urgency(d):
            try:
                return self.score(d.key)
            except Exception:
                return 0.0
        for d in sorted(self.data.values(), key=urgency, reverse=True):
            if sys.getrefcount(d.value) == 2:  # nothing else holds the array
                self.remove_data(d)
                if self.used <= self.size:
                    break


def _region_tuple(region):
    return tuple(tuple(r) for r in region)


def _root(a):
    import numpy as np
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def make_stats(m, bins=10000):
    """MatrixValueStatistics of an integer array from stats.value_counts."""
    from chimerax.map_data import MatrixValueStatistics

    from .stats import value_counts
    ms = MatrixValueStatistics.__new__(MatrixValueStatistics)
    ms.minimum, ms.maximum, ms.counts = value_counts(m, bins)
    ms.bins = bins
    ms.ccounts = None
    ms.cmass = None
    return ms


class Prefetcher:
    """Keeps the time points around the shown one in ChimeraX's data cache."""

    def __init__(self, session, series, budget):
        self.session = session
        self.series = list(series)  # one MapSeries per channel
        self.nt = len(self.series[0].maps)
        self.direction = schedule.Direction(self.nt)
        self.cache = FrameCache(budget, self._urgency)
        self.cap = budget
        self._explicit_budget = bool(os.environ.get("CHIMERAX_NGFF_PREFETCH_GB", "").strip())
        self._cur = 0
        self._dir = 1
        self._signature = None
        self._requests = {}  # channel -> (level, step, region) of the shown frame
        self._shapes = {}  # (channel, region) -> block_shape, the same for every time point
        self._frame_bytes = 1
        self._wanted = []
        self._inflight = {}  # key -> Future
        self._todo = []  # keys still to read, most urgent last
        self._empty = {}  # key -> when it was found to have no data in the store
        self._empty_checked = 0.0
        self._done = queue.SimpleQueue()
        self._stats = {}  # id(block) -> (weakref to block, MatrixValueStatistics)
        self._pool = ThreadPoolExecutor(IN_FLIGHT_BIG, thread_name_prefix="chimerax-ngff-ahead")
        self._in_flight = IN_FLIGHT
        self._failed = False
        # The region the volumes have before they are shown (what ChimeraX reads first).
        self._own_region = {c: _region_tuple(s.maps[-1].region) for c, s in enumerate(self.series)}
        self._gpu_cap = gpu_cap_bytes()
        self._ahead_ok = self._gpu_cap != 0  # bounding textures at all
        self._draw_ahead = self._ahead_ok
        self._gpu_frames = 1  # time points whose textures may be kept
        self._idle_since = time.monotonic()
        self._camera = None
        self.built = 0  # textures built ahead (for tests and timing)
        self.closed = False
        self.timings = []  # (time point, read seconds, histogram seconds) per block read
        self.reads = 0  # blocks handed to the cache (for tests and timing)
        for s in self.series:
            for v in s.maps:
                v.data.data_cache = self.cache
                _use_fast_stats(v, self)
        triggers = session.triggers
        self._frame_handler = triggers.add_handler("new frame", self._on_new_frame)
        self._close_handler = triggers.add_handler("remove models", self._on_models_closed)
        try:  # stop reads that have not started when ChimeraX quits
            self._quit_handler = triggers.add_handler("app quit", lambda *_: self.close())
        except Exception:
            self._quit_handler = None

    # -- life cycle ---------------------------------------------------------

    def status(self):
        """One line for logs and tests: what is cached, in flight and wanted."""
        return (f"t={self._cur} dir={self._dir} wanted={len(self._wanted)} "
                f"inflight={len(self._inflight)} cached={len(self.cache.data)} blocks "
                f"({self.cache.used / 2**30:.1f}/{self.cache.size / 2**30:.1f} GB) "
                f"handed over={self.reads} textures built ahead={self.built} gpu window="
                f"{self._gpu_frames} frame={self._frame_bytes / 2**20:.0f} MB "
                f"closed={self.closed}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        triggers = self.session.triggers
        for h in (self._frame_handler, self._close_handler, self._quit_handler):
            if h is None:
                continue
            try:
                triggers.remove_handler(h)
            except Exception:
                pass
        for fut in self._inflight.values():
            fut.cancel()
        self._pool.shutdown(wait=False)
        self.cache.data.clear()
        self.cache.groups.clear()
        self.cache.used = 0
        self._stats.clear()

    def _on_models_closed(self, trigger, models):
        closed = set(models)
        if any(s in closed for s in self.series):
            self.close()

    # -- per frame ----------------------------------------------------------

    def _on_new_frame(self, *_):
        if self.closed:
            return
        try:
            self.tick()
        except Exception as e:  # never let prefetch break drawing
            if not self._failed:
                self._failed = True
                self.session.logger.warning(f"NGFF prefetch stopped: {e!r}")
            self.close()

    def tick(self):
        """Take finished blocks, notice a change of frame or region, keep reads going."""
        took = self._take_results()
        sig = self._read_signature()
        sig_changed = sig != self._signature
        if sig_changed:
            self._signature = sig
            self._replan(sig)
            self._build_todo()
        elif not self._todo and (took or self._retry_due()):
            self._build_todo()
        if self._todo and len(self._inflight) < self._in_flight:
            self._submit()
        if self._ahead_ok:
            self._manage_textures(sig_changed)

    def _read_signature(self):
        t = self.series[0].last_shown_time
        parts = []
        for c, s in enumerate(self.series):
            v = s.maps[t]
            if v is None:
                parts.append((c, False, None))
                continue
            shown = bool(v.display) and bool(getattr(v, "parents_displayed", True))
            parts.append((c, shown, _region_tuple(v.region)))
        return t, tuple(parts)

    def _replan(self, sig):
        t, parts = sig
        self._dir = self.direction.update(t)
        self._cur = t
        total = 0
        self._requests = {}
        for c, shown, region in parts:
            if not shown:
                continue
            v = self.series[c].maps[t]
            grid, origin, size, step = block_request(v, region)
            self._requests[c] = (grid.level, step, region)
            total += size[0] * size[1] * size[2] * grid.value_type.itemsize
        self._frame_bytes = max(total, 1)
        big = self._frame_bytes // max(1, len(self._requests)) >= BIG_BLOCK_BYTES
        self._in_flight = IN_FLIGHT_BIG if big else IN_FLIGHT
        avail = available_bytes()
        if avail is not None and not self._explicit_budget:
            self.cap = int(DEFAULT_FRACTION * (avail + self.cache.used))
        limit = self.cap
        if avail is not None:
            limit = min(limit, self.cache.used + max(0, avail - RESERVE_BYTES))
        self.cache.resize(max(limit, self._frame_bytes))
        capacity = max(1, self.cache.size // self._frame_bytes)
        if not self.direction.playing and capacity < self.nt:
            # Not playing or scrubbing, and the series does not fit: a neighbourhood is
            # enough. The whole budget is for when the time point starts moving, so looking at
            # one frame does not read what cannot all be kept anyway. A series that fits (the
            # whole of a half-resolution one) is read in full, so it loops from memory.
            capacity = min(capacity, IDLE_FRAMES)
        self._wanted = schedule.wanted(t, self._dir, self.nt, capacity)

    # -- textures ---------------------------------------------------------------

    def _manage_textures(self, changed):
        """Keep video memory bounded and, when idle, draw the next time points ahead."""
        now = time.monotonic()
        view = self.session.main_view
        if view.render is None:  # no graphics (nogui): nothing to keep in video memory
            self._ahead_ok = False
            return
        cam = tuple(view.camera.position.matrix.flat)
        if changed or cam != self._camera:
            self._camera = cam
            self._idle_since = now
        try:
            if changed:
                self._size_gpu_window()
                self._trim_images()
            elif self._draw_ahead and (now - self._idle_since >= IDLE_S
                                       or self.direction.playing):
                self._draw_one_ahead()
        except Exception as e:
            self._ahead_ok = False
            self.session.logger.warning(f"NGFF drawing ahead stopped: {e!r}")

    def _size_gpu_window(self):
        """How many time points' textures fit the video memory budget."""
        free = free_video_memory(self.session.main_view.render)
        if free is None:
            budget = GPU_UNKNOWN_BYTES  # cannot see how much is free: be modest
        else:
            budget = self._textured_bytes() + max(0, free - GPU_RESERVE_BYTES)
        if self._gpu_cap is not None:
            budget = min(budget, self._gpu_cap)
        self._gpu_frames = max(1, int(budget // self._frame_bytes))

    def _textured_bytes(self):
        n = sum(1 for s in self.series for v in s.maps if v is not None and has_texture(v))
        return n * (self._frame_bytes // max(1, len(self._requests)))

    def _gpu_wanted(self):
        return self._wanted[:self._gpu_frames]

    def _trim_images(self):
        """Free the textures of time points outside the video memory window."""
        keep = set(self._gpu_wanted())
        keep.add(self._cur)
        for s in self.series:
            for t, v in enumerate(s.maps):
                if t in keep or v is None or v.display or v._image is None:
                    continue
                v.close_image()
                v._style_when_shown = "image"  # so it is rebuilt if it is shown again

    def _draw_one_ahead(self):
        """Build the texture of the next time point that has its data in memory."""
        for t in self._gpu_wanted():
            for c, (_, _, region) in self._requests.items():
                s = self.series[c]
                v = s.maps[t]
                if v is None or v.display or has_texture(v):
                    continue
                if self._key(c, t, region) not in self.cache.data:
                    continue  # still being read
                self.session.main_view.render.make_current()  # textures belong to this context
                s.copy_display_parameters(self._cur, t)
                im = v._image or v._update_image()
                if not (im._use_3d_texture and im._rendering_options.colormap_on_gpu):
                    self._draw_ahead = False  # 2-D planes or CPU colours: not built ahead
                    return
                pd = im._texture_3d_planes()
                pd._update_coloring()  # the colour table too: nothing left for the first draw
                table = getattr(pd, "colormap", None)
                if table is not None and table.data is not None:
                    table.fill_opengl_texture()
                self.built += 1
                return

    def _key(self, c, t, region):
        """Data cache key for time t of channel c shown with `region` (shape computed once)."""
        shape = self._shapes.get((c, region))
        v = self.series[c].maps[t]
        if shape is None:
            self._shapes[(c, region)] = shape = block_shape(self.series[c].maps[t], region)
        sub, origin, size, step = shape
        return v.data.available_subsamplings[sub], origin, size, step

    def _urgency(self, key):
        grid = key[0]
        cur = self._requests.get(grid.c)
        stale = 0.0 if cur is None or cur[0] == grid.level else 1e6
        return stale + schedule.score(grid.t, self._cur, self._dir, self.nt)

    def _retry_due(self):
        return bool(self._empty) and time.monotonic() - self._empty_checked > EMPTY_RETRY_S

    def _build_todo(self):
        """Blocks to read, most urgent first: the wanted time points, visible channels."""
        wanted = set(self._wanted)
        for key, fut in list(self._inflight.items()):
            if key[0].t not in wanted and fut.cancel():
                del self._inflight[key]
        now = time.monotonic()
        self._empty_checked = now
        self._todo = []
        for t in self._wanted:
            for c, (_, _, region) in self._requests.items():
                v = self.series[c].maps[t]
                if v is None:
                    continue
                # The block to show, and the one ChimeraX reads for this volume's own
                # region before the player copies the shown frame's region onto it.
                own = self._own_region[c]
                for key in dict.fromkeys((self._key(c, t, region), self._key(c, t, own))):
                    if key in self.cache.data or key in self._inflight:
                        continue
                    if now - self._empty.get(key, -EMPTY_RETRY_S) < EMPTY_RETRY_S:
                        continue
                    self._todo.append(key)
        self._todo.reverse()  # pop() takes the most urgent

    def _submit(self):
        """Start reads, most urgent first, until IN_FLIGHT are running."""
        while self._todo and len(self._inflight) < self._in_flight:
            key = self._todo.pop()
            if key in self.cache.data or key in self._inflight:
                continue
            self._inflight[key] = self._pool.submit(self._read, key)

    def _read(self, key):
        grid, origin, size, step = key
        try:
            t0 = time.perf_counter()
            m, complete = grid.read_ahead(origin, size, step)
            t1 = time.perf_counter()
            stats = None
            if complete:
                from .stats import can_count
                if can_count(m):
                    stats = make_stats(m)
            self.timings.append((grid.t, t1 - t0, time.perf_counter() - t1))
            self._done.put((key, m if complete else None, stats, None))
        except BaseException as e:  # reported on the main thread
            self._done.put((key, None, None, e))

    def _take_results(self):
        took = False
        while True:
            try:
                key, m, stats, error = self._done.get_nowait()
            except queue.Empty:
                break
            took = True
            self._inflight.pop(key, None)
            if error is not None:
                if not self._failed:
                    self._failed = True
                    self.session.logger.warning(f"NGFF prefetch read failed: {error!r}")
                continue
            grid, origin, size, step = key
            if m is None:
                self._empty[key] = time.monotonic()  # not written yet: look again later
                continue
            self._empty.pop(key, None)
            if key in self.cache.data or grid.t not in set(self._wanted):
                continue
            grid.cache_data(m, origin, size, step)
            self.reads += 1
            if stats is not None:
                ident = id(m)
                self._stats[ident] = (weakref.ref(m, lambda _, i=ident: self._stats.pop(i, None)),
                                      stats)
        return took

    # -- histograms -----------------------------------------------------------

    def stats_for(self, m):
        """Prefetched statistics when m is a whole prefetched block, else None."""
        root = _root(m)
        item = self._stats.get(id(root))
        if item is None or item[0]() is not root:
            return None
        if m.shape != root.shape or m.strides != root.strides:
            return None
        return item[1]


def block_shape(volume, region):
    """(subsampling, origin, size, step) of the block ChimeraX asks for to show `region`.

    Mirrors Volume.region_matrix and SubsampledGrid.matrix, so a prefetched block is stored
    under the key ChimeraX looks up. The same for every time point of a series.
    """
    origin, size, step = volume.step_aligned_region(region)
    sub, _ = volume.data.choose_subsampling(step)
    sub = tuple(sub)
    return (sub,
            tuple(i // s for i, s in zip(origin, sub)),
            tuple((n + s - 1) // s for n, s in zip(size, sub)),
            tuple(i // s for i, s in zip(step, sub)))


def block_request(volume, region):
    """(level grid, origin, size, step): the data cache key for `region` of this volume."""
    sub, origin, size, step = block_shape(volume, region)
    return volume.data.available_subsamplings[sub], origin, size, step


def _use_fast_stats(volume, prefetcher):
    """Make this volume's histogram come from stats.py, not ChimeraX's single-threaded walk."""
    import types

    from chimerax.map.volume import Volume

    from .stats import can_count

    def matrix_value_statistics(self, read_matrix=True):
        ms = self.matrix_stats
        if ms:
            return ms
        if getattr(self.data, "ignore_pad_value", None) is None:
            matrices = self.displayed_matrices(read_matrix)
            if len(matrices) == 1 and can_count(matrices[0]):
                m = matrices[0]
                ms = None if prefetcher.closed else prefetcher.stats_for(m)
                self.matrix_stats = ms or make_stats(m)
                return self.matrix_stats
        return Volume.matrix_value_statistics(self, read_matrix)

    volume.matrix_value_statistics = types.MethodType(matrix_value_statistics, volume)


def series_of(models):
    """The MapSeries (one per channel) among the opened models, if there is a time series."""
    from chimerax.map.volume import MultiChannelSeries
    from chimerax.map_series import MapSeries

    found = []
    for m in models:
        if isinstance(m, MultiChannelSeries):
            found.extend(m.map_series)
        elif isinstance(m, MapSeries):
            found.append(m)
    return [s for s in found if len(s.maps) > 1]


def start(session, models):
    """Prefetcher for the opened models, or None (no series, or prefetch turned off)."""
    budget = budget_bytes()
    series = series_of(models)
    if budget <= 0 or not series:
        return None
    prefetcher = Prefetcher(session, series, budget)
    live = [p for p in getattr(session, "ngff_prefetchers", []) if not p.closed]
    session.ngff_prefetchers = live + [prefetcher]
    session.logger.info(f"Reading time points ahead in the background "
                        f"(up to {budget / 2**30:.0f} GB of memory)")
    return prefetcher
