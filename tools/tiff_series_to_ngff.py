#!/usr/bin/env python3
"""Turn a folder of per-time-point, per-channel TIFF stacks into one OME-Zarr store.

The input is one 3-D TIFF per (time point, channel), named like the lattice light-sheet
"GPUdecon" output:

    cell3_CamB_ch0_stack0007_488nm_0065519msec_0000480407msecAbs_decon.tif

    stack<N>     time point N
    <wl>nm       excitation wavelength; channels are ordered by wavelength
    <ms>msec     milliseconds since the first time point (gives the time step)

The output is a zarr v2 / NGFF 0.4 store, axes t,c,z,y,x, uint16, blosc-lz4 chunks of
(1,1,64,256,256) and three 2x block-mean levels: the layout the Argus viewer stores use,
so `open <store> format ngff` in ChimeraX plays it with the bundle's prefetching.

TIFFs carry no voxel size. Give it with --xy-um / --z-um, or leave both out: the store then
has unit-less spatial axes with scale 1 (voxel units) and the numbers can be filled in later
by editing the three `scale` entries in .zattrs; the pixel data do not change.

Needs numpy, tifffile, zarr < 3 and numcodecs (the bioimaging venv has them).

    tiff_series_to_ngff.py SRC_DIR OUT.ome.zarr [--xy-um 0.104 --z-um 0.2] [--wavelengths 488 560]
                           [--note TEXT] [--workers 24]
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NAME = re.compile(
    r"^(?P<stem>.+?)_ch(?P<ch>\d+)_stack(?P<t>\d+)_(?P<wl>\d+)nm_(?P<ms>\d+)msec_.*\.tiff?$")

# Never red/green: 488 green, 560 magenta; the other lines get colour-blind-safe hues.
COLORS = {405: "3366FF", 488: "00FF00", 560: "FF00FF", 642: "00BFFF"}
LEVELS = 3
CHUNKS = (1, 1, 64, 256, 256)


def scan(src):
    """{wavelength: {t: (path, ms)}} for every stack file in src."""
    found = {}
    for name in sorted(os.listdir(src)):
        m = NAME.match(name)
        if m:
            found.setdefault(int(m["wl"]), {})[int(m["t"])] = (
                os.path.join(src, name), int(m["ms"]))
    if not found:
        raise SystemExit(f"no stack files in {src}")
    return found


def check_complete(found):
    times = [sorted(v) for v in found.values()]
    nt = len(times[0])
    for wl, ts in zip(found, times):
        if ts != list(range(nt)):
            raise SystemExit(f"{wl} nm: time points are not 0..{nt - 1} (have {len(ts)})")
    return nt


def time_step_seconds(series):
    """Least-squares slope of ms-since-start against stack number, in seconds."""
    ts = sorted(series)
    if len(ts) < 2:
        return 1.0
    ms = np.array([series[t][1] for t in ts], float)
    return float(np.polyfit(np.array(ts, float), ms, 1)[0]) / 1000.0


def coarsen2(vol):
    """2x block mean in z, y and x (odd trailing planes dropped), rounded, same dtype."""
    z, y, x = (n // 2 * 2 for n in vol.shape)
    v = vol[:z, :y, :x].astype(np.float32)
    v = v.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    return np.rint(v).astype(vol.dtype)


def level_shapes(zyx, levels=LEVELS):
    out = [tuple(zyx)]
    for _ in range(levels - 1):
        out.append(tuple(n // 2 for n in out[-1]))
    return out


_ARRAYS = None


def _init(store):
    global _ARRAYS
    import zarr
    root = zarr.open_group(store, mode="r+")
    _ARRAYS = [root[str(k)] for k in range(LEVELS)]


def convert_one(task):
    """Read one TIFF stack, write its three levels. Returns (t, c, wl, max, p9999)."""
    import tifffile
    t, c, wl, path, zyx = task
    vol = tifffile.imread(path)
    vol = vol.reshape(zyx) if vol.size == int(np.prod(zyx)) else vol
    if vol.shape != tuple(zyx) or vol.dtype != np.uint16:
        raise ValueError(f"{path}: {vol.shape} {vol.dtype}, expected {tuple(zyx)} uint16")
    level = vol
    for k in range(LEVELS):
        if k:
            level = coarsen2(level)
        _ARRAYS[k][t, c] = level
    sample = vol[::2, ::4, ::4]
    return t, c, wl, int(vol.max()), float(np.percentile(sample, 99.99))


def build(src, out, xy_um=None, z_um=None, workers=24, log=print, wavelengths=None, note=None):
    import numcodecs
    import tifffile
    import zarr

    found = scan(src)
    if wavelengths:
        missing = [w for w in wavelengths if w not in found]
        if missing:
            raise SystemExit(f"no {missing} nm stacks in {src} (have {sorted(found)})")
        found = {w: found[w] for w in wavelengths}
    nt = check_complete(found)
    wls = sorted(found)
    first = found[wls[0]][0][0]
    with tifffile.TiffFile(first) as tf:
        nz = len(tf.pages)
        ny, nx = tf.pages[0].shape
        if tf.pages[0].dtype != np.uint16:
            raise SystemExit(f"{first}: dtype {tf.pages[0].dtype}, this tool writes uint16")
    zyx = (nz, ny, nx)
    dt = time_step_seconds(found[wls[0]])
    log(f"{nt} time points x {len(wls)} channels {wls} nm, stack z,y,x = {zyx}, "
        f"time step {dt:.4f} s")

    tmp = out + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    root = zarr.open_group(tmp, mode="w")
    comp = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    for k, shape in enumerate(level_shapes(zyx)):
        root.create_dataset(str(k), shape=(nt, len(wls)) + shape, chunks=CHUNKS,
                            dtype="<u2", compressor=comp, fill_value=0,
                            dimension_separator="/")

    tasks = [(t, c, wl, found[wl][t][0], zyx) for t in range(nt) for c, wl in enumerate(wls)]
    top = {wl: [0, 0.0] for wl in wls}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(tmp,)) as pool:
        for i, (t, c, wl, vmax, p) in enumerate(pool.map(convert_one, tasks, chunksize=1), 1):
            top[wl][0] = max(top[wl][0], vmax)
            top[wl][1] = max(top[wl][1], p)
            if i % 30 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)} stacks, {time.time() - t0:.0f} s")

    spatial = 1.0 if xy_um is None else float(xy_um)
    zstep = spatial if z_um is None else float(z_um)
    unit = {} if xy_um is None else {"unit": "micrometer"}
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}]
    axes += [{"name": a, "type": "space", **unit} for a in "zyx"]
    datasets = []
    for k in range(LEVELS):
        f = 2 ** k
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": [dt, 1.0, zstep * f, spatial * f, spatial * f]}]})
    stem = os.path.basename(out)
    stem = stem[:-len(".ome.zarr")] if stem.endswith(".ome.zarr") else stem
    extra = {"voxel_size_note": note} if note else {}
    root.attrs.update({
        **extra,
        "multiscales": [{"version": "0.4", "name": stem, "axes": axes, "datasets": datasets}],
        "omero": {"name": stem, "channels": [
            {"label": f"{wl} nm", "color": COLORS.get(wl, "FFFFFF"), "active": True,
             "window": {"min": 0, "max": top[wl][0], "start": 0, "end": int(np.ceil(top[wl][1]))}}
            for wl in wls]}})
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not replacing it (store left at {tmp})")
    os.rename(tmp, out)
    log(f"wrote {out} in {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--xy-um", type=float, help="x and y voxel size in micrometres")
    ap.add_argument("--z-um", type=float, help="z voxel size in micrometres (default: xy)")
    ap.add_argument("--wavelengths", type=int, nargs="+", metavar="NM",
                    help="only these channels (default: every wavelength found)")
    ap.add_argument("--note", help="free text kept in .zattrs as voxel_size_note, e.g. where the voxel size "
                    "came from when it is an assumption")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    build(a.src, a.out, a.xy_um, a.z_um, a.workers, wavelengths=a.wavelengths, note=a.note)


if __name__ == "__main__":
    sys.exit(main())
