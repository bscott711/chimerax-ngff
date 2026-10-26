"""ChimeraX check: a chanalign sidecar moves the origin of the channel it lists, and `align false` / a mismatched file do not.

  HOME=<private> chimerax --nogui --exit --cmd "devel install <repo> user true"
  HOME=<private> chimerax --nogui --exit --script tests/chimerax/align_check.py

Builds a tiny two-channel store (raw numpy writes, no zarr needed) in a temp folder.
"""

import json
import os
import tempfile

import numpy as np
from chimerax.core.commands import run
from chimerax.map import Volume

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))


def write_store(root):
    os.makedirs(os.path.join(root, "0"))
    json.dump({"zarr_format": 2}, open(os.path.join(root, ".zgroup"), "w"))
    axes = [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}] + [
        {"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"]
    json.dump({"multiscales": [{"version": "0.4", "name": "cell", "axes": axes, "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 0.2, 0.2, 0.2]}]}]}],
        "omero": {"channels": [{"label": "GFP", "color": "00FF00"}, {"label": "mSc", "color": "FF00FF"}]}},
        open(os.path.join(root, ".zattrs"), "w"))
    data = np.random.default_rng(1).integers(0, 500, (1, 2, 6, 10, 12)).astype("<u2")
    json.dump({"zarr_format": 2, "shape": list(data.shape), "chunks": list(data.shape), "dtype": "<u2",
               "compressor": None, "fill_value": 0, "filters": None, "order": "C", "dimension_separator": "."},
              open(os.path.join(root, "0", ".zarray"), "w"))
    open(os.path.join(root, "0", "0.0.0.0.0"), "wb").write(data.tobytes())


def origins():
    vols = [v for m in session.models for v in m.all_models() if isinstance(v, Volume)]
    return {v.name: tuple(round(float(o), 4) for o in v.data.origin) for v in vols}


tmp = tempfile.mkdtemp(prefix="ngff-align-")
store = os.path.join(tmp, "cell_dsr.ome.zarr")
write_store(store)
sidecar = os.path.join(tmp, "cell_dsr.align.json")
good = {"version": 1, "store": store, "shape_zyx": [6, 10, 12], "voxel_um": [0.2] * 3, "reference": 0,
        "reference_label": "GFP", "moving": [{"channel": 1, "label": "mSc", "shift_um": [1.7, -0.15, -0.26]}],
        "method": "phase_xcorr"}

run(session, f"open {store} format ngff")
base = origins()
check("no sidecar: both channels at the origin", set(base.values()) == {(0.0, 0.0, 0.0)}, str(base))
run(session, "close session")

json.dump(good, open(sidecar, "w"))
run(session, f"open {store} format ngff")
o = origins()
check("sidecar: the reference stays", o["GFP"] == (0.0, 0.0, 0.0), str(o))
check("sidecar: mSc moved by (x, y, z) = (-0.26, -0.15, 1.7)", o["mSc"] == (-0.26, -0.15, 1.7), str(o))
run(session, "close session")

run(session, f"open {store} format ngff align false")
check("align false: nothing moves", set(origins().values()) == {(0.0, 0.0, 0.0)}, str(origins()))
run(session, "close session")

json.dump({**good, "shape_zyx": [6, 10, 13]}, open(sidecar, "w"))
run(session, f"open {store} format ngff")
check("a sidecar for another shape is ignored", set(origins().values()) == {(0.0, 0.0, 0.0)}, str(origins()))
run(session, "close session")

print(f"{sum(results)}/{len(results)} checks passed")
if not all(results):
    raise SystemExit(1)
