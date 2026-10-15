"""ChimeraX smoke test for the installed ChimeraX-NGFF bundle.

Run:
  NGFF_SMOKE_FLAT=<flat 2-channel store> NGFF_SMOKE_LAYOUT3=<layout-3 1-channel store> \\
  chimerax --nogui --exit --script tests/chimerax/smoke.py

The expectations match the lab stores named in docs/HANDOFF.md:
  FLAT     (150, 2, 419, 1458, 539), OMERO colours 00FF00 / FF00FF
  LAYOUT3  (150, 1, 419, 1458, 686), bioformats2raw layout 3 with a z-MIP series 1
A store whose variable is unset is skipped. Exits with status 1 if any check fails.
"""

import os
import tempfile

from chimerax.core.commands import run
from chimerax.core.errors import UserError
from chimerax.map import Volume
from chimerax.map.volume import MultiChannelSeries
from chimerax.map_series import MapSeries

FORMAT = "OME-Zarr (NGFF)"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))


def close_all():
    run(session, "close session")


def first_volume(model):
    return next(v for v in model.all_models() if isinstance(v, Volume))


# -- format registration ------------------------------------------------------

fmt = session.data_formats["ngff"]
check("format registered", fmt.name == FORMAT, fmt.name)
check("format suffix .zarr, allow_directory, default_for .zarr",
      fmt.suffixes == [".zarr"] and fmt.allow_directory and fmt.default_for == [".zarr"],
      f"{fmt.suffixes} {fmt.allow_directory} {fmt.default_for}")
others = [f.name for f in session.data_formats.formats
          if ".zarr" in f.suffixes and f.name != FORMAT]
print(f"INFO: other formats claiming .zarr: {others}")

# -- flat two-channel store ---------------------------------------------------

flat = os.environ.get("NGFF_SMOKE_FLAT")
if flat:
    models = run(session, f"open {flat}")  # no format keyword: suffix decides
    check("flat: one model", len(models) == 1, str(models))
    m = models[0]
    check("flat: resolved to our format",
          getattr(m, "opened_data_format", None) is not None
          and m.opened_data_format.name == FORMAT,
          getattr(getattr(m, "opened_data_format", None), "name", None))
    check("flat: MultiChannelSeries", isinstance(m, MultiChannelSeries), type(m).__name__)
    ms = getattr(m, "map_series", [])
    check("flat: 2 channels", len(ms) == 2, str(len(ms)))
    check("flat: 150 frames per channel", all(len(s.maps) == 150 for s in ms),
          str([len(s.maps) for s in ms]))
    v0 = ms[0].maps[0]
    check("flat: xyz size (539, 1458, 419)", tuple(v0.data.size) == (539, 1458, 419),
          str(v0.data.size))
    check("flat: step 0.136", all(abs(s - 0.136) < 1e-9 for s in v0.data.step), str(v0.data.step))
    colors = [tuple(round(c, 3) for c in s.maps[0].data.rgba) for s in ms]
    check("flat: colors green and magenta", colors == [(0, 1, 0, 1), (1, 0, 1, 1)], str(colors))
    check("flat: first frame shown in image style",
          all(s.maps[0].display and s.maps[0].image_shown for s in ms))
    check("flat: shown at a subsample step (multiscale)", tuple(v0.region[2]) == (4, 4, 4),
          str(v0.region))
    # Step to frame 10 the way the slider does, then read it.
    from chimerax.map_series.slider import MapSeriesSlider  # noqa: F401  (bundle present)
    for s in ms:
        s.show_time(10)
    v10 = ms[0].maps[10]
    a = v10.matrix()
    check("flat: frame 10 readable", a is not None and a.shape == (105, 365, 135), str(a.shape))
    close_all()

    models = run(session, f"open {flat}/ format ngff")
    check("flat: format ngff with trailing slash", len(models) == 1
          and isinstance(models[0], MultiChannelSeries))
    close_all()

# -- layout-3 single-channel store --------------------------------------------

layout3 = os.environ.get("NGFF_SMOKE_LAYOUT3")
if layout3:
    models = run(session, f"open {layout3}")
    m = models[0] if models else None
    check("layout3: MapSeries", isinstance(m, MapSeries), type(m).__name__)
    check("layout3: 150 frames", m is not None and len(m.maps) == 150,
          str(len(m.maps) if m else None))
    v0 = m.maps[0]
    check("layout3: xyz size (686, 1458, 419)", tuple(v0.data.size) == (686, 1458, 419),
          str(v0.data.size))
    check("layout3: level paths 0/0..0/2", v0.data.grid_id == "0/0 t=0 c=0", v0.data.grid_id)
    check("layout3: OMERO colour green", tuple(v0.data.rgba) == (0, 1, 0, 1), str(v0.data.rgba))
    close_all()

    models = run(session, f"open {layout3} format ngff")
    check("layout3: format ngff", len(models) == 1 and isinstance(models[0], MapSeries))
    close_all()

    models = run(session, f"open {layout3} series 1")
    v = first_volume(models[0])
    check("layout3: series 1 is the z-MIP", tuple(v.data.size) == (686, 1458, 1)
          and v.data.grid_id.startswith("1/0 "), f"{v.data.size} {v.data.grid_id}")
    close_all()

# -- errors -------------------------------------------------------------------

with tempfile.TemporaryDirectory(suffix=".zarr") as d:
    try:
        run(session, f"open {d} format ngff")
        check("error: empty folder refused", False, "no error")
    except UserError as e:
        check("error: empty folder refused with a message", "not a zarr store" in str(e), str(e))

failed = [n for n, ok in results if not ok]
print(f"SMOKE: {len(results) - len(failed)} passed, {len(failed)} failed")
if failed:
    os._exit(1)
