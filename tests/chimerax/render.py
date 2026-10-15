"""Render time point 0 of a store in image style, for checking geometry and colour by eye.

Run with a display (ChimeraX needs OpenGL for images):
  NGFF_RENDER_STORE=<store> NGFF_RENDER_OUT=<folder> NGFF_RENDER_PREFIX=<name> \\
  [NGFF_RENDER_LEVELS=both] chimerax --exit --script tests/chimerax/render.py

Writes <prefix>_1_default_<levels>.png (the view right after open, untouched), then
views of the full volume with an orthographic camera and the data's outline box:
2_xy_down_z (x right, y down, like the pipeline's MIP stills), 3_xz_side_along_y
(x right, z up) and 4_yz_side_along_x (y right, z up).
With NGFF_RENDER_LEVELS=both the default view is also saved with the OMERO window
as image levels. Images are 1200 px wide on a black background.
"""

import os

from chimerax.core.commands import run

store = os.environ["NGFF_RENDER_STORE"]
out = os.environ["NGFF_RENDER_OUT"]
prefix = os.environ["NGFF_RENDER_PREFIX"]
both = os.environ.get("NGFF_RENDER_LEVELS") == "both"
os.makedirs(out, exist_ok=True)


def save(name):
    path = os.path.join(out, f"{prefix}_{name}.png")
    run(session, f"save {path} width 1200 supersample 3")
    print(f"RENDER {path}")


run(session, "windowsize 1000 1000")
run(session, "set bgColor black")
os.environ["CHIMERAX_NGFF_LEVELS"] = "auto"
run(session, f"open {store}")
run(session, "wait 1")
save("1_default_autolevels")

run(session, "camera ortho")
run(session, "volume all showOutlineBox true")
for name, turns in [("2_xy_down_z", ["turn x 180"]),
                    ("3_xz_side_along_y", ["turn x -90"]),
                    ("4_yz_side_along_x", ["turn x -90", "turn y -90"])]:
    run(session, "view orient")
    for t in turns:
        run(session, t)
    run(session, "view")
    run(session, "wait 1")
    save(name)

if both:
    run(session, "close session")
    os.environ["CHIMERAX_NGFF_LEVELS"] = "omero"
    run(session, f"open {store}")
    run(session, "wait 1")
    save("1_default_omerolevels")
