"""GUI check: every lighting mode keeps rendering while a mesh and an image volume are shown.

Run with a display (a private Xvfb is enough; the DCV desktop is the one that matters):
  tests/chimerax/run_gui.sh --exit --script tests/chimerax/lighting.py
  NGFF_GUI_GL=dcv NGFF_GUI_STORE=<store> tests/chimerax/run_gui.sh --exit --script tests/chimerax/lighting.py

Why it exists: under DCV's GL layer (nice-dcv-gl 2025.0) ChimeraX's shadow maps raised GL error 1282 at
glDrawBuffer and stopped redrawing (`lighting soft / full / gentle`, `shadows true`, `multiShadow`), because DCV
leaves a samples query open after each depth-only pass. The Argus launchers set DCV_GL_USE_SAMPLE_QUERIES=0;
run_gui.sh does the same in dcv mode (NGFF_GUI_DCV_QUERIES=1 turns the queries back on: the negative control,
this script must then FAIL).

Setup: a gray mesh under `lighting soft` + silhouettes (the order that broke), then the NGFF store when
NGFF_GUI_STORE is set. For each lighting mode, with silhouettes off and on, both kinds of shadow map are forced to
re-render: a model is added and removed (multishadow) and the scene is rotated both ways (directional shadow).
A mode fails on any error or bug report, a blocked update loop, or a framebuffer stack deeper than one. After the
first failure later results are not trusted (the stack and the query stay stale), so the run stops there.

Environment:
  NGFF_GUI_STORE      an OME-Zarr store to show with the mesh (optional)
  NGFF_LIGHTING_MESH  an OBJ to use instead of a ChimeraX sphere (optional)
  NGFF_LIGHTING_PNG   a directory: one PNG per mode (optional)
Exits 0 when every mode renders, 1 on failure.
"""

import os
import sys

from chimerax.core.commands import run
from chimerax.core.logger import Log, PlainTextLog

MODES = [
    ("default", "lighting default"),
    ("simple", "lighting simple"),
    ("soft", "lighting soft"),
    ("full", "lighting full"),
    ("gentle", "lighting gentle"),
    ("flat", "lighting flat"),
    ("shadows", "lighting simple shadows true"),
    ("multishadow 64 / 1024", "lighting simple multiShadow 64 msMapSize 1024"),
    ("multishadow 64 / 256", "lighting simple multiShadow 64 msMapSize 256"),
    ("multishadow 8", "lighting simple multiShadow 8"),
]


class Catch(PlainTextLog):
    """Collects every error / bug report the session logs."""

    def __init__(self):
        self.messages = []

    def log(self, level, msg):
        if level >= Log.LEVEL_ERROR:
            self.messages.append(msg if len(msg) < 1500 else msg[:1500] + "...")
        return False


catch = Catch()
session.logger.add_log(catch)  # noqa: F821 - ChimeraX scripts get `session` as a global
store = os.environ.get("NGFF_GUI_STORE")
mesh = os.environ.get("NGFF_LIGHTING_MESH")
png_dir = os.environ.get("NGFF_LIGHTING_PNG")
if png_dir:
    os.makedirs(png_dir, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    msg = f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else "")
    print(msg)
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def problems():
    """Why the renderer is not healthy right now, '' when it is."""
    out = []
    if catch.messages:
        out.append("error: " + catch.messages[0].strip().splitlines()[0][:200])
    if session.update_loop.blocked():
        out.append("update loop blocked (a frame failed)")
    depth = len(session.main_view.render.framebuffer_stack)
    if depth != 1:
        out.append(f"framebuffer stack depth {depth}")
    return "; ".join(out)


def settle(frames=3):
    run(session, f"wait {frames}")


def force_shadow_redraws():
    """New and removed geometry re-renders the multishadow map, rotation the directional shadow map."""
    before = {m.id for m in session.models.list()}
    run(session, "shape sphere radius 1 center 12,0,0 name lighting_tmp color orange")
    settle()
    for m in session.models.list():
        if m.id not in before and m.name == "lighting_tmp":
            run(session, f"close #{'.'.join(str(i) for i in m.id)}")
    settle()
    run(session, "turn y 20 frames 1")
    settle()
    run(session, "turn y -20 frames 1")
    settle()


run(session, "lighting soft")
run(session, "graphics silhouettes true")
if mesh:
    run(session, f'open "{mesh}"')
else:
    run(session, "shape sphere radius 5 center 0,0,0 name lighting_mesh")
run(session, "color #1 gray")
settle()
check("soft lighting with a mesh", not problems(), problems())
if store:
    run(session, f'open "{store}" format ngff')
    settle()
    check("then the image volume opens", not problems(), problems())

done_ok = all(results)
for name, cmd in MODES:
    if not done_ok:
        break
    for sil in (False, True):
        label = f"{name}, silhouettes {'on' if sil else 'off'}"
        run(session, cmd)
        run(session, f"graphics silhouettes {'true' if sil else 'false'}")
        force_shadow_redraws()
        why = problems()
        check(label, not why, why)
        if png_dir and not why:
            run(session, f'save "{png_dir}/{name.replace(" ", "_").replace("/", "-")}_{"sil" if sil else "nosil"}.png" width 600 height 400')
        if why:
            done_ok = False
            break

sys.__stdout__.write("all lighting modes render\n" if all(results) else "lighting check FAILED\n")
sys.__stdout__.flush()
os._exit(0 if all(results) else 1)  # ChimeraX's own exit path reports 2 for any --script run on a display
