"""The first draw of a large store must already use the 3-D texture, as it does after a levels edit.

Run with a display (see run_gui.sh):
  NGFF_FIRST_STORE=<store over 256 Mvoxels> NGFF_FIRST_OUT=<folder> \\
  tests/chimerax/run_gui.sh --exit --script tests/chimerax/first_draw_check.py
Saves first.png (right after open) and same_levels.png (after re-applying the very same levels);
the two must have the same brightness. Writes the verdict (FIRST_DRAW_OK or FIRST_DRAW_FAIL) to
<folder>/first_draw_result.txt, because a script's prints and errors do not reach the shell.
"""

import os
import traceback

import numpy as np
from chimerax.core.commands import run
from chimerax.map import Volume

store = os.environ["NGFF_FIRST_STORE"]
out = os.environ["NGFF_FIRST_OUT"]


def mean_green(path):
    from PIL import Image
    return float(np.asarray(Image.open(path).convert("RGB"))[..., 1].astype(float).mean())


def check():
    run(session, "windowsize 900 700")
    run(session, f"open {store} format ngff")
    for _ in range(6):
        run(session, "wait 1")
    shown = [v for v in session.models.list(type=Volume) if v.display and v._image is not None]
    textures = [v._image._use_3d_texture for v in shown]
    run(session, f"save {out}/first.png width 600")
    v = shown[0]
    levels = " ".join(f"level {a},{b}" for a, b in v.image_levels)
    run(session, f"volume #{v.id_string} {levels}")
    for _ in range(3):
        run(session, "wait 1")
    run(session, f"save {out}/same_levels.png width 600")
    a, b = mean_green(f"{out}/first.png"), mean_green(f"{out}/same_levels.png")
    ok = bool(textures) and all(textures) and abs(a - b) <= 0.05 * max(a, b, 1e-6)
    return ok, f"3-D texture per shown volume: {textures}; mean green first={a:.2f} same levels={b:.2f}"


try:
    ok, msg = check()
    verdict = f"{msg}\n{'FIRST_DRAW_OK' if ok else 'FIRST_DRAW_FAIL'}\n"
except Exception:
    ok, verdict = False, traceback.format_exc() + "FIRST_DRAW_FAIL\n"
with open(f"{out}/first_draw_result.txt", "w") as f:
    f.write(verdict)
os._exit(0 if ok else 1)
