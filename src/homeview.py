"""The home view: the camera angle the lab's napari opens with, so both viewers start alike.

ChimeraX's own first view looks straight down the z axis. In our stores the coverslip is at the
high-z end, the cell lies at lower z, and napari's home view (argus-desktop napari_open.py,
HOME_ANGLES) is a three-quarter view with the coverslip slab at the bottom of the screen and the
cell sitting on it. This module gives ChimeraX the same picture.

napari draws data x to the right and y down, so its picture is the left-right mirror of the
right-handed view ChimeraX gets from the same direction. A maximum-intensity projection is the
same seen from either end except for that mirroring, so the camera here sits on the opposite
side (a true rotation, no mirroring) and the picture matches napari's: same dome, same
neighbouring cells on the same side.

This module must not import chimerax at the top: the portable tests run it outside ChimeraX.
"""

import math
import os

import numpy as np

# Elevation of the view axis from the z axis, and the horizontal direction (xy) it points along.
# 25 degrees above the horizon is napari's elevation (115 - 90).
ELEVATION = 25.0
HORIZONTAL = (-1.0, 1.0)


def camera_axes(elevation=ELEVATION, horizontal=HORIZONTAL):
    """(right, up, backward) unit vectors of the camera in scene (x, y, z) coordinates.

    The camera looks along -backward. up has a negative z component of sin(65 deg): scene
    -z (toward the cell) is up on the screen, so the coverslip is at the bottom.
    """
    h = np.array([horizontal[0], horizontal[1], 0.0])
    h /= np.linalg.norm(h)
    e = math.radians(elevation)
    backward = math.cos(e) * h + math.sin(e) * np.array([0.0, 0.0, 1.0])
    right = np.cross(h, [0.0, 0.0, 1.0])
    up = np.cross(backward, right)
    return right, up, backward


def enabled(option=None):
    """Whether to apply the home view: the open option wins, else CHIMERAX_NGFF_HOME_VIEW (0 opts out)."""
    if option is not None:
        return bool(option)
    return os.environ.get("CHIMERAX_NGFF_HOME_VIEW", "").strip().lower() not in ("0", "false", "no", "off")


def schedule(session, models):
    """Apply the home view once, after ChimeraX has added `models` and set its own first view.

    ChimeraX looks down z when the first model is added, which happens after the opener
    returns, so this waits for the next frame. Nothing is changed when other models were
    already open (their view is the user's), or when the volumes are never added.
    """
    if not getattr(session.models, "_initialize_camera", False) or not hasattr(session, "main_view"):
        return
    from chimerax.core.triggerset import DEREGISTER

    waited = [0]

    def apply(trigger, data):
        waited[0] += 1
        if any(m.id is None for m in models):
            return DEREGISTER if waited[0] > 60 else None  # not added yet (or never will be)
        if not any(m.display for m in models):
            return DEREGISTER
        try:
            set_view(session)
        except Exception as err:  # a failed view must never break the open
            session.logger.info(f"ngff: home view not applied ({type(err).__name__}: {err})")
        return DEREGISTER

    session.triggers.add_handler("new frame", apply)


def set_view(session):
    """Point the camera along the home direction and fit everything in the window."""
    from chimerax.geometry import Place

    v = session.main_view
    bounds = v.drawing_bounds()
    if bounds is None:
        return
    right, up, backward = camera_axes()
    center = bounds.center()
    axes = np.array([right, up, backward])  # Place wants the axis vectors as rows
    v.camera.position = Place(axes=axes, origin=center)
    v.view_all(pad=0.05)
