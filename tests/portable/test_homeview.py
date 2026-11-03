"""homeview.camera_axes: the camera frame that reproduces napari's home view in ChimeraX."""

import sys
import types

import numpy as np
import pytest
from cxngff import homeview


def test_axes_are_a_right_handed_rotation():
    right, up, backward = homeview.camera_axes()
    m = np.column_stack([right, up, backward])
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)  # no mirroring: ChimeraX rejects it


def test_matches_the_numbers_measured_from_napari():
    # napari angles (45, 0, 115): view direction (z, y, x) = (0.4226, 0.6409, -0.6409), up
    # (-0.9063, 0.2988, -0.2988). Its picture is the mirror of the right-handed view of that
    # direction, which is the view from the opposite end, so the camera looks along -d.
    d = np.array([-0.6408563820557887, 0.6408563820557887, 0.4226182617406994])  # x, y, z
    u = np.array([-0.2988362387301199, 0.2988362387301199, -0.9063077870365])
    right, up, backward = homeview.camera_axes()
    np.testing.assert_allclose(backward, d, atol=1e-9)
    np.testing.assert_allclose(up, u, atol=1e-9)
    np.testing.assert_allclose(right, -np.cross(d, u), atol=1e-9)


def test_coverslip_end_is_at_the_bottom_of_the_screen():
    # The coverslip is at high z, so scene +z must point down the screen: up . z < 0.
    right, up, backward = homeview.camera_axes()
    assert up[2] < 0
    assert abs(right[2]) < 1e-12  # horizon is level


def test_env_opt_out(monkeypatch):
    monkeypatch.delenv("CHIMERAX_NGFF_HOME_VIEW", raising=False)
    assert homeview.enabled()
    for off in ("0", "false", "No", "off"):
        monkeypatch.setenv("CHIMERAX_NGFF_HOME_VIEW", off)
        assert not homeview.enabled()
        assert homeview.enabled(True)  # the open option wins
    monkeypatch.setenv("CHIMERAX_NGFF_HOME_VIEW", "1")
    assert homeview.enabled() and not homeview.enabled(False)


class _Triggers:
    def __init__(self):
        self.handlers = []

    def add_handler(self, name, func):
        self.handlers.append((name, func))


def _session(bounds, initialize_camera=False):
    view = types.SimpleNamespace(drawing_bounds=lambda: bounds)
    return types.SimpleNamespace(
        models=types.SimpleNamespace(_initialize_camera=initialize_camera),
        main_view=view,
        triggers=_Triggers(),
    )


@pytest.fixture
def fake_triggerset(monkeypatch):
    mod = types.ModuleType("chimerax.core.triggerset")
    mod.DEREGISTER = "deregister"
    monkeypatch.setitem(sys.modules, "chimerax", types.ModuleType("chimerax"))
    monkeypatch.setitem(sys.modules, "chimerax.core", types.ModuleType("chimerax.core"))
    monkeypatch.setitem(sys.modules, "chimerax.core.triggerset", mod)


def test_schedule_fires_on_a_later_open_when_nothing_3d_is_shown(fake_triggerset):
    # ChimeraX's first-model flag is already False on the second open of a session, even after
    # everything was closed; the gate must be "no 3D model displayed", not that flag.
    session = _session(bounds=None, initialize_camera=False)
    homeview.schedule(session, [])
    assert [name for name, _ in session.triggers.handlers] == ["new frame"]


def test_schedule_leaves_the_view_alone_when_3d_models_are_shown(fake_triggerset):
    session = _session(bounds=object(), initialize_camera=True)
    homeview.schedule(session, [])
    assert session.triggers.handlers == []
