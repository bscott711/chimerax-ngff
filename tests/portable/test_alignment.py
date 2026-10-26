"""Channel alignment sidecar (chanalign): which file is read, when it is ignored, and the origin shift it gives a channel."""

import json
import os

import numpy as np
import pytest
from cxngff import alignment
from cxngff.ngff import open_image
from test_ngff import OMERO, flat_scales, make_store, tczyx

# The same file chanalign (tests/test_core.py::test_golden_sidecar) and argus-desktop pin; shape/voxel adapted below.
GOLDEN = {"version": 1, "store": "/x/viewer/x_dsr.ome.zarr", "shape_zyx": [9, 20, 17], "voxel_um": [0.136] * 3,
          "reference": 0, "reference_label": "GFP 488", "moving": [{"channel": 1, "label": "mScarlet 561",
          "shift_um": [1.7, -0.15, -0.26]}], "method": "phase_xcorr", "model": "translation",
          "axes_used": ["z", "y", "x"], "timepoints_used": [0, 50, 99], "confidence": 0.86,
          "spread_um": [0.2, 0.2, 0.1], "created": "2026-10-26T12:00:00", "tool_version": "0.1.0"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(alignment.ROOT_ENV, str(tmp_path / "central"))
    root = str(tmp_path / "viewer" / "cell_dsr.ome.zarr")
    make_store(root, tczyx(), ["0", "1"], flat_scales(2), OMERO, chunks=(1, 1, 4, 8, 8))
    return root


def write(store, data, central=False):
    real = os.path.realpath(store)
    folder = os.path.dirname(real)
    if central:
        folder = os.path.join(os.environ[alignment.ROOT_ENV], folder.lstrip("/"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "cell_dsr.align.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_no_sidecar_leaves_the_origin_alone(store):
    im = open_image(store)
    assert alignment.apply_to_image(im, store) is None
    assert im.channel_offsets == {} and im.level_origin(0, 1) == im.level_origin(0) == (0.0, 0.0, 0.0)


def test_sidecar_moves_only_the_listed_channel_xyz_order(store):
    write(store, GOLDEN)
    im = open_image(store)
    note = alignment.apply_to_image(im, store)
    assert "phase_xcorr" in note and "mScarlet 561" in note
    assert im.level_origin(0, 0) == (0.0, 0.0, 0.0)  # the reference does not move
    # shift_um is (z, y, x); ChimeraX origin is (x, y, z)
    assert im.level_origin(0, 1) == pytest.approx((-0.26, -0.15, 1.7))
    assert im.level_origin(1, 1) == pytest.approx((-0.26, -0.15, 1.7))  # every level of the channel moves alike


def test_a_disabled_alignment_is_not_read(store):
    write(store, GOLDEN)
    im = open_image(store)
    assert alignment.apply_to_image(im, store, enabled=False) is None and im.channel_offsets == {}
    assert alignment.enabled(False) is False and alignment.enabled(True) is True


@pytest.mark.parametrize("change", [{"shape_zyx": [9, 20, 18]}, {"voxel_um": [0.2] * 3}, {"version": 2},
                                    {"moving": [{"channel": 1, "label": "x", "shift_um": ["a", 0, 0]}]}])
def test_a_sidecar_for_another_store_or_a_broken_one_is_ignored(store, change):
    write(store, {**GOLDEN, **change})
    im = open_image(store)
    note = alignment.apply_to_image(im, store)
    assert im.channel_offsets == {} and "ignored" in note


def test_central_root_and_newer_file_wins(store):
    central = write(store, GOLDEN, central=True)
    assert open_image(store) and alignment.read_alignment(store, (17, 20, 9), (0.136,) * 3, "micrometer", 2)[0] == {
        1: pytest.approx((-0.26, -0.15, 1.7))}
    beside = write(store, {**GOLDEN, "moving": [{"channel": 1, "label": "m", "shift_um": [0, 0, 0.5]}]})
    os.utime(beside, (os.path.getmtime(central) + 10,) * 2)
    assert alignment.read_alignment(store, (17, 20, 9), (0.136,) * 3, "micrometer", 2)[0] == {1: (0.5, 0.0, 0.0)}


def test_units_are_converted_and_unknown_units_refuse(store):
    write(store, {**GOLDEN, "voxel_um": [0.136] * 3})
    # a store in nanometres: 136 nm voxels; the 1.7 um shift is 1700 nm
    off, _ = alignment.read_alignment(store, (17, 20, 9), (136.0,) * 3, "nanometer", 2)
    assert off[1][2] == pytest.approx(1700.0)
    off, note = alignment.read_alignment(store, (17, 20, 9), (0.136,) * 3, "parsec", 2)
    assert off == {} and "ignored" in note


def test_grid_origin_for_a_shifted_channel_reaches_chimerax_grid_data():
    # NGFFGrid passes level_origin(level, c): checked by source so this file needs no ChimeraX.
    src = open(os.path.join(os.path.dirname(alignment.__file__), "grid.py")).read()
    assert "origin=image.level_origin(level, c)" in src
