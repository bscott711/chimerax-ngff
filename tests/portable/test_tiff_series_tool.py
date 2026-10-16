"""tools/tiff_series_to_ngff.py: a folder of per-(time, channel) TIFF stacks -> one store.

Every voxel is read back through the bundle's own reader and compared with the TIFF it
came from, so a swapped channel, a shifted time point or a wrong coarsening fails.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from cxngff.ngff import open_image

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("numcodecs")
zarr = pytest.importorskip("zarr")
pytestmark = pytest.mark.skipif(int(zarr.__version__.split(".")[0]) >= 3,
                                reason="the tool writes zarr v2 with zarr-python 2")

TOOL = Path(__file__).resolve().parents[2] / "tools" / "tiff_series_to_ngff.py"
spec = importlib.util.spec_from_file_location("tiff_series_to_ngff", TOOL)
tool = importlib.util.module_from_spec(spec)
sys.modules["tiff_series_to_ngff"] = tool  # worker processes unpickle by module name
spec.loader.exec_module(tool)

ZYX = (7, 40, 34)
# camera channel number is NOT the wavelength order: ch0 = 488, ch1 = 642, ch2 = 560
CH = {488: 0, 642: 1, 560: 2}
STAMPS = [0, 9360, 18720, 28080]  # ms since the first stack


def make_series(folder, rng, nt=4):
    truth = {}
    for wl, ch in CH.items():
        for t in range(nt):
            vol = rng.integers(0, 3000, ZYX).astype("u2")
            truth[wl, t] = vol
            name = f"cell3_Cam{'A' if wl == 642 else 'B'}_ch{ch}_stack{t:04d}_{wl}nm_" \
                   f"{STAMPS[t]:07d}msec_{414888 + STAMPS[t]:010d}msecAbs_decon.tif"
            tifffile.imwrite(folder / name, vol, compression="zlib")
    return truth


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    truth = make_series(src, np.random.default_rng(5))
    out = str(tmp_path_factory.mktemp("out") / "cell3.ome.zarr")
    tool.build(src, out, workers=2, log=lambda *_: None)
    return truth, out


def block(img, k, t, c):
    nx, ny, nz = img.level_size(k)
    return img.read_block(k, t, c, (0, 0, 0), (nx, ny, nz), (1, 1, 1))


def test_shape_axes_and_time_step(built):
    _, out = built
    img = open_image(out)
    assert img.shape == (4, 3) + ZYX
    assert (img.nt, img.nc) == (4, 3)
    assert img.time_step == pytest.approx(9.36, abs=1e-3)
    assert img.unit is None  # no voxel size was given: unit-less, scale 1
    assert img.level_step(0) == (1.0, 1.0, 1.0) and img.level_step(2) == (4.0, 4.0, 4.0)
    assert [img.level_size(k) for k in range(3)] == [(34, 40, 7), (17, 20, 3), (8, 10, 1)]


def test_every_voxel_matches_its_tiff_at_level_0_and_1(built):
    truth, out = built
    img = open_image(out)
    for c, wl in enumerate(sorted(CH)):  # channels are ordered by wavelength
        for t in range(4):
            assert np.array_equal(block(img, 0, t, c), truth[wl, t]), (wl, t)
            assert np.array_equal(block(img, 1, t, c), tool.coarsen2(truth[wl, t])), (wl, t)
            assert np.array_equal(block(img, 2, t, c),
                                  tool.coarsen2(tool.coarsen2(truth[wl, t]))), (wl, t)


def test_comparison_would_catch_a_swapped_channel(built):
    truth, out = built
    img = open_image(out)
    assert not np.array_equal(block(img, 0, 1, 0), truth[642, 1])  # c0 is 488, not 642


def test_channel_labels_colours_and_windows(built):
    truth, out = built
    img = open_image(out)
    assert [ch.label for ch in img.channels] == ["488 nm", "560 nm", "642 nm"]
    green, magenta, third = (ch.rgba[:3] for ch in img.channels)
    assert green == (0.0, 1.0, 0.0) and magenta == (1.0, 0.0, 1.0)
    assert third[1] > third[0]  # not red: never red/green
    for ch, wl in zip(img.channels, sorted(CH)):
        peak = max(int(truth[wl, t].max()) for t in range(4))
        assert ch.window[0] == 0 and 0 < ch.window[1] <= peak


def test_voxel_size_flags_set_units_and_scales(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_series(src, np.random.default_rng(1), nt=2)
    out = str(tmp_path / "um.ome.zarr")
    tool.build(src, out, xy_um=0.104, z_um=0.2, workers=1, log=lambda *_: None)
    img = open_image(out)
    assert img.unit == "micrometer"
    assert img.level_step(0) == pytest.approx((0.104, 0.104, 0.2))
    assert img.level_step(1) == pytest.approx((0.208, 0.208, 0.4))


def test_missing_time_point_is_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_series(src, np.random.default_rng(2), nt=3)
    next(src.glob("*stack0001_560nm*")).unlink()
    with pytest.raises(SystemExit, match="560 nm"):
        tool.build(src, str(tmp_path / "x.ome.zarr"), workers=1, log=lambda *_: None)
    assert not (tmp_path / "x.ome.zarr").exists()


def test_no_half_written_store_under_the_final_name(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_series(src, np.random.default_rng(3), nt=2)
    # a stack of the wrong size makes a worker fail; the final name must stay absent
    bad = next(src.glob("*stack0001_488nm*"))
    tifffile.imwrite(bad, np.zeros((7, 40, 33), "u2"))
    out = tmp_path / "bad.ome.zarr"
    with pytest.raises(ValueError, match="expected"):
        tool.build(src, str(out), workers=1, log=lambda *_: None)
    assert not out.exists()
