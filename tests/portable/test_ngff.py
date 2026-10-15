"""OME-NGFF metadata: which image a store opens as, axes, transforms, channels, levels."""

import json
import os

import numpy as np
import pytest
from cxngff import ngff
from cxngff.ngff import NGFFError, open_image
from helpers import (
    axes_04,
    group_v2,
    group_v3,
    multiscales,
    pyramid,
    raw_v2,
    raw_v3,
    requires_zarr3,
    write_v3,
)

RNG = np.random.default_rng(7)


def tczyx(shape=(3, 2, 9, 20, 17)):
    return RNG.integers(0, 3000, shape).astype("u2")


def make_store(root, data, paths, scales, attrs_extra=None, axes="tczyx", version="0.4",
               group=group_v2, raw=raw_v2, ms_kw=None, chunks=None):
    """A group with multiscales over arrays made by striding the spatial axes."""
    names = list(axes)
    spatial = [i for i, n in enumerate(names) if n in "zyx"]
    levels = pyramid(data, len(paths), spatial)
    attrs = {"multiscales": multiscales(paths, scales, axes_04(names), version, **(ms_kw or {}))}
    attrs.update(attrs_extra or {})
    group(root, attrs)
    for p, a in zip(paths, levels):
        raw(os.path.join(root, *p.split("/")), a, chunks)
    return levels


def flat_scales(n, base=0.136):
    return [[1.0, 1.0] + [base * 2**k] * 3 for k in range(n)]


OMERO = {"omero": {"channels": [
    {"label": "GFP 488", "color": "00FF00", "window": {"start": 0, "end": 300}, "active": True},
    {"label": "mScarlet 561", "color": "FF00FF80", "window": {"start": 10, "end": 200}},
]}}


def test_flat_store_04(tmp_path):
    data = tczyx()
    root = str(tmp_path / "flat.ome.zarr")
    levels = make_store(root, data, ["0", "1", "2"], flat_scales(3), OMERO, chunks=(1, 1, 4, 8, 8))
    im = open_image(root + "/")
    assert [lev.path for lev in im.levels] == ["0", "1", "2"]
    assert (im.nt, im.nc) == (3, 2)
    assert im.level_size(0) == (17, 20, 9) and im.level_size(1) == (9, 10, 5)
    assert im.level_step(0) == pytest.approx((0.136,) * 3)
    assert im.level_step(2) == pytest.approx((0.544,) * 3)
    assert im.unit == "micrometer" and im.version == "0.4" and im.series is None
    assert im.subsample_levels() == [(1, (2, 2, 2)), (2, (4, 4, 4))]
    g, m = im.channels
    assert (g.label, g.rgba, g.window) == ("GFP 488", (0, 1, 0, 1), (0, 300))
    assert m.rgba == pytest.approx((1, 0, 1, 128 / 255))
    np.testing.assert_array_equal(im.read_block(0, 2, 1, (0, 0, 0), (17, 20, 9), (1, 1, 1)),
                                  data[2, 1])
    np.testing.assert_array_equal(im.read_block(1, 0, 0, (0, 0, 0), (9, 10, 5), (1, 1, 1)),
                                  levels[1][0, 0])


def test_layout3_root_multiscales_win(tmp_path):
    # Our live-lane stores: bioformats2raw layout 3 AND root multiscales at 0/0..0/2.
    data = tczyx((2, 1, 8, 16, 12))
    root = str(tmp_path / "live.ome.zarr")
    scales = [[5.0, 1.0] + [0.136 * 2**k] * 3 for k in range(3)]
    make_store(root, data, ["0/0", "0/1", "0/2"], scales,
               {"bioformats2raw.layout": 3, "omero": {"channels": [{"color": "00FF00"}]}})
    group_v2(os.path.join(root, "0"), {"multiscales": multiscales(["0", "1", "2"], scales)})
    group_v2(os.path.join(root, "OME"), {"series": ["0", "1"]})
    mip = data.max(axis=2, keepdims=True)
    group_v2(os.path.join(root, "1"), {"multiscales": multiscales(["0"], scales[:1],
                                                                   axes_04("tczyx"))})
    raw_v2(os.path.join(root, "1", "0"), mip)

    im = open_image(root)
    assert [lev.path for lev in im.levels] == ["0/0", "0/1", "0/2"] and im.series is None
    assert im.nc == 1 and im.channels[0].rgba == (0, 1, 0, 1)
    assert im.time_step == 5.0 and im.time_unit is None

    mip_im = open_image(root, series="1")
    assert mip_im.series == "1" and [lev.path for lev in mip_im.levels] == ["1/0"]
    assert mip_im.level_size(0) == (12, 16, 1)
    np.testing.assert_array_equal(mip_im.read_block(0, 1, 0, (0, 0, 0), (12, 16, 1), (1, 1, 1)),
                                  mip[1, 0])
    with pytest.raises(NGFFError, match="series: 0, 1"):
        open_image(root, series="7")


def test_bioformats2raw_without_root_multiscales(tmp_path):
    data = tczyx((1, 2, 6, 10, 8))
    root = str(tmp_path / "bf.zarr")
    group_v2(root, {"bioformats2raw.layout": 3})
    group_v2(os.path.join(root, "OME"), {"series": ["0", "1"]})
    make_store(os.path.join(root, "0"), data, ["0", "1"], flat_scales(2))
    make_store(os.path.join(root, "1"), data[:, :1], ["0"], flat_scales(1))
    im = open_image(root)
    assert im.series == "0" and [lev.path for lev in im.levels] == ["0/0", "0/1"]
    assert im.nc == 2
    im1 = open_image(root, series="1")
    assert im1.series == "1" and im1.nc == 1


def test_bioformats2raw_without_series_list_uses_0(tmp_path):
    data = tczyx((1, 1, 4, 6, 5))
    root = str(tmp_path / "bf.zarr")
    group_v2(root, {"bioformats2raw.layout": 3})
    make_store(os.path.join(root, "0"), data, ["0"], flat_scales(1))
    assert open_image(root).series == "0"


def test_ngff_03_string_axes_no_transforms(tmp_path):
    data = tczyx((2, 1, 8, 16, 12))
    root = str(tmp_path / "v03.zarr")
    levels = pyramid(data, 2, [2, 3, 4])
    group_v2(root, {"multiscales": [{"version": "0.3", "axes": list("tczyx"),
                                     "datasets": [{"path": "0"}, {"path": "1"}]}]})
    raw_v2(os.path.join(root, "0"), levels[0])
    raw_v2(os.path.join(root, "1"), levels[1])
    im = open_image(root)
    assert [a[:2] for a in im.axes] == [("t", "time"), ("c", "channel"), ("z", "space"),
                                        ("y", "space"), ("x", "space")]
    assert im.level_step(0) == (1.0, 1.0, 1.0) and im.unit is None
    # No scales: the cell size comes from the shape ratio.
    assert im.subsample_levels() == [(1, (2, 2, 2))]
    assert im.channels[0].rgba is None  # one channel, no OMERO: ChimeraX's colour


def test_ngff_01_missing_axes(tmp_path):
    data = tczyx((1, 1, 4, 6, 5))
    root = str(tmp_path / "v01.zarr")
    group_v2(root, {"multiscales": [{"version": "0.1", "datasets": [{"path": "0"}]}]})
    raw_v2(os.path.join(root, "0"), data)
    im = open_image(root)
    assert "".join(a[0] for a in im.axes) == "tczyx" and im.level_size(0) == (5, 6, 4)


@requires_zarr3
def test_ngff_05_zarr_v3_attributes_ome(tmp_path):
    data = tczyx((2, 2, 6, 10, 8))
    root = str(tmp_path / "v05.ome.zarr")
    scales = flat_scales(2, 0.5)
    ms = multiscales(["s0", "s1"], scales, axes_04("tczyx", "nanometer"), version=None)
    del ms[0]["version"]
    group_v3(root, {"ome": {"version": "0.5", "multiscales": ms, **OMERO}})
    levels = pyramid(data, 2, [2, 3, 4])
    from zarr.codecs import BloscCodec

    write_v3(os.path.join(root, "s0"), levels[0], (1, 1, 3, 5, 4), shards=(1, 1, 6, 10, 8),
             compressors=[BloscCodec()])
    write_v3(os.path.join(root, "s1"), levels[1], (1, 1, 3, 5, 4))
    im = open_image(root)
    assert im.version == "0.5" and im.unit == "nanometer"
    assert im.level_step(1) == pytest.approx((1.0, 1.0, 1.0))
    assert [c.label for c in im.channels] == ["GFP 488", "mScarlet 561"]
    np.testing.assert_array_equal(im.read_block(0, 1, 1, (1, 2, 0), (7, 8, 6), (2, 3, 1)),
                                  data[1, 1, 0:6, 2:10:3, 1:8:2])


def test_ngff_05_raw_v3_group(tmp_path):
    data = tczyx((1, 1, 4, 6, 5))
    root = str(tmp_path / "v05raw.zarr")
    ms = multiscales(["0"], flat_scales(1), axes_04("tczyx"))
    group_v3(root, {"ome": {"version": "0.5", "multiscales": ms}})
    raw_v3(os.path.join(root, "0"), data, (1, 1, 2, 3, 2))
    im = open_image(root)
    assert im.levels[0].array.zarr_format == 3
    np.testing.assert_array_equal(im.read_block(0, 0, 0, (0, 0, 0), (5, 6, 4), (1, 1, 1)),
                                  data[0, 0])


def test_raw_acquisition_4d_placeholder_scale(tmp_path):
    # Raw opym store: t,z,y,x at "p0", scale [1, 1, 1, 1], no channel axis.
    data = RNG.integers(0, 500, (3, 5, 7, 9)).astype("u2")
    root = str(tmp_path / "raw.ome.zarr")
    group_v2(root, {"multiscales": multiscales(["p0"], [[1, 1, 1, 1]], axes_04("tzyx", None)),
                    "opym": {"output_format": "both"}})
    raw_v2(os.path.join(root, "p0"), data, (1, 1, 7, 9))
    im = open_image(root)
    assert (im.nt, im.nc, im.c_axis) == (3, 1, None)
    assert im.level_step(0) == (1.0, 1.0, 1.0) and im.unit is None
    assert im.channels[0].rgba is None and im.subsample_levels() == []
    np.testing.assert_array_equal(im.read_block(0, 2, 0, (0, 0, 0), (9, 7, 5), (1, 1, 1)),
                                  data[2])


def test_bare_array(tmp_path):
    data = RNG.integers(0, 100, (4, 6, 5)).astype("u1")
    root = str(tmp_path / "bare.zarr")
    raw_v2(root, data, (2, 3, 5))
    im = open_image(root)
    assert "".join(a[0] for a in im.axes) == "zyx" and im.version is None
    assert any("plain zarr array" in w for w in im.warnings)
    np.testing.assert_array_equal(im.read_block(0, 0, 0, (0, 0, 0), (5, 6, 4), (1, 1, 1)), data)


def test_transforms_translation_and_multiscale_level(tmp_path):
    data = tczyx((1, 1, 8, 8, 8))
    root = str(tmp_path / "tr.zarr")
    scales = [[1, 1, 0.5, 0.25, 0.25], [1, 1, 1.0, 0.5, 0.5]]
    translations = [[0, 0, 1, 2, 3], [0, 0, 1.25, 2.125, 3.125]]
    ms_transforms = [{"type": "scale", "scale": [1, 1, 2, 2, 2]},
                     {"type": "translation", "translation": [0, 0, 10, 20, 30]}]
    make_store(root, data, ["0", "1"], scales,
               ms_kw={"translations": translations, "ms_transforms": ms_transforms})
    im = open_image(root)
    # scale s1*s2, translation t1*s2 + t2; ChimeraX order is x, y, z.
    assert im.level_step(0) == pytest.approx((0.5, 0.5, 1.0))
    assert im.level_origin(0) == pytest.approx((3 * 2 + 30, 2 * 2 + 20, 1 * 2 + 10))
    assert im.level_step(1) == pytest.approx((1.0, 1.0, 2.0))
    assert im.subsample_levels() == [(1, (2, 2, 2))]


def test_non_integer_level_is_skipped_with_warning(tmp_path):
    # A level whose size does not match its ratio is skipped too.
    data = tczyx((1, 1, 9, 9, 9))
    root = str(tmp_path / "bad.zarr")
    make_store(root, data, ["0", "1"], [[1, 1, 1, 1, 1], [1, 1, 4, 4, 4]])
    im = open_image(root)  # level 1 is 9 // 2 + 1 = 5 voxels, not ~9 / 4
    assert im.subsample_levels() == [] and "does not match" in im.warnings[-1]

    data = tczyx((1, 1, 9, 9, 9))
    root = str(tmp_path / "ni.zarr")
    make_store(root, data, ["0", "1", "2"], [[1, 1, 1, 1, 1], [1, 1, 1.5, 1.5, 1.5],
                                             [1, 1, 4, 4, 4]])
    im = open_image(root)
    assert im.subsample_levels() == [(2, (4, 4, 4))]  # level 1's ratio 1.5 is not integer
    assert sum("level 1 skipped" in w for w in im.warnings) == 1


def test_axes_by_position_and_names(tmp_path):
    # Space axes named a/b/c: last is x, then y, then z. Axis order x,y,z also works.
    data = RNG.integers(0, 100, (4, 6, 5)).astype("u2")
    root = str(tmp_path / "pos.zarr")
    axes = [{"name": n, "type": "space"} for n in "abc"]
    group_v2(root, {"multiscales": multiscales(["0"], [[3, 2, 1]], axes)})
    raw_v2(os.path.join(root, "0"), data)
    im = open_image(root)
    assert im.space == {"x": 2, "y": 1, "z": 0} and im.level_step(0) == (1, 2, 3)

    root2 = str(tmp_path / "xyz.zarr")
    axes = [{"name": n, "type": "space"} for n in "xyz"]
    group_v2(root2, {"multiscales": multiscales(["0"], [[1, 2, 3]], axes)})
    raw_v2(os.path.join(root2, "0"), data)  # stored as (x, y, z) = (4, 6, 5)
    im2 = open_image(root2)
    assert im2.level_size(0) == (4, 6, 5) and im2.level_step(0) == (1, 2, 3)
    np.testing.assert_array_equal(im2.read_block(0, 0, 0, (0, 0, 0), (4, 6, 5), (1, 1, 1)),
                                  data.transpose(2, 1, 0))


def test_two_dimensional_image(tmp_path):
    data = RNG.integers(0, 100, (2, 6, 5)).astype("u2")
    root = str(tmp_path / "2d.zarr")
    group_v2(root, {"multiscales": multiscales(["0"], [[1, 0.5, 0.5]], axes_04("cyx"))})
    raw_v2(os.path.join(root, "0"), data)
    im = open_image(root)
    assert im.level_size(0) == (5, 6, 1) and im.level_step(0) == (0.5, 0.5, 1.0)
    np.testing.assert_array_equal(im.read_block(0, 0, 1, (0, 0, 0), (5, 6, 1), (1, 1, 1)),
                                  data[1][None])


def test_extra_axes(tmp_path):
    data = RNG.integers(0, 100, (1, 3, 4, 5)).astype("u2")
    axes = [{"name": "angle", "type": "angle"}] + axes_04("zyx")
    root = str(tmp_path / "extra.zarr")
    group_v2(root, {"multiscales": multiscales(["0"], [[1, 1, 1, 1]], axes)})
    raw_v2(os.path.join(root, "0"), data)
    im = open_image(root)
    np.testing.assert_array_equal(im.read_block(0, 0, 0, (0, 0, 0), (5, 4, 3), (1, 1, 1)),
                                  data[0])
    root2 = str(tmp_path / "extra2.zarr")
    group_v2(root2, {"multiscales": multiscales(["0"], [[1, 1, 1, 1]], axes)})
    raw_v2(os.path.join(root2, "0"), np.zeros((2, 3, 4, 5), "u2"))
    with pytest.raises(NGFFError, match="angle"):
        open_image(root2)


@pytest.mark.parametrize("nc, expected", [
    (1, [None]),
    (2, [(0, 1, 0, 1), (1, 0, 1, 1)]),
    (5, [(0, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 0, 1), (1, 1, 1, 1)]),
])
def test_default_channel_colors(tmp_path, nc, expected):
    data = tczyx((1, nc, 2, 3, 3))
    root = str(tmp_path / "c.zarr")
    make_store(root, data, ["0"], flat_scales(1))
    assert [c.rgba for c in open_image(root).channels] == expected


def test_parse_color():
    assert ngff.parse_color("FF00FF") == (1.0, 0.0, 1.0, 1.0)
    assert ngff.parse_color("#00ff0080") == pytest.approx((0, 1, 0, 128 / 255))
    assert ngff.parse_color("red") is None and ngff.parse_color(None) is None


@pytest.mark.parametrize("attrs, match", [
    ({"plate": {"rows": []}}, "HCS"),
    ({"well": {"images": []}}, "HCS"),
    ({"something": 1}, "no OME-NGFF image"),
])
def test_refused_groups(tmp_path, attrs, match):
    root = str(tmp_path / "x.zarr")
    group_v2(root, attrs)
    with pytest.raises(NGFFError, match=match):
        open_image(root)


def test_not_a_store(tmp_path):
    with pytest.raises(NGFFError, match="not a zarr store"):
        open_image(str(tmp_path))
    with pytest.raises(NGFFError, match="no such folder"):
        open_image(str(tmp_path / "missing.zarr"))
    f = tmp_path / "file.zarr"
    f.write_text("x")
    with pytest.raises(NGFFError, match="not a folder"):
        open_image(str(f))


def test_missing_level_array(tmp_path):
    root = str(tmp_path / "m.zarr")
    group_v2(root, {"multiscales": multiscales(["0"], flat_scales(1), axes_04("tczyx"))})
    with pytest.raises(NGFFError, match="level '0'"):
        open_image(root)


def test_value_types():
    vt = ngff.chimerax_value_type
    assert vt(np.dtype(">u2")) == np.dtype("=u2")
    assert vt(np.dtype("f2")) == np.float32 and vt(np.dtype("i8")) == np.float32
    assert vt(np.dtype("u8")) == np.float32 and vt(np.dtype("?")) == np.uint8
    assert vt(np.dtype("f8")) == np.float64 and vt(np.dtype("i1")) == np.int8
    with pytest.raises(NGFFError):
        vt(np.dtype("c8"))


def test_omero_channel_count_mismatch_warns(tmp_path):
    data = tczyx((1, 3, 2, 3, 3))
    root = str(tmp_path / "mm.zarr")
    make_store(root, data, ["0"], flat_scales(1), OMERO)
    im = open_image(root)
    assert any("omero lists 2" in w for w in im.warnings)
    assert im.channels[2].rgba == (0, 1, 1, 1) and im.channels[2].label is None


def test_reading_does_not_modify_the_store(tmp_path):
    data = tczyx((1, 1, 4, 6, 5))
    root = str(tmp_path / "ro.zarr")
    make_store(root, data, ["0"], flat_scales(1))

    def snapshot():
        return {os.path.join(dp, f): os.stat(os.path.join(dp, f)).st_mtime_ns
                for dp, _, fs in os.walk(root) for f in fs}

    before = snapshot()
    im = open_image(root)
    im.read_block(0, 0, 0, (0, 0, 0), (5, 6, 4), (1, 1, 1))
    assert snapshot() == before
    assert json.loads(open(os.path.join(root, ".zattrs")).read())["multiscales"]
