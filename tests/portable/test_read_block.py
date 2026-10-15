"""Image.read_block: ChimeraX ijk regions, clamped to the level and padded with fill."""

import os

import numpy as np
import pytest
from cxngff.ngff import open_image
from helpers import axes_04, group_v2, multiscales, pyramid, raw_v2, write_v2

RNG = np.random.default_rng(99)


def store_419(tmp_path, nz=419, ny=40, nx=30):
    """t,c,z,y,x store with 3 levels made by 2x striding (419 planes -> 210 -> 105)."""
    data = RNG.integers(1, 1000, (1, 1, nz, ny, nx)).astype("u2")
    root = str(tmp_path / "s.zarr")
    levels = pyramid(data, 3, [2, 3, 4])
    scales = [[1, 1] + [0.136 * 2**k] * 3 for k in range(3)]
    group_v2(root, {"multiscales": multiscales(["0", "1", "2"], scales, axes_04("tczyx"))})
    for k, a in enumerate(levels):
        raw_v2(os.path.join(root, str(k)), a, (1, 1, 64, 16, 16))
    return root, data, levels


def expected_block(level_zyx, origin, size, step, fill=0):
    """Reference: numpy indexing with out-of-range indices replaced by fill."""
    idx = [o + st * np.arange(1 + (s - 1) // st) for o, s, st in zip(origin, size, step)]
    ix, iy, iz = idx
    out = np.full((len(iz), len(iy), len(ix)), fill, level_zyx.dtype)
    vz, vy, vx = (iz < level_zyx.shape[0]), (iy < level_zyx.shape[1]), (ix < level_zyx.shape[2])
    out[np.ix_(vz, vy, vx)] = level_zyx[np.ix_(iz[vz], iy[vy], ix[vx])]
    return out


def test_subsampled_grid_overflow_plane_is_padded(tmp_path):
    root, data, levels = store_419(tmp_path)
    im = open_image(root)
    # Level 1 by striding holds ceil(419 / 2) = 210 planes; make it 209 like a
    # floor-downsampling writer, and ask for 210 as SubsampledGrid does.
    lvl1 = levels[1][:, :, :209]
    raw_v2(os.path.join(root, "1b"), lvl1, (1, 1, 64, 16, 16))
    from cxngff.store import ZarrArray

    im.levels[1].array = ZarrArray(os.path.join(root, "1b"))
    assert im.level_size(1) == (15, 20, 209)
    m = im.read_block(1, 0, 0, (0, 0, 0), (15, 20, 210), (1, 1, 1))
    assert m.shape == (210, 20, 15)
    np.testing.assert_array_equal(m[:209], lvl1[0, 0])
    assert (m[209] == 0).all()


@pytest.mark.parametrize("origin, size, step", [
    ((0, 0, 0), (30, 40, 419), (1, 1, 1)),
    ((3, 5, 7), (20, 30, 300), (2, 3, 4)),
    ((29, 39, 418), (1, 1, 1), (1, 1, 1)),
    ((25, 35, 410), (10, 10, 20), (1, 2, 3)),  # runs past every edge
    ((40, 0, 0), (5, 40, 419), (1, 1, 1)),  # starts past the x edge: all fill
    ((0, 0, 0), (30, 40, 419), (8, 8, 8)),
])
def test_level0_regions_match_reference(tmp_path, origin, size, step):
    root, data, _ = store_419(tmp_path)
    im = open_image(root)
    m = im.read_block(0, 0, 0, origin, size, step)
    assert m.shape == tuple(1 + (s - 1) // st for s, st in zip(size, step))[::-1]
    assert m.flags.c_contiguous and m.dtype == np.dtype("=u2")
    np.testing.assert_array_equal(m, expected_block(data[0, 0], origin, size, step))


@pytest.mark.parametrize("k", [1, 2])
def test_coarse_levels_with_overflow(tmp_path, k):
    root, data, levels = store_419(tmp_path)
    im = open_image(root)
    nx, ny, nz = im.level_size(k)
    size = (nx + 1, ny + 1, nz + 1)
    m = im.read_block(k, 0, 0, (0, 0, 0), size, (1, 1, 1))
    np.testing.assert_array_equal(m, expected_block(levels[k][0, 0], (0, 0, 0), size, (1, 1, 1)))


def test_fill_value_used_for_padding(tmp_path):
    data = RNG.integers(1, 1000, (4, 5, 6)).astype("i2")
    root = str(tmp_path / "f.zarr")
    write_v2(root, data, (2, 2, 2), fill_value=-5)
    im = open_image(root)
    m = im.read_block(0, 0, 0, (4, 0, 0), (4, 5, 4), (1, 1, 1))
    np.testing.assert_array_equal(m, expected_block(data, (4, 0, 0), (4, 5, 4), (1, 1, 1), -5))


@pytest.mark.parametrize("dtype, out", [(">u2", "=u2"), ("f2", "f4"), ("i8", "f4"),
                                        ("u8", "f4"), ("bool", "u1"), (">f8", "=f8")])
def test_value_type_conversion(tmp_path, dtype, out):
    data = (RNG.integers(0, 2, (3, 4, 5)).astype(bool) if dtype == "bool"
            else RNG.integers(0, 1000, (3, 4, 5)).astype(dtype))
    root = str(tmp_path / "v.zarr")
    write_v2(root, data, (2, 2, 2))
    im = open_image(root)
    m = im.read_block(0, 0, 0, (0, 0, 0), (5, 4, 3), (1, 1, 1))
    assert m.dtype == np.dtype(out) and m.dtype.isnative
    np.testing.assert_array_equal(m, data.astype(out))


def test_background_read_matches_and_leaves_the_chunk_cache_alone(tmp_path):
    from cxngff import store

    root, data, levels = store_419(tmp_path, ny=40, nx=30)
    im = open_image(root)
    size = im.level_size(0)
    fg = im.read_block(0, 0, 0, (0, 0, 0), size, (1, 1, 1))
    store.CHUNK_CACHE.clear()
    report = {}
    bg = im.read_block(0, 0, 0, (0, 0, 0), size, (1, 1, 1), background=True, report=report)
    np.testing.assert_array_equal(bg, fg)
    assert len(store.CHUNK_CACHE) == 0
    assert report["chunks"] > 1 and report["missing"] == 0


def test_report_counts_chunks_missing_from_the_store(tmp_path):
    root, data, levels = store_419(tmp_path, ny=40, nx=30)
    # Writers skip chunks that are all fill: remove one and another stays.
    os.remove(os.path.join(root, "0", "0", "0", "0", "0", "0"))
    im = open_image(root)
    report = {}
    m = im.read_block(0, 0, 0, (0, 0, 0), im.level_size(0), (1, 1, 1), background=True,
                      report=report)
    assert report["missing"] == 1 and report["chunks"] > 1
    assert (m[:64, :16, :16] == 0).all()  # the missing chunk reads as fill
