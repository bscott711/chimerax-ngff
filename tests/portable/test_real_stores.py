"""Bit-exact reads of real lab stores against independent readers (opt-in).

Skipped unless NGFF_REAL_STORES=1: the data lives on the lab cluster. Read only.
NGFF_REAL_STORE_PATHS (os.pathsep-separated) replaces the default store list, and
OPYM_VIEWER_SRC points at opym-viewer's src folder (the level-0 reference reader).

Level 0 of time point 0 is compared with opym_viewer.volumes.ZarrStore.read(t, c);
levels 1 and 2 with chunks decoded directly by imagecodecs.blosc_decode.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from cxngff.ngff import open_image

pytestmark = pytest.mark.skipif(os.environ.get("NGFF_REAL_STORES") != "1",
                                reason="set NGFF_REAL_STORES=1 to read the lab's real stores")

DEFAULT_STORES = [
    # Backfill, flat paths 0..2, (150, 2, 419, 1458, 539). Its data is all zeros
    # (so are the pipeline's TIFFs and MIP for this cell), but layout and chunking
    # are still checked.
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/outputs/mmfs1/scratch/jacks.local/"
    "microscopy/Cell_009/viewer/Cell_009_dsr.ome.zarr",
    # Live lane, bioformats2raw layout 3, root paths 0/0..0/2, (150, 1, 419, 1458, 686).
    "/mmfs1/scratch/jacks.local/microscopy/20261014-SVO-AktPH_mSc-FLM/Cell_002/viewer/"
    "Cell_002_dsr.ome.zarr",
    # Live lane, flat, two channels with signal, (100, 2, 419, 1458, 539).
    "/mmfs1/scratch/jacks.local/microscopy/20260924-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis/"
    "Cell_006/viewer/Cell_006_dsr.ome.zarr",
]
STORES = (os.environ["NGFF_REAL_STORE_PATHS"].split(os.pathsep)
          if os.environ.get("NGFF_REAL_STORE_PATHS") else DEFAULT_STORES)
OPYM_VIEWER_SRC = os.environ.get(
    "OPYM_VIEWER_SRC", str(Path.home() / "projects" / "opym-viewer" / "src"))


def blosc_level_volume(level_dir, t, c):
    """(z, y, x) volume of one (t, c) assembled from blosc chunk files with imagecodecs."""
    import imagecodecs

    meta = json.loads((Path(level_dir) / ".zarray").read_text())
    assert meta["compressor"]["id"] == "blosc" and meta["order"] == "C"
    _, _, Z, Y, X = meta["shape"]
    _, _, cz, cy, cx = meta["chunks"]
    sep = meta.get("dimension_separator") or "."
    dtype = np.dtype(meta["dtype"])
    out = np.full((Z, Y, X), meta["fill_value"] or 0, dtype)
    for iz in range(-(-Z // cz)):
        for iy in range(-(-Y // cy)):
            for ix in range(-(-X // cx)):
                p = Path(level_dir) / sep.join(map(str, (t, c, iz, iy, ix)))
                if not p.exists():
                    continue
                chunk = np.frombuffer(imagecodecs.blosc_decode(p.read_bytes()), dtype)
                chunk = chunk.reshape(cz, cy, cx)
                z1, y1, x1 = min((iz + 1) * cz, Z), min((iy + 1) * cy, Y), min((ix + 1) * cx, X)
                out[iz * cz:z1, iy * cy:y1, ix * cx:x1] = \
                    chunk[:z1 - iz * cz, :y1 - iy * cy, :x1 - ix * cx]
    return out


@pytest.fixture(scope="module", params=STORES, ids=lambda p: Path(p).name)
def real(request):
    path = request.param
    if not os.path.isdir(path):
        pytest.skip(f"{path} not found")
    return path, open_image(path)


def test_metadata(real):
    path, im = real
    assert len(im.levels) == 3 and im.unit == "micrometer"
    assert im.level_step(0) == pytest.approx((0.136,) * 3)
    assert im.subsample_levels() == [(1, (2, 2, 2)), (2, (4, 4, 4))]
    attrs = json.loads((Path(path) / ".zattrs").read_text())
    paths = [d["path"] for d in attrs["multiscales"][0]["datasets"]]
    assert [lev.path for lev in im.levels] == paths
    if im.nc == 2:
        assert [c.rgba for c in im.channels] == [(0, 1, 0, 1), (1, 0, 1, 1)]


def test_level0_t0_matches_opym_viewer(real):
    path, im = real
    sys.path.insert(0, OPYM_VIEWER_SRC)
    try:
        from opym_viewer.volumes import ZarrStore
    finally:
        sys.path.remove(OPYM_VIEWER_SRC)
    ref = ZarrStore(Path(path))
    for c in range(im.nc):
        ours = im.read_block(0, 0, c, (0, 0, 0), im.level_size(0), (1, 1, 1))
        theirs = ref.read(0, c)
        assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
        assert np.array_equal(ours, theirs), f"channel {c} differs"


@pytest.mark.parametrize("k", [1, 2])
def test_coarse_levels_t0_match_blosc_decode(real, k):
    path, im = real
    level_dir = os.path.join(path, *im.levels[k].path.split("/"))
    for c in range(im.nc):
        ours = im.read_block(k, 0, c, (0, 0, 0), im.level_size(k), (1, 1, 1))
        assert np.array_equal(ours, blosc_level_volume(level_dir, 0, c)), f"channel {c} differs"


def test_regions_and_padding(real):
    path, im = real
    full = im.read_block(1, 0, 0, (0, 0, 0), im.level_size(1), (1, 1, 1))
    origin, size, step = (13, 101, 7), (200, 500, 150), (3, 5, 2)
    region = im.read_block(1, 0, 0, origin, size, step)
    (i0, j0, k0), (si, sj, sk) = origin, step
    np.testing.assert_array_equal(
        region, full[k0:k0 + size[2]:sk, j0:j0 + size[1]:sj, i0:i0 + size[0]:si])
    # SubsampledGrid asks level 1 for ceil(419 / 2) = 210 planes; the level has 209.
    nx, ny, nz = im.level_size(0)
    ask = tuple(-(-n // 2) for n in (nx, ny, nz))
    padded = im.read_block(1, 0, 0, (0, 0, 0), ask, (1, 1, 1))
    assert padded.shape == ask[::-1]
    lx, ly, lz = im.level_size(1)
    np.testing.assert_array_equal(padded[:lz, :ly, :lx], full)
    assert (padded[lz:] == 0).all() and (padded[:, ly:] == 0).all() and (padded[:, :, lx:] == 0).all()
