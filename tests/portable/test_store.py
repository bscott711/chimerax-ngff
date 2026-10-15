"""Round trips: arrays written by zarr-python, read by store.py, compared with numpy."""

import json
import os

import numcodecs
import numpy as np
import pytest
from cxngff import store
from cxngff.store import ChunkCache, ZarrArray, ZarrError
from helpers import ZARR3, chunk_files, requires_zarr3, write_v2, write_v3

RNG = np.random.default_rng(1234)

V2_COMPRESSORS = {
    "none": None,
    "blosc-lz4": numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE),
    "blosc-zstd-bitshuffle": numcodecs.Blosc(cname="zstd", clevel=3,
                                             shuffle=numcodecs.Blosc.BITSHUFFLE),
    "zstd": numcodecs.Zstd(level=3),
    "zlib": numcodecs.Zlib(level=5),
    "gzip": numcodecs.GZip(level=5),
    "lz4": numcodecs.LZ4(),
    "bz2": numcodecs.BZ2(level=5),
    "lzma": numcodecs.LZMA(),
}

# Strided and offset regions as (start, stop, step) per axis of a (7, 13, 11) array.
REGIONS = [
    [(0, 7, 1), (0, 13, 1), (0, 11, 1)],
    [(1, 6, 1), (2, 13, 3), (3, 10, 2)],
    [(6, 7, 1), (0, 13, 5), (10, 11, 1)],
    [(0, 7, 4), (12, 13, 1), (0, 11, 7)],
    [(2, 2, 1), (0, 13, 1), (0, 11, 1)],  # empty
]


def sample(shape=(7, 13, 11), dtype="u2"):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return RNG.integers(0, 2, shape).astype(bool)
    if dtype.kind == "f":
        return RNG.normal(100, 30, shape).astype(dtype)
    info = np.iinfo(dtype)
    return RNG.integers(max(info.min, -1000), min(info.max, 60000), shape).astype(dtype)


def check_regions(arr, data):
    np.testing.assert_array_equal(arr.read(), data)
    for region in REGIONS:
        sl = tuple(slice(*r) for r in region)
        got = arr.read(region)
        assert got.shape == data[sl].shape
        np.testing.assert_array_equal(got, data[sl])
        assert got.dtype.byteorder in "=|"


# ---------------------------------------------------------------------------
# zarr v2


@pytest.mark.parametrize("sep", [".", "/"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("compressor", list(V2_COMPRESSORS))
def test_v2_round_trip(tmp_path, compressor, order, sep):
    data = sample()
    path = tmp_path / "a.zarr"
    write_v2(path, data, (3, 5, 4), V2_COMPRESSORS[compressor], order, sep)
    arr = ZarrArray(path)
    assert arr.zarr_format == 2 and arr.shape == data.shape and arr.chunks == (3, 5, 4)
    if sep == "/":
        assert any("/" in f for f in chunk_files(path))
    check_regions(arr, data)


@pytest.mark.parametrize("dtype", ["i1", "u1", "i2", ">u2", "<i4", "u4", "f4", ">f8", "f2",
                                   "i8", "u8", "bool"])
def test_v2_dtypes(tmp_path, dtype):
    data = sample(dtype=dtype)
    write_v2(tmp_path / "a", data, (4, 4, 4), V2_COMPRESSORS["blosc-lz4"])
    arr = ZarrArray(tmp_path / "a")
    assert arr.dtype == data.dtype
    check_regions(arr, data)


def test_v2_lz4_header_matches_numcodecs():
    # numcodecs LZ4 = 4-byte little-endian size + LZ4 block = imagecodecs header=True.
    import imagecodecs

    raw = sample().tobytes()
    encoded = numcodecs.LZ4().encode(raw)
    assert int.from_bytes(bytes(encoded[:4]), "little") == len(raw)
    assert imagecodecs.lz4_decode(encoded, header=True) == raw


def test_v2_missing_chunks_give_fill_value(tmp_path):
    data = sample()
    path = tmp_path / "a"
    write_v2(path, data, (3, 5, 4), V2_COMPRESSORS["zstd"], sep="/", fill_value=7)
    os.remove(path / "0" / "1" / "2")  # chunk (0, 1, 2): z 0-2, y 5-9, x 8-10
    os.remove(path / "2" / "2" / "0")  # an edge chunk: z 6, y 10-12, x 0-3
    truth = data.copy()
    truth[0:3, 5:10, 8:11] = 7
    truth[6:7, 10:13, 0:4] = 7
    check_regions(ZarrArray(path), truth)


def test_v2_nan_fill_value(tmp_path):
    data = sample(dtype="f4")
    path = tmp_path / "a"
    write_v2(path, data, (7, 13, 4), None, fill_value=float("nan"))
    meta = json.loads((path / ".zarray").read_text())
    assert meta["fill_value"] == "NaN"
    os.remove(path / "0.0.1")
    got = ZarrArray(path).read()
    assert np.isnan(got[:, :, 4:8]).all()
    np.testing.assert_array_equal(got[:, :, :4], data[:, :, :4])


def test_v2_null_fill_value_reads_zero(tmp_path):
    data = sample()
    path = tmp_path / "a"
    write_v2(path, data, (7, 13, 4), None)
    meta = json.loads((path / ".zarray").read_text())
    meta["fill_value"] = None
    (path / ".zarray").write_text(json.dumps(meta))
    os.remove(path / "0.0.0")
    assert (ZarrArray(path).read()[:, :, :4] == 0).all()


@pytest.mark.parametrize("value, dtype, expected", [
    (None, "u2", 0), (5, "u2", 5), ("NaN", "f4", np.nan), ("Infinity", "f8", np.inf),
    ("-Infinity", "f4", -np.inf), (True, "bool", True), ("0x7fc00000", "f4", np.nan),
    ("0x3f800000", "f4", 1.0),
])
def test_parse_fill_value(value, dtype, expected):
    got = store.parse_fill_value(value, np.dtype(dtype))
    if isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def test_v2_filters_are_refused_by_name(tmp_path):
    data = sample()
    path = tmp_path / "a"
    write_v2(path, data, (7, 13, 11), None, filters=[numcodecs.Delta(dtype="u2")])
    with pytest.raises(ZarrError, match="delta"):
        ZarrArray(path)


def test_v2_unknown_compressor_is_refused(tmp_path):
    data = sample()
    path = tmp_path / "a"
    write_v2(path, data, (7, 13, 11), None)
    meta = json.loads((path / ".zarray").read_text())
    meta["compressor"] = {"id": "jpeg2k"}
    (path / ".zarray").write_text(json.dumps(meta))
    with pytest.raises(ZarrError, match="jpeg2k"):
        ZarrArray(path)


def test_v2_five_dimensional_strided(tmp_path):
    data = RNG.integers(0, 4000, (3, 2, 9, 20, 17)).astype("u2")
    write_v2(tmp_path / "a", data, (1, 1, 4, 8, 8), V2_COMPRESSORS["blosc-lz4"], sep="/")
    arr = ZarrArray(tmp_path / "a")
    sel = [(2, 3, 1), (1, 2, 1), (1, 9, 3), (5, 20, 4), (0, 17, 2)]
    np.testing.assert_array_equal(arr.read(sel), data[2:3, 1:2, 1:9:3, 5:20:4, 0:17:2])


def test_read_into_out_view_with_cast(tmp_path):
    data = sample(dtype=">u2")
    write_v2(tmp_path / "a", data, (3, 5, 4), V2_COMPRESSORS["zlib"])
    out = np.zeros((11, 13, 7), np.float32)
    ZarrArray(tmp_path / "a").read(out=out.transpose(2, 1, 0))
    np.testing.assert_array_equal(out, data.astype(np.float32).transpose(2, 1, 0))


def test_selection_validation(tmp_path):
    write_v2(tmp_path / "a", sample(), (3, 5, 4))
    arr = ZarrArray(tmp_path / "a")
    with pytest.raises(ValueError):
        arr.read([(0, 8, 1), (0, 13, 1), (0, 11, 1)])
    with pytest.raises(ValueError):
        arr.read([(0, 7, 1), (0, 13, 1)])
    assert arr.read([3, slice(None), (0, 11, 5)]).shape == (1, 13, 3)


def test_axis_plan_covers_selection():
    for start, stop, step, c in [(0, 10, 1, 3), (1, 10, 4, 3), (5, 6, 1, 4), (0, 100, 7, 8),
                                 (3, 50, 20, 5)]:
        out = np.full(len(range(start, stop, step)), -1)
        for ci, csl, osl in store._axis_plan(start, stop, step, c):
            out[osl] = np.arange(ci * c, ci * c + c)[csl]
        np.testing.assert_array_equal(out, np.arange(start, stop, step))


def test_chunk_cache_lru_and_disable():
    cache = ChunkCache(3 * 80)
    arrays = {k: np.zeros(10, np.float64) for k in "abcd"}  # 80 bytes each
    for k in "abc":
        cache.put(k, arrays[k])
    assert cache.get("a") is arrays["a"]  # a becomes most recent
    cache.put("d", arrays["d"])  # evicts b
    assert cache.get("b") is None and cache.get("a") is not None and len(cache) == 3
    assert cache.nbytes == 240
    off = ChunkCache(0)
    off.put("a", arrays["a"])
    assert off.get("a") is None


def test_cache_env(monkeypatch):
    monkeypatch.setenv("CHIMERAX_NGFF_CACHE_MB", "16")
    assert store._cache_bytes_from_env() == 16 * 2**20
    monkeypatch.setenv("CHIMERAX_NGFF_CACHE_MB", "-1")
    assert store._cache_bytes_from_env() == 0
    monkeypatch.delenv("CHIMERAX_NGFF_CACHE_MB")
    assert store._cache_bytes_from_env() == 2**30


def test_cached_chunks_are_read_only_and_reused(tmp_path):
    data = sample()
    write_v2(tmp_path / "a", data, (3, 5, 4), V2_COMPRESSORS["blosc-lz4"])
    arr = ZarrArray(tmp_path / "a")
    first = arr.read_chunk((0, 0, 0))
    assert not first.flags.writeable
    assert arr.read_chunk((0, 0, 0)) is first


# ---------------------------------------------------------------------------
# zarr v3


@requires_zarr3
@pytest.mark.parametrize("codecs", ["blosc", "zstd", "gzip", "gzip+crc32c", "none"])
def test_v3_unsharded(tmp_path, codecs):
    from zarr.codecs import BloscCodec, Crc32cCodec, GzipCodec, ZstdCodec

    compressors = {"blosc": [BloscCodec()], "zstd": [ZstdCodec()], "gzip": [GzipCodec()],
                   "gzip+crc32c": [GzipCodec(), Crc32cCodec()], "none": None}[codecs]
    data = sample()
    write_v3(tmp_path / "a", data, (3, 5, 4), compressors=compressors)
    arr = ZarrArray(tmp_path / "a")
    assert arr.zarr_format == 3 and arr.chunks == (3, 5, 4)
    assert chunk_files(tmp_path / "a")[0].startswith("c/")
    check_regions(arr, data)


@requires_zarr3
def test_v3_transpose_big_endian_crc32c(tmp_path):
    from zarr.codecs import BytesCodec, Crc32cCodec, GzipCodec, TransposeCodec

    data = sample(dtype=">u2")
    write_v3(tmp_path / "a", data, (3, 5, 4), filters=[TransposeCodec(order=(2, 0, 1))],
             serializer=BytesCodec(endian="big"), compressors=[GzipCodec(), Crc32cCodec()])
    arr = ZarrArray(tmp_path / "a")
    assert arr.dtype == np.dtype(">u2")
    check_regions(arr, data)


@requires_zarr3
@pytest.mark.parametrize("dtype", ["u1", "i2", "f4", "f8", "i8", "bool"])
def test_v3_dtypes(tmp_path, dtype):
    data = sample(dtype=dtype)
    write_v3(tmp_path / "a", data, (4, 4, 4))
    check_regions(ZarrArray(tmp_path / "a"), data)


@requires_zarr3
@pytest.mark.parametrize("encoding", [{"name": "v2", "separator": "."},
                                      {"name": "v2", "separator": "/"},
                                      {"name": "default", "separator": "."}])
def test_v3_chunk_key_encodings(tmp_path, encoding):
    data = sample()
    write_v3(tmp_path / "a", data, (3, 5, 4), chunk_key_encoding=encoding)
    check_regions(ZarrArray(tmp_path / "a"), data)


@requires_zarr3
@pytest.mark.parametrize("location", ["start", "end"])
@pytest.mark.parametrize("inner", ["blosc", "zstd+crc32c"])
def test_v3_sharded(tmp_path, location, inner):
    from zarr.codecs import BloscCodec, BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec

    codecs = {"blosc": [BytesCodec(), BloscCodec()],
              "zstd+crc32c": [BytesCodec(), ZstdCodec(), Crc32cCodec()]}[inner]
    data = sample()
    path = tmp_path / "a"
    write_v3(path, data, (6, 10, 8), compressors=None,
             serializer=ShardingCodec(chunk_shape=(3, 5, 4), codecs=codecs,
                                      index_location=location))
    arr = ZarrArray(path)
    assert arr.chunks == (3, 5, 4)  # the read unit is the inner chunk
    meta = json.loads((path / "zarr.json").read_text())
    assert meta["codecs"][0]["configuration"]["index_location"] == location
    check_regions(arr, data)


@requires_zarr3
def test_v3_sharded_missing_inner_chunks_and_shards(tmp_path):
    data = sample()
    data[0:3, 0:5, 0:4] = 0  # one all-fill inner chunk: zarr leaves it out of the shard
    path = tmp_path / "a"
    write_v3(path, data, (3, 5, 4), shards=(6, 10, 8), fill_value=0)
    arr = ZarrArray(path)
    index = arr._shard_index(arr.chunk_path((0, 0, 0)))
    assert (index[0, 0, 0] == 2**64 - 1).all()  # missing inner chunk marker
    check_regions(arr, data)
    os.remove(arr.chunk_path((1, 1, 1)))  # a whole shard missing: z 6, y 10-12, x 8-10
    store.CHUNK_CACHE.clear()
    truth = data.copy()
    truth[6:, 10:, 8:] = 0
    check_regions(ZarrArray(path), truth)


@requires_zarr3
def test_v3_nan_fill_value(tmp_path):
    data = sample(dtype="f4")
    write_v3(tmp_path / "a", data, (7, 13, 4), fill_value=float("nan"))
    os.remove(tmp_path / "a" / "c" / "0" / "0" / "1")
    got = ZarrArray(tmp_path / "a").read()
    assert np.isnan(got[:, :, 4:8]).all()


@requires_zarr3
def test_v3_numcodecs_namespace(tmp_path):
    from zarr.codecs import numcodecs as zn

    data = sample()
    write_v3(tmp_path / "a", data, (3, 5, 4), compressors=[zn.Zlib(level=3)])
    meta = json.loads((tmp_path / "a" / "zarr.json").read_text())
    assert meta["codecs"][-1]["name"] == "numcodecs.zlib"
    check_regions(ZarrArray(tmp_path / "a"), data)


def test_not_an_array(tmp_path):
    with pytest.raises(ZarrError):
        ZarrArray(tmp_path)


def test_zarr_version_marker():
    # Records which writer ran this suite (shown with pytest -v).
    assert isinstance(ZARR3, bool)
