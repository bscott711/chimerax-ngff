"""Fixture writers for the portable tests.

zarr-python writes the round-trip stores (it is only a test dependency, never a
bundle one). The raw writers below make small uncompressed v2/v3 arrays without
zarr, so the OME-NGFF metadata tests run under any zarr version.
"""

import json
import math
import os

import numpy as np
import pytest
import zarr

ZARR3 = int(zarr.__version__.split(".")[0]) >= 3
requires_zarr3 = pytest.mark.skipif(not ZARR3, reason="needs zarr-python >= 3")


def write_v2(path, data, chunks, compressor=None, order="C", sep=".", fill_value=0,
             filters=None):
    """zarr v2 array written by zarr-python (2.x or 3.x)."""
    if ZARR3:
        z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype,
                              zarr_format=2, compressors=compressor, filters=filters,
                              order=order, fill_value=fill_value,
                              chunk_key_encoding={"name": "v2", "separator": sep})
    else:
        z = zarr.open_array(str(path), mode="w", shape=data.shape, chunks=chunks,
                            dtype=data.dtype, compressor=compressor, filters=filters,
                            order=order, fill_value=fill_value, dimension_separator=sep)
    z[...] = data
    return z


def write_v3(path, data, chunks, **kw):
    """zarr v3 array written by zarr-python 3 (keywords go to zarr.create_array)."""
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype, **kw)
    z[...] = data
    return z


def chunk_files(path):
    """Relative paths of the chunk files under an array directory."""
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            if not f.startswith(".") and f != "zarr.json":
                out.append(os.path.relpath(os.path.join(root, f), path))
    return sorted(out)


# ---------------------------------------------------------------------------
# Raw writers (no zarr)


def _chunk_blocks(data, chunks, fill=0):
    grid = [math.ceil(n / c) for n, c in zip(data.shape, chunks)]
    for index in np.ndindex(*grid):
        block = np.full(chunks, fill, data.dtype)
        sl = tuple(slice(i * c, min((i + 1) * c, n)) for i, c, n in zip(index, chunks, data.shape))
        block[tuple(slice(0, s.stop - s.start) for s in sl)] = data[sl]
        yield index, block


def raw_v2(path, data, chunks=None, attrs=None):
    """Uncompressed C-order zarr v2 array with "/" chunk keys."""
    chunks = tuple(chunks or data.shape)
    os.makedirs(path, exist_ok=True)
    meta = {"zarr_format": 2, "shape": list(data.shape), "chunks": list(chunks),
            "dtype": data.dtype.str, "compressor": None, "fill_value": 0, "filters": None,
            "order": "C", "dimension_separator": "/"}
    with open(os.path.join(path, ".zarray"), "w") as f:
        json.dump(meta, f)
    if attrs is not None:
        with open(os.path.join(path, ".zattrs"), "w") as f:
            json.dump(attrs, f)
    for index, block in _chunk_blocks(data, chunks):
        p = os.path.join(path, *map(str, index))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(block.tobytes())


def raw_v3(path, data, chunks=None):
    """Uncompressed zarr v3 array (bytes codec, little endian, default "c/" keys)."""
    chunks = tuple(chunks or data.shape)
    names = {"u1": "uint8", "u2": "uint16", "i2": "int16", "f4": "float32"}
    os.makedirs(path, exist_ok=True)
    meta = {"zarr_format": 3, "node_type": "array", "shape": list(data.shape),
            "data_type": names[data.dtype.str[1:]],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {}}
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(meta, f)
    for index, block in _chunk_blocks(data, chunks):
        p = os.path.join(path, "c", *map(str, index))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(block.astype(block.dtype.newbyteorder("<")).tobytes())


def group_v2(path, attrs=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ".zgroup"), "w") as f:
        json.dump({"zarr_format": 2}, f)
    if attrs is not None:
        with open(os.path.join(path, ".zattrs"), "w") as f:
            json.dump(attrs, f)


def group_v3(path, attrs=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attrs or {}}, f)


def axes_04(names, unit="micrometer"):
    types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    out = []
    for n in names:
        a = {"name": n, "type": types[n]}
        if types[n] == "space" and unit:
            a["unit"] = unit
        out.append(a)
    return out


def multiscales(paths, scales, axes=None, version="0.4", translations=None, name="img",
                ms_transforms=None):
    datasets = []
    for i, (p, s) in enumerate(zip(paths, scales)):
        ct = [{"type": "scale", "scale": list(s)}]
        if translations is not None:
            ct.append({"type": "translation", "translation": list(translations[i])})
        datasets.append({"path": p, "coordinateTransformations": ct})
    ms = {"version": version, "name": name, "datasets": datasets}
    if axes is not None:
        ms["axes"] = axes
    if ms_transforms is not None:
        ms["coordinateTransformations"] = ms_transforms
    return [ms]


def pyramid(data, levels, spatial_axes):
    """Level arrays made by striding the spatial axes by 2**k (exact, for tests)."""
    out = []
    for k in range(levels):
        sl = [slice(None)] * data.ndim
        for a in spatial_axes:
            sl[a] = slice(None, None, 2**k)
        out.append(np.ascontiguousarray(data[tuple(sl)]))
    return out
