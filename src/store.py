"""Read local zarr v2 and v3 arrays with numpy and imagecodecs only.

This module must not import chimerax: the portable tests run it outside ChimeraX.

Supported:
  zarr v2  .zarray, dimension_separator "." or "/", order C or F, compressor
           null/blosc/zstd/zlib/gzip/lz4/bz2/lzma, no filters.
  zarr v3  zarr.json arrays, chunk_key_encoding default or v2, codecs transpose,
           bytes, blosc, zstd, gzip, crc32c (stripped, not verified) and
           sharding_indexed (one level).

Reads decode only the chunks a region touches, in parallel on a module thread
pool (imagecodecs releases the GIL), and keep decoded chunks in a bounded LRU
cache shared by all arrays (CHIMERAX_NGFF_CACHE_MB, default 1024, <= 0 disables).
For sharded arrays the unit that is read, decoded and cached is the inner chunk.
"""

import itertools
import json
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ZarrError(ValueError):
    """The store or array cannot be read by this reader."""


# ---------------------------------------------------------------------------
# Metadata files


def load_json(path):
    """Parsed JSON file, or None when the file does not exist."""
    try:
        with open(path, "rb") as f:
            text = f.read()
    except (FileNotFoundError, NotADirectoryError):
        return None
    try:
        return json.loads(text)
    except ValueError as e:
        raise ZarrError(f"{path}: invalid JSON ({e})") from None


def node_info(path):
    """(zarr_format, node_type, metadata) of a zarr node, or (None, None, None).

    node_type is "array" or "group". For v2 groups the metadata is .zgroup
    (or {} when only .zattrs exists).
    """
    meta = load_json(os.path.join(path, "zarr.json"))
    if meta is not None:
        if meta.get("zarr_format") != 3:
            raise ZarrError(f"{path}/zarr.json: zarr_format is {meta.get('zarr_format')!r}, not 3")
        return 3, meta.get("node_type"), meta
    meta = load_json(os.path.join(path, ".zarray"))
    if meta is not None:
        return 2, "array", meta
    group = load_json(os.path.join(path, ".zgroup"))
    attrs_exist = os.path.exists(os.path.join(path, ".zattrs"))
    if group is not None or attrs_exist:
        return 2, "group", group or {}
    return None, None, None


def read_attributes(path):
    """User attributes of a node: v3 zarr.json "attributes" or v2 .zattrs ({} if none)."""
    meta = load_json(os.path.join(path, "zarr.json"))
    if meta is not None:
        return meta.get("attributes") or {}
    return load_json(os.path.join(path, ".zattrs")) or {}


# ---------------------------------------------------------------------------
# Decoded-chunk cache


class ChunkCache:
    """Thread-safe LRU of decoded chunks, bounded in bytes."""

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self._items = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            a = self._items.get(key)
            if a is not None:
                self._items.move_to_end(key)
            return a

    def put(self, key, a):
        n = a.nbytes
        if self.max_bytes <= 0 or n > self.max_bytes:
            return
        with self._lock:
            if key in self._items:
                return
            self._items[key] = a
            self._bytes += n
            while self._bytes > self.max_bytes:
                _, old = self._items.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def nbytes(self):
        return self._bytes

    def __len__(self):
        return len(self._items)


def _cache_bytes_from_env():
    value = os.environ.get("CHIMERAX_NGFF_CACHE_MB", "").strip()
    try:
        mb = float(value) if value else 1024.0
    except ValueError:
        mb = 1024.0
    return max(0, int(mb * 2**20))


CHUNK_CACHE = ChunkCache(_cache_bytes_from_env())

_POOL = ThreadPoolExecutor(min(32, os.cpu_count() or 4),
                           thread_name_prefix="chimerax-ngff")


def _background_threads():
    value = os.environ.get("CHIMERAX_NGFF_PREFETCH_THREADS", "").strip()
    try:
        n = int(value) if value else 16
    except ValueError:
        n = 16
    return max(1, min(n, os.cpu_count() or 4))


# Decoding for prefetch runs apart from the pool above, so a read for the frame on screen
# never waits behind chunks read ahead.
_BACKGROUND_POOL = ThreadPoolExecutor(_background_threads(),
                                      thread_name_prefix="chimerax-ngff-prefetch")


def no_hugepage_stalls():
    """Stop numpy asking for transparent huge pages on big arrays.

    numpy madvises every large allocation for huge pages. Where the kernel's THP defrag is
    "madvise" (Argus), each such page fault waits on direct memory compaction: opym measured
    30 GB of 127 MB arrays taking 101 s, single allocations up to 1.9 s, against 15 s without.
    A full-resolution time point here is 882 MB per channel. Same as NUMPY_MADVISE_HUGEPAGE=0.
    Returns True when the switch was thrown.
    """
    try:
        from numpy._core import multiarray
    except ImportError:  # numpy < 2
        from numpy.core import multiarray
    try:
        multiarray._set_madvise_hugepage(False)
    except AttributeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Codecs


def _imagecodecs():
    try:
        import imagecodecs
    except ImportError:
        raise ZarrError("imagecodecs is needed to decode compressed chunks") from None
    return imagecodecs


def _decoder(codec_id):
    """Function bytes -> bytes for a compressor id (zarr v2 id or v3 codec name)."""
    ic = _imagecodecs()
    decoders = {
        "blosc": ic.blosc_decode,
        "zstd": ic.zstd_decode,
        "zlib": ic.zlib_decode,
        "gzip": ic.gzip_decode,
        # numcodecs LZ4 prefixes the block with its 4-byte little-endian size.
        "lz4": lambda data: ic.lz4_decode(data, header=True),
        "bz2": ic.bz2_decode,
        "lzma": ic.lzma_decode,
    }
    return decoders.get(codec_id)


def _strip_crc32c(data):
    # imagecodecs in ChimeraX has no CRC32C, so the checksum is dropped unverified.
    return memoryview(data)[:-4]


def _v2_decompressor(compressor, where):
    if compressor is None:
        return None
    cid = compressor.get("id")
    f = _decoder(cid)
    if f is None:
        raise ZarrError(f"{where}: unsupported zarr v2 compressor {cid!r}")
    return f


# zarr v3 core data types (extension types are not supported).
_V3_DTYPES = {
    "bool": "?",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "uint8": "u1",
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
    "float16": "f2",
    "float32": "f4",
    "float64": "f8",
    "complex64": "c8",
    "complex128": "c16",
}


class _CodecPipeline:
    """Decoder for one zarr v3 codec list (no sharding) on chunks of a fixed shape."""

    def __init__(self, codecs, chunk_shape, base_dtype, where):
        self.chunk_shape = tuple(chunk_shape)
        self.transposes = []  # array -> array, in encoding order
        self.dtype = None
        self.bytes_decoders = []  # bytes -> bytes, in encoding order
        self.byte_overhead = 0  # fixed bytes added by checksums (used for shard indexes)
        self.fixed_size = True  # the encoded size depends only on the shape
        seen_array_to_bytes = False
        for codec in codecs:
            name, config = _codec_name_config(codec, where)
            if name == "transpose":
                if seen_array_to_bytes:
                    raise ZarrError(f"{where}: transpose codec after the bytes codec")
                order = _transpose_order(config.get("order"), len(chunk_shape), where)
                self.transposes.append(order)
            elif name in ("bytes", "endian"):
                endian = config.get("endian", "little")
                if endian not in ("little", "big"):
                    raise ZarrError(f"{where}: bytes codec endian {endian!r}")
                self.dtype = base_dtype.newbyteorder("<" if endian == "little" else ">")
                seen_array_to_bytes = True
            elif name == "sharding_indexed":
                raise ZarrError(f"{where}: nested sharding is not supported")
            elif name == "crc32c":
                self.bytes_decoders.append(_strip_crc32c)
                self.byte_overhead += 4
            else:
                short = name[len("numcodecs."):] if name.startswith("numcodecs.") else name
                f = _decoder(short)
                if f is None:
                    raise ZarrError(f"{where}: unsupported zarr v3 codec {name!r}")
                self.bytes_decoders.append(f)
                self.fixed_size = False
        if self.dtype is None:
            if not seen_array_to_bytes and base_dtype.itemsize == 1:
                self.dtype = base_dtype
            else:
                raise ZarrError(f"{where}: codecs have no bytes codec")
        # Shape of the array as it was serialized, after the transposes.
        shape = self.chunk_shape
        for order in self.transposes:
            shape = tuple(shape[i] for i in order)
        self.encoded_shape = shape

    def encoded_nbytes(self):
        """Encoded size when it is fixed (shard index codecs), else None."""
        if not self.fixed_size:
            return None
        return math.prod(self.chunk_shape) * self.dtype.itemsize + self.byte_overhead

    def decode(self, data, where):
        for f in reversed(self.bytes_decoders):
            data = f(data)
        n = math.prod(self.chunk_shape) * self.dtype.itemsize
        if len(data) != n:
            raise ZarrError(f"{where}: decoded {len(data)} bytes, expected {n}")
        a = np.frombuffer(data, self.dtype).reshape(self.encoded_shape)
        for order in reversed(self.transposes):
            a = a.transpose(np.argsort(order))
        return a


def _codec_name_config(codec, where):
    if isinstance(codec, str):
        return codec, {}
    if not isinstance(codec, dict) or "name" not in codec:
        raise ZarrError(f"{where}: bad codec entry {codec!r}")
    return codec["name"], codec.get("configuration") or {}


def _transpose_order(order, ndim, where):
    if order == "C":
        return tuple(range(ndim))
    if order == "F":
        return tuple(reversed(range(ndim)))
    if order is None or sorted(order) != list(range(ndim)):
        raise ZarrError(f"{where}: bad transpose order {order!r}")
    return tuple(int(i) for i in order)


# ---------------------------------------------------------------------------
# Fill values


def parse_fill_value(value, dtype):
    """numpy scalar of dtype for a zarr fill_value (None means 0)."""
    if value is None:
        value = 0
    elif isinstance(value, str):
        special = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
        if value in special:
            value = special[value]
        elif value.startswith("0x") and dtype.kind == "f":
            # zarr v3: the float's bit pattern as big-endian hex.
            raw = int(value, 16).to_bytes(dtype.itemsize, "big")
            return np.frombuffer(raw, dtype.newbyteorder(">"))[0].astype(dtype.newbyteorder("="))
        else:
            raise ZarrError(f"unsupported fill_value {value!r}")
    elif isinstance(value, (list, tuple)):
        if dtype.kind == "c" and len(value) == 2:
            return np.array(complex(parse_fill_value(value[0], np.dtype("f8")),
                                    parse_fill_value(value[1], np.dtype("f8"))), dtype)
        raise ZarrError(f"unsupported fill_value {value!r}")
    native = dtype.newbyteorder("=")
    if native.kind in "iub" and isinstance(value, float) and not math.isfinite(value):
        value = 0
    return np.array(value).astype(native)[()]


# ---------------------------------------------------------------------------
# Arrays


class ZarrArray:
    """A local zarr v2 or v3 array.

    Attributes: path, zarr_format, shape, dtype (as stored, with byte order),
    chunks (the read unit: the inner chunk for sharded arrays), fill_value.
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)
        fmt, kind, meta = node_info(self.path)
        if kind != "array":
            raise ZarrError(f"{path}: not a zarr array")
        self.zarr_format = fmt
        self._shard = None
        if fmt == 2:
            self._init_v2(meta)
        else:
            self._init_v3(meta)
        self.ndim = len(self.shape)
        if len(self.chunks) != self.ndim:
            raise ZarrError(f"{path}: chunk shape {self.chunks} does not match shape {self.shape}")
        if any(c <= 0 for c in self.chunks):
            raise ZarrError(f"{path}: bad chunk shape {self.chunks}")

    def __repr__(self):
        return (f"ZarrArray({self.path!r}, shape={self.shape}, dtype={self.dtype}, "
                f"v{self.zarr_format})")

    # -- metadata -----------------------------------------------------------

    def _init_v2(self, meta):
        where = self.path
        self.shape = tuple(int(s) for s in meta["shape"])
        self.chunks = tuple(int(c) for c in meta["chunks"])
        try:
            self.dtype = np.dtype(meta["dtype"])
        except TypeError:
            raise ZarrError(f"{where}: unsupported dtype {meta['dtype']!r}") from None
        if self.dtype.fields is not None or self.dtype.kind not in "biufc":
            raise ZarrError(f"{where}: unsupported dtype {meta['dtype']!r}")
        order = meta.get("order", "C")
        if order not in ("C", "F"):
            raise ZarrError(f"{where}: unsupported order {order!r}")
        self._order = order
        filters = meta.get("filters")
        if filters:
            ids = ", ".join(str(f.get("id") if isinstance(f, dict) else f) for f in filters)
            raise ZarrError(f"{where}: zarr v2 filters are not supported ({ids})")
        self._decompress = _v2_decompressor(meta.get("compressor"), where)
        sep = meta.get("dimension_separator") or "."
        if sep not in (".", "/"):
            raise ZarrError(f"{where}: unsupported dimension_separator {sep!r}")
        self._key = _v2_key(sep)
        self.fill_value = parse_fill_value(meta.get("fill_value"), self.dtype)

    def _init_v3(self, meta):
        where = self.path
        self.shape = tuple(int(s) for s in meta["shape"])
        data_type = meta.get("data_type")
        if not isinstance(data_type, str) or data_type not in _V3_DTYPES:
            raise ZarrError(f"{where}: unsupported data_type {data_type!r}")
        base = np.dtype(_V3_DTYPES[data_type])
        grid = meta.get("chunk_grid") or {}
        if grid.get("name") != "regular":
            raise ZarrError(f"{where}: unsupported chunk_grid {grid.get('name')!r}")
        chunk_shape = tuple(int(c) for c in grid["configuration"]["chunk_shape"])
        if meta.get("storage_transformers"):
            raise ZarrError(f"{where}: storage_transformers are not supported")
        cke = meta.get("chunk_key_encoding") or {"name": "default"}
        cke_config = cke.get("configuration") or {}
        if cke.get("name") == "default":
            self._key = _v3_default_key(cke_config.get("separator", "/"))
        elif cke.get("name") == "v2":
            self._key = _v2_key(cke_config.get("separator", "."))
        else:
            raise ZarrError(f"{where}: unsupported chunk_key_encoding {cke.get('name')!r}")

        codecs = meta.get("codecs") or []
        names = [_codec_name_config(c, where)[0] for c in codecs]
        if "sharding_indexed" in names:
            # Inner chunks are read as byte ranges of the shard file, so sharding must be
            # the only outer codec (what zarr-python writes).
            if names != ["sharding_indexed"]:
                raise ZarrError(f"{where}: codecs besides sharding_indexed ({names}) are not "
                                "supported on sharded arrays")
            config = _codec_name_config(codecs[0], where)[1]
            inner = tuple(int(c) for c in config["chunk_shape"])
            if len(inner) != len(chunk_shape) or any(
                    s % c for s, c in zip(chunk_shape, inner)):
                raise ZarrError(f"{where}: inner chunks {inner} do not tile shards {chunk_shape}")
            per_shard = tuple(s // c for s, c in zip(chunk_shape, inner))
            pipeline = _CodecPipeline(config.get("codecs") or [{"name": "bytes"}], inner, base,
                                      where)
            index_codecs = config.get("index_codecs") or [
                {"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
            index = _CodecPipeline(index_codecs, per_shard + (2,), np.dtype("u8"), where)
            index_nbytes = index.encoded_nbytes()
            if index_nbytes is None:
                raise ZarrError(f"{where}: shard index codecs must have a fixed size")
            location = config.get("index_location", "end")
            if location not in ("start", "end"):
                raise ZarrError(f"{where}: bad shard index_location {location!r}")
            self._shard = {
                "per_shard": per_shard,
                "index": index,
                "index_nbytes": index_nbytes,
                "location": location,
                "indexes": OrderedDict(),
                "lock": threading.Lock(),
            }
            self.chunks = inner
        else:
            pipeline = _CodecPipeline(codecs, chunk_shape, base, where)
            self.chunks = chunk_shape
        self._pipeline = pipeline
        self.dtype = pipeline.dtype
        self._order = "C"
        self.fill_value = parse_fill_value(meta.get("fill_value"), self.dtype)

    # -- chunk access -------------------------------------------------------

    def chunk_path(self, chunk_index):
        """File holding a (storage) chunk: the shard file for sharded arrays."""
        return os.path.join(self.path, self._key(chunk_index))

    def _decode_v2(self, data, where):
        if self._decompress is not None:
            data = self._decompress(data)
        n = math.prod(self.chunks) * self.dtype.itemsize
        if len(data) != n:
            raise ZarrError(f"{where}: decoded {len(data)} bytes, expected {n}")
        return np.frombuffer(data, self.dtype).reshape(self.chunks, order=self._order)

    def _read_unit(self, index):
        """Decoded read unit (chunk or inner chunk) at index, or None when missing."""
        if self._shard is not None:
            return self._read_inner_chunk(index)
        path = self.chunk_path(index)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return None
        if self.zarr_format == 2:
            return self._decode_v2(data, path)
        return self._pipeline.decode(data, path)

    def _shard_index(self, path):
        """Shard index array (*per_shard, 2) of (offset, nbytes), or None if no shard file."""
        sh = self._shard
        with sh["lock"]:
            idx = sh["indexes"].get(path)
            if idx is not None:
                sh["indexes"].move_to_end(path)
                return idx
        n = sh["index_nbytes"]
        try:
            with open(path, "rb") as f:
                if sh["location"] == "end":
                    f.seek(-n, os.SEEK_END)
                data = f.read(n)
        except FileNotFoundError:
            return None  # missing shards are not remembered: a writer may add them later
        except OSError as e:
            raise ZarrError(f"{path}: cannot read shard index ({e})") from None
        idx = sh["index"].decode(data, path + " (shard index)")
        with sh["lock"]:
            sh["indexes"][path] = idx
            while len(sh["indexes"]) > 4096:
                sh["indexes"].popitem(last=False)
        return idx

    def _read_inner_chunk(self, index):
        sh = self._shard
        per = sh["per_shard"]
        shard = tuple(i // n for i, n in zip(index, per))
        inner = tuple(i % n for i, n in zip(index, per))
        path = self.chunk_path(shard)
        idx = self._shard_index(path)
        if idx is None:
            return None
        offset, nbytes = (int(v) for v in idx[inner])
        if offset == 2**64 - 1 and nbytes == 2**64 - 1:
            return None
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(nbytes)
        if len(data) != nbytes:
            raise ZarrError(f"{path}: inner chunk {inner} is truncated")
        return self._pipeline.decode(data, f"{path} inner chunk {inner}")

    def read_chunk(self, index, cache=True):
        """Decoded read unit at index (read-only), or None when missing.

        Chunks are kept in the shared cache unless cache is False (used by prefetch, whose
        blocks are cached whole by ChimeraX).
        """
        key = (self.path, tuple(index))
        a = CHUNK_CACHE.get(key)
        if a is None:
            a = self._read_unit(index)
            if a is None:
                return None
            a.flags.writeable = False
            if cache:
                CHUNK_CACHE.put(key, a)
        return a

    # -- regions ------------------------------------------------------------

    def normalize_selection(self, selection):
        """Per-axis (start, stop, step) from slices, (start, stop[, step]) tuples or ints."""
        if selection is None:
            selection = [None] * self.ndim
        if len(selection) != self.ndim:
            raise ValueError(f"selection has {len(selection)} axes, array has {self.ndim}")
        out = []
        for s, n in zip(selection, self.shape):
            if s is None:
                s = slice(None)
            if isinstance(s, slice):
                start, stop, step = s.indices(n)
                if step < 1:
                    raise ValueError("negative steps are not supported")
            elif isinstance(s, (int, np.integer)):
                start, stop, step = int(s), int(s) + 1, 1
            else:
                start, stop = int(s[0]), int(s[1])
                step = int(s[2]) if len(s) > 2 else 1
            if not (0 <= start <= n and start <= stop <= n and step >= 1):
                raise ValueError(f"selection ({start}, {stop}, {step}) outside axis of size {n}")
            out.append((start, stop, step))
        return out

    def read(self, selection=None, out=None, background=False, report=None):
        """Read a strided region into a new array (native byte order) or into out.

        selection: per axis a slice, an int (kept as an axis of length 1) or
        (start, stop[, step]). out may be any writable array (or view) of the region's
        shape; values are cast to its dtype. Chunks are decoded in parallel; missing
        chunks give fill_value. A dict passed as `report` gets "chunks" (how many the region
        touches) and "missing" (how many of those are not in the store; zarr writers skip
        chunks that are all fill_value, so some missing is normal).
        background=True decodes on the prefetch pool and leaves the chunk cache alone.
        """
        sel = self.normalize_selection(selection)
        shape = tuple(len(range(*s)) for s in sel)
        if out is None:
            out = np.empty(shape, self.dtype.newbyteorder("="))
        elif tuple(out.shape) != shape:
            raise ValueError(f"out has shape {out.shape}, region has shape {shape}")
        if 0 in shape:
            return out

        plans = [_axis_plan(*s, c) for s, c in zip(sel, self.chunks)]
        fill = cast_fill(self.fill_value, out.dtype)

        def one(task):
            parts = task
            index = tuple(p[0] for p in parts)
            out_sl = tuple(p[2] for p in parts)
            a = self.read_chunk(index, cache=not background)
            if a is None:
                out[out_sl] = fill
                absent.append(index)
            else:
                out[out_sl] = a[tuple(p[1] for p in parts)]

        tasks = list(itertools.product(*plans))
        absent = []
        if len(tasks) == 1:
            one(tasks[0])
        else:
            # list() re-raises the first worker exception here.
            list((_BACKGROUND_POOL if background else _POOL).map(one, tasks))
        if report is not None:
            report["chunks"] = len(tasks)
            report["missing"] = len(absent)
        return out


def cast_fill(fill_value, dtype):
    """fill_value cast to dtype, with NaN/inf mapped to 0 for integer outputs."""
    v = fill_value
    if dtype.kind in "iub" and np.asarray(v).dtype.kind in "fc" and not np.isfinite(v):
        v = 0
    with np.errstate(invalid="ignore", over="ignore"):
        return np.array(v).astype(dtype)[()]


def _axis_plan(start, stop, step, csize):
    """For one axis: [(chunk index, slice in chunk, slice in output)] for start:stop:step."""
    plan = []
    if start >= stop:
        return plan
    for ci in range(start // csize, (stop - 1) // csize + 1):
        c0 = ci * csize
        c1 = min(c0 + csize, stop)
        i0 = start if c0 <= start else start + -(-(c0 - start) // step) * step
        if i0 >= c1:
            continue
        n = (c1 - 1 - i0) // step + 1
        o0 = (i0 - start) // step
        plan.append((ci, slice(i0 - c0, i0 - c0 + (n - 1) * step + 1, step), slice(o0, o0 + n)))
    return plan


def _v2_key(sep):
    def key(index):
        return sep.join(str(i) for i in index) if len(index) else "0"
    return key


def _v3_default_key(sep):
    def key(index):
        return "c" + "".join(sep + str(i) for i in index)
    return key


def open_array(path):
    """ZarrArray for the array at path."""
    return ZarrArray(path)
