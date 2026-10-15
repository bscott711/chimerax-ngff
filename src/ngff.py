"""OME-NGFF metadata: find the image in a store and describe its levels, axes and channels.

This module must not import chimerax: the portable tests run it outside ChimeraX.

Which image a store opens as:
  1. a root "multiscales" (OME-NGFF 0.1-0.5; 0.5 nests attributes under "ome"),
     unless a series is requested. Our live-lane stores have bioformats2raw layout 3
     AND root multiscales pointing at 0/0..0/2; the root multiscales win.
  2. "bioformats2raw.layout": the requested series group, else the first one listed
     in OME/ "series" (or "0").
  3. a bare zarr array: one level, axes "tczyx"[-ndim:], scale 1.
  HCS plates and wells are refused with a clear message.

Physical units are kept as given (micrometres stay micrometres), as ChimeraX's own
OME-TIFF reader does.
"""

import math
import os
import posixpath

import numpy as np

from .store import ZarrArray, ZarrError, cast_fill, node_info, read_attributes


class NGFFError(ValueError):
    """The path is not an OME-Zarr image this reader can open."""


# Channel colours without OMERO metadata, by channel index (lab rule: channel 2 is
# magenta, never red/green). Channels past the list are gray.
DEFAULT_COLORS = ((0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0),
                  (1.0, 1.0, 0.0, 1.0))
GRAY = (1.0, 1.0, 1.0, 1.0)

_TYPE_FROM_NAME = {"t": "time", "time": "time", "c": "channel", "channel": "channel",
                   "x": "space", "y": "space", "z": "space"}


class Channel:
    def __init__(self, label=None, rgba=None, window=None, active=True):
        self.label = label  # str or None
        self.rgba = rgba  # (r, g, b, a) floats 0-1, or None for ChimeraX's default
        self.window = window  # (start, end) from OMERO, or None
        self.active = active

    def __repr__(self):
        return f"Channel({self.label!r}, rgba={self.rgba}, window={self.window})"


class Level:
    def __init__(self, path, array, scale, translation, explicit_scale):
        self.path = path  # relative to the store root, "/"-separated
        self.array = array  # ZarrArray
        self.scale = scale  # per array axis
        self.translation = translation  # per array axis
        self.explicit_scale = explicit_scale  # False when no coordinateTransformations


class Image:
    """One multiscale image: levels, axes, channels, and xyz geometry per level."""

    def __init__(self, root, name, version, series, axes, levels, channels, warnings):
        self.root = root
        self.name = name
        self.version = version
        self.series = series
        self.axes = axes  # [(name, type, unit)] per array axis
        self.levels = levels
        self.channels = channels
        self.warnings = warnings
        self._classify_axes()
        self.value_type = chimerax_value_type(levels[0].array.dtype)

    # -- axes -------------------------------------------------------------

    def _classify_axes(self):
        shape = self.levels[0].array.shape
        self.t_axis = self.c_axis = None
        space = []
        for i, (name, typ, _) in enumerate(self.axes):
            if typ == "time":
                if self.t_axis is not None:
                    raise NGFFError(f"{self.root}: more than one time axis")
                self.t_axis = i
            elif typ == "channel":
                if self.c_axis is not None:
                    raise NGFFError(f"{self.root}: more than one channel axis")
                self.c_axis = i
            elif typ == "space":
                space.append(i)
            elif shape[i] != 1:
                raise NGFFError(f"{self.root}: axis {name!r} (type {typ!r}) has size {shape[i]}; "
                                "only time, channel and space axes can be longer than 1")
        if len(space) not in (2, 3):
            raise NGFFError(f"{self.root}: {len(space)} space axes; 2 or 3 are supported")
        names = [self.axes[i][0].lower() for i in space]
        wanted = ["x", "y"] if len(space) == 2 else ["x", "y", "z"]
        if sorted(names) == sorted(wanted):
            self.space = {n: space[names.index(n)] for n in wanted}
        else:
            # No x/y/z names: the last space axis is x, then y, then z.
            self.space = {"x": space[-1], "y": space[-2]}
            if len(space) == 3:
                self.space["z"] = space[-3]
        self.space.setdefault("z", None)
        units = {self.axes[self.space[a]][2] for a in "xyz" if self.space[a] is not None}
        if len(units) > 1:
            self.warnings.append(f"space axes have different units {sorted(map(str, units))}; "
                                 "values are used as given")
        self.unit = self.axes[self.space["x"]][2]
        self.time_unit = self.axes[self.t_axis][2] if self.t_axis is not None else None

    @property
    def shape(self):
        return self.levels[0].array.shape

    @property
    def nt(self):
        return 1 if self.t_axis is None else self.shape[self.t_axis]

    @property
    def nc(self):
        return 1 if self.c_axis is None else self.shape[self.c_axis]

    @property
    def time_step(self):
        return None if self.t_axis is None else self.levels[0].scale[self.t_axis]

    # -- per-level xyz geometry -------------------------------------------

    def _xyz(self, values, missing):
        return tuple(missing if self.space[a] is None else values[self.space[a]] for a in "xyz")

    def level_size(self, k):
        """(nx, ny, nz) of level k; nz is 1 for 2-D images."""
        return tuple(int(n) for n in self._xyz(self.levels[k].array.shape, 1))

    def level_step(self, k):
        """Voxel size (x, y, z) of level k in the axes' own units."""
        return tuple(float(s) for s in self._xyz(self.levels[k].scale, 1.0))

    def level_origin(self, k):
        return tuple(float(t) for t in self._xyz(self.levels[k].translation, 0.0))

    def subsample_levels(self, tol=1e-3):
        """[(k, (cx, cy, cz))] for the levels usable as ChimeraX subsamples of level 0.

        The cell size is the per-axis integer ratio of level k's voxel size to level 0's
        (from the shape ratio when the store gives no scales). Levels whose ratio is not
        an integer within tol, whose size does not match it, or whose time/channel sizes
        differ from level 0, are skipped and a reason is added to self.warnings.
        """
        s0, n0 = self.level_step(0), self.level_size(0)
        shape0 = self.shape
        out, used = [], {(1, 1, 1)}
        for k in range(1, len(self.levels)):
            lev = self.levels[k]
            nk = self.level_size(k)
            fixed = [a for a in (self.t_axis, self.c_axis) if a is not None]
            if any(lev.array.shape[a] != shape0[a] for a in fixed):
                self.warnings.append(f"level {lev.path} skipped: time/channel sizes "
                                     f"{lev.array.shape} differ from level 0 {shape0}")
                continue
            if lev.explicit_scale or self.levels[0].explicit_scale:
                ratios = [b / a for a, b in zip(s0, self.level_step(k))]
            else:
                ratios = [a / b for a, b in zip(n0, nk)]
            cell = tuple(int(round(r)) for r in ratios)
            if any(c < 1 or abs(r - c) > tol for r, c in zip(ratios, cell)):
                self.warnings.append(f"level {lev.path} skipped: scale ratio to level 0 "
                                     f"{tuple(round(r, 4) for r in ratios)} is not integer")
                continue
            if any(abs(n - m / c) > 1 for n, m, c in zip(nk, n0, cell)):
                self.warnings.append(f"level {lev.path} skipped: size {nk} does not match "
                                     f"level 0 size {n0} / {cell}")
                continue
            if cell in used:
                continue
            used.add(cell)
            out.append((k, cell))
        return out

    # -- reading ------------------------------------------------------------

    def read_block(self, k, t, c, ijk_origin, ijk_size, ijk_step, background=False,
                   report=None):
        """Region of level k at time t, channel c, as a (nk, nj, ni) array of value_type.

        background=True reads on the prefetch pool without touching the chunk cache. A dict
        passed as `report` is filled in as ZarrArray.read describes.

        ijk are ChimeraX (x, y, z) indices. Each axis returns 1 + (size - 1) // step values.
        Indices past the end of the array are padded with the fill value: ChimeraX asks
        a subsample level for ceil(size / cell) voxels, which can be one more than the
        level holds (419 planes -> level 1 has 209 but is asked for 210).
        """
        arr = self.levels[k].array
        dims = self.level_size(k)
        counts = [1 + (int(s) - 1) // int(st) for s, st in zip(ijk_size, ijk_step)]
        out = np.empty(counts[::-1], self.value_type)
        valid = []
        for o, st, n, dim in zip(ijk_origin, ijk_step, counts, dims):
            o, st = int(o), int(st)
            if o < 0 or st < 1:
                raise ValueError(f"bad region origin {ijk_origin} / step {ijk_step}")
            valid.append(0 if o >= dim else min(n, (dim - 1 - o) // st + 1))
        if valid != counts:
            out[...] = cast_fill(arr.fill_value, out.dtype)
        if min(valid) == 0:
            return out

        sel = [(0, 1, 1)] * arr.ndim
        if self.t_axis is not None:
            sel[self.t_axis] = (t, t + 1, 1)
        if self.c_axis is not None:
            sel[self.c_axis] = (c, c + 1, 1)
        for a, o, st, nv in zip("xyz", ijk_origin, ijk_step, valid):
            ai = self.space[a]
            if ai is not None:
                o, st = int(o), int(st)
                sel[ai] = (o, o + (nv - 1) * st + 1, st)
        view = out[:valid[2], :valid[1], :valid[0]]
        arr.read(sel, out=self._array_order_view(view), background=background, report=report)
        return out

    def _array_order_view(self, view):
        """View of a (z, y, x) block with the array's axis order (other axes length 1)."""
        if self.space["z"] is None:
            view = view[0]
            cur = [self.space["y"], self.space["x"]]
        else:
            cur = [self.space["z"], self.space["y"], self.space["x"]]
        others = [a for a in range(len(self.axes)) if a not in cur]
        view = view[(Ellipsis,) + (np.newaxis,) * len(others)]
        return view.transpose(np.argsort(cur + others))


def chimerax_value_type(dtype):
    """Native dtype ChimeraX can display for a stored dtype.

    Keeps (u)int8/16/32 and float32/64; float16 and (u)int64 become float32, bool uint8.
    """
    kind, size = dtype.kind, dtype.itemsize
    if kind == "b":
        return np.dtype(np.uint8)
    if (kind in "iu" and size <= 4) or (kind == "f" and size in (4, 8)):
        return dtype.newbyteorder("=")
    if kind in "iuf":
        return np.dtype(np.float32)
    raise NGFFError(f"data type {dtype} cannot be shown in ChimeraX")


# ---------------------------------------------------------------------------
# Opening


def open_image(path, series=None):
    """Image for the OME-Zarr store (or plain zarr array) at a local path."""
    root = os.path.abspath(os.path.expanduser(path.rstrip("/") or "/"))
    if not os.path.isdir(root):
        if os.path.exists(root):
            raise NGFFError(f"{path}: not a folder; an OME-Zarr store is a folder")
        raise NGFFError(f"{path}: no such folder (only local OME-Zarr stores can be opened)")
    try:
        fmt, kind, _ = node_info(root)
        if kind is None:
            raise NGFFError(f"{root}: not a zarr store (no zarr.json, .zgroup, .zattrs or "
                            ".zarray)")
        if kind == "array":
            if series is not None:
                raise NGFFError(f"{root}: is a single array; it has no series")
            return _bare_array_image(root)
        attrs = ngff_attributes(root)
        if series is not None:
            return _series_image(root, attrs, str(series))
        if "multiscales" in attrs:
            return _image_from_group(root, "", None)
        if "bioformats2raw.layout" in attrs:
            names = bioformats2raw_series(root)
            return _image_from_group(root, names[0], names[0])
        if "plate" in attrs or "well" in attrs:
            raise NGFFError(f"{root}: is an HCS plate/well; open one image group inside it "
                            "(HCS is not supported in this version)")
        raise NGFFError(f"{root}: no OME-NGFF image here (no multiscales, "
                        "bioformats2raw.layout or array)")
    except ZarrError as e:
        raise NGFFError(str(e)) from None


def ngff_attributes(path):
    """Group attributes with OME-NGFF 0.5's "ome" namespace merged into the top level."""
    attrs = dict(read_attributes(path))
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        attrs.update(ome)
    return attrs


def bioformats2raw_series(root):
    """Series group names of a bioformats2raw store: OME/ "series", else ["0"]."""
    names = ngff_attributes(os.path.join(root, "OME")).get("series")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return ["0"]


def _series_image(root, attrs, series):
    rel = series.strip("/")
    gpath = os.path.join(root, *rel.split("/"))
    _, kind, _ = node_info(gpath) if os.path.isdir(gpath) else (None, None, None)
    if kind != "group":
        if "bioformats2raw.layout" in attrs:
            avail = ", ".join(bioformats2raw_series(root))
            raise NGFFError(f"{root}: no series {series!r} (series: {avail})")
        raise NGFFError(f"{root}: no group {series!r} to open as a series")
    return _image_from_group(root, rel, rel)


def _bare_array_image(root):
    arr = ZarrArray(root)
    if not 2 <= arr.ndim <= 5:
        raise NGFFError(f"{root}: array has {arr.ndim} dimensions; 2 to 5 are supported")
    axes = _parse_axes(None, arr.ndim, root)
    one = [1.0] * arr.ndim
    level = Level("", arr, one, [0.0] * arr.ndim, False)
    channels = _channels({}, _channel_count(axes, arr.shape), [])
    return Image(root, os.path.basename(root), None, None, axes, [level], channels,
                 ["plain zarr array without OME-NGFF metadata: axes "
                  f"{''.join(a[0] for a in axes)}, voxel size 1"])


def _image_from_group(root, group_rel, series):
    gpath = os.path.join(root, *group_rel.split("/")) if group_rel else root
    attrs = ngff_attributes(gpath)
    ms_list = attrs.get("multiscales")
    if not ms_list:
        if "plate" in attrs or "well" in attrs:
            raise NGFFError(f"{gpath}: is an HCS plate/well (not supported in this version)")
        raise NGFFError(f"{gpath}: no multiscales metadata")
    warnings = []
    if len(ms_list) > 1:
        warnings.append(f"{len(ms_list)} multiscales in {gpath}; opening the first")
    ms = ms_list[0]
    version = str(ms.get("version") or attrs.get("version") or "")
    datasets = ms.get("datasets") or []
    if not datasets:
        raise NGFFError(f"{gpath}: multiscales has no datasets")

    arrays = []
    for ds in datasets:
        rel = posixpath.join(group_rel, ds["path"]) if group_rel else ds["path"]
        apath = os.path.join(root, *rel.split("/"))
        try:
            arrays.append((rel, ZarrArray(apath)))
        except (ZarrError, OSError, KeyError) as e:
            raise NGFFError(f"{root}: level {rel!r}: {e}") from None
    ndim = arrays[0][1].ndim
    for rel, a in arrays:
        if a.ndim != ndim:
            raise NGFFError(f"{root}: level {rel!r} has {a.ndim} dimensions, level 0 has {ndim}")
    axes = _parse_axes(ms.get("axes"), ndim, gpath)

    ms_transforms = ms.get("coordinateTransformations") or []
    levels = []
    for (rel, arr), ds in zip(arrays, datasets):
        transforms = ds.get("coordinateTransformations") or []
        scale, translation = _scale_translation(list(transforms) + list(ms_transforms), ndim,
                                                f"{gpath} level {rel!r}")
        levels.append(Level(rel, arr, scale, translation, bool(transforms or ms_transforms)))

    name = ms.get("name") or os.path.basename(root)
    channels = _channels(attrs, _channel_count(axes, arrays[0][1].shape), warnings)
    return Image(root, name, version, series, axes, levels, channels, warnings)


def _channel_count(axes, shape):
    for (_, typ, _), n in zip(axes, shape):
        if typ == "channel":
            return n
    return 1


def _parse_axes(axes, ndim, where):
    """[(name, type, unit)] from v0.4+ dicts, v0.3 strings, or "tczyx"[-ndim:] if missing."""
    if not axes:
        if ndim > 5:
            raise NGFFError(f"{where}: {ndim} dimensions and no axes metadata")
        axes = list("tczyx"[-ndim:])
    if len(axes) != ndim:
        raise NGFFError(f"{where}: {len(axes)} axes for {ndim}-dimensional arrays")
    out = []
    for a in axes:
        if isinstance(a, str):
            name, typ, unit = a, None, None
        elif isinstance(a, dict) and "name" in a:
            name, typ, unit = str(a["name"]), a.get("type"), a.get("unit")
        else:
            raise NGFFError(f"{where}: bad axis entry {a!r}")
        if not typ:
            typ = _TYPE_FROM_NAME.get(name.lower())
        out.append((name, typ, unit))
    return out


def _scale_translation(transforms, ndim, where):
    """Per-axis scale and translation of a sequence of NGFF transforms applied in order.

    Applying scale s after (s0, t0) gives (s0 * s, t0 * s); a translation d gives
    (s0, t0 + d). So dataset transforms followed by multiscale ones combine as
    scale s1 * s2 and translation t1 * s2 + t2.
    """
    scale, trans = [1.0] * ndim, [0.0] * ndim
    for tr in transforms:
        typ = tr.get("type")
        if typ == "identity":
            continue
        if typ not in ("scale", "translation"):
            raise NGFFError(f"{where}: unsupported coordinate transformation {typ!r}")
        values = tr.get(typ)
        if values is None:
            raise NGFFError(f"{where}: {typ} transformation without values "
                            "(transforms stored in files are not supported)")
        if len(values) != ndim:
            raise NGFFError(f"{where}: {typ} has {len(values)} values for {ndim} axes")
        values = [float(v) for v in values]
        if typ == "scale":
            scale = [s * v for s, v in zip(scale, values)]
            trans = [t * v for t, v in zip(trans, values)]
        else:
            trans = [t + v for t, v in zip(trans, values)]
    if any(not (s > 0 and math.isfinite(s)) for s in scale):
        raise NGFFError(f"{where}: scale {scale} must be positive")
    return scale, trans


def parse_color(value):
    """(r, g, b, a) floats from an OMERO "RRGGBB" or "RRGGBBAA" hex string, else None."""
    if not isinstance(value, str):
        return None
    h = value.strip().lstrip("#")
    if len(h) not in (6, 8):
        return None
    try:
        parts = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
    except ValueError:
        return None
    if len(parts) == 3:
        parts.append(1.0)
    return tuple(parts)


def _channels(attrs, nc, warnings):
    """Channel label, colour and window from OMERO metadata, with lab defaults."""
    omero = attrs.get("omero") if isinstance(attrs.get("omero"), dict) else {}
    entries = omero.get("channels") or []
    if entries and len(entries) != nc:
        warnings.append(f"omero lists {len(entries)} channels, the image has {nc}")
    channels = []
    for c in range(nc):
        e = entries[c] if c < len(entries) and isinstance(entries[c], dict) else {}
        rgba = parse_color(e.get("color"))
        if rgba is None and nc > 1:
            rgba = DEFAULT_COLORS[c] if c < len(DEFAULT_COLORS) else GRAY
        window = None
        w = e.get("window")
        if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None:
            try:
                start, end = float(w["start"]), float(w["end"])
            except (TypeError, ValueError):
                start = end = None
            if start is not None and end > start:
                window = (start, end)
        label = e.get("label")
        channels.append(Channel(str(label) if label else None, rgba, window,
                                bool(e.get("active", True))))
    return channels
